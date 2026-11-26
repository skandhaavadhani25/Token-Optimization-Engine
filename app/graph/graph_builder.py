"""Constructs a NetworkX MultiDiGraph code graph from MetadataStore."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import networkx as nx

from app.graph.call_resolver import CallResolver
from app.models.schemas import (
    CallResolutionStats,
    GraphBuildReport,
    ParsedFile,
)
from app.storage.metadata_store import MetadataStore


def build_graph(
    store: MetadataStore,
    repo_name: str,
    include_ambiguous: bool = False,
    include_external_calls: bool | None = None,
) -> tuple[nx.MultiDiGraph, GraphBuildReport]:
    """Build a NetworkX MultiDiGraph from repository metadata stored in the metadata index.

    Args:
        store: Opened MetadataStore holding repository AST records.
        repo_name: Repository name string.
        include_ambiguous: If True, link ambiguous calls to all candidate nodes.

    Returns:
        Tuple of (populated MultiDiGraph, GraphBuildReport).
    """
    start_time = time.time()
    graph = nx.MultiDiGraph()

    # Read last_indexed_at from metadata store for staleness tracking
    stats = store.get_stats()
    graph.graph["repo_name"] = repo_name
    graph.graph["last_indexed_at"] = stats.last_indexed_at
    graph.graph["repo_root"] = str(store.repo_root)

    # Load all parsed files from store
    all_files: list[ParsedFile] = store.load_all()

    if include_external_calls is None:
        include_external_calls = (len(all_files) <= 2)

    # Pre-index through CallResolver
    resolver = CallResolver(all_files)

    # -------------------------------------------------------------------------
    # 1. Create Nodes
    # -------------------------------------------------------------------------
    project_id = f"project:{repo_name}"
    graph.add_node(project_id, type="project", name=repo_name)

    # Collect folder hierarchy
    folders: set[str] = set()
    for pf in all_files:
        p = Path(pf.file_path).parent
        while p != Path(".") and str(p) != "":
            folders.add(p.as_posix())
            p = p.parent

    # Add folder nodes
    for fld in sorted(folders):
        graph.add_node(f"folder:{fld}", type="folder", path=fld)

    # Add file, class, function, method nodes
    for pf in all_files:
        file_id = f"file:{pf.file_path}"
        graph.add_node(
            file_id,
            type="file",
            path=pf.file_path,
            module_name=pf.module_name,
            line_count=pf.line_count,
            content_hash=pf.content_hash,
        )

        for cls in pf.classes:
            class_id = f"class:{cls.qualified_name}"
            graph.add_node(
                class_id,
                type="class",
                name=cls.name,
                qualified_name=cls.qualified_name,
                file=cls.file_path,
                start_line=cls.start_line,
                end_line=cls.end_line,
            )

            for method in cls.methods:
                method_id = f"method:{method.qualified_name}"
                graph.add_node(
                    method_id,
                    type="method",
                    name=method.name,
                    qualified_name=method.qualified_name,
                    file=method.file_path,
                    start_line=method.start_line,
                    end_line=method.end_line,
                    is_async=method.is_async,
                    parent_class=method.parent_class,
                )

        for fn in pf.functions:
            func_id = f"func:{fn.qualified_name}"
            graph.add_node(
                func_id,
                type="function",
                name=fn.name,
                qualified_name=fn.qualified_name,
                file=fn.file_path,
                start_line=fn.start_line,
                end_line=fn.end_line,
                is_async=fn.is_async,
            )

    # -------------------------------------------------------------------------
    # 2. Add CONTAINS and DEFINES Edges
    # -------------------------------------------------------------------------
    # CONTAINS: project -> folder, folder -> folder, folder -> file
    for fld in folders:
        parent_dir = Path(fld).parent.as_posix()
        if parent_dir in folders:
            graph.add_edge(f"folder:{parent_dir}", f"folder:{fld}", rel="CONTAINS")
        else:
            graph.add_edge(project_id, f"folder:{fld}", rel="CONTAINS")

    for pf in all_files:
        file_id = f"file:{pf.file_path}"
        parent_dir = Path(pf.file_path).parent.as_posix()
        if parent_dir in folders:
            graph.add_edge(f"folder:{parent_dir}", file_id, rel="CONTAINS")
        else:
            graph.add_edge(project_id, file_id, rel="CONTAINS")

        # DEFINES: file -> class, file -> function, class -> method
        for cls in pf.classes:
            class_id = f"class:{cls.qualified_name}"
            graph.add_edge(file_id, class_id, rel="DEFINES")
            for method in cls.methods:
                method_id = f"method:{method.qualified_name}"
                graph.add_edge(class_id, method_id, rel="DEFINES")

        for fn in pf.functions:
            func_id = f"func:{fn.qualified_name}"
            graph.add_edge(file_id, func_id, rel="DEFINES")

    # -------------------------------------------------------------------------
    # 3. Add IMPORTS and INHERITS Edges
    # -------------------------------------------------------------------------
    # Track cross-file references for DEPENDS_ON derivation
    # file_refs[(source_file_node, target_file_node)] = reference count
    file_refs: dict[tuple[str, str], int] = defaultdict(int)

    for pf in all_files:
        file_id = f"file:{pf.file_path}"
        import_table = resolver.file_imports.get(pf.file_path, {})
        seen_import_targets: set[str] = set()

        for local_name, r_imp in import_table.items():
            if r_imp.is_internal and r_imp.target_file:
                target_file_id = f"file:{r_imp.target_file}"
                if target_file_id != file_id:
                    file_refs[(file_id, target_file_id)] += 1
                if target_file_id not in seen_import_targets and target_file_id != file_id:
                    seen_import_targets.add(target_file_id)
                    graph.add_edge(file_id, target_file_id, rel="IMPORTS")
            elif not r_imp.is_internal:
                ext_name = r_imp.target_module.split(".")[0] if r_imp.target_module else local_name
                ext_id = f"external:{ext_name}"
                if ext_id not in graph:
                    graph.add_node(ext_id, type="external", name=ext_name)
                if ext_id not in seen_import_targets:
                    seen_import_targets.add(ext_id)
                    graph.add_edge(file_id, ext_id, rel="IMPORTS")

        # INHERITS edges
        for cls in pf.classes:
            class_id = f"class:{cls.qualified_name}"
            for base_expr in cls.bases:
                base_expr = base_expr.strip()
                if not base_expr:
                    continue

                # Find resolved base
                if "." not in base_expr:
                    if base_expr in resolver.classes_by_short_name_in_file.get(pf.file_path, {}):
                        cand = resolver.classes_by_short_name_in_file[pf.file_path][base_expr]
                        graph.add_edge(class_id, f"class:{cand.qualified_name}", rel="INHERITS")
                    elif base_expr in import_table:
                        r_imp = import_table[base_expr]
                        if r_imp.is_internal:
                            target_qname = f"{r_imp.target_module}.{r_imp.target_symbol or base_expr}"
                            if target_qname in resolver.classes_by_qname:
                                graph.add_edge(class_id, f"class:{target_qname}", rel="INHERITS")
                            else:
                                ext_id = f"external:{base_expr}"
                                if ext_id not in graph:
                                    graph.add_node(ext_id, type="external", name=base_expr)
                                graph.add_edge(class_id, ext_id, rel="INHERITS")
                        else:
                            ext_id = f"external:{base_expr}"
                            if ext_id not in graph:
                                graph.add_node(ext_id, type="external", name=base_expr)
                            graph.add_edge(class_id, ext_id, rel="INHERITS")
                    else:
                        ext_id = f"external:{base_expr}"
                        if ext_id not in graph:
                            graph.add_node(ext_id, type="external", name=base_expr)
                        graph.add_edge(class_id, ext_id, rel="INHERITS")
                else:
                    parts = base_expr.split(".")
                    prefix = parts[0]
                    suffix = ".".join(parts[1:])
                    if prefix in import_table:
                        r_imp = import_table[prefix]
                        if r_imp.is_internal:
                            target_qname = f"{r_imp.target_module}.{suffix}"
                            if target_qname in resolver.classes_by_qname:
                                graph.add_edge(class_id, f"class:{target_qname}", rel="INHERITS")
                            else:
                                ext_id = f"external:{base_expr}"
                                if ext_id not in graph:
                                    graph.add_node(ext_id, type="external", name=base_expr)
                                graph.add_edge(class_id, ext_id, rel="INHERITS")
                        else:
                            ext_id = f"external:{base_expr}"
                            if ext_id not in graph:
                                graph.add_node(ext_id, type="external", name=base_expr)
                            graph.add_edge(class_id, ext_id, rel="INHERITS")
                    else:
                        ext_id = f"external:{base_expr}"
                        if ext_id not in graph:
                            graph.add_node(ext_id, type="external", name=base_expr)
                        graph.add_edge(class_id, ext_id, rel="INHERITS")

    # -------------------------------------------------------------------------
    # 4. Resolve and Add CALLS Edges
    # -------------------------------------------------------------------------
    total_calls = 0
    resolved_calls = 0
    resolved_by_rule: dict[str, int] = defaultdict(int)
    skipped_ambiguous = 0
    external_calls = 0
    builtin_calls = 0
    unresolved_calls = 0

    for pf in all_files:
        caller_file_id = f"file:{pf.file_path}"
        all_callers = list(pf.functions) + [m for c in pf.classes for m in c.methods]

        for caller in all_callers:
            caller_node_id = (
                f"method:{caller.qualified_name}"
                if caller.is_method
                else f"func:{caller.qualified_name}"
            )

            for call in caller.calls:
                total_calls += 1
                result = resolver.resolve_call(
                    caller=caller,
                    call=call,
                    file_path=pf.file_path,
                    include_ambiguous=include_ambiguous,
                )

                if result.targets:
                    resolved_calls += 1
                    for target in result.targets:
                        resolved_by_rule[target.resolution] += 1
                        graph.add_edge(
                            caller_node_id,
                            target.node_id,
                            rel="CALLS",
                            lineno=call.lineno,
                            confidence=target.confidence,
                            resolution=target.resolution,
                        )
                        # Add cross-file reference for DEPENDS_ON
                        target_node_data = graph.nodes.get(target.node_id, {})
                        target_file = target_node_data.get("file")
                        if target_file:
                            target_file_id = f"file:{target_file}"
                            if target_file_id != caller_file_id:
                                file_refs[(caller_file_id, target_file_id)] += 1
                elif result.is_ambiguous_skipped:
                    skipped_ambiguous += 1
                elif result.is_external:
                    external_calls += 1
                    if include_external_calls:
                        ext_id = f"external:{call.name}"
                        if ext_id not in graph:
                            graph.add_node(
                                ext_id,
                                type="external",
                                name=call.name,
                                label=call.name,
                                note="external",
                            )
                        graph.add_edge(
                            caller_node_id,
                            ext_id,
                            rel="CALLS",
                            lineno=call.lineno,
                            confidence=0.3,
                            resolution="external",
                        )
                elif result.is_builtin:
                    builtin_calls += 1
                else:
                    unresolved_calls += 1
                    if include_external_calls:
                        unres_id = f"unresolved:{call.name}"
                        if unres_id not in graph:
                            graph.add_node(
                                unres_id,
                                type="unresolved",
                                name=call.name,
                                label=call.name,
                                note="unresolved",
                            )
                        graph.add_edge(
                            caller_node_id,
                            unres_id,
                            rel="CALLS",
                            lineno=call.lineno,
                            confidence=0.2,
                            resolution="unresolved",
                        )

    # -------------------------------------------------------------------------
    # 4b. Include text-only files as file nodes
    # -------------------------------------------------------------------------
    try:
        import os
        from app.workspace.manager import TEXT_ONLY_EXTENSIONS
        if store.repo_root.exists():
            for root_str, _, files in os.walk(store.repo_root):
                for f_name in files:
                    p = Path(root_str) / f_name
                    ext = p.suffix.lower()
                    if ext in TEXT_ONLY_EXTENSIONS:
                        try:
                            rel_p = p.relative_to(store.repo_root).as_posix()
                            file_id = f"file:{rel_p}"
                            if file_id not in graph:
                                graph.add_node(
                                    file_id,
                                    type="text_only",
                                    path=rel_p,
                                    file=rel_p,
                                    name=f_name,
                                    label=f"{f_name} (text-only)",
                                    line_count=0,
                                    note="text-only (no call graph)",
                                )
                                parent_dir = Path(rel_p).parent.as_posix()
                                if parent_dir in folders:
                                    graph.add_edge(f"folder:{parent_dir}", file_id, rel="CONTAINS")
                                else:
                                    graph.add_edge(project_id, file_id, rel="CONTAINS")
                        except Exception:
                            pass
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 5. Derive DEPENDS_ON Edges
    # -------------------------------------------------------------------------
    for (src_file_id, tgt_file_id), weight in file_refs.items():
        if src_file_id != tgt_file_id and weight > 0:
            graph.add_edge(src_file_id, tgt_file_id, rel="DEPENDS_ON", weight=weight)

    # -------------------------------------------------------------------------
    # 6. Build Report
    # -------------------------------------------------------------------------
    node_counts: dict[str, int] = defaultdict(int)
    for _, data in graph.nodes(data=True):
        node_counts[data.get("type", "unknown")] += 1

    edge_counts: dict[str, int] = defaultdict(int)
    for _, _, data in graph.edges(data=True):
        edge_counts[data.get("rel", "unknown")] += 1

    call_stats = CallResolutionStats(
        total_calls=total_calls,
        resolved_calls=resolved_calls,
        resolved_by_rule=dict(resolved_by_rule),
        skipped_ambiguous=skipped_ambiguous,
        external_calls=external_calls,
        builtin_calls=builtin_calls,
        unresolved_calls=unresolved_calls,
    )

    build_time = time.time() - start_time
    report = GraphBuildReport(
        repo_name=repo_name,
        total_nodes=graph.number_of_nodes(),
        total_edges=graph.number_of_edges(),
        node_counts=dict(node_counts),
        edge_counts=dict(edge_counts),
        call_resolution=call_stats,
        build_time_seconds=build_time,
    )

    return graph, report
