"""Production-grade incremental indexing for metadata store, code graph, and vector embeddings."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from app.embeddings.embedding_generator import BaseEmbeddingGenerator
from app.embeddings.indexer import build_embeddings
from app.graph.call_resolver import CallResolver
from app.graph.graph_builder import build_graph
from app.graph.graph_io import graph_exists, load_graph, save_graph
from app.indexer import build_index
from app.models.schemas import ParsedFile
from app.parser.python_parser import parse_file
from app.scanner.repository_scanner import scan_repository
from app.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class IncrementalReport(BaseModel):
    """Execution report for an incremental update operation."""

    model_config = ConfigDict(frozen=True)

    repo_path: str = Field(description="Root path of the repository")
    added: list[str] = Field(default_factory=list, description="Files newly discovered and indexed")
    modified: list[str] = Field(default_factory=list, description="Files whose content changed")
    deleted: list[str] = Field(default_factory=list, description="Files removed from the repository")
    files_reparsed: int = Field(default=0, description="Total files parsed during this update")
    graph_nodes_changed: int = Field(default=0, description="Graph nodes added, updated, or removed")
    graph_edges_rechecked: int = Field(default=0, description="Graph edges rechecked or re-resolved")
    items_reembedded: int = Field(default=0, description="Embedding items updated or generated")
    store_seconds: float = Field(default=0.0, description="Time spent updating the metadata store")
    graph_seconds: float = Field(default=0.0, description="Time spent updating code graph")
    embed_seconds: float = Field(default=0.0, description="Time spent updating embeddings")
    total_seconds: float = Field(default=0.0, description="Total wall-clock duration of update")
    verified: bool | None = Field(default=None, description="Equivalence verification result against full rebuild")


def _surgically_update_graph(
    store: MetadataStore,
    repo_name: str,
    repo_path: Path,
    added: list[str],
    modified: list[str],
    deleted: list[str],
    all_files: list[ParsedFile],
) -> tuple[nx.MultiDiGraph, int, int]:
    """Surgically update existing code graph for added, modified, and deleted files.

    Returns:
        tuple of (updated_graph, nodes_changed_count, edges_rechecked_count)
    """
    if not graph_exists(repo_path):
        # First-time build
        graph, _ = build_graph(store, repo_name)
        return graph, graph.number_of_nodes(), graph.number_of_edges()

    try:
        graph = load_graph(repo_path)
    except Exception as exc:
        logger.warning(f"Could not load existing graph ({exc}), rebuilding full graph.")
        graph, _ = build_graph(store, repo_name)
        return graph, graph.number_of_nodes(), graph.number_of_edges()

    changed_files = set(added) | set(modified) | set(deleted)
    if not changed_files:
        return graph, 0, 0

    nodes_changed = 0
    edges_rechecked = 0

    # 1. Identify callers into changed files to ensure cross-file CALLS edges are updated
    callers_into_changed: set[str] = set()
    for u, v, data in graph.edges(data=True):
        if data.get("rel") == "CALLS":
            v_file = graph.nodes[v].get("file") if v in graph else None
            u_file = graph.nodes[u].get("file") if u in graph else None
            if v_file in changed_files and u_file and u_file not in changed_files:
                callers_into_changed.add(u_file)

    # 2. Identify files that import any of the changed files
    importers_of_changed: set[str] = set()
    for u, v, data in graph.edges(data=True):
        if data.get("rel") in ("IMPORTS", "DEPENDS_ON"):
            u_path = graph.nodes[u].get("path") if u in graph else None
            v_path = graph.nodes[v].get("path") if v in graph else None
            if v_path in changed_files and u_path and u_path not in changed_files:
                importers_of_changed.add(u_path)

    files_to_relink = set(added) | set(modified) | callers_into_changed | importers_of_changed

    # 3. Remove old nodes belonging to modified or deleted files
    nodes_to_remove = []
    for node_id, data in list(graph.nodes(data=True)):
        f = data.get("file") or data.get("path")
        if f in (set(modified) | set(deleted)):
            nodes_to_remove.append(node_id)

    for n in nodes_to_remove:
        edges_rechecked += graph.degree(n)
        graph.remove_node(n)
        nodes_changed += 1

    # 4. Remove outgoing/incoming edges for files_to_relink that need re-resolution
    edges_to_remove = []
    for u, v, key, data in list(graph.edges(keys=True, data=True)):
        rel = data.get("rel")
        u_file = graph.nodes[u].get("file") or graph.nodes[u].get("path") if u in graph else None
        v_file = graph.nodes[v].get("file") or graph.nodes[v].get("path") if v in graph else None

        if rel in ("CALLS", "IMPORTS", "INHERITS"):
            if u_file in files_to_relink or v_file in changed_files:
                edges_to_remove.append((u, v, key))
        elif rel == "DEPENDS_ON":
            edges_to_remove.append((u, v, key))

    for u, v, key in edges_to_remove:
        if graph.has_edge(u, v, key=key):
            graph.remove_edge(u, v, key=key)
            edges_rechecked += 1

    # 5. Build full CallResolver over current metadata
    resolver = CallResolver(all_files)
    parsed_by_path = {pf.file_path: pf for pf in all_files}

    # 6. Add/re-add nodes for added and modified files
    project_id = f"project:{repo_name}"
    for rel_p in (set(added) | set(modified)):
        pf = parsed_by_path.get(rel_p)
        if not pf:
            continue

        file_id = f"file:{pf.file_path}"
        graph.add_node(
            file_id,
            type="file",
            path=pf.file_path,
            module_name=pf.module_name,
            line_count=pf.line_count,
            content_hash=pf.content_hash,
        )
        nodes_changed += 1

        # Parent folder hierarchy
        p = Path(pf.file_path).parent
        curr_p = p
        while curr_p != Path(".") and str(curr_p) != "":
            fld_str = curr_p.as_posix()
            fld_id = f"folder:{fld_str}"
            if fld_id not in graph:
                graph.add_node(fld_id, type="folder", path=fld_str)
                nodes_changed += 1
                parent_dir = curr_p.parent
                if parent_dir != Path(".") and str(parent_dir) != "":
                    graph.add_edge(f"folder:{parent_dir.as_posix()}", fld_id, rel="CONTAINS")
                else:
                    graph.add_edge(project_id, fld_id, rel="CONTAINS")
            curr_p = curr_p.parent

        if p != Path(".") and str(p) != "":
            graph.add_edge(f"folder:{p.as_posix()}", file_id, rel="CONTAINS")
        else:
            graph.add_edge(project_id, file_id, rel="CONTAINS")

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
            graph.add_edge(file_id, class_id, rel="DEFINES")
            nodes_changed += 1

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
                graph.add_edge(class_id, method_id, rel="DEFINES")
                nodes_changed += 1

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
            graph.add_edge(file_id, func_id, rel="DEFINES")
            nodes_changed += 1

    # 7. Re-link IMPORTS, INHERITS, CALLS, and DEPENDS_ON for files_to_relink
    file_refs: dict[tuple[str, str], int] = defaultdict(int)

    for rel_p in files_to_relink:
        pf = parsed_by_path.get(rel_p)
        if not pf:
            continue

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
                    edges_rechecked += 1
            elif not r_imp.is_internal:
                ext_name = r_imp.target_module.split(".")[0] if r_imp.target_module else local_name
                ext_id = f"external:{ext_name}"
                if ext_id not in graph:
                    graph.add_node(ext_id, type="external", name=ext_name)
                    nodes_changed += 1
                if ext_id not in seen_import_targets:
                    seen_import_targets.add(ext_id)
                    graph.add_edge(file_id, ext_id, rel="IMPORTS")
                    edges_rechecked += 1

        # INHERITS edges
        for cls in pf.classes:
            class_id = f"class:{cls.qualified_name}"
            for base_expr in cls.bases:
                base_expr = base_expr.strip()
                if not base_expr:
                    continue
                if "." not in base_expr:
                    if base_expr in resolver.classes_by_short_name_in_file.get(pf.file_path, {}):
                        cand = resolver.classes_by_short_name_in_file[pf.file_path][base_expr]
                        graph.add_edge(class_id, f"class:{cand.qualified_name}", rel="INHERITS")
                        edges_rechecked += 1
                    elif base_expr in import_table:
                        r_imp = import_table[base_expr]
                        if r_imp.is_internal:
                            target_qname = f"{r_imp.target_module}.{r_imp.target_symbol or base_expr}"
                            if target_qname in resolver.classes_by_qname:
                                graph.add_edge(class_id, f"class:{target_qname}", rel="INHERITS")
                                edges_rechecked += 1
                            else:
                                ext_id = f"external:{base_expr}"
                                if ext_id not in graph:
                                    graph.add_node(ext_id, type="external", name=base_expr)
                                graph.add_edge(class_id, ext_id, rel="INHERITS")
                                edges_rechecked += 1
                        else:
                            ext_id = f"external:{base_expr}"
                            if ext_id not in graph:
                                graph.add_node(ext_id, type="external", name=base_expr)
                            graph.add_edge(class_id, ext_id, rel="INHERITS")
                            edges_rechecked += 1
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
                                edges_rechecked += 1
                            else:
                                ext_id = f"external:{base_expr}"
                                if ext_id not in graph:
                                    graph.add_node(ext_id, type="external", name=base_expr)
                                graph.add_edge(class_id, ext_id, rel="INHERITS")
                                edges_rechecked += 1

        # CALLS edges
        all_callers = list(pf.functions) + [m for c in pf.classes for m in c.methods]
        for caller in all_callers:
            caller_node_id = (
                f"method:{caller.qualified_name}"
                if caller.is_method
                else f"func:{caller.qualified_name}"
            )
            for call in caller.calls:
                res = resolver.resolve_call(caller=caller, call=call, file_path=pf.file_path)
                if res.targets:
                    for target in res.targets:
                        graph.add_edge(
                            caller_node_id,
                            target.node_id,
                            rel="CALLS",
                            lineno=call.lineno,
                            confidence=target.confidence,
                            resolution=target.resolution,
                        )
                        edges_rechecked += 1
                        target_node_data = graph.nodes.get(target.node_id, {})
                        target_file = target_node_data.get("file")
                        if target_file:
                            target_file_id = f"file:{target_file}"
                            if target_file_id != file_id:
                                file_refs[(file_id, target_file_id)] += 1

    # Re-derive DEPENDS_ON edges from IMPORTS and CALLS
    file_refs: dict[tuple[str, str], int] = defaultdict(int)
    for u, v, data in graph.edges(data=True):
        rel = data.get("rel")
        if rel == "IMPORTS":
            if u.startswith("file:") and v.startswith("file:") and u != v:
                file_refs[(u, v)] += 1
        elif rel == "CALLS":
            u_node = graph.nodes.get(u, {})
            v_node = graph.nodes.get(v, {})
            u_file = u_node.get("file")
            v_file = v_node.get("file")
            if u_file and v_file and u_file != v_file:
                u_fid = f"file:{u_file}"
                v_fid = f"file:{v_file}"
                file_refs[(u_fid, v_fid)] += 1

    for (src_file_id, tgt_file_id), weight in file_refs.items():
        if src_file_id != tgt_file_id and weight > 0:
            graph.add_edge(src_file_id, tgt_file_id, rel="DEPENDS_ON", weight=weight)
            edges_rechecked += 1

    stats = store.get_stats()
    graph.graph["last_indexed_at"] = stats.last_indexed_at
    return graph, nodes_changed, edges_rechecked


def verify_equivalence(repo_path: Path | str) -> bool:
    """Verify that current incremental state is byte-for-byte/entity-equivalent to a full rebuild.

    Runs a full rebuild into a scratch namespace (dropped afterwards) and asserts:
    1. Metadata store entity counts match.
    2. Graph nodes and edges match identically.
    """
    repo = Path(repo_path).resolve()
    scratch_ns = f"verify-{uuid.uuid4().hex[:10]}"
    try:
        # 1. Full build into a scratch namespace
        build_index(repo_path=repo, force=True, namespace=scratch_ns)
        with MetadataStore.open(repo, namespace=scratch_ns) as temp_store:
            temp_graph, _ = build_graph(temp_store, repo.name)

        # 2. Compare against active store
        with MetadataStore.open(repo) as active_store:
            active_stats = active_store.get_stats()
            with MetadataStore.open(repo, namespace=scratch_ns) as temp_store:
                temp_stats = temp_store.get_stats()

            assert active_stats.total_files == temp_stats.total_files, (
                f"File count mismatch: active={active_stats.total_files}, full={temp_stats.total_files}"
            )
            assert active_stats.total_functions == temp_stats.total_functions, (
                f"Function count mismatch: active={active_stats.total_functions}, full={temp_stats.total_functions}"
            )
            assert active_stats.total_classes == temp_stats.total_classes, (
                f"Class count mismatch: active={active_stats.total_classes}, full={temp_stats.total_classes}"
            )

        # 3. Compare graph
        active_graph = load_graph(repo)
        active_nodes = set(active_graph.nodes())
        temp_nodes = set(temp_graph.nodes())
        assert active_nodes == temp_nodes, (
            f"Graph node set mismatch: missing in active={temp_nodes - active_nodes}, "
            f"extra in active={active_nodes - temp_nodes}"
        )

        active_edges = {
            (u, v, data.get("rel"))
            for u, v, data in active_graph.edges(data=True)
        }
        temp_edges = {
            (u, v, data.get("rel"))
            for u, v, data in temp_graph.edges(data=True)
        }
        assert active_edges == temp_edges, (
            f"Graph edge set mismatch: missing in active={temp_edges - active_edges}, "
            f"extra in active={active_edges - temp_edges}"
        )
    finally:
        MetadataStore.drop(repo, scratch_ns)

    return True


def update_index(
    repo_path: Path | str,
    verify: bool = False,
    generator: BaseEmbeddingGenerator | None = None,
    namespace: str | None = None,
) -> IncrementalReport:
    """Perform production incremental update across metadata store, graph, and embeddings.

    Args:
        repo_path: Path to root repository directory.
        verify: If True, asserts equivalence against a fresh full rebuild in temp storage.
        generator: Optional embedding generator (useful for tests using FakeEmbeddingGenerator).
        namespace: Optional explicit namespace for the stored vector index.

    Returns:
        IncrementalReport detailing changed files, nodes, edges, embeddings, and timings.
    """
    total_start = time.perf_counter()
    repo = Path(repo_path).resolve()

    # 1. Scan and detect diff
    scan_res = scan_repository(repo)
    scan_map = {f.path: f for f in scan_res.files}

    store_start = time.perf_counter()
    with MetadataStore.open(repo) as store:
        changes = store.diff_against_scan(scan_res)
        added = changes.added
        modified = changes.modified
        deleted = changes.deleted

        if not added and not modified and not deleted:
            # Idempotent fast exit
            store_sec = time.perf_counter() - store_start
            verified_result = verify_equivalence(repo) if verify else None
            return IncrementalReport(
                repo_path=str(repo),
                added=[],
                modified=[],
                deleted=[],
                files_reparsed=0,
                graph_nodes_changed=0,
                graph_edges_rechecked=0,
                items_reembedded=0,
                store_seconds=round(store_sec, 4),
                graph_seconds=0.0,
                embed_seconds=0.0,
                total_seconds=round(time.perf_counter() - total_start, 4),
                verified=verified_result,
            )

        # Apply store updates
        for d in deleted:
            store.delete_file(d)

        for m in modified:
            store.delete_file(m)

        to_parse = added + modified
        for p in to_parse:
            f_info = scan_map[p]
            parsed = parse_file(abs_path=f_info.absolute_path, rel_path=p)
            store.save_parsed_file(
                parsed=parsed,
                size_bytes=f_info.size_bytes,
                last_modified=f_info.last_modified,
            )

        all_parsed = store.load_all()
    store_sec = time.perf_counter() - store_start

    # 2. Update Code Graph
    graph_start = time.perf_counter()
    with MetadataStore.open(repo) as store:
        graph, nodes_changed, edges_rechecked = _surgically_update_graph(
            store=store,
            repo_name=repo.name,
            repo_path=repo,
            added=added,
            modified=modified,
            deleted=deleted,
            all_files=all_parsed,
        )
        save_graph(graph, repo_path=repo)
    graph_sec = time.perf_counter() - graph_start

    # 3. Update Vector Embeddings
    embed_start = time.perf_counter()
    emb_report = build_embeddings(
        repo_path=repo,
        force=False,
        generator=generator,
        show_progress=False,
        namespace=namespace,
    )
    items_reembedded = emb_report.items_new + emb_report.items_changed
    embed_sec = time.perf_counter() - embed_start

    # 4. Optional Equivalence Verification
    verified_result = None
    if verify:
        verified_result = verify_equivalence(repo)

    total_sec = time.perf_counter() - total_start
    return IncrementalReport(
        repo_path=str(repo),
        added=added,
        modified=modified,
        deleted=deleted,
        files_reparsed=len(to_parse),
        graph_nodes_changed=nodes_changed,
        graph_edges_rechecked=edges_rechecked,
        items_reembedded=items_reembedded,
        store_seconds=round(store_sec, 4),
        graph_seconds=round(graph_sec, 4),
        embed_seconds=round(embed_sec, 4),
        total_seconds=round(total_sec, 4),
        verified=verified_result,
    )
