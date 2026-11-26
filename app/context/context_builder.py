"""Context builder assembling deterministic, token-budgeted prompt contexts."""

from __future__ import annotations

import re
import time
from typing import Any

from app.context.context_compressor import compress, compress_to_fit
from app.context.formatter import render_context
from app.context.token_budget import BudgetTooSmallError, TokenBudget
from app.context.token_counter import TokenCounter, get_counter
from app.graph.graph_search import CodeGraph
from app.models.schemas import (
    BuilderOptions,
    BuiltContext,
    ClassInfo,
    DetailLevel,
    FunctionInfo,
    IncludedItem,
    RankedCandidate,
    RetrievalResult,
)
from app.storage.metadata_store import MetadataStore


class ContextBuilder:
    """Orchestrates progressive compression and greedy packing into a token budget."""

    def __init__(
        self,
        metadata_store: MetadataStore,
        graph: CodeGraph | Any | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.graph = graph if isinstance(graph, CodeGraph) else (CodeGraph(graph) if graph is not None else None)
        self.counter = counter or get_counter()

    def build(
        self,
        retrieval_result: RetrievalResult,
        budget: TokenBudget | int | None = None,
        options: BuilderOptions | None = None,
    ) -> BuiltContext:
        """Assemble structured, token-bounded context from retrieval results.

        Args:
            retrieval_result: Ranked candidates and task query from hybrid retrieval.
            budget: Optional TokenBudget or integer token limit (default: 4000).
            options: Optional BuilderOptions configuration.

        Returns:
            BuiltContext holding final rendered text and detailed item statistics.

        Raises:
            BudgetTooSmallError: If budget cannot fit minimal task + target context.
        """
        start_time = time.time()
        options = options or BuilderOptions()

        if budget is None:
            tb = TokenBudget(total_budget=4000)
        elif isinstance(budget, int):
            tb = TokenBudget(total_budget=budget)
        else:
            tb = budget

        # 1. Reserved budget for task
        task_tokens = self.counter.count(retrieval_result.query)
        tb.reserved_for_task = task_tokens

        # Check for impossibly small budget
        min_target_cost = 10
        min_overhead = 25
        if tb.total_budget < task_tokens + min_target_cost + min_overhead:
            raise BudgetTooSmallError(
                f"Budget {tb.total_budget} is too small to fit minimal task + target ({task_tokens + min_target_cost + min_overhead} tokens required)."
            )

        # 2. Identify Target Entity
        target_ref: str | None = retrieval_result.target
        target_uncertain: bool = False

        if not target_ref and retrieval_result.candidates:
            target_ref = retrieval_result.candidates[0].ref
            target_uncertain = True
        elif retrieval_result.target_confidence == "low":
            target_uncertain = True

        target_entity: FunctionInfo | ClassInfo | None = None
        target_location: str | None = None
        if target_ref and self.metadata_store is not None:
            target_entity = self.metadata_store.get_function(target_ref)
            if not target_entity:
                target_entity = self.metadata_store.get_class(target_ref)
            if target_entity:
                target_location = f"{target_entity.file_path}:{target_entity.start_line}-{target_entity.end_line}"

        # 3. Target Level Selection
        target_max_tokens = tb.target_max_tokens()
        target_level = DetailLevel.FULL

        if target_entity is not None:
            full_cost = self.counter.count(target_entity.source)
            if full_cost > target_max_tokens:
                snip = compress_to_fit(target_entity, target_max_tokens, counter=self.counter)
                target_level = snip.level if snip else DetailLevel.REFERENCE
        else:
            target_level = DetailLevel.REFERENCE

        # Target item record
        included_items: list[IncludedItem] = []
        downgraded_count = 0
        omitted_items: list[dict[str, str]] = []
        warnings: list[str] = []
        if target_uncertain:
            warnings.append("Low confidence: no strong match found")

        if target_ref and target_entity:
            t_snippet = compress(target_entity, target_level, counter=self.counter)
            included_items.append(
                IncludedItem(
                    ref=target_ref,
                    file=target_entity.file_path,
                    lines=f"{target_entity.start_line}-{target_entity.end_line}",
                    level=target_level.value,
                    tokens=t_snippet.tokens,
                    original_tokens=t_snippet.original_tokens,
                    role="target",
                    score=1.0,
                    reason="Primary task target",
                )
            )

        # 4. Check if target is a method inside a class (record class context)
        parent_class_entity: ClassInfo | None = None
        if isinstance(target_entity, FunctionInfo) and target_entity.parent_class:
            parent_qname = target_entity.qualified_name.rsplit(".", 1)[0]
            parent_cls = self.metadata_store.get_class(parent_qname)
            if not parent_cls:
                try:
                    for cand_cls in self.metadata_store.load_parsed_file(target_entity.file_path).classes:
                        if cand_cls.name == target_entity.parent_class:
                            parent_cls = cand_cls
                            break
                except Exception:
                    pass
            if parent_cls:
                other_methods = [m for m in parent_cls.methods if m.name != target_entity.name]
                parent_class_entity = ClassInfo(
                    name=parent_cls.name,
                    qualified_name=parent_cls.qualified_name,
                    file_path=parent_cls.file_path,
                    start_line=parent_cls.start_line,
                    end_line=parent_cls.end_line,
                    bases=parent_cls.bases,
                    decorators=parent_cls.decorators,
                    docstring=parent_cls.docstring,
                    methods=other_methods,
                    source=parent_cls.source,
                )

        # 5. Analyze Callees and Callers for Role Assignment
        target_callees: set[str] = set()
        target_callers: set[str] = set()

        if target_ref and self.graph:
            try:
                target_nids = self.graph.resolve_node(target_ref)
                if target_nids:
                    t_nid = target_nids[0]
                    for c in self.graph.callees(t_nid, depth=1):
                        target_callees.add(c.qualified_name)
                        target_callees.add(c.name)
                        target_callees.add(c.node_id)
                    for c in self.graph.callers(t_nid, depth=1):
                        target_callers.add(c.qualified_name)
                        target_callers.add(c.name)
                        target_callers.add(c.node_id)
            except Exception:
                pass

        if target_entity and isinstance(target_entity, FunctionInfo):
            for call in target_entity.calls:
                target_callees.add(call.name)

        # 6. Candidate Pool Processing
        pinned_set = set(options.must_include)
        exclude_set = set(options.exclude)

        candidates_to_pack: list[dict[str, Any]] = []

        # If target has a parent class, prioritize class context
        if parent_class_entity is not None and parent_class_entity.qualified_name not in exclude_set:
            sig_snip = compress(parent_class_entity, DetailLevel.SIGNATURE, counter=self.counter)
            candidates_to_pack.append({
                "cand": RankedCandidate(
                    ref=parent_class_entity.qualified_name,
                    score=0.99,
                    entity_type="class",
                    file=parent_class_entity.file_path,
                    start_line=parent_class_entity.start_line,
                    end_line=parent_class_entity.end_line,
                ),
                "entity": parent_class_entity,
                "role": "related",
                "aim_level": DetailLevel.SIGNATURE,
                "snippet": sig_snip,
                "score": 0.99,
                "is_pinned": True,
                "vpt": 0.99 / max(1, sig_snip.tokens),
            })

        for cand in retrieval_result.candidates:
            ref = cand.ref
            if ref == target_ref or ref in exclude_set:
                continue

            entity = self.metadata_store.get_function(ref)
            if not entity:
                entity = self.metadata_store.get_class(ref)
            if not entity:
                continue

            # Role
            if ref in target_callees or any(c in ref for c in target_callees):
                role = "callee"
            elif ref in target_callers or any(c in ref for c in target_callers):
                role = "caller"
            else:
                role = "related"

            if role == "caller" and not options.include_callers and ref not in pinned_set:
                continue

            # Aim detail level
            if ref in pinned_set:
                aim_level = DetailLevel.TRIMMED
            elif role == "callee" or cand.match_type == "exact":
                aim_level = DetailLevel(options.callee_detail.lower())
            elif role == "caller":
                # Intent-aware: for explain / fix_bug, callers promoted to trimmed
                if retrieval_result.intent in ("explain", "fix_bug"):
                    aim_level = DetailLevel.TRIMMED
                else:
                    aim_level = DetailLevel(options.caller_detail.lower())
            else:
                if cand.score >= 0.7:
                    aim_level = DetailLevel.SIGNATURE
                else:
                    aim_level = DetailLevel(options.other_detail.lower())

            snippet = compress(entity, aim_level, counter=self.counter)
            vpt = cand.score / max(1, snippet.tokens)

            candidates_to_pack.append({
                "cand": cand,
                "entity": entity,
                "role": role,
                "aim_level": aim_level,
                "snippet": snippet,
                "score": cand.score,
                "is_pinned": ref in pinned_set,
                "vpt": vpt,
            })

        # Add any must_include pinned items that were not in retrieval candidates
        if self.metadata_store is not None:
            existing_packed_refs = {c["cand"].ref for c in candidates_to_pack}
            for pinned_ref in options.must_include:
                if pinned_ref == target_ref or pinned_ref in exclude_set or pinned_ref in existing_packed_refs:
                    continue
                pinned_ent = self.metadata_store.get_function(pinned_ref) or self.metadata_store.get_class(pinned_ref)
                if pinned_ent:
                    p_snip = compress(pinned_ent, DetailLevel.TRIMMED, counter=self.counter)
                    candidates_to_pack.append({
                        "cand": RankedCandidate(
                            ref=pinned_ref,
                            score=1.0,
                            entity_type="function" if isinstance(pinned_ent, FunctionInfo) else "class",
                            file=pinned_ent.file_path,
                            start_line=pinned_ent.start_line,
                            end_line=pinned_ent.end_line,
                            match_type="pinned",
                        ),
                        "entity": pinned_ent,
                        "role": "related",
                        "aim_level": DetailLevel.TRIMMED,
                        "snippet": p_snip,
                        "score": 1.0,
                        "is_pinned": True,
                        "vpt": 1.0 / max(1, p_snip.tokens),
                    })

        # Order candidates: pinned first, then descending value-per-token
        candidates_to_pack.sort(key=lambda c: (1 if c["is_pinned"] else 0, c["vpt"]), reverse=True)

        # 7. Pack Candidates into Budget (Greedy with Progressive Downgrade)
        # Conservative budget ceiling for greedy pass
        available_budget = int(tb.total_budget * options.max_utilization) - tb.reserved_overhead - tb.reserved_for_task

        current_tokens = sum(item.tokens for item in included_items)
        level_order = [
            DetailLevel.TRIMMED,
            DetailLevel.SIGNATURE,
            DetailLevel.STUB,
            DetailLevel.REFERENCE,
        ]

        for item_data in candidates_to_pack:
            if len(included_items) >= options.max_items and not item_data["is_pinned"]:
                omitted_items.append({"ref": item_data["cand"].ref, "reason": "max_items_limit"})
                continue

            cand = item_data["cand"]
            entity = item_data["entity"]
            role = item_data["role"]
            aim_level = item_data["aim_level"]

            # Try from aim_level downwards
            start_idx = level_order.index(aim_level) if aim_level in level_order else 0
            placed = False

            for lvl in level_order[start_idx:]:
                snip = compress(entity, lvl, counter=self.counter)
                cost = snip.tokens
                if current_tokens + cost <= available_budget or item_data["is_pinned"]:
                    included_items.append(
                        IncludedItem(
                            ref=cand.ref,
                            file=entity.file_path,
                            lines=f"{entity.start_line}-{entity.end_line}",
                            level=lvl.value,
                            tokens=cost,
                            original_tokens=snip.original_tokens,
                            role=role,
                            score=cand.score,
                            reason=(
                                f"Selected as {role} (aim {aim_level.value})"
                                if lvl == aim_level
                                else f"Selected as {role}; downgraded {aim_level.value} -> {lvl.value} to fit budget"
                            ),
                        )
                    )
                    current_tokens += cost
                    if lvl != aim_level:
                        downgraded_count += 1
                    placed = True
                    break

            if not placed:
                omitted_items.append({"ref": cand.ref, "reason": "budget_exceeded"})

        # 7. Upgrade Pass
        # If budget remains, upgrade highest-scoring downgraded items one level up
        for _ in range(3):
            upgraded_any = False
            for item in sorted(included_items, key=lambda it: it.score, reverse=True):
                if item.role == "target":
                    continue
                cur_lvl = DetailLevel(item.level)
                if cur_lvl in (DetailLevel.FULL, DetailLevel.TRIMMED):
                    continue

                entity = self.metadata_store.get_function(item.ref) or self.metadata_store.get_class(item.ref)
                if not entity:
                    continue

                higher_lvl = DetailLevel.TRIMMED if cur_lvl == DetailLevel.SIGNATURE else DetailLevel.SIGNATURE
                higher_snip = compress(entity, higher_lvl, counter=self.counter)
                diff = higher_snip.tokens - item.tokens

                if current_tokens + diff <= available_budget:
                    item.level = higher_lvl.value
                    item.tokens = higher_snip.tokens
                    current_tokens += diff
                    if downgraded_count > 0:
                        downgraded_count -= 1
                    upgraded_any = True
            if not upgraded_any:
                break

        # 8. Dependency Paths
        dependency_paths: list[str] = []
        if options.include_dependency_path and target_ref and self.graph:
            target_nids = self.graph.resolve_node(target_ref)
            if target_nids:
                t_nid = target_nids[0]
                related_refs = [
                    it.ref for it in included_items
                    if it.role in ("callee", "related") and it.ref != target_ref
                ]
                for rref in related_refs[:3]:
                    r_nids = self.graph.resolve_node(rref)
                    if r_nids:
                        path = self.graph.dependency_path(t_nid, r_nids[0])
                        if path and len(path) > 1:
                            clean_parts = []
                            for p in path:
                                p_data = self.graph.graph.nodes.get(p, {})
                                name_val = p_data.get("name") or p.split(":")[-1].split(".")[-1]
                                clean_parts.append(name_val)
                            clean_path = " -> ".join(clean_parts)
                            if clean_path not in dependency_paths:
                                dependency_paths.append(clean_path)

        # 9. Extract Relevant Imports
        relevant_imports: dict[str, list[str]] = {}
        if options.include_imports:
            files_used = {it.file for it in included_items}
            for fpath in sorted(files_used):
                file_imports = self.metadata_store.find_imports_by_file(fpath)
                if not file_imports:
                    continue

                # Filter imports to symbols that appear in the included snippets for this file
                file_snippets = [
                    self._get_snippet_text(it)
                    for it in included_items
                    if it.file == fpath
                ]
                combined_text = "\n".join(file_snippets)

                used_names: list[str] = []
                for imp in file_imports:
                    name_to_check = imp.alias or imp.name or imp.module.split(".")[-1]
                    # Word boundary check
                    if re.search(r"\b" + re.escape(name_to_check) + r"\b", combined_text):
                        used_names.append(imp.name or imp.module)

                if used_names:
                    relevant_imports[fpath] = sorted(list(dict.fromkeys(used_names)))

        # 10. Group Code Snippets by File
        code_snippets_by_file = self._assemble_snippets_by_file(included_items)

        # 11. Final Verification & Downgrade Loop (Enforce Hard Invariant)
        rendered_text = render_context(
            task=retrieval_result.query,
            target_ref=target_ref,
            target_location=target_location,
            dependency_paths=dependency_paths,
            related_items=[it for it in included_items if it.role != "target"],
            relevant_imports=relevant_imports,
            code_snippets_by_file=code_snippets_by_file,
            omitted_items=omitted_items,
            format_type="text",
        )
        total_tokens = self.counter.count(rendered_text)

        # Progressive post-render downgrade loop if over budget
        iteration = 0
        while total_tokens > tb.total_budget and iteration < 100:
            iteration += 1
            # Downgrade lowest scoring, highest token non-target item first
            non_targets = [it for it in included_items if it.role != "target"]
            if not non_targets:
                # Even target + task is over budget; downgrade target progressively
                if included_items:
                    t_it = included_items[0]
                    t_cur = DetailLevel(t_it.level)
                    if t_cur == DetailLevel.FULL:
                        t_it.level = DetailLevel.TRIMMED.value
                        downgraded_count += 1
                    elif t_cur == DetailLevel.TRIMMED:
                        t_it.level = DetailLevel.SIGNATURE.value
                        downgraded_count += 1
                    elif t_cur == DetailLevel.SIGNATURE:
                        t_it.level = DetailLevel.STUB.value
                        downgraded_count += 1
                    elif t_cur == DetailLevel.STUB:
                        t_it.level = DetailLevel.REFERENCE.value
                        downgraded_count += 1
                    else:
                        raise BudgetTooSmallError(
                            f"Budget {tb.total_budget} is too small to fit minimal context ({total_tokens} tokens)."
                        )
                else:
                    raise BudgetTooSmallError(
                        f"Budget {tb.total_budget} is too small to fit minimal context ({total_tokens} tokens)."
                    )
            else:
                # Sort lowest score first, breaking ties by highest tokens
                non_targets.sort(key=lambda it: (it.score, -it.tokens))
                downgraded = False
                for it in non_targets:
                    cur = DetailLevel(it.level)
                    if cur == DetailLevel.FULL:
                        it.level = DetailLevel.TRIMMED.value
                        downgraded = True
                    elif cur == DetailLevel.TRIMMED:
                        it.level = DetailLevel.SIGNATURE.value
                        downgraded = True
                    elif cur == DetailLevel.SIGNATURE:
                        it.level = DetailLevel.STUB.value
                        downgraded = True
                    elif cur == DetailLevel.STUB:
                        it.level = DetailLevel.REFERENCE.value
                        downgraded = True
                    elif cur == DetailLevel.REFERENCE:
                        # Drop completely
                        included_items.remove(it)
                        omitted_items.append({"ref": it.ref, "reason": "post_verification_budget_cutoff"})
                        downgraded = True

                    if downgraded:
                        downgraded_count += 1
                        break

            # Re-assemble snippets and re-render
            code_snippets_by_file = self._assemble_snippets_by_file(included_items)
            rendered_text = render_context(
                task=retrieval_result.query,
                target_ref=target_ref,
                target_location=target_location,
                dependency_paths=dependency_paths,
                related_items=[it for it in included_items if it.role != "target"],
                relevant_imports=relevant_imports,
                code_snippets_by_file=code_snippets_by_file,
                omitted_items=omitted_items,
                format_type="text",
            )
            total_tokens = self.counter.count(rendered_text)

        if total_tokens > tb.total_budget:
            raise BudgetTooSmallError(
                f"Budget {tb.total_budget} is too small to fit minimal context ({total_tokens} tokens)."
            )

        build_time = time.time() - start_time
        files_included = sorted(list({it.file for it in included_items}))
        utilization = round(total_tokens / tb.total_budget, 4)

        return BuiltContext(
            text=rendered_text,
            tokens=total_tokens,
            budget_total=tb.total_budget,
            utilization=utilization,
            target=target_ref,
            target_uncertain=target_uncertain,
            items=included_items,
            omitted_items=omitted_items,
            downgraded_count=downgraded_count,
            files_included=files_included,
            counter_name=self.counter.name,
            build_time_seconds=round(build_time, 4),
            warnings=warnings,
        )

    def _get_snippet_text(self, item: IncludedItem) -> str:
        """Helper to get snippet source text for an included item."""
        entity = self.metadata_store.get_function(item.ref) or self.metadata_store.get_class(item.ref)
        if not entity:
            return ""
        snip = compress(entity, DetailLevel(item.level), counter=self.counter)
        return snip.text

    def _assemble_snippets_by_file(self, included_items: list[IncludedItem]) -> dict[str, list[str]]:
        """Group and order code snippets by file and start line."""
        snippets_by_file: dict[str, list[tuple[int, str]]] = {}

        # De-duplication set: check if class is already included at FULL/TRIMMED
        full_classes = {
            it.ref for it in included_items
            if it.level in (DetailLevel.FULL.value, DetailLevel.TRIMMED.value)
        }

        seen_refs: set[str] = set()
        for item in included_items:
            if item.ref in seen_refs:
                continue
            seen_refs.add(item.ref)

            # REFERENCE items don't have code in SOURCE CODE section
            if item.level == DetailLevel.REFERENCE.value:
                continue

            entity = self.metadata_store.get_function(item.ref) or self.metadata_store.get_class(item.ref)
            if not entity:
                continue

            # Deduplication: if method's parent class is already FULL/TRIMMED, skip method snippet
            if isinstance(entity, FunctionInfo) and entity.parent_class:
                parent_qname = entity.qualified_name.rsplit(".", 1)[0]
                if (
                    entity.parent_class in full_classes
                    or f"{entity.file_path}.{entity.parent_class}" in full_classes
                    or parent_qname in full_classes
                ):
                    continue

            snip = compress(entity, DetailLevel(item.level), counter=self.counter)
            start_l = entity.start_line or 0
            if entity.file_path not in snippets_by_file:
                snippets_by_file[entity.file_path] = []
            snippets_by_file[entity.file_path].append((start_l, snip.text))

        # Sort snippets inside each file by line number
        result: dict[str, list[str]] = {}
        for fpath, snip_list in sorted(snippets_by_file.items()):
            snip_list.sort(key=lambda t: t[0])
            result[fpath] = [s[1] for s in snip_list]

        return result
