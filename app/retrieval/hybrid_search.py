"""Hybrid retrieval combining semantic vector search, code graph, and exact matches."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from app.embeddings.semantic_search import SemanticSearch, load_semantic_search
from app.graph.graph_io import load_graph
from app.graph.graph_search import CodeGraph
from app.models.schemas import (
    ExecutionTrace,
    GraphEdgeTrace,
    RankedCandidate,
    RetrievalResult,
    SignalHit,
    TraceOperation,
)
from app.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "is", "it", "how", "what", "does", "function", "method", "class",
}



def _is_test_path(path: str | None) -> bool:
    parts = (path or "").replace("\\", "/").lower().split("/")
    name = parts[-1]
    return any(p in ("tests", "test") for p in parts[:-1]) or name.startswith("test_") or name.endswith("_test.py")


class HybridSearch:
    """Retrieval engine producing RankedCandidate lists and target detection."""

    def __init__(
        self,
        metadata_store: MetadataStore,
        graph: CodeGraph | Any | None = None,
        semantic_search: SemanticSearch | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.graph = graph if isinstance(graph, CodeGraph) else (CodeGraph(graph) if graph is not None else None)
        self.semantic_search = semantic_search

    @classmethod
    def from_repo(
        cls,
        repo_path: Path | str,
        metadata_store: MetadataStore | None = None,
    ) -> HybridSearch:
        """Instantiate HybridSearch for a repository path, loading store, graph, and vector store."""
        repo = Path(repo_path).resolve()
        store = metadata_store or MetadataStore(repo)

        # Try loading graph
        graph: CodeGraph | None = None
        try:
            nx_graph = load_graph(repo)
            if nx_graph is not None:
                graph = CodeGraph(nx_graph)
        except Exception:
            pass

        ssearch = load_semantic_search(repo, store)

        return cls(
            metadata_store=store,
            graph=graph,
            semantic_search=ssearch,
        )

    def search(
        self,
        query: str,
        top_k: int = 15,
        target_override: str | None = None,
        use_semantic: bool = True,
        use_keyword: bool = True,
        use_graph: bool = True,
    ) -> RetrievalResult:
        """Execute hybrid search on repository for a natural language task or symbol.

        The returned result carries an ExecutionTrace describing each signal that
        ran, every hit it produced, and each code-graph edge that was walked.
        """
        trace = ExecutionTrace()
        ops: list[TraceOperation] = []
        hits: list[SignalHit] = []
        edges: list[GraphEdgeTrace] = []

        def record(name: str, t_start: float, detail: str, count: int | None = None, status: str = "ran") -> None:
            ops.append(
                TraceOperation(
                    step=len(ops) + 1,
                    name=name,
                    status=status,
                    detail=detail,
                    count=count,
                    seconds=round(time.perf_counter() - t_start, 4),
                )
            )

        # 1. Intent Detection
        t0 = time.perf_counter()
        query_lower = query.lower()
        if any(w in query_lower for w in ("optimize", "performance", "faster", "speed", "latency")):
            intent = "optimize"
        elif any(w in query_lower for w in ("explain", "how does", "what is", "walkthrough", "understand")):
            intent = "explain"
        elif any(w in query_lower for w in ("fix", "bug", "error", "issue", "crash", "exception", "broken")):
            intent = "fix_bug"
        else:
            intent = "general"
        record("intent_detection", t0, f"Query classified as '{intent}'")

        # 2. Target Identification
        t0 = time.perf_counter()
        target_ref: str | None = target_override
        target_confidence: str = "low"
        target_reason = "Target pinned by caller" if target_override else ""

        if not target_ref:
            # Check for exact matches mentioned in query words or dotted identifiers
            identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", query)
            for ident in identifiers:
                if len(ident) < 3:
                    continue
                # Try exact function
                fn = self.metadata_store.get_function(ident)
                if fn:
                    target_ref = fn.qualified_name
                    target_confidence = "high"
                    target_reason = f"'{ident}' in the query is an exact function name"
                    break
                cls_ent = self.metadata_store.get_class(ident)
                if cls_ent:
                    target_ref = cls_ent.qualified_name
                    target_confidence = "high"
                    target_reason = f"'{ident}' in the query is an exact class name"
                    break
                # Try short name match
                matches = self.metadata_store.find_functions_by_name(ident, exact=True)
                if matches:
                    target_ref = matches[0].qualified_name
                    target_confidence = "high"
                    target_reason = f"'{ident}' in the query matches a function short name"
                    break
        if target_ref and target_confidence == "high":
            hits.append(SignalHit(ref=target_ref, signal="exact_target", score=1.0, note=target_reason))
        record(
            "target_identification",
            t0,
            target_reason or "No symbol named in the query; target will come from the top-ranked candidate",
            count=1 if target_ref else 0,
        )

        candidate_map: dict[str, RankedCandidate] = {}
        signals: dict[str, str] = {}

        # 3. Semantic Search
        if not use_semantic:
            signals["semantic"] = "skipped: disabled by request"
            record("semantic_search", time.perf_counter(), "Disabled by request", status="skipped")
        elif self.semantic_search is None:
            signals["semantic"] = "skipped: no embedding index"
            record("semantic_search", time.perf_counter(), "No embedding index available", status="skipped")
            trace.warnings.append("Semantic embeddings unavailable; relying on keyword and graph signals.")
        else:
            t0 = time.perf_counter()
            try:
                sem_matches = self.semantic_search.search(query, top_k=top_k * 2)
                for sm in sem_matches:
                    ref = sm.ref
                    entity = self.metadata_store.get_function(ref) or self.metadata_store.get_class(ref)
                    file_path = sm.file or (entity.file_path if entity else "")
                    start_line = sm.start_line if sm.start_line is not None else (entity.start_line if entity else None)
                    end_line = sm.end_line if sm.end_line is not None else (entity.end_line if entity else None)
                    etype = sm.entity_type or ("class" if hasattr(entity, "methods") else ("method" if getattr(entity, "is_method", False) else "function"))
                    candidate_map[ref] = RankedCandidate(
                        ref=ref,
                        score=round(sm.score, 4),
                        entity_type=etype,
                        file=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        matched_kinds=sm.matched_kinds,
                        match_type="semantic",
                    )
                    hits.append(
                        SignalHit(
                            ref=ref,
                            signal="semantic",
                            score=round(sm.score, 4),
                            file=file_path,
                            note=("matched " + ", ".join(sm.matched_kinds)) if sm.matched_kinds else "",
                        )
                    )
                signals["semantic"] = "used"
                record("semantic_search", t0, f"Vector search returned {len(sem_matches)} entities", len(sem_matches))
            except Exception as exc:
                logger.warning(f"Semantic search failed during hybrid retrieval: {exc}")
                signals["semantic"] = f"failed: {exc}"
                record("semantic_search", t0, f"Failed: {exc}", status="failed")
                trace.warnings.append(f"Semantic search failed: {exc}")

        # 4. Keyword matching on function names
        if not use_keyword:
            signals["keyword"] = "skipped: disabled by request"
            record("keyword_search", time.perf_counter(), "Disabled by request", status="skipped")
        else:
            t0 = time.perf_counter()
            tokens = [w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query) if w.lower() not in STOPWORDS]
            kw_found = 0
            for token in dict.fromkeys(tokens):
                for fn in self.metadata_store.find_functions_by_name(token, exact=False):
                    fn_parts = set(fn.name.lower().split("_"))
                    if fn.name.lower() == token:
                        score = 0.95
                    elif token in fn_parts or fn.name.lower().startswith(token):
                        score = 0.85
                    else:
                        score = 0.4
                    existing = candidate_map.get(fn.qualified_name)
                    if existing is not None and existing.score >= score:
                        # Agreement between signals strengthens the candidate.
                        candidate_map[fn.qualified_name] = existing.model_copy(
                            update={"score": min(1.0, round(existing.score + 0.1, 4)), "match_type": "hybrid"}
                        )
                    elif existing is not None:
                        candidate_map[fn.qualified_name] = existing.model_copy(
                            update={"score": score, "match_type": "hybrid"}
                        )
                    else:
                        candidate_map[fn.qualified_name] = RankedCandidate(
                            ref=fn.qualified_name,
                            score=score,
                            entity_type="method" if fn.is_method else "function",
                            file=fn.file_path,
                            start_line=fn.start_line,
                            end_line=fn.end_line,
                            matched_kinds=[],
                            match_type="keyword",
                        )
                    hits.append(
                        SignalHit(
                            ref=fn.qualified_name,
                            signal="keyword",
                            score=score,
                            file=fn.file_path,
                            note=f"name matches '{token}'",
                        )
                    )
                    kw_found += 1
            signals["keyword"] = "used"
            record("keyword_search", t0, f"{kw_found} name matches for {len(set(tokens))} query terms", kw_found)

        # If target still undetermined, use top candidate
        if not target_ref and candidate_map:
            pool = list(candidate_map.values())
            if "test" not in query.lower():
                # A question about behaviour should anchor on source code, not on its tests.
                pool = [c for c in pool if not _is_test_path(c.file)] or pool
            top_cand = max(pool, key=lambda c: c.score)
            target_ref = top_cand.ref
            target_confidence = "high" if top_cand.score >= 0.7 else ("medium" if top_cand.score >= 0.5 else "low")
            target_reason = f"Top-ranked candidate ({top_cand.match_type}, score {top_cand.score:.2f})"

        # 5. AST call expansion from target function
        t0 = time.perf_counter()
        ast_added = 0
        if target_ref:
            target_fn = self.metadata_store.get_function(target_ref)
            if target_fn:
                for call in target_fn.calls:
                    call_name = call.name.split(".")[-1]
                    for fn in self.metadata_store.find_functions_by_name(call_name, exact=True):
                        if fn.qualified_name != target_ref and fn.qualified_name not in candidate_map:
                            candidate_map[fn.qualified_name] = RankedCandidate(
                                ref=fn.qualified_name,
                                score=0.88,
                                entity_type="method" if fn.is_method else "function",
                                file=fn.file_path,
                                start_line=fn.start_line,
                                end_line=fn.end_line,
                                matched_kinds=[],
                                match_type="exact",
                            )
                            hits.append(
                                SignalHit(
                                    ref=fn.qualified_name,
                                    signal="ast_call",
                                    score=0.88,
                                    file=fn.file_path,
                                    note=f"called by {target_ref} (line {call.lineno})" if getattr(call, "lineno", None) else f"called by {target_ref}",
                                )
                            )
                            ast_added += 1
        record(
            "ast_call_expansion",
            t0,
            f"{ast_added} functions called by the target found via parsed call sites",
            ast_added,
            status="ran" if target_ref else "skipped",
        )

        # 6. Graph expansion (Callees and Callers of target)
        if not use_graph:
            signals["graph"] = "skipped: disabled by request"
            record("graph_expansion", time.perf_counter(), "Disabled by request", status="skipped")
        elif self.graph is None:
            signals["graph"] = "skipped: no code graph"
            record("graph_expansion", time.perf_counter(), "No code graph built", status="skipped")
            trace.warnings.append("Code graph unavailable; run indexing to build it.")
        elif not target_ref:
            signals["graph"] = "skipped: no target"
            record("graph_expansion", time.perf_counter(), "No target to expand from", status="skipped")
        else:
            t0 = time.perf_counter()
            trace.graph_nodes = self.graph.graph.number_of_nodes()
            trace.graph_edges_total = self.graph.graph.number_of_edges()
            try:
                target_nids = self.graph.find_nodes(target_ref)
                if not target_nids:
                    signals["graph"] = "skipped: target not in graph"
                    record("graph_expansion", t0, f"'{target_ref}' is not a node in the code graph", status="skipped")
                else:
                    t_nid = target_nids[0]
                    walked = 0
                    for direction, results, new_score in (
                        ("callee", self.graph.callees(t_nid, depth=1), 0.85),
                        ("caller", self.graph.callers(t_nid, depth=1), 0.75),
                    ):
                        for c in results:
                            node_type = self.graph.graph.nodes.get(c.node_id, {}).get("type", "function")
                            if node_type not in ("function", "method", "class"):
                                continue
                            ref = c.qualified_name
                            entity = self.metadata_store.get_function(ref) or self.metadata_store.get_class(ref)
                            existing = candidate_map.get(ref)
                            if existing is None:
                                if not entity:
                                    continue
                                candidate_map[ref] = RankedCandidate(
                                    ref=ref,
                                    score=new_score,
                                    entity_type=node_type,
                                    file=c.file or entity.file_path,
                                    start_line=entity.start_line,
                                    end_line=entity.end_line,
                                    matched_kinds=[],
                                    match_type="graph",
                                )
                                outcome = "added"
                            elif direction == "callee":
                                # Boost existing candidate score
                                candidate_map[ref] = existing.model_copy(
                                    update={"score": min(1.0, round(existing.score + 0.15, 4)), "match_type": "hybrid"}
                                )
                                outcome = "boosted"
                            else:
                                outcome = "already_present"
                            walked += 1
                            edges.append(
                                GraphEdgeTrace(
                                    source=target_ref if direction == "callee" else ref,
                                    target=ref if direction == "callee" else target_ref,
                                    rel="CALLS",
                                    direction=direction,
                                    depth=c.depth,
                                    confidence=c.confidence,
                                    outcome=outcome,
                                )
                            )
                            hits.append(
                                SignalHit(
                                    ref=ref,
                                    signal=f"graph_{direction}",
                                    score=new_score,
                                    file=c.file or (entity.file_path if entity else ""),
                                    note=f"{direction} of {target_ref} ({c.confidence} confidence)",
                                )
                            )
                    signals["graph"] = "used"
                    record("graph_expansion", t0, f"Walked {walked} CALLS edges around '{target_ref}' (callees and callers)", walked)
            except Exception as exc:
                logger.warning(f"Graph expansion failed during hybrid retrieval: {exc}")
                signals["graph"] = f"failed: {exc}"
                record("graph_expansion", t0, f"Failed: {exc}", status="failed")
                trace.warnings.append(f"Graph expansion failed: {exc}")

        # Sort candidates descending by score
        all_candidates = sorted(candidate_map.values(), key=lambda c: c.score, reverse=True)
        candidates = all_candidates[:top_k]

        trace.intent = intent
        trace.target = target_ref
        trace.target_confidence = target_confidence
        trace.target_reason = target_reason
        trace.signals = signals
        trace.hits = hits
        trace.graph_edges = edges
        trace.candidates_considered = len(all_candidates)
        trace.candidates_returned = len(candidates)
        trace.operations = ops

        return RetrievalResult(
            query=query,
            target=target_ref,
            target_confidence=target_confidence,
            candidates=candidates,
            intent=intent,
            trace=trace,
        )
