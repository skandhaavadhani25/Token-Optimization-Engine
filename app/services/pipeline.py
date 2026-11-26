"""RepositoryService pipeline coordinating index, graph, embeddings, retrieval, context, and AI completions."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any
import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.ai.base_provider import AIProvider, ChatMessage, CompletionResult
from app.ai.pricing import get_cost_info
from app.ai.prompts import build_messages
from app.ai.provider_factory import get_provider
from app.ai.response_parser import ParsedAnswer, parse_response
from app.config import get_settings
from app.context.context_builder import ContextBuilder
from app.context.token_counter import TokenCounter, get_counter
from app.embeddings.indexer import build_embeddings
from app.embeddings.semantic_search import SemanticSearch, load_semantic_search
from app.embeddings.vector_store import VectorStore
from app.graph.graph_builder import build_graph
from app.graph.graph_io import is_graph_stale, load_graph, save_graph
from app.graph.graph_search import CodeGraph
from app.indexer import build_index
from app.models.schemas import (
    BuilderOptions,
    BuiltContext,
    EmbeddingReport,
    ExecutionTrace,
    GraphBuildReport,
    IndexReport,
    RetrievalResult,
    SavingsReport,
    SelectionTrace,
    SemanticHit,
    TraceOperation,
)
from app.retrieval.hybrid_search import HybridSearch
from app.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class IndexingInProgressError(Exception):
    """Raised when an operation is attempted while indexing is concurrently running."""


class PrerequisiteMissingError(Exception):
    """Raised when an index, graph, or embedding artifact required for an operation is missing."""


class TokenReport(BaseModel):
    """Token consumption, budget utilization, and timing report."""

    model_config = ConfigDict(frozen=True)

    context_tokens: int = Field(description="Token count of context prompt (counted by local tokenizer)")
    provider_input_tokens: int | None = Field(default=None, description="Actual input tokens reported by provider API")
    provider_output_tokens: int | None = Field(default=None, description="Actual output tokens reported by provider API")
    provider_total_tokens: int | None = Field(default=None, description="Actual total tokens reported by provider API")
    exact_count_tokens: int | None = Field(default=None, description="Tokens counted by provider count_tokens API")
    budget: int = Field(description="Configured token budget")
    utilization: float = Field(description="Context tokens / budget fraction")
    estimated_cost_usd: float | None = Field(default=None, description="Estimated completion cost in USD")
    is_free_tier: bool = Field(default=False, description="Whether completion was on a free tier or local model")
    cost_display: str | None = Field(default=None, description="Formatted cost string ('free tier', '$0.0012', or 'unknown')")
    provider: str | None = Field(default=None, description="AI provider used")
    model: str | None = Field(default=None, description="AI model identifier used")
    key_source: str | None = Field(default=None, description="API key source ('user' or 'server')")
    usage_estimated: bool = Field(default=False, description="Whether token usage is estimated rather than reported by API")
    files_in_repo: int = Field(description="Total files indexed in repository")
    files_in_context: int = Field(description="Files included in final prompt context")
    retrieval_seconds: float = Field(description="Time spent in hybrid candidate retrieval")
    context_build_seconds: float = Field(description="Time spent in context packing and compression")
    llm_seconds: float = Field(description="Time spent waiting for AI provider completion")
    total_seconds: float = Field(description="Total end-to-end processing time")


class AskResult(BaseModel):
    """Result of an /ask coding task query."""

    model_config = ConfigDict(frozen=True)

    query: str
    context: BuiltContext
    completion: CompletionResult | None
    answer: ParsedAnswer | None
    token_report: TokenReport
    dry_run: bool
    messages: list[ChatMessage]
    system_prompt: str
    provider: str | None = None
    model: str | None = None
    key_source: str | None = None
    usage_estimated: bool = False
    trace: ExecutionTrace | None = None


class CombinedIndexReport(BaseModel):
    """Aggregated report across the three indexing stages."""

    model_config = ConfigDict(frozen=True)

    repo_path: str
    scan_and_store: IndexReport
    graph_build: GraphBuildReport
    embeddings: EmbeddingReport
    total_seconds: float


def get_configured_embedding_model() -> str:
    """Return configured embedding model identifier, respecting active generator mocks."""
    import sys
    for mod_path in (
        "app.embeddings.semantic_search",
        "app.embeddings.indexer",
        "app.embeddings.embedding_generator",
    ):
        try:
            mod = sys.modules.get(mod_path)
            if mod and hasattr(mod, "EmbeddingGenerator"):
                cls_or_mock = getattr(mod, "EmbeddingGenerator")
                inst = cls_or_mock()
                if hasattr(inst, "model_name") and inst.model_name:
                    return inst.model_name
        except Exception:
            pass
    return get_settings().embedding_model


class RepositoryService:
    """Thread-safe unified pipeline service for a single repository."""

    def __init__(self, repo_path: Path | str) -> None:
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise FileNotFoundError(f"Repository directory does not exist: {self.repo_path}")

        self._index_lock = threading.Lock()
        self._is_indexing = False

        # Lazy caches
        self._store: MetadataStore | None = None
        self._graph: CodeGraph | None = None
        self._semantic_search: SemanticSearch | None = None
        self._counter: TokenCounter | None = None
        # rel_path -> (mtime_ns, size, tokens); avoids re-tokenizing the repo on every request
        self._file_token_cache: dict[str, tuple[int, int, int]] = {}

    @property
    def is_indexing(self) -> bool:
        """Check if an indexing operation is currently running."""
        return self._is_indexing

    def _ensure_not_indexing(self) -> None:
        """Raise IndexingInProgressError if indexing is running."""
        if self._is_indexing:
            raise IndexingInProgressError(
                f"Indexing is currently in progress for repository '{self.repo_path.name}'. Please wait."
            )

    def _get_store(self) -> MetadataStore:
        """Get or lazily initialize MetadataStore."""
        if self._store is None:
            if not MetadataStore.exists(self.repo_path):
                raise PrerequisiteMissingError(
                    f"Metadata store not found for '{self.repo_path.name}'. "
                    f"Please run 'POST /index' or 'python -m app.main index <repo>' first."
                )
            self._store = MetadataStore(self.repo_path)
        return self._store

    def _get_graph(self) -> CodeGraph | None:
        """Get or lazily load CodeGraph from disk."""
        if self._graph is None:
            try:
                nx_g = load_graph(self.repo_path)
                if nx_g is not None:
                    self._graph = CodeGraph(nx_g)
            except Exception as e:
                logger.debug(f"Could not load graph for {self.repo_path}: {e}")
        return self._graph

    def _get_semantic_search(self) -> SemanticSearch | None:
        """Get or lazily initialize SemanticSearch."""
        if self._semantic_search is None:
            self._semantic_search = load_semantic_search(self.repo_path, self._get_store())
        return self._semantic_search

    def invalidate_caches(self) -> None:
        """Clear cached components so fresh data from disk is reloaded."""
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None
        self._graph = None
        self._semantic_search = None

    def index(self, force: bool = False) -> CombinedIndexReport:
        """Run the full indexing pipeline: scanner/store -> graph -> embeddings."""
        if not self._index_lock.acquire(blocking=False):
            raise IndexingInProgressError(
                f"Indexing is already in progress for repository '{self.repo_path.name}'."
            )

        self._is_indexing = True
        t0 = time.perf_counter()
        try:
            self.invalidate_caches()

            # Stage 1: AST Scan & metadata index
            scan_report = build_index(self.repo_path, force=force)

            # Stage 2: Code Graph
            with MetadataStore.open(self.repo_path) as store:
                nx_graph, graph_report = build_graph(store, repo_name=self.repo_path.name)
                save_graph(nx_graph, self.repo_path)

            # Stage 3: Semantic Embeddings
            embed_report = build_embeddings(
                repo_path=self.repo_path,
                force=force,
            )

            total_elapsed = time.perf_counter() - t0
            return CombinedIndexReport(
                repo_path=str(self.repo_path),
                scan_and_store=scan_report,
                graph_build=graph_report,
                embeddings=embed_report,
                total_seconds=total_elapsed,
            )
        finally:
            self.invalidate_caches()
            self._is_indexing = False
            self._index_lock.release()

    def _get_counter(self) -> TokenCounter:
        """Shared token counter so repeated requests reuse its cache."""
        if self._counter is None:
            self._counter = get_counter("tiktoken")
        return self._counter

    def _file_tokens(self, rel_path: str) -> int:
        """Token count of a repository file as a naive pipeline would paste it."""
        full = self.repo_path / rel_path
        try:
            st = full.stat()
        except OSError:
            return 0
        cached = self._file_token_cache.get(rel_path)
        if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return cached[2]
        try:
            tokens = self._get_counter().count(full.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return 0
        self._file_token_cache[rel_path] = (st.st_mtime_ns, st.st_size, tokens)
        return tokens

    def measure_savings(
        self,
        store: MetadataStore,
        built_ctx: BuiltContext,
        prompt_tokens: int,
        provider_input_tokens: int | None = None,
    ) -> SavingsReport:
        """Compare the prompt we send against pasting every file or every contributing file."""
        counter = self._get_counter()
        repo_paths = store.get_all_file_paths()
        naive_full = sum(self._file_tokens(p) for p in repo_paths)
        whole_files = sum(self._file_tokens(p) for p in built_ctx.files_included)

        def pct(saved: int, base: int) -> float:
            return round(saved / base * 100.0, 1) if base > 0 else 0.0

        saved_full = naive_full - prompt_tokens
        saved_whole = whole_files - prompt_tokens
        before = sum(it.original_tokens for it in built_ctx.items)
        after = sum(it.tokens for it in built_ctx.items)
        is_estimate = bool(getattr(counter, "is_estimate", False)) or counter.name == "char"
        return SavingsReport(
            counter="char (len/4 estimate)" if is_estimate else counter.name,
            is_estimate=is_estimate,
            context_tokens=built_ctx.tokens,
            prompt_tokens=prompt_tokens,
            provider_input_tokens=provider_input_tokens,
            naive_full_repo_tokens=naive_full,
            whole_files_tokens=whole_files,
            repo_files=len(repo_paths),
            context_files=len(built_ctx.files_included),
            saved_vs_full_repo_tokens=saved_full,
            saved_vs_full_repo_pct=pct(saved_full, naive_full),
            saved_vs_whole_files_tokens=saved_whole,
            saved_vs_whole_files_pct=pct(saved_whole, whole_files),
            compression_saved_tokens=max(0, before - after),
            reduction_factor=round(naive_full / prompt_tokens, 1) if prompt_tokens > 0 else 0.0,
        )

    def measure_prompt_savings(self, built_ctx: BuiltContext, prompt_tokens: int) -> SavingsReport:
        """Savings of a prompt of `prompt_tokens` that embeds `built_ctx`, versus naive baselines."""
        return self.measure_savings(self._get_store(), built_ctx, prompt_tokens=prompt_tokens)

    def _retrieve_and_build(
        self,
        query: str,
        top_k: int,
        budget: int,
        use_semantic: bool = True,
        use_keyword: bool = True,
        use_graph: bool = True,
        options: BuilderOptions | None = None,
    ) -> tuple[RetrievalResult, BuiltContext, float, float]:
        """Hybrid retrieval then context packing; returns results and per-stage seconds.

        The returned RetrievalResult has its trace completed with selection details,
        a token-savings report, and the context-building operation.
        """
        store = self._get_store()
        graph = self._get_graph() if use_graph else None

        # Check semantic search and embedding model consistency
        sem_search = None
        warnings: list[str] = []
        if use_semantic:
            manifest = VectorStore.read_manifest(self.repo_path)
            if manifest is not None:
                try:
                    m_model = manifest.get("model_name")
                    configured_model = get_configured_embedding_model()
                    if m_model and m_model != configured_model:
                        raise PrerequisiteMissingError(
                            f"Embeddings were built with model '{m_model}', but configured model is '{configured_model}'. "
                            f"Please run 'python -m app.main embed {self.repo_path} --force' or 'POST /index' to rebuild embeddings."
                        )
                except PrerequisiteMissingError:
                    raise
                except Exception as e:
                    warnings.append(f"Could not verify embedding model: {e}")

            sem_search = self._get_semantic_search()
            if sem_search is None:
                warnings.append(
                    f"Semantic embeddings index not found for '{self.repo_path.name}'. "
                    f"Degrading to keyword retrieval. Please run 'POST /index' or 'python -m app.main embed <repo>' to build embeddings."
                )

        t_ret = time.perf_counter()
        hybrid = HybridSearch(metadata_store=store, graph=graph, semantic_search=sem_search)
        retrieval_res = hybrid.search(
            query,
            top_k=top_k,
            use_semantic=use_semantic,
            use_keyword=use_keyword,
            use_graph=use_graph,
        )
        retrieval_secs = time.perf_counter() - t_ret

        t_ctx = time.perf_counter()
        builder = ContextBuilder(metadata_store=store, graph=graph, counter=self._get_counter())
        built_ctx = builder.build(retrieval_res, budget=budget, options=options)
        context_secs = time.perf_counter() - t_ctx
        if warnings:
            built_ctx.warnings.extend(warnings)

        trace = retrieval_res.trace or ExecutionTrace()
        trace.warnings.extend(w for w in warnings if w not in trace.warnings)
        trace.selection = SelectionTrace(
            candidates_in=len(retrieval_res.candidates),
            included=list(built_ctx.items),
            omitted=list(built_ctx.omitted_items),
            downgraded_count=built_ctx.downgraded_count,
            tokens_before_compression=sum(it.original_tokens for it in built_ctx.items),
            tokens_after_compression=sum(it.tokens for it in built_ctx.items),
        )
        by_level: dict[str, int] = {}
        for it in built_ctx.items:
            by_level[it.level] = by_level.get(it.level, 0) + 1
        level_text = ", ".join(f"{n} {lvl}" for lvl, n in by_level.items()) or "nothing"
        trace.operations.append(
            TraceOperation(
                step=len(trace.operations) + 1,
                name="context_packing",
                detail=(
                    f"Packed {len(built_ctx.items)} of {len(retrieval_res.candidates)} candidates ({level_text}) "
                    f"into {built_ctx.tokens} / {built_ctx.budget_total} tokens; "
                    f"{len(built_ctx.omitted_items)} pruned, {built_ctx.downgraded_count} downgraded"
                ),
                count=len(built_ctx.items),
                seconds=round(context_secs, 4),
            )
        )
        # Retrieval-only requests have no model prompt; the task text is the only addition.
        task_tokens = self._get_counter().count(query)
        trace.savings = self.measure_savings(store, built_ctx, prompt_tokens=built_ctx.tokens + task_tokens)
        trace.operations.append(
            TraceOperation(
                step=len(trace.operations) + 1,
                name="token_accounting",
                detail=(
                    f"Prompt {trace.savings.prompt_tokens} tokens vs {trace.savings.naive_full_repo_tokens} "
                    f"for all {trace.savings.repo_files} files ({trace.savings.saved_vs_full_repo_pct}% saved)"
                ),
                count=trace.savings.prompt_tokens,
            )
        )
        return retrieval_res, built_ctx, retrieval_secs, context_secs

    def search(
        self,
        query: str,
        top_k: int = 10,
        kinds: list[str] | None = None,
    ) -> list[SemanticHit]:
        """Perform semantic similarity search on the repository."""
        self._ensure_not_indexing()
        sem = self._get_semantic_search()
        if sem is None:
            raise PrerequisiteMissingError(
                f"Semantic embeddings index not found for '{self.repo_path.name}'. "
                f"Please run 'POST /index' or 'python -m app.main embed <repo>' first."
            )
        return sem.search(query, top_k=top_k, kinds=kinds)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        max_tokens: int | None = None,
        use_semantic: bool = True,
        use_keyword: bool = True,
        use_graph: bool = True,
        explain: bool = False,
        options: BuilderOptions | None = None,
    ) -> tuple[RetrievalResult, BuiltContext]:
        """Perform hybrid retrieval and token-budgeted context assembly."""
        self._ensure_not_indexing()
        budget = max_tokens if max_tokens is not None else get_settings().context_token_budget
        retrieval_res, built_ctx, _, _ = self._retrieve_and_build(
            query,
            top_k=top_k,
            budget=budget,
            use_semantic=use_semantic,
            use_keyword=use_keyword,
            use_graph=use_graph,
            options=options,
        )
        return retrieval_res, built_ctx

    def ask(
        self,
        query: str,
        budget: int | None = None,
        provider: AIProvider | None = None,
        provider_override: str | None = None,
        dry_run: bool = False,
        model_override: str | None = None,
        temperature_override: float | None = None,
        max_output_tokens_override: int | None = None,
        api_key: str | SecretStr | None = None,
        key_source: str | None = None,
    ) -> AskResult:
        """Run the end-to-end coding assistant pipeline: retrieve -> context -> LLM -> parse."""
        self._ensure_not_indexing()
        t0 = time.perf_counter()

        settings = get_settings()
        target_budget = budget if budget is not None else settings.context_token_budget

        target_provider_name = (
            provider_override
            or (provider.name if provider else settings.ai_provider)
        ).lower()
        target_model = model_override or settings.get_model_for_provider(
            target_provider_name, raise_if_missing=False
        )

        # 1-2. Retrieval and context packing (trace is attached to retrieval_res)
        store = self._get_store()
        retrieval_res, built_ctx, retrieval_secs, context_build_secs = self._retrieve_and_build(
            query, top_k=15, budget=target_budget
        )

        # 3. Assemble Prompt Messages with prompt_style
        style = settings.prompt_style or "full"
        system_prompt, messages = build_messages(
            context_text=built_ctx.text,
            task_text=query,
            style=style,
        )

        # Re-measure savings against the real prompt (system prompt + context + task)
        trace = retrieval_res.trace
        if trace is not None:
            prompt_tokens = self._get_counter().count(system_prompt) + sum(
                self._get_counter().count(m.content) for m in messages
            )
            trace.savings = self.measure_savings(store, built_ctx, prompt_tokens=prompt_tokens)

        # Total files in repository
        stats = store.get_stats()
        files_in_repo = stats.total_files

        if dry_run:
            total_secs = time.perf_counter() - t0
            dry_model = target_model or "dry-run-model"
            cost_usd, is_free, cost_str = get_cost_info(dry_model, 0, 0, provider=target_provider_name)
            token_report = TokenReport(
                context_tokens=built_ctx.tokens,
                provider_input_tokens=None,
                provider_output_tokens=None,
                provider_total_tokens=None,
                exact_count_tokens=None,
                budget=target_budget,
                utilization=built_ctx.utilization,
                estimated_cost_usd=cost_usd,
                is_free_tier=is_free,
                cost_display=cost_str,
                provider=target_provider_name,
                model=target_model,
                key_source=key_source,
                usage_estimated=False,
                files_in_repo=files_in_repo,
                files_in_context=len(built_ctx.files_included),
                retrieval_seconds=retrieval_secs,
                context_build_seconds=context_build_secs,
                llm_seconds=0.0,
                total_seconds=total_secs,
            )
            return AskResult(
                query=query,
                context=built_ctx,
                completion=None,
                answer=None,
                token_report=token_report,
                dry_run=True,
                messages=messages,
                system_prompt=system_prompt,
                provider=target_provider_name,
                model=target_model,
                key_source=key_source,
                usage_estimated=False,
                trace=trace,
            )

        # 4. Invoke Provider
        try:
            active_provider = provider or get_provider(
                provider_name=target_provider_name,
                model=target_model,
                api_key=api_key,
                raise_if_model_missing=True,
            )
        except ValueError as err:
            from app.ai.base_provider import ProviderAuthError
            raise ProviderAuthError(str(err)) from err

        if not active_provider.is_configured():
            from app.ai.base_provider import ApiKeyRequiredError, ProviderAuthError
            hint = active_provider.get_setup_hint() or f"Please configure credentials for '{active_provider.name}'."
            if active_provider.is_local:
                raise ProviderAuthError(f"Local AI Provider '{active_provider.name}' is not ready: {hint}")
            raise ApiKeyRequiredError(
                f"API key required for provider '{active_provider.name}'. Please enter your API key in the web page."
            )

        # Exact token count check if supported
        exact_tokens: int | None = None
        try:
            exact_tokens = active_provider.count_tokens(system_prompt, messages)
        except Exception:
            pass

        t_llm_start = time.perf_counter()
        completion = active_provider.complete(
            system=system_prompt,
            messages=messages,
            max_output_tokens=max_output_tokens_override,
            temperature=temperature_override,
        )
        llm_secs = time.perf_counter() - t_llm_start
        if trace is not None:
            if trace.savings is not None:
                trace.savings = trace.savings.model_copy(
                    update={"provider_input_tokens": completion.usage.input_tokens}
                )
            trace.operations.append(
                TraceOperation(
                    step=len(trace.operations) + 1,
                    name="llm_completion",
                    detail=(
                        f"{active_provider.name} returned {completion.usage.output_tokens} output tokens "
                        f"for {completion.usage.input_tokens} input tokens"
                    ),
                    count=completion.usage.output_tokens,
                    seconds=round(llm_secs, 4),
                )
            )

        # 5. Parse Model Response
        known_symbols = set()
        for it in built_ctx.items:
            known_symbols.add(it.ref)
        parsed_answer = parse_response(completion.text, known_targets=known_symbols)

        total_secs = time.perf_counter() - t0

        actual_model = completion.model or target_model or "unknown"
        cost_usd, is_free, cost_str = get_cost_info(
            actual_model,
            completion.usage.input_tokens,
            completion.usage.output_tokens,
            provider=active_provider.name,
        )

        token_report = TokenReport(
            context_tokens=built_ctx.tokens,
            provider_input_tokens=completion.usage.input_tokens,
            provider_output_tokens=completion.usage.output_tokens,
            provider_total_tokens=completion.usage.total_tokens,
            exact_count_tokens=exact_tokens,
            budget=target_budget,
            utilization=built_ctx.utilization,
            estimated_cost_usd=cost_usd,
            is_free_tier=is_free,
            cost_display=cost_str,
            provider=active_provider.name,
            model=actual_model,
            key_source=key_source,
            usage_estimated=completion.usage.usage_estimated,
            files_in_repo=files_in_repo,
            files_in_context=len(built_ctx.files_included),
            retrieval_seconds=retrieval_secs,
            context_build_seconds=context_build_secs,
            llm_seconds=llm_secs,
            total_seconds=total_secs,
        )

        return AskResult(
            query=query,
            context=built_ctx,
            completion=completion,
            answer=parsed_answer,
            token_report=token_report,
            dry_run=False,
            messages=messages,
            system_prompt=system_prompt,
            provider=active_provider.name,
            model=actual_model,
            key_source=key_source,
            usage_estimated=completion.usage.usage_estimated,
            trace=trace,
        )

    def graph_view(
        self,
        center: str | None = None,
        hops: int = 2,
        max_nodes: int = 150,
        rels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return nodes and edges formatted for visualization UI, capped at max_nodes."""
        self._ensure_not_indexing()
        graph = self._get_graph()
        if graph is None:
            raise PrerequisiteMissingError(
                f"Code graph not found for '{self.repo_path.name}'. "
                f"Please run 'POST /index' or 'python -m app.main graph-build <repo>' first."
            )

        nx_g: nx.MultiDiGraph = graph.graph
        total_nodes = nx_g.number_of_nodes()
        total_edges = nx_g.number_of_edges()

        # Select subgraph
        if center:
            matched_nodes = graph.find_nodes(center)
            if not matched_nodes:
                matched_nodes = [n for n in nx_g.nodes() if center.lower() in str(n).lower()]
            if not matched_nodes:
                return {
                    "nodes": [],
                    "edges": [],
                    "truncated": False,
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "center": center,
                }
            seed = matched_nodes[0]
            # BFS up to hops
            visited = {seed}
            current_layer = {seed}
            for _ in range(hops):
                next_layer = set()
                for node in current_layer:
                    next_layer.update(nx_g.successors(node))
                    next_layer.update(nx_g.predecessors(node))
                visited.update(next_layer)
                current_layer = next_layer
                if len(visited) >= max_nodes:
                    break
            sub_nodes = list(visited)[:max_nodes]
        else:
            # Overview: top degree nodes
            sorted_nodes = sorted(nx_g.degree(), key=lambda x: x[1], reverse=True)
            sub_nodes = [node for node, _ in sorted_nodes[:max_nodes]]

        truncated = len(sub_nodes) < total_nodes

        # Build node and edge payload
        node_set = set(sub_nodes)
        nodes_payload = []
        for n in sub_nodes:
            data = nx_g.nodes.get(n, {})
            node_file = data.get("file") or data.get("file_path") or data.get("path")
            nodes_payload.append({
                "id": str(n),
                "label": str(data.get("label") or data.get("name") or str(n).split(":")[-1]),
                "type": str(data.get("type", "unknown")),
                "file": node_file,
                "start_line": data.get("start_line"),
                "end_line": data.get("end_line"),
                "degree": nx_g.degree(n),
                "note": data.get("note"),
            })

        edges_payload = []
        allowed_rels = set(rels) if rels else None
        for u, v, k, data in nx_g.edges(sub_nodes, data=True, keys=True):
            if v in node_set:
                rel = data.get("rel", "RELATED")
                if allowed_rels and rel not in allowed_rels:
                    continue
                edges_payload.append({
                    "source": str(u),
                    "target": str(v),
                    "rel": rel,
                    "confidence": data.get("confidence", 1.0),
                })

        return {
            "nodes": nodes_payload,
            "edges": edges_payload,
            "truncated": truncated,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "center": center,
        }

    def graph_node(self, node_id: str) -> dict[str, Any]:
        """Return detailed node information, callers, callees, dependencies, and source snippet."""
        self._ensure_not_indexing()
        graph = self._get_graph()
        if graph is None:
            raise PrerequisiteMissingError("Code graph is not built.")

        nx_g = graph.graph
        if node_id not in nx_g:
            # Try fuzzy search
            candidates = graph.find_nodes(node_id)
            if not candidates:
                raise KeyError(f"Node '{node_id}' not found in code graph.")
            node_id = candidates[0]

        data = nx_g.nodes[node_id]
        callers = [str(u) for u, _, d in nx_g.in_edges(node_id, data=True) if d.get("rel") == "CALLS"]
        callees = [str(v) for _, v, d in nx_g.out_edges(node_id, data=True) if d.get("rel") == "CALLS"]
        file_deps = [str(v) for _, v, d in nx_g.out_edges(node_id, data=True) if d.get("rel") == "DEPENDS_ON"]

        source_code = None
        store = self._get_store()
        file_path = data.get("file") or data.get("file_path") or data.get("path")
        if file_path:
            name = data.get("name")
            # Lookup entity in store
            funcs = store.find_functions_by_name(name or "")
            for f in funcs:
                if f.file_path == file_path:
                    source_code = f.source
                    break

            # Fallback: Read lines directly from file if on disk
            if not source_code:
                full_path = (self.repo_path / file_path).resolve()
                if full_path.is_file():
                    try:
                        content = full_path.read_text(encoding="utf-8", errors="replace")
                        s_line = data.get("start_line")
                        e_line = data.get("end_line")
                        if s_line is not None and e_line is not None:
                            lines = content.splitlines()
                            source_code = "\n".join(lines[max(0, s_line - 1):min(len(lines), e_line)])
                        else:
                            source_code = content[:2000]
                    except Exception:
                        pass

        return {
            "id": node_id,
            "label": data.get("label") or data.get("name", node_id),
            "type": data.get("type", "unknown"),
            "file": file_path,
            "start_line": data.get("start_line"),
            "end_line": data.get("end_line"),
            "source": source_code,
            "callers": callers,
            "callees": callees,
            "dependencies": file_deps,
            "note": data.get("note"),
        }

    def stats(self) -> dict[str, Any]:
        """Aggregate database stats, graph stats, embedding manifest, and staleness flags."""
        self._ensure_not_indexing()
        store = self._get_store()
        store_stats = store.get_stats()

        graph = self._get_graph()
        graph_data: dict[str, Any] = {}
        if graph is not None:
            resolved = sum(1 for _, _, rel in graph.graph.edges(data="rel") if rel == "CALLS")
            graph_data = {
                "nodes": graph.graph.number_of_nodes(),
                "edges": graph.graph.number_of_edges(),
                "resolved_calls": resolved,
                "resolution_rate": min(1.0, resolved / store_stats.total_calls) if store_stats.total_calls else None,
            }

        embedding_data: dict[str, Any] = VectorStore.read_manifest(self.repo_path) or {}

        # Staleness detection
        stale_reasons: list[str] = []
        is_stale = False
        get_settings()

        if store_stats.total_files == 0:
            is_stale = True
            stale_reasons.append("Metadata index has 0 files indexed.")

        if not graph_data or graph_data.get("nodes", 0) == 0:
            is_stale = True
            stale_reasons.append("Code graph has not been generated.")
        else:
            if is_graph_stale(self.repo_path):
                is_stale = True
                stale_reasons.append("Code graph is stale relative to metadata store.")

        emb_items = embedding_data.get("item_count", embedding_data.get("total_items", 0))
        emb_model = embedding_data.get("model_name")
        active_model = get_configured_embedding_model()
        if not embedding_data or emb_items == 0:
            is_stale = True
            stale_reasons.append("Semantic embeddings have not been generated.")
        elif emb_model and emb_model != active_model:
            is_stale = True
            stale_reasons.append(
                f"Embeddings model '{emb_model}' differs from configured '{active_model}'."
            )

        return {
            "repository": str(self.repo_path),
            "database": {
                "files": store_stats.total_files,
                "classes": store_stats.total_classes,
                "functions": store_stats.total_functions,
                "methods": store_stats.total_methods,
                "imports": store_stats.total_imports,
                "calls": store_stats.total_calls,
                "files_with_errors": store_stats.files_with_errors,
            },
            "graph": graph_data,
            "embeddings": {
                "total_items": emb_items,
                "model_name": emb_model,
            },
            "is_stale": is_stale,
            "stale_reasons": stale_reasons,
        }

