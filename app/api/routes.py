"""FastAPI API endpoints and routing handlers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request, Response
from pydantic import SecretStr

from app.ai.base_provider import (
    ApiKeyRequiredError,
)
from app.ai.provider_factory import get_provider, get_registered_providers
from app.api.schemas import (
    AskRequest,
    ConfigResponse,
    GraphNodeResponse,
    GraphViewResponse,
    HealthResponse,
    IndexJobResponse,
    IndexJobStatusResponse,
    IndexRequest,
    ProviderInfoResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchRequest,
    SearchResponse,
    IncrementalUpdateRequest,
    BenchmarkOneRequest,
    BenchmarkOneResponse,
    QuerySetInfo,
    BenchmarkRunRequest,
    ValidateKeyRequest,
    ValidateKeyResponse,
)
from app.api.state import (
    create_background_job,
    get_background_job,
    get_service_for_target,
    resolve_repo_or_workspace,
    update_job_status,
)
from app.security import validation_rate_limiter
from app.benchmark.baselines import ALL_STRATEGIES, GraphRetrievalStrategy, WholeFileOfTargetStrategy
from app.benchmark.runner import BenchmarkQuery, BenchmarkResult, run_benchmark
from app.benchmark.token_meter import measure_tokens
from app.config import get_settings
from app.services.incremental import IncrementalReport, update_index
from app.services.pipeline import (
    AskResult,
    CombinedIndexReport,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryTooLongError(Exception):
    """Raised when a query string exceeds configured MAX_QUERY_CHARS."""

    def __init__(self, query_length: int, max_length: int) -> None:
        self.query_length = query_length
        self.max_length = max_length
        super().__init__(
            f"Query is {query_length:,} characters; the maximum is {max_length:,}. Shorten it or paste only the task."
        )


def validate_query_length(query: str) -> None:
    """Validate query character length against configured maximum."""
    settings = get_settings()
    if len(query) > settings.max_query_chars:
        raise QueryTooLongError(len(query), settings.max_query_chars)


def _run_background_index(
    job_id: str,
    repo_path: str | None,
    workspace_id: str | None,
    force: bool,
) -> None:
    """Worker function for asynchronous background indexing."""
    update_job_status(job_id, "running")
    try:
        svc = get_service_for_target(repo_path=repo_path, workspace_id=workspace_id)
        report = svc.index(force=force)
        update_job_status(job_id, "completed", result=report)
    except Exception as exc:
        logger.exception(f"Background indexing job {job_id} failed: {exc}")
        update_job_status(job_id, "failed", error=str(exc))


@router.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Check service health and provider configuration readiness."""
    settings = get_settings()
    provider = get_provider(raise_if_model_missing=False)

    provider_statuses: dict[str, bool] = {}
    for p_name in get_registered_providers():
        try:
            p_inst = get_provider(p_name, raise_if_model_missing=False)
            provider_statuses[p_name] = p_inst.is_configured()
        except Exception:
            provider_statuses[p_name] = False

    active_model = settings.get_active_model() or settings.ai_model or ""
    return HealthResponse(
        status="ok",
        version="0.8.0",
        provider_configured=provider.is_configured(),
        provider_name=provider.name,
        model_name=active_model,
        embedding_model=settings.embedding_model,
        max_query_chars=settings.max_query_chars,
        providers=provider_statuses,
    )


@router.get("/config", response_model=ConfigResponse, tags=["System"])
def get_config() -> ConfigResponse:
    """Return public system configuration parameters."""
    settings = get_settings()
    provider = get_provider(raise_if_model_missing=False)
    active_model = settings.get_active_model() or settings.ai_model or ""
    return ConfigResponse(
        ai_provider=settings.ai_provider,
        ai_model=active_model,
        embedding_model=settings.embedding_model,
        provider_configured=provider.is_configured(),
        context_token_budget=settings.context_token_budget,
        max_query_chars=settings.max_query_chars,
    )


_SERVER_MODELS_CACHE: dict[str, list[str]] = {}


@router.get("/providers", response_model=list[ProviderInfoResponse], tags=["System"])
def list_providers() -> list[ProviderInfoResponse]:
    """List all supported AI completion providers, server configuration status, and whether user keys are accepted."""
    settings = get_settings()
    results: list[ProviderInfoResponse] = []
    for p_name in get_registered_providers():
        try:
            prov = get_provider(p_name, raise_if_model_missing=False)
            default_m = settings.get_model_for_provider(p_name, raise_if_missing=False)
            hint = prov.get_setup_hint()
            is_loc = prov.is_local
            available_models = (
                prov.get_available_models()
                if hasattr(prov, "get_available_models")
                else []
            )
            if is_loc:
                configured = prov.is_configured()
                accepts_user_key = False
            else:
                server_key = settings.get_key_for_provider(p_name)
                configured = bool(
                    server_key and (server_key.get_secret_value() if isinstance(server_key, SecretStr) else server_key)
                )
                accepts_user_key = True
        except Exception as exc:
            configured = False
            default_m = None
            hint = str(exc)
            is_loc = False
            available_models = []
            accepts_user_key = p_name != "ollama"

        results.append(
            ProviderInfoResponse(
                name=p_name,
                configured=configured,
                default_model=default_m,
                is_local=is_loc,
                setup_hint=hint,
                available_models=available_models,
                accepts_user_key=accepts_user_key,
            )
        )
    return results


@router.post(
    "/providers/{name}/validate",
    response_model=ValidateKeyResponse,
    tags=["System"],
)
def validate_provider_key(
    name: str,
    request: Request,
    response: Response,
    body: ValidateKeyRequest | None = None,
    x_provider_key: str | None = Header(default=None, alias="X-Provider-Key"),
) -> ValidateKeyResponse:
    """Validate credentials for a provider making the cheapest possible call.

    Rate limited to 10 requests per minute per client IP.
    """
    response.headers["Cache-Control"] = "no-store"
    settings = get_settings()

    # 1. In-memory IP rate limiting
    client_ip = getattr(getattr(request, "client", None), "host", "unknown")
    if not validation_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many validation attempts. Please wait a minute and retry.",
        )

    # 2. Canonical name check
    raw_name = name.lower()
    canonical_name = "openai_compat" if raw_name == "openai" else ("ollama" if raw_name == "local" else raw_name)

    valid_providers = get_registered_providers() + ["openai", "local"]
    if canonical_name not in valid_providers:
        return ValidateKeyResponse(
            valid=False,
            models=[],
            message=f"Unknown provider '{name}'.",
        )

    # 3. Local Ollama validation (no key required)
    if canonical_name == "ollama":
        try:
            prov = get_provider("ollama", raise_if_model_missing=False)
            is_valid, models, msg = prov.validate_key()
            return ValidateKeyResponse(valid=is_valid, models=models, message=msg)
        except Exception as exc:
            return ValidateKeyResponse(
                valid=False,
                models=[],
                message=f"Ollama error: {exc}",
            )

    # 4. Cloud providers key precedence: Header key > server .env key
    header_key = x_provider_key.strip() if x_provider_key else None
    server_key = settings.get_key_for_provider(canonical_name)
    raw_server_key = (
        server_key.get_secret_value() if isinstance(server_key, SecretStr) else (str(server_key) if server_key else None)
    )
    effective_key = header_key or raw_server_key

    if not effective_key:
        return ValidateKeyResponse(
            valid=False,
            models=[],
            message=f"No API key provided. Please enter your {canonical_name} API key.",
        )

    # 5. Build stateless instance with the key and validate
    target_model = body.model if body and body.model else None
    try:
        prov = get_provider(
            provider_name=canonical_name,
            api_key=effective_key,
            model=target_model,
            raise_if_model_missing=False,
        )
        is_valid, models, msg = prov.validate_key(model=target_model)
        return ValidateKeyResponse(valid=is_valid, models=models, message=msg)
    except Exception as exc:
        return ValidateKeyResponse(
            valid=False,
            models=[],
            message=f"Validation error: {exc}",
        )


@router.get(
    "/providers/{name}/models",
    response_model=list[str],
    tags=["System"],
)
def get_provider_models(
    name: str,
    response: Response,
    x_provider_key: str | None = Header(default=None, alias="X-Provider-Key"),
) -> list[str]:
    """Retrieve available models for a provider, optionally using user-supplied key."""
    response.headers["Cache-Control"] = "no-store"
    settings = get_settings()

    raw_name = name.lower()
    canonical_name = "openai_compat" if raw_name == "openai" else ("ollama" if raw_name == "local" else raw_name)

    if canonical_name == "ollama":
        prov = get_provider("ollama", raise_if_model_missing=False)
        return prov.get_available_models()

    header_key = x_provider_key.strip() if x_provider_key else None
    if header_key:
        # Fetched with user-supplied key: NEVER share across users / shared cache
        prov = get_provider(
            provider_name=canonical_name,
            api_key=header_key,
            raise_if_model_missing=False,
        )
        return prov.get_available_models()

    # Server-side key fallback: cache allowed for server key
    if canonical_name in _SERVER_MODELS_CACHE:
        return _SERVER_MODELS_CACHE[canonical_name]

    server_key = settings.get_key_for_provider(canonical_name)
    raw_server_key = (
        server_key.get_secret_value() if isinstance(server_key, SecretStr) else (str(server_key) if server_key else None)
    )
    if raw_server_key:
        prov = get_provider(
            provider_name=canonical_name,
            api_key=raw_server_key,
            raise_if_model_missing=False,
        )
        models = prov.get_available_models()
        _SERVER_MODELS_CACHE[canonical_name] = models
        return models

    # Fallback to provider static defaults
    prov = get_provider(provider_name=canonical_name, raise_if_model_missing=False)
    return prov.get_available_models()


@router.post(
    "/index",
    response_model=CombinedIndexReport | IndexJobResponse,
    tags=["Indexing"],
)
def index_repository(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    background: bool = Query(default=False, description="Run indexing as an asynchronous background job"),
) -> CombinedIndexReport | IndexJobResponse:
    """Index an entire repository (scan, code graph, semantic embeddings)."""
    if background:
        job_target = request.workspace_id or request.repo_path or "workspace"
        job_id = create_background_job(job_target)
        background_tasks.add_task(
            _run_background_index,
            job_id,
            request.repo_path,
            request.workspace_id,
            request.force,
        )
        return IndexJobResponse(job_id=job_id, status="queued")

    svc = get_service_for_target(repo_path=request.repo_path, workspace_id=request.workspace_id)
    return svc.index(force=request.force)


@router.get(
    "/index/status/{job_id}",
    response_model=IndexJobStatusResponse,
    tags=["Indexing"],
)
def get_index_job_status(job_id: str) -> IndexJobStatusResponse:
    """Retrieve execution status of a background indexing job."""
    job = get_background_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    return IndexJobStatusResponse(**job)


@router.post("/search", response_model=SearchResponse, tags=["Retrieval"])
def search_repository(request: SearchRequest) -> SearchResponse:
    """Perform semantic code search across indexed repository entities."""
    validate_query_length(request.query)
    svc = get_service_for_target(repo_path=request.repo_path, workspace_id=request.workspace_id)
    hits = svc.search(query=request.query, top_k=request.top_k, kinds=request.kinds)
    return SearchResponse(query=request.query, total_hits=len(hits), hits=hits)


@router.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
def retrieve_context(request: RetrieveRequest) -> RetrieveResponse:
    """Run hybrid candidate retrieval and assemble token-budgeted prompt context."""
    validate_query_length(request.query)
    svc = get_service_for_target(repo_path=request.repo_path, workspace_id=request.workspace_id)
    ret_res, built_ctx = svc.retrieve(
        query=request.query,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        use_semantic=request.use_semantic,
        use_keyword=request.use_keyword,
        use_graph=request.use_graph,
        explain=request.explain,
    )
    return RetrieveResponse(retrieval=ret_res, context=built_ctx)


@router.post("/ask", response_model=AskResult, tags=["Assistant"])
def ask_coding_assistant(
    request: AskRequest,
    response: Response,
    x_provider_key: str | None = Header(default=None, alias="X-Provider-Key"),
) -> AskResult:
    """Run full pipeline: retrieve context, invoke AI provider, and return parsed solution."""
    validate_query_length(request.query)
    response.headers["Cache-Control"] = "no-store"
    settings = get_settings()

    target_provider = (request.provider or settings.ai_provider).lower()
    valid_providers = get_registered_providers() + ["openai", "local"]
    if target_provider not in valid_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown AI provider '{request.provider}'. Supported providers: {', '.join(sorted(get_registered_providers()))}",
        )

    canonical_provider = "openai_compat" if target_provider == "openai" else ("ollama" if target_provider == "local" else target_provider)

    # Key precedence: Header key (if present) > server .env key (if configured) > error
    header_key = x_provider_key.strip() if x_provider_key else None
    server_key = settings.get_key_for_provider(canonical_provider)
    raw_server_key = (
        server_key.get_secret_value() if isinstance(server_key, SecretStr) else (str(server_key) if server_key else None)
    )

    if header_key:
        api_key = header_key
        key_source = "user"
    elif raw_server_key:
        api_key = raw_server_key
        key_source = "server"
    else:
        api_key = None
        key_source = None

    if canonical_provider != "ollama" and not api_key and not request.dry_run:
        raise ApiKeyRequiredError(
            f"API key required for provider '{canonical_provider}'. Please enter your API key in the web page."
        )

    svc = get_service_for_target(repo_path=request.repo_path, workspace_id=request.workspace_id)
    return svc.ask(
        query=request.query,
        budget=request.max_tokens,
        dry_run=request.dry_run,
        provider_override=canonical_provider,
        model_override=request.model,
        temperature_override=request.temperature,
        max_output_tokens_override=request.max_output_tokens,
        api_key=api_key,
        key_source=key_source,
    )


@router.get("/graph", response_model=GraphViewResponse, tags=["Graph"])
def get_graph_subgraph(
    repo_path: str | None = Query(default=None, description="Repository path"),
    workspace_id: str | None = Query(default=None, description="Workspace ID"),
    center: str | None = Query(default=None, description="Center node qualified name"),
    hops: int = Query(default=2, ge=1, le=5, description="Search radius hops"),
    max_nodes: int = Query(default=150, ge=1, le=500, description="Max nodes to include"),
    rels: str | None = Query(default=None, description="Comma-separated relationship filter"),
) -> GraphViewResponse:
    """Query code graph nodes and edges for visualization."""
    svc = get_service_for_target(repo_path=repo_path, workspace_id=workspace_id)
    rel_list = [r.strip() for r in rels.split(",")] if rels else None
    view_data = svc.graph_view(center=center, hops=hops, max_nodes=max_nodes, rels=rel_list)

    ws_name = None
    if workspace_id:
        from app.workspace.manager import WorkspaceManager
        ws_rec = WorkspaceManager().get_workspace(workspace_id)
        if ws_rec:
            ws_name = ws_rec.get("name")
    view_data["workspace_name"] = ws_name

    return GraphViewResponse(**view_data)


@router.get("/graph/node", response_model=GraphNodeResponse, tags=["Graph"])
def get_graph_node_details(
    repo_path: str | None = Query(default=None, description="Repository path"),
    workspace_id: str | None = Query(default=None, description="Workspace ID"),
    id: str = Query(..., description="Node ID or qualified name"),
) -> GraphNodeResponse:
    """Query comprehensive node details, callers, callees, dependencies, and source snippet."""
    svc = get_service_for_target(repo_path=repo_path, workspace_id=workspace_id)
    node_data = svc.graph_node(node_id=id)
    return GraphNodeResponse(**node_data)


@router.get("/stats", tags=["System"])
def get_repository_stats(
    repo_path: str | None = Query(default=None, description="Repository path"),
    workspace_id: str | None = Query(default=None, description="Workspace ID"),
) -> dict[str, Any]:
    """Retrieve indexed entity counts, graph metrics, and staleness diagnostics."""
    svc = get_service_for_target(repo_path=repo_path, workspace_id=workspace_id)
    return svc.stats()


@router.post(
    "/index/update",
    response_model=IncrementalReport,
    tags=["Indexing"],
)
def update_index_endpoint(request: IncrementalUpdateRequest) -> IncrementalReport:
    """Perform one-shot incremental indexing updating metadata, graph, and embeddings."""
    target_path = resolve_repo_or_workspace(repo_path=request.repo_path, workspace_id=request.workspace_id)
    return update_index(repo_path=target_path, verify=request.verify)


@router.get(
    "/benchmark/querysets",
    response_model=list[QuerySetInfo],
    tags=["Benchmark"],
)
def list_benchmark_querysets() -> list[QuerySetInfo]:
    """List available benchmark query set JSON files in eval/ directory."""
    import json
    eval_dir = Path("eval")
    results: list[QuerySetInfo] = []
    if not eval_dir.exists():
        return results
    for p in sorted(eval_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                sample = data[0].get("query") if isinstance(data[0], dict) else None
                results.append(
                    QuerySetInfo(
                        filename=p.name,
                        path=str(p),
                        query_count=len(data),
                        sample_query=sample,
                    )
                )
        except Exception:
            continue
    return results


@router.post(
    "/benchmark-one",
    response_model=BenchmarkOneResponse,
    tags=["Benchmark"],
)
def benchmark_one_endpoint(request: BenchmarkOneRequest) -> BenchmarkOneResponse:
    """Run fast single-query token size comparison (whole_file vs graph retrieval)."""
    validate_query_length(request.query)
    target_path = resolve_repo_or_workspace(repo_path=request.repo_path, workspace_id=request.workspace_id)
    wf_strat = WholeFileOfTargetStrategy()
    gr_strat = GraphRetrievalStrategy()

    wf_ctx = wf_strat.build(target_path, request.query, budget=request.budget)
    gr_ctx = gr_strat.build(target_path, request.query, budget=request.budget)

    wf_meas = measure_tokens(wf_ctx.context_text)
    gr_meas = measure_tokens(gr_ctx.context_text)

    wf_tokens = wf_meas.effective_tokens
    gr_tokens = gr_meas.effective_tokens

    reduction_pct = 0.0
    if wf_tokens > 0:
        reduction_pct = round(((wf_tokens - gr_tokens) / wf_tokens) * 100.0, 2)

    return BenchmarkOneResponse(
        query=request.query,
        whole_file_tokens=wf_tokens,
        graph_tokens=gr_tokens,
        reduction_pct=reduction_pct,
        files_inspected=gr_ctx.files_inspected,
        counter=gr_meas.counter,
    )


@router.post(
    "/benchmark",
    response_model=BenchmarkResult,
    tags=["Benchmark"],
)
def run_benchmark_endpoint(request: BenchmarkRunRequest) -> BenchmarkResult:
    """Run full benchmark across requested strategies."""
    import json
    target_path = resolve_repo_or_workspace(repo_path=request.repo_path, workspace_id=request.workspace_id)

    query_dicts = []
    if request.queries:
        query_dicts = request.queries
    elif request.query_file:
        q_path = Path(request.query_file)
        if not q_path.is_absolute():
            if not q_path.exists() and (Path("eval") / q_path).exists():
                q_path = Path("eval") / q_path
        if not q_path.exists():
            raise HTTPException(status_code=404, detail=f"Query file '{request.query_file}' not found.")
        with open(q_path, "r", encoding="utf-8") as f:
            query_dicts = json.load(f)
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'queries' or 'query_file'.")

    queries = [BenchmarkQuery(**q) for q in query_dicts]

    selected_strats = None
    if request.strategies:
        strat_map = {s.name: s for s in ALL_STRATEGIES}
        selected_strats = [strat_map[s] for s in request.strategies if s in strat_map]

    result = run_benchmark(
        repo_path=target_path,
        queries=queries,
        strategies=selected_strats or ALL_STRATEGIES,
        budget=request.budget,
        call_ai=request.call_ai,
    )
    return result

