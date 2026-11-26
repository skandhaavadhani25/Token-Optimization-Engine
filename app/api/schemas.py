"""Pydantic schemas for FastAPI request and response payloads."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import BuiltContext, RetrievalResult, SemanticHit
from app.services.pipeline import CombinedIndexReport


class ErrorDetail(BaseModel):
    """Structured error descriptor."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable description of the error")
    details: Any | None = Field(default=None, description="Optional extra diagnostic details")
    request_id: str | None = Field(default=None, description="Correlation request ID")


class ErrorResponse(BaseModel):
    """Standard error response format across all endpoints."""

    model_config = ConfigDict(frozen=True)

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Service health and capability status."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="ok", description="Service status")
    version: str = Field(default="0.8.0", description="API version")
    provider_configured: bool = Field(description="Whether default LLM provider credentials are ready")
    provider_name: str = Field(description="Configured default AI provider name")
    model_name: str = Field(description="Configured default model identifier")
    embedding_model: str = Field(description="Configured embedding model identifier")
    max_query_chars: int = Field(default=4000, description="Configured maximum query character limit")
    providers: dict[str, bool] = Field(default_factory=dict, description="Configuration readiness boolean per provider")


class ProviderInfoResponse(BaseModel):
    """Information and readiness status of an AI provider."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Canonical provider identifier name")
    configured: bool = Field(description="Whether provider credentials and dependencies are ready")
    default_model: str | None = Field(default=None, description="Configured default model for this provider")
    is_local: bool = Field(default=False, description="Whether the provider executes locally without cloud APIs")
    setup_hint: str | None = Field(default=None, description="Actionable configuration hint if unconfigured")
    available_models: list[str] = Field(default_factory=list, description="Locally available or suggested model IDs")
    accepts_user_key: bool = Field(default=False, description="Whether this provider accepts a user-provided API key")


class ValidateKeyRequest(BaseModel):
    """Payload to validate provider credentials."""

    model_config = ConfigDict(frozen=True)

    model: str | None = Field(default=None, description="Optional target model identifier to validate")


class ValidateKeyResponse(BaseModel):
    """Result of provider credential validation."""

    model_config = ConfigDict(frozen=True)

    valid: bool = Field(description="Whether the credentials are valid and functional")
    models: list[str] = Field(default_factory=list, description="Available models detected for this key")
    message: str | None = Field(default=None, description="Human-friendly validation message")


class ConfigResponse(BaseModel):
    """Public service configuration parameters."""

    model_config = ConfigDict(frozen=True)

    ai_provider: str = Field(description="Configured AI provider name")
    ai_model: str = Field(description="Configured default AI model identifier")
    embedding_model: str = Field(description="Configured embedding model name")
    provider_configured: bool = Field(description="Whether provider credentials are set")
    context_token_budget: int = Field(description="Default context token budget")
    max_query_chars: int = Field(description="Configured maximum query character limit")


class IndexRequest(BaseModel):
    """Request payload to index a repository."""

    repo_path: str | None = Field(default=None, description="Path to local repository")
    workspace_id: str | None = Field(default=None, description="Workspace ID")
    force: bool = Field(default=False, description="Whether to bypass caches and force re-index")


class IndexJobResponse(BaseModel):
    """Response returned when indexing is started as a background job."""

    job_id: str = Field(description="Unique job identifier")
    status: str = Field(default="queued", description="Initial job status")


class IndexJobStatusResponse(BaseModel):
    """Status query response for an asynchronous index job."""

    job_id: str
    repo_path: str | None = None
    status: str
    result: CombinedIndexReport | None = None
    error: str | None = None


class SearchRequest(BaseModel):
    """Request payload for semantic code search."""

    repo_path: str | None = Field(default=None, description="Repository path")
    workspace_id: str | None = Field(default=None, description="Workspace ID")
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Max results")
    kinds: list[str] | None = Field(default=None, description="Filter by vector kind")


class SearchResponse(BaseModel):
    """Search results response."""

    query: str
    total_hits: int
    hits: list[SemanticHit]


class RetrieveRequest(BaseModel):
    """Request payload for hybrid retrieval and context packing."""

    repo_path: str | None = Field(default=None, description="Repository path")
    workspace_id: str | None = Field(default=None, description="Workspace ID")
    query: str = Field(..., description="Task query string")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of candidates to retrieve")
    max_tokens: int | None = Field(default=None, ge=100, description="Context token budget override")
    use_semantic: bool = Field(default=True, description="Enable semantic vector search")
    use_keyword: bool = Field(default=True, description="Enable keyword search")
    use_graph: bool = Field(default=True, description="Enable graph neighborhood search")
    explain: bool = Field(default=False, description="Include retrieval score explanations")


class RetrieveResponse(BaseModel):
    """Hybrid retrieval and token-budgeted context response."""

    retrieval: RetrievalResult
    context: BuiltContext


class AskRequest(BaseModel):
    """Request payload to execute an end-to-end AI coding query."""

    repo_path: str | None = Field(default=None, description="Repository path")
    workspace_id: str | None = Field(default=None, description="Workspace ID")
    query: str = Field(..., description="User coding task instruction")
    max_tokens: int | None = Field(default=None, ge=100, description="Context token budget")
    dry_run: bool = Field(default=False, description="Assemble prompt without invoking model")
    provider: str | None = Field(default=None, description="AI provider override: claude | gemini | openai_compat | ollama")
    model: str | None = Field(default=None, description="Provider model override")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
    max_output_tokens: int | None = Field(default=None, ge=1, description="Max generated tokens")


class GraphViewResponse(BaseModel):
    """Graph structure formatted for visual representation."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    truncated: bool
    total_nodes: int
    total_edges: int
    center: str | None = None
    workspace_name: str | None = None


class GraphNodeResponse(BaseModel):
    """Detailed metadata and local connectivity for an individual node."""

    id: str
    label: str
    type: str
    file: str | None
    start_line: int | None
    end_line: int | None
    source: str | None
    callers: list[str]
    callees: list[str]
    dependencies: list[str]


class IncrementalUpdateRequest(BaseModel):
    """Request payload for one-shot incremental index update."""

    repo_path: str | None = Field(default=None, description="Repository path")
    workspace_id: str | None = Field(default=None, description="Workspace ID")
    verify: bool = Field(default=False, description="Verify equivalence against full rebuild")


class BenchmarkOneRequest(BaseModel):
    """Request payload to compute single-query reduction vs whole_file baseline."""

    repo_path: str | None = Field(default=None, description="Repository path")
    workspace_id: str | None = Field(default=None, description="Workspace ID")
    query: str = Field(..., description="Query string to evaluate")
    budget: int = Field(default=4000, ge=100, description="Token budget")


class BenchmarkOneResponse(BaseModel):
    """Fast single-query comparison response."""

    query: str
    whole_file_tokens: int
    graph_tokens: int
    reduction_pct: float
    files_inspected: list[str]
    counter: str = "tiktoken"


class QuerySetInfo(BaseModel):
    """Information about an evaluation query set."""

    filename: str
    path: str
    query_count: int
    sample_query: str | None = None


class BenchmarkRunRequest(BaseModel):
    """Request payload to execute a full benchmark run."""

    repo_path: str | None = Field(default=None, description="Repository path")
    workspace_id: str | None = Field(default=None, description="Workspace ID")
    query_file: str | None = Field(default=None, description="Query file name or path in eval/")
    queries: list[dict[str, Any]] | None = Field(default=None, description="Inline query set array")
    budget: int = Field(default=4000, ge=100, description="Token budget")
    call_ai: bool = Field(default=False, description="Execute live AI completions")
    strategies: list[str] | None = Field(default=None, description="Strategies to run")

