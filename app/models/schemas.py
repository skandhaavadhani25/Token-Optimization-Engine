"""Pydantic schemas for the code-context-engine."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FileInfo(BaseModel):
    """Metadata for a single source file in the repository."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(
        ...,
        description="Path relative to the repository root in POSIX format (e.g. app/main.py)",
    )
    absolute_path: str = Field(
        ...,
        description="Normalized absolute path on the host filesystem",
    )
    size_bytes: int = Field(
        ...,
        ge=0,
        description="File size in bytes",
    )
    line_count: int = Field(
        ...,
        ge=0,
        description="Total number of lines in the file",
    )
    extension: str = Field(
        ...,
        description="File extension including leading dot (e.g. .py)",
    )
    last_modified: float = Field(
        ...,
        description="Last modification timestamp in epoch seconds",
    )


class ScanResult(BaseModel):
    """Aggregate statistics and collected file metadata from a repository scan."""

    model_config = ConfigDict(frozen=True)

    repo_root: str = Field(
        ...,
        description="Normalized path to the repository root",
    )
    files: list[FileInfo] = Field(
        default_factory=list,
        description="List of discovered and filtered source files",
    )
    total_files: int = Field(
        ...,
        ge=0,
        description="Total number of included source files",
    )
    total_size_bytes: int = Field(
        ...,
        ge=0,
        description="Sum of file sizes in bytes across all included files",
    )
    total_lines: int = Field(
        ...,
        ge=0,
        description="Sum of line counts across all included files",
    )
    skipped_dirs: int = Field(
        ...,
        ge=0,
        description="Number of directories skipped or pruned during traversal",
    )
    skipped_files: int = Field(
        ...,
        ge=0,
        description="Number of files skipped during scanning",
    )
    scan_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total time taken for the repository scan in seconds",
    )


class ImportInfo(BaseModel):
    """Structured representation of an import statement."""

    model_config = ConfigDict(frozen=True)

    module: str = Field(description="Imported module path, or empty string for bare relative imports")
    name: str | None = Field(default=None, description="Imported object/symbol name if from-import")
    alias: str | None = Field(default=None, description="Local alias as defined by 'as <alias>'")
    is_from_import: bool = Field(description="True for 'from x import y', False for 'import x'")
    level: int = Field(default=0, ge=0, description="Relative import level (0 for absolute, 1 for ., 2 for ..)")
    lineno: int = Field(ge=1, description="Source line number where import occurs")


class ArgumentInfo(BaseModel):
    """Information about a function or method parameter."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Parameter name")
    annotation: str | None = Field(default=None, description="Type annotation source expression")
    default: str | None = Field(default=None, description="Default value source expression")
    kind: str = Field(description="Argument kind: positional, keyword_only, vararg, kwarg")


class CallInfo(BaseModel):
    """Information about a function or method invocation."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Dotted callee name as written, e.g. get_user, self.verify, utils.hash")
    lineno: int = Field(ge=1, description="Source line number of the call")


class FunctionInfo(BaseModel):
    """Metadata and code structure for a function or class method."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Function or method name")
    qualified_name: str = Field(description="Dotted qualified name, e.g. module.func or module.Class.method")
    file_path: str = Field(description="POSIX relative file path within repository")
    start_line: int = Field(ge=1, description="Starting line number including decorators")
    end_line: int = Field(ge=1, description="Ending line number")
    arguments: list[ArgumentInfo] = Field(default_factory=list, description="Function arguments")
    return_annotation: str | None = Field(default=None, description="Return type annotation source expression")
    decorators: list[str] = Field(default_factory=list, description="Decorators as source expressions")
    docstring: str | None = Field(default=None, description="Function docstring if present")
    calls: list[CallInfo] = Field(default_factory=list, description="De-duplicated list of function calls made")
    source: str = Field(description="Exact source code slice including decorators")
    is_async: bool = Field(default=False, description="Whether the function is defined with async def")
    is_method: bool = Field(default=False, description="Whether the function is defined inside a class")
    parent_class: str | None = Field(default=None, description="Name of enclosing class if is_method is True")


class ClassInfo(BaseModel):
    """Metadata and structure for a class definition."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Class name")
    qualified_name: str = Field(description="Dotted qualified name, e.g. module.ClassName")
    file_path: str = Field(description="POSIX relative file path within repository")
    start_line: int = Field(ge=1, description="Starting line number including decorators")
    end_line: int = Field(ge=1, description="Ending line number")
    bases: list[str] = Field(default_factory=list, description="Base class expressions as strings")
    decorators: list[str] = Field(default_factory=list, description="Decorators as source expressions")
    docstring: str | None = Field(default=None, description="Class docstring if present")
    methods: list[FunctionInfo] = Field(default_factory=list, description="Methods defined in class")
    source: str = Field(description="Exact source code slice including decorators")


class ParsedFile(BaseModel):
    """Structured extraction of an individual Python file."""

    model_config = ConfigDict(frozen=True)

    file_path: str = Field(description="POSIX relative file path within repository")
    module_name: str = Field(description="Dotted Python module name derived from relative path")
    imports: list[ImportInfo] = Field(default_factory=list, description="Imports found in file")
    classes: list[ClassInfo] = Field(default_factory=list, description="Classes defined at module level")
    functions: list[FunctionInfo] = Field(default_factory=list, description="Functions defined at module level")
    module_docstring: str | None = Field(default=None, description="Module-level docstring if present")
    line_count: int = Field(ge=0, description="Total line count of the file")
    content_hash: str = Field(description="SHA-256 hash of raw file bytes")
    parse_error: str | None = Field(default=None, description="Error message if parsing failed, else None")


class ParseSummary(BaseModel):
    """Aggregate statistics for an entire repository parsing run."""

    model_config = ConfigDict(frozen=True)

    total_files: int = Field(ge=0, description="Total number of files processed")
    parsed_ok: int = Field(ge=0, description="Count of successfully parsed files")
    failed: int = Field(ge=0, description="Count of files that failed parsing")
    total_functions: int = Field(ge=0, description="Total module-level functions across files")
    total_classes: int = Field(ge=0, description="Total classes across files")
    total_methods: int = Field(ge=0, description="Total class methods across files")
    total_imports: int = Field(ge=0, description="Total import records across files")
    total_calls: int = Field(ge=0, description="Total function and method calls extracted")
    parse_time_seconds: float = Field(ge=0.0, description="Total parsing duration in seconds")
    errors: list[dict[str, str]] = Field(default_factory=list, description="List of {file, error} dicts")


class StoreStats(BaseModel):
    """Aggregate statistics of the metadata database."""

    model_config = ConfigDict(frozen=True)

    total_files: int = Field(ge=0, description="Total indexed files in database")
    total_functions: int = Field(ge=0, description="Total module-level functions")
    total_classes: int = Field(ge=0, description="Total indexed classes")
    total_methods: int = Field(ge=0, description="Total class methods")
    total_imports: int = Field(ge=0, description="Total import records")
    total_calls: int = Field(ge=0, description="Total recorded function calls")
    files_with_errors: int = Field(ge=0, description="Files with parse errors recorded")
    db_size_bytes: int = Field(default=0, ge=0, description="Reserved; MongoDB does not expose per-repo size here")
    last_indexed_at: float | None = Field(default=None, description="Timestamp of last indexing run in epoch seconds")


class ChangeSet(BaseModel):
    """File change detection results comparing repository scan against the store."""

    model_config = ConfigDict(frozen=True)

    added: list[str] = Field(default_factory=list, description="Files found on disk but absent from index")
    modified: list[str] = Field(default_factory=list, description="Files whose content hash changed")
    deleted: list[str] = Field(default_factory=list, description="Files present in index but removed from disk")
    unchanged: list[str] = Field(default_factory=list, description="Files whose content hash is unchanged")


class IndexReport(BaseModel):
    """Execution summary report for an indexing run."""

    model_config = ConfigDict(frozen=True)

    files_scanned: int = Field(ge=0, description="Total source files discovered during scan")
    added: int = Field(ge=0, description="Number of newly indexed files")
    modified: int = Field(ge=0, description="Number of modified files updated")
    deleted: int = Field(ge=0, description="Number of removed files purged from index")
    unchanged: int = Field(ge=0, description="Number of unmodified files skipped")
    parsed: int = Field(ge=0, description="Number of files actually parsed in this run")
    parse_errors: int = Field(ge=0, description="Number of files that produced parse errors")
    scan_time: float = Field(ge=0.0, description="Seconds spent scanning filesystem")
    parse_time: float = Field(ge=0.0, description="Seconds spent parsing ASTs")
    store_time: float = Field(ge=0.0, description="Seconds spent persisting to database")
    total_time: float = Field(ge=0.0, description="Total elapsed seconds for entire index operation")


class CallEdgeResult(BaseModel):
    """Result item representing a caller or callee graph relationship."""

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(description="Unique node identifier in code graph")
    qualified_name: str = Field(description="Fully qualified name of function/method")
    file: str = Field(description="POSIX relative file path where function/method resides")
    depth: int = Field(ge=1, description="Traversal distance from source node")
    confidence: str = Field(description="Resolution confidence: exact, high, medium, low")
    resolution: str = Field(description="Resolution strategy rule that matched")
    lineno: int = Field(ge=1, description="Call site line number")


class CallResolutionStats(BaseModel):
    """Metrics tracking quality and coverage of call edge resolution."""

    model_config = ConfigDict(frozen=True)

    total_calls: int = Field(default=0, ge=0, description="Total calls extracted across repository")
    resolved_calls: int = Field(default=0, ge=0, description="Total calls resolved to internal definitions")
    resolved_by_rule: dict[str, int] = Field(
        default_factory=dict,
        description="Counts of resolved calls broken down by rule name",
    )
    skipped_ambiguous: int = Field(default=0, ge=0, description="Ambiguous calls omitted")
    external_calls: int = Field(default=0, ge=0, description="Calls to external or stdlib modules")
    builtin_calls: int = Field(default=0, ge=0, description="Calls to Python language built-ins")
    unresolved_calls: int = Field(default=0, ge=0, description="Unresolved function or method invocations")


class GraphBuildReport(BaseModel):
    """Execution report describing graph structure and call resolution performance."""

    model_config = ConfigDict(frozen=True)

    repo_name: str = Field(description="Name of the repository")
    total_nodes: int = Field(ge=0, description="Total number of nodes in graph")
    total_edges: int = Field(ge=0, description="Total number of edges in graph")
    node_counts: dict[str, int] = Field(default_factory=dict, description="Node counts keyed by type")
    edge_counts: dict[str, int] = Field(default_factory=dict, description="Edge counts keyed by rel type")
    call_resolution: CallResolutionStats = Field(
        default_factory=CallResolutionStats,
        description="Detailed call resolution statistics",
    )
    build_time_seconds: float = Field(ge=0.0, description="Time taken to construct graph in seconds")


class EmbeddedItem(BaseModel):
    """An embeddable code or metadata item representing an entity."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(description="Unique item identifier, e.g. function_desc:<qualified_name>")
    kind: str = Field(description="Vector kind: function_desc, function_code, class_desc, file_summary")
    ref: str = Field(description="Target entity reference: qualified name or file path")
    file: str = Field(description="POSIX relative file path where entity resides")
    text_hash: str = Field(description="SHA-256 hash of the embedded text")
    text: str = Field(default="", description="The raw natural language or code text to embed")


class ScoredItem(BaseModel):
    """A vector search result from the index before entity deduplication."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(description="Item identifier")
    kind: str = Field(description="Item kind")
    ref: str = Field(description="Entity reference")
    file: str = Field(description="File path")
    text_hash: str = Field(description="Content hash")
    score: float = Field(description="Cosine similarity score")


class SemanticHit(BaseModel):
    """Deduplicated and ranked semantic search result for an entity."""

    model_config = ConfigDict(frozen=True)

    ref: str = Field(description="Qualified name or file path of entity")
    entity_type: str = Field(description="Entity type: function, method, class, file")
    file: str = Field(description="POSIX relative file path")
    score: float = Field(description="Relevance similarity score")
    matched_kinds: list[str] = Field(default_factory=list, description="Vector kinds that matched this entity")
    kind_scores: dict[str, float] = Field(default_factory=dict, description="Raw scores per matched kind")
    start_line: int | None = Field(default=None, description="Starting line in source file if applicable")
    end_line: int | None = Field(default=None, description="Ending line in source file if applicable")


class EmbeddingReport(BaseModel):
    """Execution report describing embedding indexing operation."""

    model_config = ConfigDict(frozen=True)

    items_total: int = Field(default=0, ge=0, description="Total items in vector index")
    items_new: int = Field(default=0, ge=0, description="New items embedded in this run")
    items_changed: int = Field(default=0, ge=0, description="Modified items updated in index")
    items_removed: int = Field(default=0, ge=0, description="Deleted items removed from index")
    items_unchanged: int = Field(default=0, ge=0, description="Unchanged items skipped")
    items_skipped: int = Field(default=0, ge=0, description="Trivially small items skipped")
    by_kind: dict[str, int] = Field(default_factory=dict, description="Counts of total items by kind")
    model_name: str = Field(default="", description="Embedding model name used")
    embedding_time_seconds: float = Field(default=0.0, ge=0.0, description="Time spent generating embeddings")
    total_time_seconds: float = Field(default=0.0, ge=0.0, description="Total indexing duration")
    index_size_bytes: int = Field(default=0, ge=0, description="Total size of vector index and metadata on disk")


class RankedCandidate(BaseModel):
    """Ranked code entity candidate from hybrid retrieval."""

    model_config = ConfigDict(frozen=True)

    ref: str = Field(description="Qualified name or POSIX file path")
    score: float = Field(description="Relevance or hybrid ranking score")
    entity_type: str = Field(description="Entity type: function, method, class, file")
    file: str = Field(description="POSIX relative file path")
    start_line: int | None = Field(default=None, description="Starting line in source file")
    end_line: int | None = Field(default=None, description="Ending line in source file")
    matched_kinds: list[str] = Field(default_factory=list, description="Vector kinds that matched this entity")
    match_type: str = Field(default="semantic", description="Origin match mechanism: semantic, exact, graph")


class TraceOperation(BaseModel):
    """One step the engine ran while serving a request, in execution order."""

    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=1, description="1-based execution order")
    name: str = Field(description="Operation identifier, e.g. 'semantic_search'")
    status: str = Field(default="ran", description="ran, skipped, or failed")
    detail: str = Field(default="", description="Human readable outcome of the operation")
    count: int | None = Field(default=None, description="Items produced by the operation, when applicable")
    seconds: float = Field(default=0.0, ge=0.0, description="Wall-clock duration of the operation")


class SignalHit(BaseModel):
    """A candidate surfaced by one retrieval signal."""

    model_config = ConfigDict(frozen=True)

    ref: str = Field(description="Qualified name of the matched entity")
    signal: str = Field(description="semantic, keyword, ast_call, graph_callee, graph_caller, or exact_target")
    score: float = Field(description="Score assigned by that signal")
    file: str = Field(default="", description="POSIX relative file path")
    note: str = Field(default="", description="Extra explanation, e.g. the keyword that matched")


class GraphEdgeTrace(BaseModel):
    """A code-graph edge walked during retrieval."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Qualified name where the walk started")
    target: str = Field(description="Qualified name reached")
    rel: str = Field(default="CALLS", description="Edge relationship type")
    direction: str = Field(description="callee (outgoing) or caller (incoming)")
    depth: int = Field(default=1, ge=1, description="Hop distance from the target")
    confidence: str = Field(default="", description="Edge resolution confidence")
    outcome: str = Field(description="added, boosted, or already_present")


class SelectionTrace(BaseModel):
    """What the context builder kept, compressed, and pruned."""

    model_config = ConfigDict(frozen=True)

    candidates_in: int = Field(default=0, ge=0, description="Ranked candidates handed to the builder")
    included: list[IncludedItem] = Field(default_factory=list, description="Items placed in the prompt")
    omitted: list[dict[str, str]] = Field(default_factory=list, description="Items pruned, with reason")
    downgraded_count: int = Field(default=0, ge=0, description="Items rendered below their aimed detail level")
    tokens_before_compression: int = Field(default=0, ge=0, description="Full source tokens of included items")
    tokens_after_compression: int = Field(default=0, ge=0, description="Tokens of included items as rendered")


class SavingsReport(BaseModel):
    """Token cost of the engine's prompt versus naive baselines."""

    model_config = ConfigDict(frozen=True)

    counter: str = Field(default="tiktoken", description="Tokenizer used; 'char' means a len/4 estimate")
    is_estimate: bool = Field(default=False, description="True when counts use the char heuristic")
    context_tokens: int = Field(ge=0, description="Tokens of the assembled repository context")
    prompt_tokens: int = Field(ge=0, description="Context plus task text, as sent to the model")
    provider_input_tokens: int | None = Field(default=None, description="Input tokens the provider billed, when known")
    naive_full_repo_tokens: int = Field(ge=0, description="Tokens to paste every indexed file")
    whole_files_tokens: int = Field(ge=0, description="Tokens to paste each file that contributed context")
    repo_files: int = Field(ge=0, description="Files in the repository index")
    context_files: int = Field(ge=0, description="Files that contributed context")
    saved_vs_full_repo_tokens: int = Field(default=0, description="naive_full_repo_tokens - prompt_tokens")
    saved_vs_full_repo_pct: float = Field(default=0.0, description="Percent saved against pasting the full repo")
    saved_vs_whole_files_tokens: int = Field(default=0, description="whole_files_tokens - prompt_tokens")
    saved_vs_whole_files_pct: float = Field(default=0.0, description="Percent saved against pasting whole files")
    compression_saved_tokens: int = Field(default=0, description="Tokens removed by compressing included items")
    reduction_factor: float = Field(default=0.0, description="naive_full_repo_tokens / prompt_tokens")


class ExecutionTrace(BaseModel):
    """Explains how a request used the code graph and what it cost."""

    model_config = ConfigDict(frozen=False)

    intent: str | None = Field(default=None, description="Detected task intent")
    target: str | None = Field(default=None, description="Chosen target entity")
    target_confidence: str | None = Field(default=None, description="Confidence in the chosen target")
    target_reason: str = Field(default="", description="Why that target was chosen")
    signals: dict[str, str] = Field(default_factory=dict, description="Per signal: used, skipped (reason), or failed")
    hits: list[SignalHit] = Field(default_factory=list, description="Every candidate each signal produced")
    graph_edges: list[GraphEdgeTrace] = Field(default_factory=list, description="Graph edges walked")
    graph_nodes: int = Field(default=0, ge=0, description="Nodes in the loaded code graph")
    graph_edges_total: int = Field(default=0, ge=0, description="Edges in the loaded code graph")
    candidates_considered: int = Field(default=0, ge=0, description="Unique candidates before the top_k cut")
    candidates_returned: int = Field(default=0, ge=0, description="Candidates kept after the top_k cut")
    operations: list[TraceOperation] = Field(default_factory=list, description="Ordered operations that ran")
    selection: SelectionTrace | None = Field(default=None, description="Context packing outcome")
    savings: SavingsReport | None = Field(default=None, description="Token savings versus naive baselines")
    warnings: list[str] = Field(default_factory=list, description="Degradations such as missing embeddings")


class RetrievalResult(BaseModel):
    """Result of hybrid search query before context construction."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="User natural language task or question")
    target: str | None = Field(default=None, description="Primary target entity qualified name if confident")
    target_confidence: str | None = Field(default=None, description="Confidence level: high, medium, low")
    candidates: list[RankedCandidate] = Field(default_factory=list, description="Ranked entity candidates")
    intent: str | None = Field(default=None, description="Detected task intent: optimize, explain, fix_bug, general")
    trace: ExecutionTrace | None = Field(default=None, description="How the candidates were found (graph, semantic, keyword)")


class DetailLevel(str, Enum):
    """Progressive code compression detail level."""

    FULL = "full"
    TRIMMED = "trimmed"
    SIGNATURE = "signature"
    STUB = "stub"
    REFERENCE = "reference"


class CompressedSnippet(BaseModel):
    """A compressed code or reference snippet representing an entity."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Compressed Python code or reference line")
    level: DetailLevel = Field(description="Detail compression level")
    tokens: int = Field(ge=0, description="Token count of compressed text")
    original_tokens: int = Field(ge=0, description="Token count of original source code")
    ref: str = Field(description="Entity qualified name")
    file: str = Field(description="POSIX relative file path")
    start_line: int | None = Field(default=None, description="Starting line number")
    end_line: int | None = Field(default=None, description="Ending line number")
    degraded: bool = Field(default=False, description="Whether fallback parsing was used")


class IncludedItem(BaseModel):
    """Record of an entity included in the final assembled context."""

    model_config = ConfigDict(frozen=False)

    ref: str = Field(description="Qualified name of included entity")
    file: str = Field(description="POSIX relative file path")
    lines: str = Field(description="Line range representation, e.g. '10-24'")
    level: str = Field(description="Compression level: full, trimmed, signature, stub, reference")
    tokens: int = Field(ge=0, description="Tokens consumed by this item snippet")
    original_tokens: int = Field(default=0, ge=0, description="Tokens of the entity's full source before compression")
    role: str = Field(description="Role: target, callee, caller, related, reference")
    score: float = Field(description="Retrieval relevance score")
    reason: str = Field(description="Inclusion or detail level rationale")


class BuilderOptions(BaseModel):
    """Configurable options and tuning knobs for ContextBuilder."""

    model_config = ConfigDict(frozen=True)

    max_utilization: float = Field(default=0.95, ge=0.1, le=1.0, description="Target utilization cap for upgrade pass")
    include_callers: bool = Field(default=True, description="Whether to include caller entities")
    include_imports: bool = Field(default=True, description="Whether to extract and show used imports")
    include_dependency_path: bool = Field(default=True, description="Whether to include graph CALLS paths")
    include_file_map: bool = Field(default=True, description="Whether to group snippets by file")
    caller_detail: str = Field(default="signature", description="Default aim detail level for callers")
    callee_detail: str = Field(default="trimmed", description="Default aim detail level for callees")
    other_detail: str = Field(default="stub", description="Default aim detail level for related entities")
    max_items: int = Field(default=30, ge=1, description="Maximum total items to include")
    must_include: list[str] = Field(default_factory=list, description="User-pinned qualified names")
    exclude: list[str] = Field(default_factory=list, description="Explicitly excluded qualified names")


class BuiltContext(BaseModel):
    """Final rendered prompt context ready for LLM consumption."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Final rendered prompt context string")
    tokens: int = Field(ge=0, description="Tokens in final rendered context text")
    budget_total: int = Field(ge=0, description="Allocated token budget ceiling")
    utilization: float = Field(ge=0.0, le=1.0, description="Fraction of token budget consumed")
    target: str | None = Field(default=None, description="Primary target entity qualified name")
    target_uncertain: bool = Field(default=False, description="True if target was selected with low confidence")
    items: list[IncludedItem] = Field(default_factory=list, description="List of included items with roles")
    omitted_items: list[dict[str, str]] = Field(default_factory=list, description="Lower-relevance items omitted")
    downgraded_count: int = Field(default=0, ge=0, description="Count of items downgraded to fit budget")
    files_included: list[str] = Field(default_factory=list, description="List of files represented in context")
    counter_name: str = Field(default="tiktoken", description="Token counter name used for verification")
    build_time_seconds: float = Field(default=0.0, ge=0.0, description="Assembly duration in seconds")
    warnings: list[str] = Field(default_factory=list, description="Warnings generated during context building")





# IncludedItem is declared after the trace models that reference it.
SelectionTrace.model_rebuild()
ExecutionTrace.model_rebuild()
RetrievalResult.model_rebuild()
