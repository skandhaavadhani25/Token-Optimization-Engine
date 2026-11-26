"""CLI entry point for code-context-engine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

# Ensure UTF-8 output encoding for Unicode tree characters on Windows terminals
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import networkx as nx

from app.config import get_settings
from app.context.context_builder import ContextBuilder
from app.context.context_compressor import compress
from app.context.formatter import format_context
from app.context.token_budget import BudgetTooSmallError
from app.context.token_counter import get_counter
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.embeddings.indexer import build_embeddings
from app.embeddings.semantic_search import SemanticSearch
from app.embeddings.vector_store import VectorStore
from app.graph.graph_builder import build_graph
from app.graph.graph_io import load_graph, save_graph
from app.graph.graph_search import CodeGraph
from app.indexer import build_index
from app.models.schemas import BuilderOptions, DetailLevel, ParsedFile
from app.parser.python_parser import parse_file, parse_repository
from app.retrieval.hybrid_search import HybridSearch
from app.scanner.repository_scanner import InvalidRepositoryError, scan_repository
from app.storage.metadata_store import MetadataStore


def format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes:,} bytes"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes:,} bytes ({size_bytes / 1024:.2f} KB)"
    return f"{size_bytes:,} bytes ({size_bytes / (1024 * 1024):.2f} MB)"


def format_file_tree(parsed: ParsedFile) -> str:
    """Format a ParsedFile structure into a human-readable tree diagram."""
    lines: list[str] = [f"File: {parsed.file_path}"]

    top_items: list[tuple[str, any]] = []
    if parsed.imports:
        import_strs: list[str] = []
        for imp in parsed.imports:
            if imp.is_from_import:
                prefix = ("." * imp.level) + imp.module
                target = f"{prefix}.{imp.name}" if prefix else str(imp.name or "*")
                if imp.alias:
                    import_strs.append(f"{target} as {imp.alias}")
                else:
                    import_strs.append(target)
            else:
                if imp.alias:
                    import_strs.append(f"{imp.module} as {imp.alias}")
                else:
                    import_strs.append(imp.module)
        top_items.append(("imports", ", ".join(import_strs)))

    for cls in parsed.classes:
        top_items.append(("class", cls))

    for fn in parsed.functions:
        top_items.append(("function", fn))

    num_top = len(top_items)
    for i, (item_type, item) in enumerate(top_items):
        is_last_top = i == num_top - 1
        top_branch = " └── " if is_last_top else " ├── "
        child_indent = "      " if is_last_top else " │    "

        if item_type == "imports":
            lines.append(f"{top_branch}imports: {item}")
        elif item_type == "class":
            cls = item
            bases_str = f" (bases: {', '.join(cls.bases)})" if cls.bases else ""
            lines.append(f"{top_branch}class {cls.name}{bases_str}")
            num_methods = len(cls.methods)
            for j, method in enumerate(cls.methods):
                is_last_method = j == num_methods - 1
                method_branch = "└── " if is_last_method else "├── "
                sub_indent = child_indent + ("     " if is_last_method else "│    ")

                args_str = ", ".join(a.name for a in method.arguments)
                lines.append(f"{child_indent}{method_branch}method {method.name}({args_str})")
                if method.calls:
                    call_names = ", ".join(c.name for c in method.calls)
                    lines.append(f"{sub_indent}└── calls: {call_names}")
        elif item_type == "function":
            fn = item
            args_str = ", ".join(a.name for a in fn.arguments)
            lines.append(f"{top_branch}function {fn.name}({args_str})")
            if fn.calls:
                call_names = ", ".join(c.name for c in fn.calls)
                lines.append(f"{child_indent}└── calls: {call_names}")

    return "\n".join(lines)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Code Context Engine - CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a repository for source files")
    scan_parser.add_argument("repo_path", help="Path to repository root to scan")
    scan_parser.add_argument("--json", action="store_true", help="Output full ScanResult as JSON")
    scan_parser.add_argument(
        "--max-size",
        type=int,
        default=1_000_000,
        help="Maximum file size in bytes (default: 1,000,000)",
    )
    scan_parser.add_argument(
        "--ext",
        nargs="+",
        default=[".py"],
        help="Allowed file extensions (default: .py)",
    )

    # 2. parse command
    parse_parser = subparsers.add_parser("parse", help="Scan and parse an entire repository")
    parse_parser.add_argument("repo_path", help="Path to repository root to parse")
    parse_parser.add_argument("--json", action="store_true", help="Output full parse results as JSON")
    parse_parser.add_argument(
        "--max-size",
        type=int,
        default=1_000_000,
        help="Maximum file size in bytes (default: 1,000,000)",
    )
    parse_parser.add_argument(
        "--ext",
        nargs="+",
        default=[".py"],
        help="Allowed file extensions (default: .py)",
    )

    # 3. parse-file command
    parse_file_parser = subparsers.add_parser("parse-file", help="Parse a single Python file into a tree")
    parse_file_parser.add_argument("file_path", help="Path to the Python file to parse")
    parse_file_parser.add_argument("--json", action="store_true", help="Output ParsedFile model as JSON")

    # 4. index command
    index_parser = subparsers.add_parser("index", help="Index a repository into the MongoDB metadata store")
    index_parser.add_argument("repo_path", help="Path to repository root to index")
    index_parser.add_argument("--force", action="store_true", help="Force full re-index from scratch")
    index_parser.add_argument("--json", action="store_true", help="Output IndexReport as JSON")

    # 5. stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics of the indexed repository")
    stats_parser.add_argument("repo_path", help="Path to repository root")
    stats_parser.add_argument("--json", action="store_true", help="Output StoreStats as JSON")

    # 6. find command
    find_parser = subparsers.add_parser("find", help="Find functions or classes by name")
    find_parser.add_argument("repo_path", help="Path to repository root")
    find_parser.add_argument("name", help="Name of function, method, or class to find")
    find_parser.add_argument("--json", action="store_true", help="Output matches as JSON")

    # 7. show command
    show_parser = subparsers.add_parser("show", help="Show details and source of a qualified symbol")
    show_parser.add_argument("repo_path", help="Path to repository root")
    show_parser.add_argument("qualified_name", help="Fully qualified name of function, method, or class")
    show_parser.add_argument("--json", action="store_true", help="Output symbol as JSON")

    # 8. graph-build command
    gb_parser = subparsers.add_parser("graph-build", help="Build and persist NetworkX code graph")
    gb_parser.add_argument("repo_path", help="Path to repository root")
    gb_parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="Include low-confidence edges for ambiguous call targets",
    )
    gb_parser.add_argument("--json", action="store_true", help="Output GraphBuildReport as JSON")

    # 9. graph-stats command
    gs_parser = subparsers.add_parser("graph-stats", help="Display statistics of saved code graph")
    gs_parser.add_argument("repo_path", help="Path to repository root")
    gs_parser.add_argument("--json", action="store_true", help="Output stats as JSON")

    # 10. callees command
    callees_parser = subparsers.add_parser("callees", help="Find functions/methods called by target")
    callees_parser.add_argument("repo_path", help="Path to repository root")
    callees_parser.add_argument("name", help="Target function or method name")
    callees_parser.add_argument("--depth", type=int, default=1, help="Search depth (default: 1)")
    callees_parser.add_argument("--json", action="store_true", help="Output callees as JSON")

    # 11. callers command
    callers_parser = subparsers.add_parser("callers", help="Find functions/methods calling target")
    callers_parser.add_argument("repo_path", help="Path to repository root")
    callers_parser.add_argument("name", help="Target function or method name")
    callers_parser.add_argument("--depth", type=int, default=1, help="Search depth (default: 1)")
    callers_parser.add_argument("--json", action="store_true", help="Output callers as JSON")

    # 12. neighbors command
    nb_parser = subparsers.add_parser("neighbors", help="Find graph neighborhood around target node")
    nb_parser.add_argument("repo_path", help="Path to repository root")
    nb_parser.add_argument("name", help="Target node name or identifier")
    nb_parser.add_argument("--hops", type=int, default=2, help="Neighborhood radius (default: 2)")
    nb_parser.add_argument("--json", action="store_true", help="Output neighborhood as JSON")

    # 13. path command
    path_parser = subparsers.add_parser("path", help="Find shortest CALLS dependency path between two functions")
    path_parser.add_argument("repo_path", help="Path to repository root")
    path_parser.add_argument("from_name", help="Source function or method name")
    path_parser.add_argument("to_name", help="Target destination function or method name")
    path_parser.add_argument("--json", action="store_true", help="Output path as JSON")

    # 14. file-deps command
    fd_parser = subparsers.add_parser("file-deps", help="Show file-level imports and dependencies")
    fd_parser.add_argument("repo_path", help="Path to repository root")
    fd_parser.add_argument("file_path", help="Relative file path")
    fd_parser.add_argument("--json", action="store_true", help="Output dependencies as JSON")

    # 15. embed command
    embed_parser = subparsers.add_parser("embed", help="Generate or update semantic vector embeddings")
    embed_parser.add_argument("repo_path", help="Path to repository root")
    embed_parser.add_argument("--force", action="store_true", help="Re-embed all entities from scratch")
    embed_parser.add_argument("--model", type=str, default=None, help="Embedding model name override")
    embed_parser.add_argument("--json", action="store_true", help="Output EmbeddingReport as JSON")

    # 16. embed-stats command
    es_parser = subparsers.add_parser("embed-stats", help="Display vector store manifest and item counts")
    es_parser.add_argument("repo_path", help="Path to repository root")
    es_parser.add_argument("--json", action="store_true", help="Output stats as JSON")

    # 17. semantic command
    sem_parser = subparsers.add_parser("semantic", help="Search repository using semantic natural language query")
    sem_parser.add_argument("repo_path", help="Path to repository root")
    sem_parser.add_argument("query", help="Search query string")
    sem_parser.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    sem_parser.add_argument(
        "--kinds",
        type=str,
        default=None,
        help="Comma-separated vector kinds to filter by (e.g. function_desc,function_code)",
    )
    sem_parser.add_argument("--json", action="store_true", help="Output search results as JSON")

    # 18. context command
    ctx_parser = subparsers.add_parser("context", help="Build token-budgeted prompt context for a query")
    ctx_parser.add_argument("repo_path", help="Path to repository root")
    ctx_parser.add_argument("query", help="User task or search query")
    ctx_parser.add_argument("--budget", type=int, default=4000, help="Maximum token budget (default: 4000)")
    ctx_parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "markdown", "json"],
        help="Output format: text, markdown, or json (default: text)",
    )
    ctx_parser.add_argument("--pin", nargs="+", default=[], help="Pin entity qualified names to must_include")
    ctx_parser.add_argument("--exclude", nargs="+", default=[], help="Exclude entity qualified names")
    ctx_parser.add_argument("--no-callers", action="store_true", help="Exclude caller entities")
    ctx_parser.add_argument("--no-imports", action="store_true", help="Exclude imports section")
    ctx_parser.add_argument("--show-items", action="store_true", help="Print table of included items and metrics")
    ctx_parser.add_argument("--output", type=str, default=None, help="Save final context text to file")
    ctx_parser.add_argument(
        "--counter",
        type=str,
        default="tiktoken",
        choices=["tiktoken", "char"],
        help="Tokenizer counter to use (default: tiktoken)",
    )

    # 19. compress command
    cmp_parser = subparsers.add_parser("compress", help="Compress a code entity to a detail level")
    cmp_parser.add_argument("repo_path", help="Path to repository root")
    cmp_parser.add_argument("qualified_name", help="Fully qualified name of function or class")
    cmp_parser.add_argument(
        "--level",
        type=str,
        default="full",
        choices=["full", "trimmed", "signature", "stub", "reference"],
        help="Compression detail level (default: full)",
    )
    cmp_parser.add_argument(
        "--counter",
        type=str,
        default="tiktoken",
        choices=["tiktoken", "char"],
        help="Tokenizer counter to use (default: tiktoken)",
    )

    # 20. budget-sweep command
    bs_parser = subparsers.add_parser("budget-sweep", help="Sweep multiple budgets and display degradation table")
    bs_parser.add_argument("repo_path", help="Path to repository root")
    bs_parser.add_argument("query", help="User task or search query")
    bs_parser.add_argument(
        "--budgets",
        type=str,
        default="500,1000,2000,4000,8000",
        help="Comma-separated budget limits (default: 500,1000,2000,4000,8000)",
    )
    bs_parser.add_argument(
        "--counter",
        type=str,
        default="tiktoken",
        choices=["tiktoken", "char"],
        help="Tokenizer counter to use (default: tiktoken)",
    )

    # 21. serve command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI service")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # 22. ask command
    ask_parser = subparsers.add_parser("ask", help="Run the coding assistant on a repository query")
    ask_parser.add_argument("repo_path", help="Path to repository root")
    ask_parser.add_argument("query", help="User task or coding question")
    ask_parser.add_argument("--budget", type=int, default=None, help="Context token budget (default: 4000)")
    ask_parser.add_argument("--provider", type=str, default=None, help="AI provider override: claude | gemini | openai_compat | ollama")
    ask_parser.add_argument("--model", type=str, default=None, help="Model override")
    ask_parser.add_argument("--dry-run", action="store_true", help="Assemble prompt without invoking model")
    ask_parser.add_argument("--show-context", action="store_true", help="Print the built context text")
    ask_parser.add_argument("--json", action="store_true", help="Output AskResult as JSON")

    # 22b. providers command
    prov_parser = subparsers.add_parser("providers", help="List available AI providers and configuration readiness")
    prov_parser.add_argument("--json", action="store_true", help="Output providers list as JSON")

    # 23. benchmark command
    bm_parser = subparsers.add_parser("benchmark", help="Run before/after token reduction benchmark")
    bm_parser.add_argument("repo_path", help="Path to repository root")
    bm_parser.add_argument("queries_file", help="Path to evaluation queries JSON file")
    bm_parser.add_argument("--budget", type=int, default=4000, help="Context token budget (default: 4000)")
    bm_parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategies: whole_repo,whole_file,naive_keyword,fixed_window,graph",
    )
    bm_parser.add_argument("--call-ai", action="store_true", help="Invoke configured AI model for completion metrics")
    bm_parser.add_argument("--ai-output-tokens", type=int, default=512, help="Max AI output tokens (default: 512)")
    bm_parser.add_argument("--repeat", type=int, default=1, help="Repetitions for live AI calls (default: 1)")
    bm_parser.add_argument("--json", type=str, default=None, dest="json_file", help="Output JSON report file path")
    bm_parser.add_argument("--md", type=str, default=None, dest="md_file", help="Output Markdown report file path")

    # 24. repo-tokens command
    rt_parser = subparsers.add_parser("repo-tokens", help="Measure repository total tokens and file distribution")
    rt_parser.add_argument("repo_path", help="Path to repository root")
    rt_parser.add_argument("--json", action="store_true", help="Output distribution as JSON")

    # 25. update command (incremental indexing)
    up_parser = subparsers.add_parser("update", help="Run incremental indexing update")
    up_parser.add_argument("repo_path", help="Path to repository root")
    up_parser.add_argument("--verify", action="store_true", help="Assert equivalence against fresh full rebuild")
    up_parser.add_argument("--json", action="store_true", help="Output IncrementalReport as JSON")

    # 26. watch command (automatic incremental indexing on change)
    w_parser = subparsers.add_parser("watch", help="Watch repository and trigger debounced incremental indexing")
    w_parser.add_argument("repo_path", help="Path to repository root")
    w_parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds (default: 2.0)")
    w_parser.add_argument("--verify", action="store_true", help="Assert equivalence on each update")

    return parser.parse_args(args)


def run_scan(
    repo_path: str,
    as_json: bool,
    max_size: int = 1_000_000,
    extensions: Sequence[str] = (".py",),
) -> int:
    """Execute scan and print human-readable or JSON summary."""
    try:
        result = scan_repository(
            repo_path=repo_path,
            extensions=extensions,
            max_file_size=max_size,
        )
    except (InvalidRepositoryError, FileNotFoundError, NotADirectoryError, ValueError) as err:
        sys.stderr.write(f"Error scanning repository: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
        return 0

    print("=" * 64)
    print("Code Context Engine - Repository Scan")
    print("=" * 64)
    print(f"Repository Root:      {result.repo_root}")
    print(f"Total Python Files:   {result.total_files:,}")
    print(f"Total Lines:          {result.total_lines:,}")
    print(f"Total Size:           {format_size(result.total_size_bytes)}")
    print(f"Skipped Directories:  {result.skipped_dirs:,}")
    print(f"Skipped Files:        {result.skipped_files:,}")
    print(f"Scan Duration:        {result.scan_time_seconds:.4f}s")
    print("\nTop 10 Largest Files:")
    if not result.files:
        print("  (No files found)")
    else:
        sorted_by_size = sorted(result.files, key=lambda item: item.size_bytes, reverse=True)
        for idx, item in enumerate(sorted_by_size[:10], start=1):
            print(f"  {idx:>2}. {item.path} ({format_size(item.size_bytes)}, {item.line_count:,} lines)")
    print("=" * 64)
    return 0


def run_parse(
    repo_path: str,
    as_json: bool,
    max_size: int = 1_000_000,
    extensions: Sequence[str] = (".py",),
) -> int:
    """Scan and parse repository, outputting summary and top callables."""
    try:
        scan_result = scan_repository(
            repo_path=repo_path,
            extensions=extensions,
            max_file_size=max_size,
        )
    except (InvalidRepositoryError, FileNotFoundError, NotADirectoryError, ValueError) as err:
        sys.stderr.write(f"Error scanning repository: {err}\n")
        return 1

    parsed_files, summary = parse_repository(scan_result)

    if as_json:
        output_payload = {
            "summary": summary.model_dump(),
            "files": [f.model_dump() for f in parsed_files],
        }
        sys.stdout.write(json.dumps(output_payload, indent=2) + "\n")
        return 0

    print("=" * 64)
    print("Code Context Engine - Repository Parse Summary")
    print("=" * 64)
    print(f"Repository Root:      {scan_result.repo_root}")
    print(f"Total Files:          {summary.total_files:,}")
    print(f"Parsed Successfully:  {summary.parsed_ok:,}")
    print(f"Failed Parse:         {summary.failed:,}")
    print(f"Total Functions:      {summary.total_functions:,}")
    print(f"Total Classes:        {summary.total_classes:,}")
    print(f"Total Methods:        {summary.total_methods:,}")
    print(f"Total Imports:        {summary.total_imports:,}")
    print(f"Total Calls:          {summary.total_calls:,}")
    print(f"Parse Duration:       {summary.parse_time_seconds:.4f}s")

    all_callables = []
    for f in parsed_files:
        for fn in f.functions:
            all_callables.append(fn)
        for cls in f.classes:
            for method in cls.methods:
                all_callables.append(method)

    all_callables.sort(key=lambda item: len(item.calls), reverse=True)
    print("\nTop 10 Functions by Call Count:")
    if not all_callables:
        print("  (No functions or methods found)")
    else:
        for idx, item in enumerate(all_callables[:10], start=1):
            print(
                f"  {idx:>2}. {item.qualified_name} ({len(item.calls)} calls) - {item.file_path}:{item.start_line}"
            )

    if summary.errors:
        print(f"\nParse Errors ({len(summary.errors)}):")
        for err in summary.errors:
            print(f"  - {err['file']}: {err['error']}")

    print("=" * 64)
    return 0


def run_parse_file(file_path: str, as_json: bool) -> int:
    """Parse a single file and print its structure as a tree or JSON."""
    target_path = Path(file_path)
    if not target_path.exists():
        sys.stderr.write(f"Error: File not found: {file_path}\n")
        return 1
    if not target_path.is_file():
        sys.stderr.write(f"Error: Path is not a file: {file_path}\n")
        return 1

    parsed = parse_file(abs_path=target_path, rel_path=file_path)

    if parsed.parse_error:
        sys.stderr.write(f"Parse error in {file_path}: {parsed.parse_error}\n")
        return 1

    if as_json:
        sys.stdout.write(parsed.model_dump_json(indent=2) + "\n")
        return 0

    print(format_file_tree(parsed))
    return 0


def run_index(repo_path: str, force: bool, as_json: bool) -> int:
    """Scan and index repository into the MongoDB metadata store."""
    try:
        report = build_index(repo_path, force=force)
    except Exception as err:
        sys.stderr.write(f"Error indexing repository: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(report.model_dump_json(indent=2) + "\n")
        return 0

    print("=" * 64)
    print("Code Context Engine - Index Report")
    print("=" * 64)
    print(f"Repository Root:  {Path(repo_path).resolve()}")
    print(f"Files Scanned:    {report.files_scanned:,}")
    print(f"Added:            {report.added:,}")
    print(f"Modified:         {report.modified:,}")
    print(f"Deleted:          {report.deleted:,}")
    print(f"Unchanged:        {report.unchanged:,}")
    print(f"Files Parsed:     {report.parsed:,}")
    print(f"Parse Errors:     {report.parse_errors:,}")
    print(f"Scan Time:        {report.scan_time:.4f}s")
    print(f"Parse Time:       {report.parse_time:.4f}s")
    print(f"Store Time:       {report.store_time:.4f}s")
    print(f"Total Time:       {report.total_time:.4f}s")
    print("=" * 64)
    return 0


def run_stats(repo_path: str, as_json: bool) -> int:
    """Show aggregate statistics from existing metadata database."""
    try:
        with MetadataStore.open(repo_path) as store:
            stats = store.get_stats()
            store_key = store.repo
    except Exception as err:
        sys.stderr.write(f"Error accessing metadata store: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(stats.model_dump_json(indent=2) + "\n")
        return 0

    last_indexed_str = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats.last_indexed_at))
        if stats.last_indexed_at
        else "Never"
    )

    print("=" * 64)
    print("Code Context Engine - Metadata Store Stats")
    print("=" * 64)
    print(f"Store Key:          {store_key}")
    print(f"Total Files:        {stats.total_files:,}")
    print(f"Total Functions:    {stats.total_functions:,}")
    print(f"Total Classes:      {stats.total_classes:,}")
    print(f"Total Methods:      {stats.total_methods:,}")
    print(f"Total Imports:      {stats.total_imports:,}")
    print(f"Total Calls:        {stats.total_calls:,}")
    print(f"Files with Errors:  {stats.files_with_errors:,}")
    print(f"Database Size:      {format_size(stats.db_size_bytes)}")
    print(f"Last Indexed:       {last_indexed_str}")
    print("=" * 64)
    return 0


def run_find(repo_path: str, name: str, as_json: bool) -> int:
    """Lookup functions or classes by name."""
    try:
        with MetadataStore.open(repo_path) as store:
            funcs = store.find_functions_by_name(name, exact=True)
            if not funcs:
                funcs = store.find_functions_by_name(name, exact=False)

            cls = store.get_class(name)
            classes = [cls] if cls else store.find_classes_by_name(name, exact=False)
    except Exception as err:
        sys.stderr.write(f"Error searching store: {err}\n")
        return 1

    if as_json:
        data = {
            "functions": [f.model_dump() for f in funcs],
            "classes": [c.model_dump() for c in classes],
        }
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
        return 0

    if not funcs and not classes:
        print(f"No functions or classes found matching '{name}'.")
        return 0

    print("=" * 64)
    print(f"Search Results for '{name}'")
    print("=" * 64)

    if classes:
        print(f"Classes ({len(classes)}):")
        for c in classes:
            print(f"  • {c.qualified_name} ({c.file_path}:{c.start_line}-{c.end_line})")
            if c.methods:
                print(f"    Methods ({len(c.methods)}): {', '.join(m.name for m in c.methods)}")

    if funcs:
        print(f"\nFunctions & Methods ({len(funcs)}):")
        for f in funcs:
            kind = "method" if f.is_method else "function"
            calls_str = ", ".join(call.name for call in f.calls) if f.calls else "(none)"
            print(f"  • [{kind}] {f.qualified_name} ({f.file_path}:{f.start_line}-{f.end_line})")
            print(f"    Calls: {calls_str}")

    print("=" * 64)
    return 0


def run_show(repo_path: str, qualified_name: str, as_json: bool) -> int:
    """Print full source and metadata for a qualified symbol."""
    try:
        with MetadataStore.open(repo_path) as store:
            fn = store.get_function(qualified_name)
            cls = store.get_class(qualified_name) if not fn else None
    except Exception as err:
        sys.stderr.write(f"Error querying store: {err}\n")
        return 1

    if fn:
        if as_json:
            sys.stdout.write(fn.model_dump_json(indent=2) + "\n")
            return 0

        kind = "Method" if fn.is_method else "Function"
        args_str = ", ".join(a.name for a in fn.arguments)
        calls_str = ", ".join(c.name for c in fn.calls) if fn.calls else "(none)"
        print("=" * 64)
        print(f"{kind}: {fn.qualified_name}")
        print("=" * 64)
        print(f"File:         {fn.file_path}:{fn.start_line}-{fn.end_line}")
        print(f"Arguments:    ({args_str})")
        print(f"Return:       {fn.return_annotation or 'None'}")
        print(f"Async:        {fn.is_async}")
        if fn.parent_class:
            print(f"Class:        {fn.parent_class}")
        if fn.decorators:
            print(f"Decorators:   {', '.join(fn.decorators)}")
        if fn.docstring:
            print(f"Docstring:    {fn.docstring}")
        print(f"Calls:        {calls_str}")
        print("\nSource Code:")
        print("-" * 64)
        print(fn.source.rstrip())
        print("=" * 64)
        return 0

    if cls:
        if as_json:
            sys.stdout.write(cls.model_dump_json(indent=2) + "\n")
            return 0

        methods_str = ", ".join(m.name for m in cls.methods) if cls.methods else "(none)"
        bases_str = ", ".join(cls.bases) if cls.bases else "(none)"
        print("=" * 64)
        print(f"Class: {cls.qualified_name}")
        print("=" * 64)
        print(f"File:         {cls.file_path}:{cls.start_line}-{cls.end_line}")
        print(f"Bases:        {bases_str}")
        if cls.decorators:
            print(f"Decorators:   {', '.join(cls.decorators)}")
        if cls.docstring:
            print(f"Docstring:    {cls.docstring}")
        print(f"Methods:      {methods_str}")
        print("\nSource Code:")
        print("-" * 64)
        print(cls.source.rstrip())
        print("=" * 64)
        return 0

    sys.stderr.write(f"Error: Symbol not found: {qualified_name}\n")
    return 1


def resolve_target_node(cg: CodeGraph, name: str) -> tuple[str | None, str | None]:
    """Resolve a user-provided name or ID to a single node ID or return an error message."""
    matches = cg.find_nodes(name, exact=True)
    if not matches:
        matches = cg.find_nodes(name, exact=False)

    if not matches:
        return None, f"Error: No node found matching '{name}'."
    if len(matches) > 1:
        lines = [f"Multiple nodes found matching '{name}':"]
        for m in matches:
            lines.append(f"  - {m}")
        lines.append("Please specify the exact qualified name or node ID.")
        return None, "\n".join(lines)
    return matches[0], None


def run_graph_build(repo_path: str, include_ambiguous: bool, as_json: bool) -> int:
    """Build and persist the NetworkX code graph."""
    try:
        repo = Path(repo_path).resolve()
        if not MetadataStore.exists(repo):
            build_index(repo)

        with MetadataStore.open(repo) as store:
            if store.get_stats().total_files == 0:
                build_index(repo)

            repo_name = repo.name or "root"
            graph, report = build_graph(
                store=store,
                repo_name=repo_name,
                include_ambiguous=include_ambiguous,
            )
            saved_path = save_graph(graph, repo)
    except Exception as err:
        sys.stderr.write(f"Error building graph: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(report.model_dump_json(indent=2) + "\n")
        return 0

    print("=" * 64)
    print(f"Code Context Engine - Graph Build Report: {report.repo_name}")
    print("=" * 64)
    print(f"Total Nodes:         {report.total_nodes:,}")
    print(f"Total Edges:         {report.total_edges:,}")
    print(f"Build Time:          {report.build_time_seconds:.3f}s")
    print(f"Saved Path:          {saved_path}")
    print("\nNode Counts by Type:")
    print("-" * 64)
    for ntype, count in sorted(report.node_counts.items()):
        print(f"  {ntype:<15} {count:>6,}")
    print("\nEdge Counts by Rel:")
    print("-" * 64)
    for rel, count in sorted(report.edge_counts.items()):
        print(f"  {rel:<15} {count:>6,}")
    print("\nCall Resolution Statistics:")
    print("-" * 64)
    cr = report.call_resolution
    pct = (cr.resolved_calls / cr.total_calls * 100.0) if cr.total_calls > 0 else 0.0
    print(f"  Total Calls:       {cr.total_calls:,}")
    print(f"  Resolved Calls:    {cr.resolved_calls:,} ({pct:.1f}%)")
    for rule, count in sorted(cr.resolved_by_rule.items()):
        print(f"    - {rule:<15} {count:>6,}")
    print(f"  Skipped Ambiguous: {cr.skipped_ambiguous:,}")
    print(f"  External Calls:    {cr.external_calls:,}")
    print(f"  Builtin Calls:     {cr.builtin_calls:,}")
    print(f"  Unresolved Calls:  {cr.unresolved_calls:,}")
    print("=" * 64)
    return 0


def run_graph_stats(repo_path: str, as_json: bool) -> int:
    """Load and print code graph summary metrics."""
    try:
        graph = load_graph(repo_path)
        cg = CodeGraph(graph)
        stats = cg.stats()
    except Exception as err:
        sys.stderr.write(f"Error loading graph stats: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(json.dumps(stats, indent=2) + "\n")
        return 0

    print("=" * 64)
    print("Code Context Engine - Graph Statistics")
    print("=" * 64)
    print(f"Total Nodes:         {stats['total_nodes']:,}")
    print(f"Total Edges:         {stats['total_edges']:,}")
    print("\nNode Counts by Type:")
    print("-" * 64)
    for ntype, count in sorted(stats["node_counts"].items()):
        print(f"  {ntype:<15} {count:>6,}")
    print("\nEdge Counts by Rel:")
    print("-" * 64)
    for rel, count in sorted(stats["edge_counts"].items()):
        print(f"  {rel:<15} {count:>6,}")
    print("=" * 64)
    return 0


def run_callees(repo_path: str, name: str, depth: int, as_json: bool) -> int:
    """Find and display callees of the target node."""
    try:
        graph = load_graph(repo_path)
        cg = CodeGraph(graph)
    except Exception as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1

    node_id, err_msg = resolve_target_node(cg, name)
    if err_msg:
        sys.stderr.write(err_msg + "\n")
        return 1

    results = cg.callees(node_id, depth=depth)

    if as_json:
        sys.stdout.write(json.dumps([r.model_dump() for r in results], indent=2) + "\n")
        return 0

    print("=" * 64)
    print(f"Callees of: {node_id} (depth={depth})")
    print("=" * 64)
    if not results:
        print("  (no callees found)")
    else:
        for r in results:
            indent = "  " * r.depth
            print(f"{indent}[depth {r.depth}] {r.qualified_name} ({r.node_id})")
            print(f"{indent}  line {r.lineno} in {r.file} | confidence: {r.confidence} | rule: {r.resolution}")
    print("=" * 64)
    return 0


def run_callers(repo_path: str, name: str, depth: int, as_json: bool) -> int:
    """Find and display callers of the target node."""
    try:
        graph = load_graph(repo_path)
        cg = CodeGraph(graph)
    except Exception as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1

    node_id, err_msg = resolve_target_node(cg, name)
    if err_msg:
        sys.stderr.write(err_msg + "\n")
        return 1

    results = cg.callers(node_id, depth=depth)

    if as_json:
        sys.stdout.write(json.dumps([r.model_dump() for r in results], indent=2) + "\n")
        return 0

    print("=" * 64)
    print(f"Callers of: {node_id} (depth={depth})")
    print("=" * 64)
    if not results:
        print("  (no callers found)")
    else:
        for r in results:
            indent = "  " * r.depth
            print(f"{indent}[depth {r.depth}] {r.qualified_name} ({r.node_id})")
            print(f"{indent}  line {r.lineno} in {r.file} | confidence: {r.confidence} | rule: {r.resolution}")
    print("=" * 64)
    return 0


def run_neighbors(repo_path: str, name: str, hops: int, as_json: bool) -> int:
    """Find and display neighborhood around target node."""
    try:
        graph = load_graph(repo_path)
        cg = CodeGraph(graph)
    except Exception as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1

    node_id, err_msg = resolve_target_node(cg, name)
    if err_msg:
        sys.stderr.write(err_msg + "\n")
        return 1

    sub = cg.neighborhood(node_id, hops=hops)

    if as_json:
        data = nx.node_link_data(sub)
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
        return 0

    print("=" * 64)
    print(f"Neighborhood of: {node_id} (hops={hops})")
    print("=" * 64)
    print(f"Total Subgraph Nodes: {sub.number_of_nodes()}")
    print(f"Total Subgraph Edges: {sub.number_of_edges()}")
    print("\nConnected Nodes:")
    for nid, data in sorted(sub.nodes(data=True)):
        ntype = data.get("type", "")
        qname = data.get("qualified_name") or data.get("path") or data.get("name") or nid
        print(f"  [{ntype}] {nid} ({qname})")
    print("\nEdges:")
    for u, v, data in sorted(sub.edges(data=True)):
        rel = data.get("rel", "")
        extra = f" (line {data.get('lineno')})" if "lineno" in data else ""
        print(f"  {u} --[{rel}{extra}]--> {v}")
    print("=" * 64)
    return 0


def run_path(repo_path: str, from_name: str, to_name: str, as_json: bool) -> int:
    """Find and display shortest dependency path between two nodes."""
    try:
        graph = load_graph(repo_path)
        cg = CodeGraph(graph)
    except Exception as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1

    from_id, err_from = resolve_target_node(cg, from_name)
    if err_from:
        sys.stderr.write(err_from + "\n")
        return 1

    to_id, err_to = resolve_target_node(cg, to_name)
    if err_to:
        sys.stderr.write(err_to + "\n")
        return 1

    path = cg.dependency_path(from_id, to_id)

    if as_json:
        sys.stdout.write(json.dumps({"from": from_id, "to": to_id, "path": path}, indent=2) + "\n")
        return 0

    print("=" * 64)
    print(f"Dependency Path (CALLS): {from_id} -> {to_id}")
    print("=" * 64)
    if path is None:
        print(f"No path found between '{from_id}' and '{to_id}' over CALLS edges.")
    else:
        print(f"Path found ({len(path) - 1} hops):")
        for i, step in enumerate(path):
            arrow = "  -> " if i > 0 else "     "
            print(f"{arrow}{step}")
    print("=" * 64)
    return 0


def run_file_deps(repo_path: str, file_path: str, as_json: bool) -> int:
    """Find and display file-level dependencies."""
    try:
        graph = load_graph(repo_path)
        cg = CodeGraph(graph)
    except Exception as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1

    deps = cg.file_dependencies(file_path)

    if as_json:
        sys.stdout.write(json.dumps(deps, indent=2) + "\n")
        return 0

    print("=" * 64)
    print(f"File Dependencies for: {file_path}")
    print("=" * 64)
    print(f"Imports ({len(deps['imports'])}):")
    for item in deps["imports"]:
        print(f"  -> {item}")
    print(f"\nImported By ({len(deps['imported_by'])}):")
    for item in deps["imported_by"]:
        print(f"  <- {item}")
    print(f"\nDepends On ({len(deps['depends_on'])}):")
    for item in deps["depends_on"]:
        print(f"  -> {item}")
    print(f"\nDepended On By ({len(deps['depended_on_by'])}):")
    for item in deps["depended_on_by"]:
        print(f"  <- {item}")
    print("=" * 64)
    return 0


def run_embed(repo_path: str, force: bool, model: str | None, as_json: bool) -> int:
    """Build or incrementally update semantic vector embeddings."""
    try:
        report = build_embeddings(repo_path=repo_path, force=force, model_name=model)
    except Exception as err:
        sys.stderr.write(f"Error building embeddings: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(report.model_dump_json(indent=2) + "\n")
        return 0

    speed = (
        (report.items_new + report.items_changed) / report.embedding_time_seconds
        if report.embedding_time_seconds > 0
        else 0.0
    )

    print("=" * 64)
    print("Code Context Engine - Semantic Embedding Report")
    print("=" * 64)
    print(f"Repository Root:    {Path(repo_path).resolve()}")
    print(f"Model Name:         {report.model_name}")
    print(f"Total Items:        {report.items_total:,}")
    print(f"New Items:          {report.items_new:,}")
    print(f"Changed Items:      {report.items_changed:,}")
    print(f"Removed Items:      {report.items_removed:,}")
    print(f"Unchanged Items:    {report.items_unchanged:,}")
    print(f"Skipped Items:      {report.items_skipped:,}")
    print("\nItems by Kind:")
    print("-" * 64)
    for kind, count in sorted(report.by_kind.items()):
        print(f"  {kind:<18} {count:>6,}")
    print("\nPerformance & Storage:")
    print("-" * 64)
    print(f"Embedding Time:     {report.embedding_time_seconds:.3f}s ({speed:.1f} items/sec)")
    print(f"Total Time:         {report.total_time_seconds:.3f}s")
    print(f"Index Disk Size:    {format_size(report.index_size_bytes)}")
    print("=" * 64)
    return 0


def run_embed_stats(repo_path: str, as_json: bool) -> int:
    """Display vector store manifest and item counts without loading model."""
    try:
        manifest = VectorStore.read_manifest(repo_path)
        if manifest is None:
            sys.stderr.write(
                f"Error: No vector embeddings found for {repo_path}.\n"
                f"Please run 'python -m app.main embed {repo_path}' first.\n"
            )
            return 1

        store = VectorStore(
            dimension=manifest.get("dimension", 384),
            model_name=manifest.get("model_name", ""),
            repo_path=repo_path,
        )
        store.load()
        by_kind = store.stats()["by_kind"]
        total_bytes = manifest.get("index_size_bytes", 0)
        manifest["by_kind"] = by_kind
        manifest["index_size_bytes"] = total_bytes
    except Exception as err:
        sys.stderr.write(f"Error reading embedding stats: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(json.dumps(manifest, indent=2) + "\n")
        return 0

    last_updated_str = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(manifest.get("last_updated_at", 0)))
        if manifest.get("last_updated_at")
        else "Never"
    )

    print("=" * 64)
    print("Code Context Engine - Vector Store Stats")
    print("=" * 64)
    print(f"Store Key:          {store.key}")
    print(f"Model Name:         {manifest.get('model_name')}")
    print(f"Dimension:          {manifest.get('dimension')}")
    print(f"Total Items:        {manifest.get('item_count', 0):,}")
    print(f"Index Disk Size:    {format_size(total_bytes)}")
    print(f"Last Updated:       {last_updated_str}")
    print("\nItems by Kind:")
    print("-" * 64)
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind:<18} {count:>6,}")
    print("=" * 64)
    return 0


def run_semantic(
    repo_path: str,
    query: str,
    top_k: int,
    kinds_str: str | None,
    as_json: bool,
) -> int:
    """Execute semantic query against vector index."""
    try:
        repo = Path(repo_path).resolve()
        manifest = VectorStore.read_manifest(repo)
        if manifest is None:
            sys.stderr.write(
                f"Error: Vector index not found for {repo}.\n"
                f"Please run 'python -m app.main embed {repo_path}' first.\n"
            )
            return 1

        model_name = manifest.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        dimension = manifest.get("dimension", 384)

        generator = EmbeddingGenerator(model_name=model_name)
        vector_store = VectorStore(
            dimension=dimension,
            model_name=model_name,
            repo_path=repo,
        )
        vector_store.load()

        with MetadataStore.open(repo) as store:
            searcher = SemanticSearch(
                vector_store=vector_store,
                metadata_store=store,
                generator=generator,
            )
            kinds = [k.strip() for k in kinds_str.split(",") if k.strip()] if kinds_str else None
            hits = searcher.search(query=query, top_k=top_k, kinds=kinds)
    except Exception as err:
        sys.stderr.write(f"Error performing semantic search: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(json.dumps([h.model_dump() for h in hits], indent=2) + "\n")
        return 0

    print("=" * 64)
    print(f"Semantic Search Results for: '{query}'")
    print(f"Filter Kinds: {kinds_str or 'all'}")
    print("=" * 64)

    if not hits:
        print("  (No relevant entities found)")
        print("=" * 64)
        return 0

    for idx, hit in enumerate(hits, start=1):
        lines_str = f":{hit.start_line}-{hit.end_line}" if hit.start_line and hit.end_line else ""
        kinds_joined = ", ".join(
            f"{k} ({hit.kind_scores.get(k, 0):.2f})" for k in hit.matched_kinds
        )
        print(f"{idx:>2}. [{hit.entity_type:<8}] score: {hit.score:.4f}  {hit.ref}")
        print(f"    File: {hit.file}{lines_str}")
        print(f"    Matched: {kinds_joined}")
    print("=" * 64)
    return 0


def run_context(
    repo_path: str,
    query: str,
    budget: int = 4000,
    format_type: str = "text",
    pin: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    no_callers: bool = False,
    no_imports: bool = False,
    show_items: bool = False,
    output_file: str | None = None,
    counter_name: str = "tiktoken",
) -> int:
    """Assemble token-bounded context for a natural language coding request."""
    repo = Path(repo_path).resolve()
    if not repo.exists():
        sys.stderr.write(f"Repository path does not exist: {repo}\n")
        return 1

    try:
        counter = get_counter(counter_name)
    except Exception as err:
        sys.stderr.write(f"Error initializing token counter: {err}\n")
        return 1

    with MetadataStore.open(repo) as store:
        graph = None
        try:
            nx_g = load_graph(repo)
            if nx_g is not None:
                graph = CodeGraph(nx_g)
        except Exception:
            pass

        retriever = HybridSearch.from_repo(repo, metadata_store=store)
        retrieval_result = retriever.search(query)

        options = BuilderOptions(
            must_include=list(pin or []),
            exclude=list(exclude or []),
            include_callers=not no_callers,
            include_imports=not no_imports,
        )

        builder = ContextBuilder(metadata_store=store, graph=graph, counter=counter)
        try:
            built = builder.build(retrieval_result, budget=budget, options=options)
        except BudgetTooSmallError as err:
            sys.stderr.write(f"Error: {err}\n")
            return 1

        formatted = format_context(built, fmt=format_type)

        if output_file:
            out_p = Path(output_file).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(formatted, encoding="utf-8")

        if format_type == "json":
            sys.stdout.write(formatted + "\n")
            return 0

        # Print context text
        sys.stdout.write(formatted + "\n")

        # Print summary
        print("=" * 64)
        print("CONTEXT BUILD SUMMARY")
        print("=" * 64)
        print(f"Tokens Used / Budget:  {built.tokens} / {built.budget_total} ({built.utilization * 100:.1f}%)")
        print(f"Target:                {built.target or '(none)'} (uncertain={built.target_uncertain})")
        print(f"Files Included:        {len(built.files_included)}")
        print(f"Items Included:        {len(built.items)}")

        level_counts: dict[str, int] = {}
        for it in built.items:
            level_counts[it.level] = level_counts.get(it.level, 0) + 1
        level_summary = ", ".join(f"{k}: {v}" for k, v in sorted(level_counts.items()))
        print(f"Items by Level:        {level_summary}")
        print(f"Items Downgraded:      {built.downgraded_count}")
        print(f"Items Omitted:         {len(built.omitted_items)}")
        print(f"Build Time:            {built.build_time_seconds:.4f}s")
        print(f"Token Counter:         {built.counter_name}")
        if output_file:
            print(f"Output File:           {output_file}")
        print("=" * 64)

        if show_items:
            print("\nINCLUDED ITEMS TABLE:")
            print(f"{'Ref':<40} {'Role':<10} {'Level':<10} {'Tokens':<8} {'Score':<6}")
            print("-" * 78)
            for it in built.items:
                short_ref = it.ref if len(it.ref) <= 40 else "..." + it.ref[-37:]
                print(f"{short_ref:<40} {it.role:<10} {it.level:<10} {it.tokens:<8} {it.score:<6.2f}")
            print("=" * 64)

    return 0


def run_compress(
    repo_path: str,
    qualified_name: str,
    level: str = "full",
    counter_name: str = "tiktoken",
) -> int:
    """Compress a single code entity to a given detail level for inspection."""
    repo = Path(repo_path).resolve()
    if not repo.exists():
        sys.stderr.write(f"Repository path does not exist: {repo}\n")
        return 1

    try:
        counter = get_counter(counter_name)
    except Exception as err:
        sys.stderr.write(f"Error initializing token counter: {err}\n")
        return 1

    with MetadataStore.open(repo) as store:
        entity = store.get_function(qualified_name)
        if not entity:
            entity = store.get_class(qualified_name)
        if not entity:
            # Try searching by short name
            matches = store.find_functions_by_name(qualified_name, exact=True)
            if matches:
                entity = matches[0]

        if not entity:
            sys.stderr.write(f"Entity not found: {qualified_name}\n")
            return 1

        try:
            target_level = DetailLevel(level.lower())
        except ValueError:
            sys.stderr.write(f"Invalid detail level: {level}\n")
            return 1

        snip = compress(entity, target_level, counter=counter)
        orig_tokens = snip.original_tokens
        tokens = snip.tokens
        ratio = round(tokens / max(1, orig_tokens), 3)
        savings = round((1 - ratio) * 100, 1)

        print("=" * 64)
        print(f"COMPRESSED SNIPPET: {entity.qualified_name}")
        print(f"Level:            {snip.level.value.upper()}")
        print(f"File & Lines:     {snip.file}:{snip.start_line}-{snip.end_line}")
        print(f"Tokens:           {tokens} (Original: {orig_tokens})")
        print(f"Compression:      {ratio:.1%} of original ({savings}% savings)")
        if snip.degraded:
            print("Degraded:         True (fallback parser used)")
        print("=" * 64)
        print(snip.text)
        print("=" * 64)

    return 0


def run_budget_sweep(
    repo_path: str,
    query: str,
    budgets_str: str = "500,1000,2000,4000,8000",
    counter_name: str = "tiktoken",
) -> int:
    """Build context across multiple token budgets and display comparison table."""
    repo = Path(repo_path).resolve()
    if not repo.exists():
        sys.stderr.write(f"Repository path does not exist: {repo}\n")
        return 1

    try:
        counter = get_counter(counter_name)
    except Exception as err:
        sys.stderr.write(f"Error initializing token counter: {err}\n")
        return 1

    try:
        budgets = [int(b.strip()) for b in budgets_str.split(",") if b.strip()]
    except ValueError:
        sys.stderr.write(f"Invalid budgets string: {budgets_str}\n")
        return 1

    with MetadataStore.open(repo) as store:
        graph = None
        try:
            nx_g = load_graph(repo)
            if nx_g is not None:
                graph = CodeGraph(nx_g)
        except Exception:
            pass

        retriever = HybridSearch.from_repo(repo, metadata_store=store)
        retrieval_result = retriever.search(query)
        builder = ContextBuilder(metadata_store=store, graph=graph, counter=counter)

        print("=" * 86)
        print(f"BUDGET SWEEP FOR QUERY: '{query}'")
        print(f"Target: {retrieval_result.target or '(none)'} (confidence={retrieval_result.target_confidence})")
        print("=" * 86)
        header = f"{'Budget':<8} {'Tokens':<8} {'Util %':<8} {'Items':<7} {'Target Level':<14} {'Levels (F/T/S/St/R)':<22} {'Omitted':<8}"
        print(header)
        print("-" * 86)

        for b in sorted(budgets):
            try:
                built = builder.build(retrieval_result, budget=b)
                t_item = next((it for it in built.items if it.role == "target"), None)
                t_lvl = t_item.level.upper() if t_item else "N/A"

                counts = {"full": 0, "trimmed": 0, "signature": 0, "stub": 0, "reference": 0}
                for it in built.items:
                    lvl_lower = it.level.lower()
                    if lvl_lower in counts:
                        counts[lvl_lower] += 1

                lvl_str = f"{counts['full']}/{counts['trimmed']}/{counts['signature']}/{counts['stub']}/{counts['reference']}"
                util_pct = f"{built.utilization * 100:.1f}%"
                print(f"{b:<8} {built.tokens:<8} {util_pct:<8} {len(built.items):<7} {t_lvl:<14} {lvl_str:<22} {len(built.omitted_items):<8}")
            except BudgetTooSmallError:
                print(f"{b:<8} {'TOO SMALL':<8} {'-':<8} {'0':<7} {'NONE':<14} {'-':<22} {'-':<8}")

        print("=" * 86)

    return 0


def run_serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> int:
    """Start the FastAPI service and Web UI dashboard using uvicorn."""
    import uvicorn
    url = f"http://{host}:{port}/"
    print("=" * 64)
    print("Code Context Engine Web Dashboard")
    print(f"Web UI Dashboard:      {url}")
    print(f"REST API Docs:         http://{host}:{port}/docs")
    print("=" * 64)
    uvicorn.run("app.api.server:app", host=host, port=port, reload=reload)
    return 0


def run_update(repo_path: str, verify: bool = False, as_json: bool = False) -> int:
    """Run incremental indexing update on repository."""
    from app.services.incremental import update_index
    try:
        report = update_index(repo_path, verify=verify)
    except Exception as err:
        sys.stderr.write(f"Error during incremental update: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(report.model_dump_json(indent=2) + "\n")
        return 0

    print("=" * 64)
    print("Code Context Engine - Incremental Update Report")
    print("=" * 64)
    print(f"Repository Root:        {report.repo_path}")
    print(f"Added Files:            {len(report.added)}")
    print(f"Modified Files:         {len(report.modified)}")
    print(f"Deleted Files:          {len(report.deleted)}")
    print(f"Files Reparsed:         {report.files_reparsed}")
    print(f"Graph Nodes Changed:    {report.graph_nodes_changed}")
    print(f"Graph Edges Rechecked:  {report.graph_edges_rechecked}")
    print(f"Items Re-embedded:      {report.items_reembedded}")
    print(f"Store Update Time:      {report.store_seconds:.4f}s")
    print(f"Graph Update Time:      {report.graph_seconds:.4f}s")
    print(f"Embed Update Time:      {report.embed_seconds:.4f}s")
    print(f"Total Duration:         {report.total_seconds:.4f}s")
    if report.verified is not None:
        print(f"Full Rebuild Match:     {'PASS' if report.verified else 'FAIL'}")
    print("=" * 64)
    return 0


def run_watch(repo_path: str, interval: float = 2.0, verify: bool = False) -> int:
    """Start watcher mode for automatic repository incremental indexing."""
    from app.services.watcher import RepositoryWatcher
    print("=" * 64)
    print(f"Starting Repository Watcher: {Path(repo_path).resolve()}")
    print(f"Poll Interval: {interval}s | Equivalence Verification: {verify}")
    print("Press Ctrl+C to stop.")
    print("=" * 64)
    watcher = RepositoryWatcher(repo_path, interval=interval, verify=verify)
    watcher.watch()
    return 0



def run_providers(as_json: bool = False) -> int:
    """List available AI completion providers, readiness status, and default models."""
    from app.config import get_settings
    from app.ai.provider_factory import get_provider, get_registered_providers

    settings = get_settings()
    data = []
    for p_name in get_registered_providers():
        try:
            prov = get_provider(p_name, raise_if_model_missing=False)
            configured = prov.is_configured()
            model_id = settings.get_model_for_provider(p_name, raise_if_missing=False)
            hint = prov.get_setup_hint()
            is_loc = prov.is_local
            available_models = (
                prov.get_available_models()
                if hasattr(prov, "get_available_models")
                else []
            )
        except Exception as exc:
            configured = False
            model_id = None
            hint = str(exc)
            is_loc = False
            available_models = []

        data.append({
            "name": p_name,
            "configured": configured,
            "default_model": model_id,
            "is_local": is_loc,
            "setup_hint": hint,
            "available_models": available_models,
        })

    if as_json:
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
        return 0

    print("=" * 72)
    print("AI COMPLETION PROVIDERS")
    print("=" * 72)
    for p in data:
        status_tag = "[CONFIGURED]" if p["configured"] else "[NOT CONFIGURED]"
        loc_tag = "(local)" if p["is_local"] else "(cloud)"
        print(f"• {p['name']:<16} {status_tag:<18} {loc_tag}")
        if p["default_model"]:
            print(f"    Default Model:    {p['default_model']}")
        if p["setup_hint"] and not p["configured"]:
            print(f"    Setup Hint:       {p['setup_hint']}")
        if p["available_models"]:
            print(f"    Pulled Models:    {', '.join(p['available_models'])}")
        print()
    print("=" * 72)
    return 0


def run_ask(
    repo_path: str,
    query: str,
    budget: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    dry_run: bool = False,
    show_context: bool = False,
    as_json: bool = False,
) -> int:
    """Execute end-to-end coding assistant pipeline locally."""
    from app.services.pipeline import RepositoryService, PrerequisiteMissingError, IndexingInProgressError
    from app.ai.base_provider import ProviderError

    try:
        service = RepositoryService(repo_path)
        result = service.ask(
            query=query,
            budget=budget,
            dry_run=dry_run,
            provider_override=provider,
            model_override=model,
        )
    except (FileNotFoundError, PrerequisiteMissingError, IndexingInProgressError, ProviderError, ValueError) as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected error: {err}\n")
        return 1

    if as_json:
        sys.stdout.write(json.dumps(result.model_dump(), indent=2) + "\n")
        return 0

    if show_context:
        print("=" * 72)
        print("BUILT CONTEXT PROMPT")
        print("=" * 72)
        print(result.context.text)
        print("=" * 72)
        print()

    if dry_run:
        print("=" * 72)
        print("DRY RUN: EXACT PROMPT MESSAGES")
        print("=" * 72)
        print(f"PROVIDER: {result.provider or 'default'}")
        print(f"MODEL:    {result.model or 'default'}")
        print("\nSYSTEM PROMPT:")
        print(result.system_prompt)
        print("\nCONVERSATION MESSAGES:")
        for msg in result.messages:
            print(f"[{msg.role.upper()}]:")
            print(msg.content)
        print("=" * 72)
    else:
        print("=" * 72)
        print("AI CODING ASSISTANT RESPONSE")
        print("=" * 72)
        if result.answer:
            if result.answer.analysis:
                print("### ANALYSIS")
                print(result.answer.analysis)
                print()
            if result.answer.changes:
                print("### PROPOSED CHANGES")
                for ch in result.answer.changes:
                    syn_tag = "valid" if ch.syntax_valid else "INVALID SYNTAX"
                    unk_tag = " (UNKNOWN TARGET)" if ch.unknown_target else ""
                    print(f"### CHANGE: {ch.qualified_name} ({ch.file}) [syntax: {syn_tag}]{unk_tag}")
                    print(f"```{ch.language}")
                    print(ch.code)
                    print("```")
                    print()
            if result.answer.risks:
                print("### RISKS")
                for r in result.answer.risks:
                    print(f"- {r}")
                print()
            if result.answer.needs_context:
                print("### NEEDS_CONTEXT")
                for nc in result.answer.needs_context:
                    print(f"- {nc}")
                print()
            if result.answer.parse_warnings:
                print("### WARNINGS")
                for w in result.answer.parse_warnings:
                    print(f"* {w}")
                print()
        elif result.completion:
            print(result.completion.text)
        print("=" * 72)

    tr = result.token_report
    print("\n" + "=" * 72)
    print("TOKEN & LATENCY REPORT")
    print("=" * 72)
    print(f"Provider:              {tr.provider or 'N/A'}")
    print(f"Model:                 {tr.model or 'N/A'}")
    print(f"Context Tokens:        {tr.context_tokens} / {tr.budget} ({tr.utilization * 100:.1f}%)")
    est_tag = " (estimated)" if tr.usage_estimated else ""
    if tr.provider_input_tokens is not None:
        print(f"Provider Input Tokens: {tr.provider_input_tokens}{est_tag}")
        print(f"Provider Output Tokens:{tr.provider_output_tokens}{est_tag}")
        print(f"Provider Total Tokens: {tr.provider_total_tokens}{est_tag}")
    elif tr.usage_estimated:
        print("Provider Tokens:       Not reported by API (usage estimated)")
    if tr.exact_count_tokens is not None:
        print(f"Exact Count Tokens:    {tr.exact_count_tokens}")
    if tr.is_free_tier or tr.cost_display == "free tier":
        print("Estimated Cost:        free tier")
    elif tr.estimated_cost_usd is not None:
        print(f"Estimated Cost (USD):  ${tr.estimated_cost_usd:.6f}")
    elif tr.cost_display:
        print(f"Estimated Cost:        {tr.cost_display}")
    else:
        print("Estimated Cost:        unknown (no pricing configured)")
    print(f"Files (Context/Repo):  {tr.files_in_context} / {tr.files_in_repo}")
    print(f"Latency Breakdown:     Retrieval: {tr.retrieval_seconds:.3f}s | Context: {tr.context_build_seconds:.3f}s | LLM: {tr.llm_seconds:.3f}s | Total: {tr.total_seconds:.3f}s")
    print("=" * 72)
    return 0



def run_repo_tokens(repo_path: str, as_json: bool = False) -> int:
    """Calculate and display total tokens and per-file distribution across repository."""
    import statistics
    from app.context.token_counter import get_token_counter
    from app.storage.metadata_store import MetadataStore
    from app.scanner.repository_scanner import scan_repository

    repo_root = Path(repo_path).resolve()
    if not repo_root.is_dir():
        sys.stderr.write(f"Error: {repo_root} is not a directory.\n")
        return 1

    counter = get_token_counter("cl100k_base")
    file_tokens: list[tuple[str, int]] = []

    try:
        with MetadataStore.open(repo_root) as store:
            files = store.get_all_files()
            for f in files:
                src = f.source_code
                if not src:
                    full = repo_root / f.path
                    if full.is_file():
                        src = full.read_text(encoding="utf-8", errors="replace")
                if src:
                    file_tokens.append((f.path, counter.count_tokens(src)))
    except Exception:
        pass

    if not file_tokens:
        scan_res = scan_repository(repo_root)
        for sf in scan_res.files:
            full = repo_root / sf.path
            if full.is_file():
                try:
                    src = full.read_text(encoding="utf-8", errors="replace")
                    file_tokens.append((sf.path, counter.count_tokens(src)))
                except Exception:
                    pass

    if not file_tokens:
        sys.stderr.write("No Python source files found to measure.\n")
        return 1

    token_counts = [tok for _, tok in file_tokens]
    total_tokens = sum(token_counts)
    file_count = len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    median_tokens = statistics.median(token_counts)
    sorted_counts = sorted(token_counts)
    p95_idx = int(0.95 * len(sorted_counts))
    p95_tokens = sorted_counts[min(p95_idx, len(sorted_counts) - 1)]
    times_over_200k = total_tokens / 200_000.0

    if as_json:
        data = {
            "repo_path": str(repo_root),
            "file_count": file_count,
            "total_tokens": total_tokens,
            "min_tokens": min_tokens,
            "median_tokens": median_tokens,
            "max_tokens": max_tokens,
            "p95_tokens": p95_tokens,
            "times_over_200k_window": round(times_over_200k, 2),
        }
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
        return 0

    print("=" * 72)
    print(f"REPOSITORY SOURCE TOKENS REPORT: {repo_root.name.upper()}")
    print("=" * 72)
    print(f"Repository Root:             {repo_root}")
    print(f"Total Python Source Files:   {file_count:,}")
    print(f"Total cl100k_base Tokens:    {total_tokens:,}")
    print("-" * 72)
    print("Per-File Token Distribution:")
    print(f"  Minimum Tokens:            {min_tokens:,}")
    print(f"  Median Tokens:             {median_tokens:,.1f}")
    print(f"  95th Percentile (p95):     {p95_tokens:,}")
    print(f"  Maximum Tokens:            {max_tokens:,}")
    print("-" * 72)
    print(f"Context Window Scale:        {times_over_200k:.2f}x of a standard 200,000 token context window")
    print("=" * 72)
    return 0


def run_benchmark_cli(
    repo_path: str,
    queries_file: str,
    budget: int = 4000,
    strategies_str: str | None = None,
    call_ai: bool = False,
    ai_output_tokens: int = 512,
    repeat: int = 1,
    json_file: str | None = None,
    md_file: str | None = None,
) -> int:
    """Execute benchmark run and output reports."""
    from app.benchmark.baselines import (
        ALL_STRATEGIES,
        FixedWindowStrategy,
        GraphRetrievalStrategy,
        NaiveKeywordFilesStrategy,
        WholeFileOfTargetStrategy,
        WholeRepoStrategy,
    )
    from app.benchmark.report import generate_reports
    from app.benchmark.runner import BenchmarkQuery, run_benchmark

    repo_root = Path(repo_path).resolve()
    if not repo_root.is_dir():
        sys.stderr.write(f"Error: Repository path {repo_root} does not exist.\n")
        return 1

    q_path = Path(queries_file).resolve()
    if not q_path.is_file():
        sys.stderr.write(f"Error: Queries file {q_path} does not exist.\n")
        return 1

    try:
        raw_data = json.loads(q_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list):
            sys.stderr.write("Error: Queries file must contain a JSON array of query objects.\n")
            return 1
        queries = [BenchmarkQuery(**item) for item in raw_data]
    except Exception as err:
        sys.stderr.write(f"Error reading queries file: {err}\n")
        return 1

    active_strats = ALL_STRATEGIES
    if strategies_str:
        strat_map = {
            "whole_repo": WholeRepoStrategy(),
            "whole_file": WholeFileOfTargetStrategy(),
            "naive_keyword": NaiveKeywordFilesStrategy(),
            "fixed_window": FixedWindowStrategy(),
            "graph": GraphRetrievalStrategy(),
        }
        chosen = [s.strip() for s in strategies_str.split(",") if s.strip()]
        active_strats = [strat_map[k] for k in chosen if k in strat_map]

    print(f"Starting benchmark across {len(queries)} queries and {len(active_strats)} strategies...")
    result = run_benchmark(
        repo_path=repo_root,
        queries=queries,
        strategies=active_strats,
        budget=budget,
        call_ai=call_ai,
        ai_output_tokens=ai_output_tokens,
        repeat=repeat,
    )

    table_text, out_json, out_md = generate_reports(result, json_path=json_file, md_path=md_file)
    print("\n" + table_text + "\n")
    print(f"Written benchmark JSON report: {out_json}")
    print(f"Written benchmark Markdown summary: {out_md}")
    return 0


# Commands that work from the filesystem alone and need no persisted state.
_STATELESS_COMMANDS = {"scan", "parse", "parse-file", "serve", "providers", "repo-tokens"}


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI execution routine."""
    args = parse_args(argv)
    if args.command not in _STATELESS_COMMANDS and not get_settings().mongodb_uri:
        sys.stderr.write(
            "Error: this command keeps its index, graph, and embeddings in MongoDB between runs.\n"
            "Set MONGODB_URI (for example mongodb://localhost:27017) in the environment or .env file.\n"
        )
        sys.exit(2)
    if args.command == "scan":
        sys.exit(run_scan(
            repo_path=args.repo_path,
            as_json=args.json,
            max_size=args.max_size,
            extensions=args.ext,
        ))
    elif args.command == "parse":
        sys.exit(run_parse(
            repo_path=args.repo_path,
            as_json=args.json,
            max_size=args.max_size,
            extensions=args.ext,
        ))
    elif args.command == "parse-file":
        sys.exit(run_parse_file(
            file_path=args.file_path,
            as_json=args.json,
        ))
    elif args.command == "index":
        sys.exit(run_index(
            repo_path=args.repo_path,
            force=args.force,
            as_json=args.json,
        ))
    elif args.command == "stats":
        sys.exit(run_stats(
            repo_path=args.repo_path,
            as_json=args.json,
        ))
    elif args.command == "find":
        sys.exit(run_find(
            repo_path=args.repo_path,
            name=args.name,
            as_json=args.json,
        ))
    elif args.command == "show":
        sys.exit(run_show(
            repo_path=args.repo_path,
            qualified_name=args.qualified_name,
            as_json=args.json,
        ))
    elif args.command == "graph-build":
        sys.exit(run_graph_build(
            repo_path=args.repo_path,
            include_ambiguous=args.include_ambiguous,
            as_json=args.json,
        ))
    elif args.command == "graph-stats":
        sys.exit(run_graph_stats(
            repo_path=args.repo_path,
            as_json=args.json,
        ))
    elif args.command == "callees":
        sys.exit(run_callees(
            repo_path=args.repo_path,
            name=args.name,
            depth=args.depth,
            as_json=args.json,
        ))
    elif args.command == "callers":
        sys.exit(run_callers(
            repo_path=args.repo_path,
            name=args.name,
            depth=args.depth,
            as_json=args.json,
        ))
    elif args.command == "neighbors":
        sys.exit(run_neighbors(
            repo_path=args.repo_path,
            name=args.name,
            hops=args.hops,
            as_json=args.json,
        ))
    elif args.command == "path":
        sys.exit(run_path(
            repo_path=args.repo_path,
            from_name=args.from_name,
            to_name=args.to_name,
            as_json=args.json,
        ))
    elif args.command == "file-deps":
        sys.exit(run_file_deps(
            repo_path=args.repo_path,
            file_path=args.file_path,
            as_json=args.json,
        ))
    elif args.command == "embed":
        sys.exit(run_embed(
            repo_path=args.repo_path,
            force=args.force,
            model=args.model,
            as_json=args.json,
        ))
    elif args.command == "embed-stats":
        sys.exit(run_embed_stats(
            repo_path=args.repo_path,
            as_json=args.json,
        ))
    elif args.command == "semantic":
        sys.exit(run_semantic(
            repo_path=args.repo_path,
            query=args.query,
            top_k=args.top_k,
            kinds_str=args.kinds,
            as_json=args.json,
        ))
    elif args.command == "context":
        sys.exit(run_context(
            repo_path=args.repo_path,
            query=args.query,
            budget=args.budget,
            format_type=args.format,
            pin=args.pin,
            exclude=args.exclude,
            no_callers=args.no_callers,
            no_imports=args.no_imports,
            show_items=args.show_items,
            output_file=args.output,
            counter_name=args.counter,
        ))
    elif args.command == "compress":
        sys.exit(run_compress(
            repo_path=args.repo_path,
            qualified_name=args.qualified_name,
            level=args.level,
            counter_name=args.counter,
        ))
    elif args.command == "budget-sweep":
        sys.exit(run_budget_sweep(
            repo_path=args.repo_path,
            query=args.query,
            budgets_str=args.budgets,
            counter_name=args.counter,
        ))
    elif args.command == "serve":
        sys.exit(run_serve(
            host=args.host,
            port=args.port,
            reload=args.reload,
        ))
    elif args.command == "ask":
        sys.exit(run_ask(
            repo_path=args.repo_path,
            query=args.query,
            budget=args.budget,
            provider=args.provider,
            model=args.model,
            dry_run=args.dry_run,
            show_context=args.show_context,
            as_json=args.json,
        ))
    elif args.command == "providers":
        sys.exit(run_providers(as_json=args.json))
    elif args.command == "benchmark":
        sys.exit(run_benchmark_cli(
            repo_path=args.repo_path,
            queries_file=args.queries_file,
            budget=args.budget,
            strategies_str=args.strategies,
            call_ai=args.call_ai,
            ai_output_tokens=args.ai_output_tokens,
            repeat=args.repeat,
            json_file=args.json_file,
            md_file=args.md_file,
        ))
    elif args.command == "repo-tokens":
        sys.exit(run_repo_tokens(
            repo_path=args.repo_path,
            as_json=args.json,
        ))
    elif args.command == "update":
        sys.exit(run_update(
            repo_path=args.repo_path,
            verify=args.verify,
            as_json=args.json,
        ))
    elif args.command == "watch":
        sys.exit(run_watch(
            repo_path=args.repo_path,
            interval=args.interval,
            verify=args.verify,
        ))


if __name__ == "__main__":
    main()


