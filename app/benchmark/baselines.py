"""Baseline and graph context strategies for benchmark comparisons."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from app.services.pipeline import RepositoryService
from app.storage.metadata_store import MetadataStore

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "it", "this", "that", "how", "what", "why", "can", "does",
    "explain", "optimize", "fix", "bug", "implement", "refactor",
    "with", "from", "by", "as", "into", "through", "about",
}


class StrategyContext(BaseModel):
    """Context built by a context extraction strategy."""

    strategy_name: str
    context_text: str
    files_inspected: list[str] = Field(default_factory=list)
    file_count: int = 0
    truncated: bool = False
    notes: str = ""


class ContextStrategy(ABC):
    """Abstract interface for a context construction strategy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this strategy."""

    @abstractmethod
    def build(
        self,
        repo_path: Path | str,
        query: str,
        budget: int | None = None,
    ) -> StrategyContext:
        """Construct the context payload for a given query and repo."""


def _resolve_target_file(repo_path: Path | str, query: str) -> tuple[str | None, str | None]:
    """Find the primary target file and symbol for a query using the index."""
    svc = RepositoryService(repo_path)
    try:
        retrieval_res, _ = svc.retrieve(query=query, max_tokens=1000)
        if retrieval_res.candidates:
            primary = retrieval_res.candidates[0]
            return primary.file, primary.ref
    except Exception:
        pass

    # Fallback to direct index lookup
    with MetadataStore.open(repo_path) as store:
        # Check if any query word matches a function or class
        words = re.findall(r"[A-Za-z0-9_]+", query)
        for w in words:
            funcs = store.find_functions_by_name(w)
            if funcs:
                return funcs[0].file_path, funcs[0].qualified_name
        files = store.get_all_files()
        if files:
            return files[0].file_path, None
    return None, None


def _read_file_content(repo_root: Path, rel_path: str, store: MetadataStore) -> str:
    """Read file content from disk or metadata store."""
    full_path = repo_root / rel_path
    if full_path.is_file():
        try:
            return full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    db_file = store.get_file(rel_path)
    if db_file and db_file.source_code:
        return db_file.source_code
    return ""


class WholeRepoStrategy(ContextStrategy):
    """Concatenate every indexed source file in the repository (capped at safety ceiling)."""

    def __init__(self, max_safety_chars: int = 2_000_000) -> None:
        self.max_safety_chars = max_safety_chars

    @property
    def name(self) -> str:
        return "whole_repo"

    def build(
        self,
        repo_path: Path | str,
        query: str,
        budget: int | None = None,
    ) -> StrategyContext:
        repo_root = Path(repo_path).resolve()
        pieces: list[str] = []
        files_inspected: list[str] = []
        current_chars = 0
        truncated = False

        with MetadataStore.open(repo_root) as store:
            all_files = store.get_all_file_paths()
            for p in all_files:
                content = _read_file_content(repo_root, p, store)
                file_text = f"# File: {p}\n{content}\n\n"
                file_len = len(file_text)

                if current_chars + file_len > self.max_safety_chars:
                    rem = max(0, self.max_safety_chars - current_chars)
                    if rem > 0:
                        pieces.append(file_text[:rem])
                        current_chars += rem
                    truncated = True
                    files_inspected.append(p)
                    break

                pieces.append(file_text)
                current_chars += file_len
                files_inspected.append(p)

        context_text = "".join(pieces)
        return StrategyContext(
            strategy_name=self.name,
            context_text=context_text,
            files_inspected=files_inspected,
            file_count=len(files_inspected),
            truncated=truncated,
            notes=f"Concatenated {len(files_inspected)} files ({len(context_text)} chars)",
        )


class WholeFileOfTargetStrategy(ContextStrategy):
    """Locate the file containing the target entity and send that entire file."""

    @property
    def name(self) -> str:
        return "whole_file"

    def build(
        self,
        repo_path: Path | str,
        query: str,
        budget: int | None = None,
    ) -> StrategyContext:
        repo_root = Path(repo_path).resolve()
        target_file, target_ref = _resolve_target_file(repo_root, query)

        if not target_file:
            return StrategyContext(
                strategy_name=self.name,
                context_text="",
                files_inspected=[],
                file_count=0,
                truncated=False,
                notes="No target file found",
            )

        with MetadataStore.open(repo_root) as store:
            content = _read_file_content(repo_root, target_file, store)

        context_text = f"# File: {target_file}\n{content}"
        return StrategyContext(
            strategy_name=self.name,
            context_text=context_text,
            files_inspected=[target_file],
            file_count=1,
            truncated=False,
            notes=f"Target: {target_ref or 'unknown'} in {target_file}",
        )


class NaiveKeywordFilesStrategy(ContextStrategy):
    """Find all files whose text contains any query keyword and concatenate them in full."""

    def __init__(self, max_safety_chars: int = 2_000_000) -> None:
        self.max_safety_chars = max_safety_chars

    @property
    def name(self) -> str:
        return "naive_keyword"

    def build(
        self,
        repo_path: Path | str,
        query: str,
        budget: int | None = None,
    ) -> StrategyContext:
        repo_root = Path(repo_path).resolve()
        # Extract keywords
        raw_tokens = re.findall(r"[A-Za-z0-9_]+", query)
        keywords = [t.lower() for t in raw_tokens if t.lower() not in STOP_WORDS and len(t) >= 3]
        if not keywords:
            keywords = [t.lower() for t in raw_tokens if len(t) >= 2]

        matched_files: list[tuple[str, str]] = []

        with MetadataStore.open(repo_root) as store:
            all_files = store.get_all_file_paths()
            for p in all_files:
                content = _read_file_content(repo_root, p, store)
                lower_content = content.lower()
                if any(kw in lower_content for kw in keywords):
                    matched_files.append((p, content))

        # Fallback if no files matched
        if not matched_files:
            target_file, _ = _resolve_target_file(repo_root, query)
            if target_file:
                with MetadataStore.open(repo_root) as store:
                    content = _read_file_content(repo_root, target_file, store)
                    matched_files = [(target_file, content)]

        pieces: list[str] = []
        files_inspected: list[str] = []
        current_chars = 0
        truncated = False

        for path, content in matched_files:
            file_text = f"# File: {path}\n{content}\n\n"
            file_len = len(file_text)

            if current_chars + file_len > self.max_safety_chars:
                rem = max(0, self.max_safety_chars - current_chars)
                if rem > 0:
                    pieces.append(file_text[:rem])
                    current_chars += rem
                truncated = True
                files_inspected.append(path)
                break

            pieces.append(file_text)
            current_chars += file_len
            files_inspected.append(path)

        context_text = "".join(pieces)
        return StrategyContext(
            strategy_name=self.name,
            context_text=context_text,
            files_inspected=files_inspected,
            file_count=len(files_inspected),
            truncated=truncated,
            notes=f"Matched {len(files_inspected)} files on keywords: {keywords}",
        )


class FixedWindowStrategy(ContextStrategy):
    """Target file plus sibling files in the same directory, capped at N files."""

    def __init__(self, n: int = 10, rule: str = "sibling", max_safety_chars: int = 2_000_000) -> None:
        self.n = n
        self.rule = rule  # "sibling" or "alphabetical"
        self.max_safety_chars = max_safety_chars

    @property
    def name(self) -> str:
        return "fixed_window"

    def build(
        self,
        repo_path: Path | str,
        query: str,
        budget: int | None = None,
    ) -> StrategyContext:
        repo_root = Path(repo_path).resolve()
        target_file, target_ref = _resolve_target_file(repo_root, query)

        selected_files: list[str] = []

        with MetadataStore.open(repo_root) as store:
            all_files = store.get_all_file_paths()

            if not target_file and all_files:
                target_file = all_files[0]

            if target_file:
                selected_files.append(target_file)
                target_dir = Path(target_file).parent

                if self.rule == "sibling":
                    siblings = [f for f in all_files if Path(f).parent == target_dir and f != target_file]
                    selected_files.extend(siblings[: self.n - 1])

                # Fill up to N if fewer siblings
                if len(selected_files) < self.n:
                    remainder = [f for f in all_files if f not in selected_files]
                    selected_files.extend(remainder[: self.n - len(selected_files)])

            selected_files = sorted(list(dict.fromkeys(selected_files)))

            pieces: list[str] = []
            files_inspected: list[str] = []
            current_chars = 0
            truncated = False

            for path in selected_files:
                content = _read_file_content(repo_root, path, store)
                file_text = f"# File: {path}\n{content}\n\n"
                file_len = len(file_text)

                if current_chars + file_len > self.max_safety_chars:
                    rem = max(0, self.max_safety_chars - current_chars)
                    if rem > 0:
                        pieces.append(file_text[:rem])
                        current_chars += rem
                    truncated = True
                    files_inspected.append(path)
                    break

                pieces.append(file_text)
                current_chars += file_len
                files_inspected.append(path)

        context_text = "".join(pieces)
        return StrategyContext(
            strategy_name=self.name,
            context_text=context_text,
            files_inspected=files_inspected,
            file_count=len(files_inspected),
            truncated=truncated,
            notes=f"Selected {len(files_inspected)} files (rule={self.rule}, N={self.n})",
        )


class GraphRetrievalStrategy(ContextStrategy):
    """Our real system: hybrid BM25 + FAISS + Graph retrieval packed under token budget."""

    @property
    def name(self) -> str:
        return "graph"

    def build(
        self,
        repo_path: Path | str,
        query: str,
        budget: int | None = None,
    ) -> StrategyContext:
        svc = RepositoryService(repo_path)
        token_budget = budget if budget is not None else 4000
        _, built_ctx = svc.retrieve(query=query, max_tokens=token_budget)

        files_inspected = sorted(list(set(item.file for item in built_ctx.items if item.file)))

        return StrategyContext(
            strategy_name=self.name,
            context_text=built_ctx.text,
            files_inspected=files_inspected,
            file_count=len(files_inspected),
            truncated=False,
            notes=(
                f"Graph retrieval tokens: {built_ctx.tokens}/{built_ctx.budget_total} "
                f"({built_ctx.utilization:.1%} budget utilized)"
            ),
        )


ALL_STRATEGIES: list[ContextStrategy] = [
    WholeRepoStrategy(),
    WholeFileOfTargetStrategy(),
    NaiveKeywordFilesStrategy(),
    FixedWindowStrategy(),
    GraphRetrievalStrategy(),
]
