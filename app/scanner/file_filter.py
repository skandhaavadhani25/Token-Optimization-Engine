"""File filtering logic for repository scanning."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Collection, Sequence

DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset({
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    "node_modules",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "build",
    "dist",
    "site-packages",
    ".eggs",
})

DEFAULT_IGNORED_DIR_PATTERNS: tuple[str, ...] = ("*.egg-info",)


class GitIgnoreRule:
    """Represents a single rule parsed from a .gitignore file."""

    def __init__(self, raw_line: str) -> None:
        pattern = raw_line.strip()
        self.negation: bool = pattern.startswith("!")
        if self.negation:
            pattern = pattern[1:].strip()

        self.dir_only: bool = pattern.endswith("/")
        if self.dir_only:
            pattern = pattern[:-1]

        if pattern.startswith("/"):
            self.anchored: bool = True
            pattern = pattern[1:]
        elif "/" in pattern:
            self.anchored = True
        else:
            self.anchored = False

        self.pattern: str = pattern

    def matches(self, rel_path: str, is_dir: bool = False) -> bool:
        """Check whether a POSIX relative path matches this rule."""
        clean = rel_path.strip("/")
        if not clean:
            return False

        parts = clean.split("/")
        name = parts[-1]

        if self.dir_only:
            if is_dir:
                if self.anchored:
                    return fnmatch.fnmatch(clean, self.pattern) or clean.startswith(f"{self.pattern}/")
                return any(fnmatch.fnmatch(part, self.pattern) for part in parts)
            else:
                parent_parts = parts[:-1]
                if not parent_parts:
                    return False
                if self.anchored:
                    parent_path = "/".join(parent_parts)
                    return fnmatch.fnmatch(parent_path, self.pattern) or parent_path.startswith(f"{self.pattern}/")
                return any(fnmatch.fnmatch(part, self.pattern) for part in parent_parts)

        if self.anchored:
            return fnmatch.fnmatch(clean, self.pattern) or clean.startswith(f"{self.pattern}/")

        if fnmatch.fnmatch(name, self.pattern):
            return True
        return any(fnmatch.fnmatch(part, self.pattern) for part in parts)


class GitIgnoreMatcher:
    """Matcher for a collection of .gitignore rules."""

    def __init__(self, rules: list[GitIgnoreRule] | None = None) -> None:
        self.rules: list[GitIgnoreRule] = rules or []

    @classmethod
    def from_file(cls, gitignore_path: Path | str) -> GitIgnoreMatcher:
        """Parse rules from a .gitignore file if it exists."""
        path = Path(gitignore_path)
        if not path.is_file():
            return cls([])

        rules: list[GitIgnoreRule] = []
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                rules.append(GitIgnoreRule(stripped))
        except OSError:
            pass

        return cls(rules)

    def is_ignored(self, rel_path: str, is_dir: bool = False) -> bool:
        """Return True if rel_path matches ignore rules and is not overridden by negation."""
        ignored = False
        for rule in self.rules:
            if rule.matches(rel_path, is_dir=is_dir):
                ignored = not rule.negation
        return ignored


def is_binary_file(file_path: Path | str) -> bool:
    """Detect binary files by reading the first 1024 bytes and checking for null bytes.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if the file contains at least one null byte in the initial 1024 bytes.
    """
    path = Path(file_path)
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except OSError:
        return False


def should_skip_dir(
    name: str,
    rel_path: str | None = None,
    gitignore_matcher: GitIgnoreMatcher | None = None,
) -> bool:
    """Check if a directory name should be skipped.

    Args:
        name: Base name of the directory (e.g. '__pycache__', 'node_modules').
        rel_path: Optional relative path from repository root in POSIX format.
        gitignore_matcher: Optional GitIgnoreMatcher instance.

    Returns:
        True if the directory should be skipped, False otherwise.
    """
    lowered = name.lower()
    if lowered in DEFAULT_IGNORED_DIRS or name in DEFAULT_IGNORED_DIRS:
        return True

    for pattern in DEFAULT_IGNORED_DIR_PATTERNS:
        if fnmatch.fnmatch(lowered, pattern) or fnmatch.fnmatch(name, pattern):
            return True

    if gitignore_matcher is not None:
        target_path = rel_path if rel_path is not None else name
        if gitignore_matcher.is_ignored(target_path, is_dir=True):
            return True

    return False


def should_include_file(
    file_path: Path | str,
    allowed_extensions: Sequence[str] = (".py",),
    max_file_size: int = 1_000_000,
    rel_path: str | None = None,
    gitignore_matcher: GitIgnoreMatcher | None = None,
) -> bool:
    """Check if a file should be included in repository analysis.

    Args:
        file_path: Absolute or relative path to the file.
        allowed_extensions: Tuple or sequence of extensions to include.
        max_file_size: Maximum allowed file size in bytes (default 1MB).
        rel_path: Optional relative POSIX path for gitignore evaluation.
        gitignore_matcher: Optional GitIgnoreMatcher instance.

    Returns:
        True if the file meets all inclusion criteria, False otherwise.
    """
    path = Path(file_path)

    # 1. Extension check
    normalized_exts = {
        (ext.lower() if ext.startswith(".") else f".{ext.lower()}")
        for ext in allowed_extensions
    }
    if path.suffix.lower() not in normalized_exts:
        return False

    # 2. Gitignore check
    if gitignore_matcher is not None:
        check_path = rel_path if rel_path is not None else path.as_posix()
        if gitignore_matcher.is_ignored(check_path, is_dir=False):
            return False

    # 3. File existence and size check
    try:
        stat_info = path.stat()
    except OSError:
        return False

    if stat_info.st_size > max_file_size:
        return False

    # 4. Binary check
    if is_binary_file(path):
        return False

    return True


class FileFilter:
    """Encapsulates repository-level file and directory filtering configuration."""

    def __init__(
        self,
        repo_root: Path | str | None = None,
        allowed_extensions: Sequence[str] = (".py",),
        max_file_size: int = 1_000_000,
        respect_gitignore: bool = True,
        custom_ignored_dirs: Collection[str] | None = None,
    ) -> None:
        self.repo_root: Path | None = Path(repo_root).resolve() if repo_root else None
        self.allowed_extensions: tuple[str, ...] = tuple(
            (ext.lower() if ext.startswith(".") else f".{ext.lower()}")
            for ext in allowed_extensions
        )
        self.max_file_size: int = max_file_size
        self.custom_ignored_dirs: frozenset[str] = frozenset(custom_ignored_dirs or ())

        self.gitignore_matcher: GitIgnoreMatcher | None = None
        if respect_gitignore and self.repo_root:
            gitignore_file = self.repo_root / ".gitignore"
            if gitignore_file.is_file():
                self.gitignore_matcher = GitIgnoreMatcher.from_file(gitignore_file)

    def should_skip_dir(self, name: str, rel_path: str | None = None) -> bool:
        """Check if a directory should be pruned from traversal."""
        if name in self.custom_ignored_dirs or name.lower() in self.custom_ignored_dirs:
            return True
        return should_skip_dir(
            name=name,
            rel_path=rel_path,
            gitignore_matcher=self.gitignore_matcher,
        )

    def should_include_file(self, path: Path | str, rel_path: str | None = None) -> bool:
        """Check if a file should be included in the repository index."""
        target = Path(path)
        if rel_path is None and self.repo_root:
            try:
                rel_path = target.resolve().relative_to(self.repo_root).as_posix()
            except ValueError:
                rel_path = target.as_posix()

        return should_include_file(
            file_path=target,
            allowed_extensions=self.allowed_extensions,
            max_file_size=self.max_file_size,
            rel_path=rel_path,
            gitignore_matcher=self.gitignore_matcher,
        )
