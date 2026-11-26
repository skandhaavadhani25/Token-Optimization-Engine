"""Repository scanner for traversing directories and collecting Python file metadata."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Sequence

from app.models.schemas import FileInfo, ScanResult
from app.scanner.file_filter import FileFilter


class InvalidRepositoryError(ValueError):
    """Base exception for invalid repository path inputs."""


class RepositoryNotFoundError(InvalidRepositoryError, FileNotFoundError):
    """Raised when repository path does not exist."""


class RepositoryNotADirectoryError(InvalidRepositoryError, NotADirectoryError):
    """Raised when repository path is not a directory."""


def count_lines(file_path: Path | str) -> int:
    """Count lines in a UTF-8 text file.

    Raises:
        UnicodeDecodeError: If file cannot be decoded as UTF-8.
        OSError: If an I/O error occurs while reading.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def scan_repository(
    repo_path: Path | str,
    extensions: Sequence[str] = (".py",),
    max_file_size: int = 1_000_000,
    respect_gitignore: bool = True,
) -> ScanResult:
    """Scan a repository directory and return metadata for all matched files.

    Performs an efficient top-down directory walk, pruning ignored directories
    in place to prevent traversing unnecessary trees (e.g. node_modules, .git, venv).
    Gracefully catches and skips unreadable or malformed files.

    Args:
        repo_path: Path to the local repository root directory.
        extensions: Allowed file extensions (default: ('.py',)).
        max_file_size: Maximum allowed file size in bytes (default: 1,000,000).
        respect_gitignore: Whether to parse and respect .gitignore rules at repo root.

    Returns:
        ScanResult containing aggregate statistics and sorted FileInfo records.

    Raises:
        RepositoryNotFoundError: If repo_path does not exist on disk.
        RepositoryNotADirectoryError: If repo_path exists but is not a directory.
    """
    start_time = time.perf_counter()
    repo = Path(repo_path).resolve()

    if not repo.exists():
        raise RepositoryNotFoundError(f"Repository path does not exist: {repo}")
    if not repo.is_dir():
        raise RepositoryNotADirectoryError(f"Repository path is not a directory: {repo}")

    file_filter = FileFilter(
        repo_root=repo,
        allowed_extensions=extensions,
        max_file_size=max_file_size,
        respect_gitignore=respect_gitignore,
    )

    scanned_files: list[FileInfo] = []
    skipped_dirs: int = 0
    skipped_files: int = 0

    for root_str, dirs, filenames in os.walk(repo, topdown=True):
        root = Path(root_str)
        try:
            rel_root = root.relative_to(repo).as_posix()
            if rel_root == ".":
                rel_root = ""
        except ValueError:
            rel_root = ""

        # In-place directory pruning to avoid descending into ignored trees
        kept_dirs: list[str] = []
        for d in dirs:
            dir_rel = f"{rel_root}/{d}".lstrip("/") if rel_root else d
            if file_filter.should_skip_dir(name=d, rel_path=dir_rel):
                skipped_dirs += 1
            else:
                kept_dirs.append(d)
        dirs[:] = kept_dirs

        # Process candidate files
        for fname in filenames:
            file_path = root / fname
            file_rel = f"{rel_root}/{fname}".lstrip("/") if rel_root else fname

            if not file_filter.should_include_file(path=file_path, rel_path=file_rel):
                skipped_files += 1
                continue

            try:
                stat_info = file_path.stat()
                line_count = count_lines(file_path)
            except (UnicodeDecodeError, OSError):
                skipped_files += 1
                continue

            scanned_files.append(
                FileInfo(
                    path=file_rel,
                    absolute_path=str(file_path.resolve()),
                    size_bytes=stat_info.st_size,
                    line_count=line_count,
                    extension=file_path.suffix,
                    last_modified=stat_info.st_mtime,
                )
            )

    # Sort files deterministically by relative POSIX path
    scanned_files.sort(key=lambda item: item.path)

    scan_duration = time.perf_counter() - start_time
    total_files = len(scanned_files)
    total_size = sum(item.size_bytes for item in scanned_files)
    total_lines = sum(item.line_count for item in scanned_files)

    return ScanResult(
        repo_root=str(repo),
        files=scanned_files,
        total_files=total_files,
        total_size_bytes=total_size,
        total_lines=total_lines,
        skipped_dirs=skipped_dirs,
        skipped_files=skipped_files,
        scan_time_seconds=round(scan_duration, 4),
    )
