"""Workspace management for Code Context Engine.

Manages workspace lifecycle, local path copy, archive upload with zip-slip & symlink
safety validation, secret scanning, text-only indexing, and clean deletion.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from app.api.state import validate_and_resolve_repo_path
from app.chat.storage import ChatStorage
from app.config import get_settings
from app.scanner.file_filter import DEFAULT_IGNORED_DIRS, FileFilter
from app.workspace.secret_scanner import is_secret_filename, scan_file_for_secrets

logger = logging.getLogger(__name__)

# Extensions considered text-only for keyword and semantic chunk indexing
TEXT_ONLY_EXTENSIONS: frozenset[str] = frozenset({
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".log",
    ".rst",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sql",
    ".sh",
    ".bash",
    ".xml",
})

# Extensions considered binary and skipped from indexing
BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc",
    ".pyd",
    ".pyo",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".sqlite",
    ".db",
    ".class",
    ".jar",
    ".wasm",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
})


class WorkspaceError(Exception):
    """Base exception for workspace operations."""


class UploadSecurityError(WorkspaceError):
    """Raised when an uploaded archive violates security rules (zip-slip, symlinks, size limits)."""


def _single_top_level_folder(infolist: list[zipfile.ZipInfo]) -> str | None:
    """Return the folder name when every file in the archive lives under one folder."""
    tops: set[str] = set()
    for member in infolist:
        if member.is_dir():
            continue
        parts = member.filename.replace("\\", "/").split("/")
        if len(parts) < 2:
            return None
        tops.add(parts[0])
        if len(tops) > 1:
            return None
    return next(iter(tops)) if tops else None


class WorkspaceManager:
    """Manages workspaces, files, indexing triggers, and cleanup."""

    def __init__(self, storage: ChatStorage | None = None) -> None:
        self.settings = get_settings()
        self.storage = storage or ChatStorage()
        self.workspaces_base: Path = self.settings.workspaces_dir.resolve()
        self.workspaces_base.mkdir(parents=True, exist_ok=True)

    def _generate_workspace_id(self) -> str:
        return f"ws_{uuid.uuid4().hex[:10]}"

    def _get_workspace_repo_path(self, workspace_id: str) -> Path:
        return self.workspaces_base / workspace_id / "repo"

    def list_workspaces(self) -> list[dict[str, Any]]:
        return self.storage.list_workspaces()

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        return self.storage.get_workspace(workspace_id)

    # =========================================================================
    # Workspace Creation: Local Path
    # =========================================================================

    def create_from_local_path(
        self,
        local_path: str | Path,
        name: str | None = None,
        edit_in_place: bool = False,
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        """Create a workspace by copying a local repository or pointing to it.

        Args:
            local_path: Path to existing repository.
            name: Optional human-readable name.
            edit_in_place: If True, uses the original directory directly (requires confirmation).
                           If False (default), safely copies the code into a workspace copy.

        Returns:
            Tuple of (workspace_record, secret_warnings, skip_report).
        """
        resolved_src = validate_and_resolve_repo_path(local_path)
        ws_id = self._generate_workspace_id()
        ws_name = name or resolved_src.name or "Local Repo"

        secret_warnings: list[str] = []
        skip_report: dict[str, list[str]] = {"ignored_dirs": [], "binary_files": [], "secrets": []}
        file_counts = {"python": 0, "text_only": 0, "skipped": 0}

        if edit_in_place:
            working_path = resolved_src
        else:
            working_path = self._get_workspace_repo_path(ws_id)
            working_path.mkdir(parents=True, exist_ok=True)

            file_filter = FileFilter(resolved_src, respect_gitignore=True)

            # Copy files while respecting filters
            for root_str, dirs, filenames in os.walk(resolved_src, topdown=True):
                root = Path(root_str)
                try:
                    rel_root_str = root.relative_to(resolved_src).as_posix()
                    if rel_root_str == ".":
                        rel_root_str = ""
                except ValueError:
                    rel_root_str = ""

                # Directory pruning
                kept_dirs: list[str] = []
                for d in dirs:
                    d_rel = f"{rel_root_str}/{d}".lstrip("/") if rel_root_str else d
                    if file_filter.should_skip_dir(name=d, rel_path=d_rel):
                        skip_report["ignored_dirs"].append(d_rel)
                    else:
                        kept_dirs.append(d)
                dirs[:] = kept_dirs

                dest_dir = working_path / rel_root_str
                dest_dir.mkdir(parents=True, exist_ok=True)

                for file_name in filenames:
                    src_file = root / file_name
                    rel_file_str = f"{rel_root_str}/{file_name}".lstrip("/") if rel_root_str else file_name
                    rel_file = Path(rel_file_str)

                    if file_filter.gitignore_matcher and file_filter.gitignore_matcher.is_ignored(rel_file_str):
                        skip_report["secrets"].append(rel_file_str)
                        file_counts["skipped"] += 1
                        continue

                    # Scan for secrets
                    if is_secret_filename(src_file):
                        secret_warnings.append(f"Excluded secret config file from index: {rel_file}")
                        skip_report["secrets"].append(str(rel_file))
                        file_counts["skipped"] += 1
                        # We still copy it so workspace is complete, but it is excluded from prompt/indexing
                        shutil.copy2(src_file, dest_dir / file_name)
                        continue

                    ext = src_file.suffix.lower()
                    if ext in BINARY_EXTENSIONS:
                        skip_report["binary_files"].append(str(rel_file))
                        file_counts["skipped"] += 1
                        continue

                    # Check file size (cap 1MB)
                    try:
                        sz = src_file.stat().st_size
                        if sz > 1_000_000:
                            file_counts["skipped"] += 1
                            continue
                    except OSError:
                        file_counts["skipped"] += 1
                        continue

                    # Copy text/python file
                    shutil.copy2(src_file, dest_dir / file_name)

                    if ext == ".py":
                        file_counts["python"] += 1
                    else:
                        file_counts["text_only"] += 1

                    # Scan content for secret keys
                    try:
                        content_sample = src_file.read_bytes()[:50_000]
                        warns = scan_file_for_secrets(rel_file, content_sample)
                        if warns:
                            secret_warnings.extend(warns)
                    except Exception:
                        pass

        record = self.storage.create_workspace(
            name=ws_name,
            source="local_path",
            working_path=working_path,
            original_path=resolved_src,
            edit_in_place=edit_in_place,
            file_counts=file_counts,
            workspace_id=ws_id,
        )

        return record, secret_warnings, skip_report

    # =========================================================================
    # Workspace Creation: Upload (Zip Archive)
    # =========================================================================

    def create_from_upload(
        self,
        file_bytes: bytes,
        filename: str,
        name: str | None = None,
    ) -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
        """Create a workspace by extracting an uploaded .zip archive with strict safety checks.

        Args:
            file_bytes: Raw bytes of the uploaded zip archive.
            filename: Original uploaded filename.
            name: Optional workspace name.

        Returns:
            Tuple of (workspace_record, secret_warnings, skip_report).
        """
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise UploadSecurityError(
                f"Uploaded file exceeds maximum limit of {self.settings.max_upload_mb} MB."
            )

        try:
            zf = zipfile.ZipFile(io.BytesIO(file_bytes))
        except zipfile.BadZipFile as exc:
            raise UploadSecurityError(f"Invalid or corrupted zip archive: {exc}")

        ws_id = self._generate_workspace_id()
        dest_repo = self._get_workspace_repo_path(ws_id)
        dest_repo.mkdir(parents=True, exist_ok=True)
        resolved_dest = dest_repo.resolve()

        file_counts = {"python": 0, "text_only": 0, "skipped": 0}
        secret_warnings: list[str] = []
        skip_report: dict[str, list[str]] = {
            "zip_slip_rejected": [],
            "symlinks_rejected": [],
            "ignored_dirs": [],
            "binary_files": [],
            "oversize_files": [],
            "secrets": [],
        }

        infolist = zf.infolist()
        if len(infolist) > self.settings.max_files:
            shutil.rmtree(dest_repo.parent, ignore_errors=True)
            raise UploadSecurityError(
                f"Archive contains {len(infolist):,} entries; maximum allowed is {self.settings.max_files:,}."
            )

        # First pass: Validate security (Zip-slip, symlinks, absolute paths)
        for member in infolist:
            m_name = member.filename

            # 1. Reject absolute paths or Windows drive letters
            if m_name.startswith("/") or m_name.startswith("\\") or re.match(r"^[a-zA-Z]:", m_name):
                shutil.rmtree(dest_repo.parent, ignore_errors=True)
                raise UploadSecurityError(
                    f"Zip archive contains unsafe absolute path entry: '{m_name}'"
                )

            # 2. Reject path traversal (Zip-slip)
            parts = Path(m_name.replace("\\", "/")).parts
            if ".." in parts:
                shutil.rmtree(dest_repo.parent, ignore_errors=True)
                raise UploadSecurityError(
                    f"Zip archive contains unsafe path traversal ('..') entry: '{m_name}'"
                )

            # 3. Reject symlinks and device files
            is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
            if is_symlink:
                shutil.rmtree(dest_repo.parent, ignore_errors=True)
                raise UploadSecurityError(
                    f"Symlinks and device entries are forbidden: '{m_name}'"
                )

        # A single wrapping folder (my_project/...) is dropped so imports like
        # "from auth.login import x" resolve against the project root.
        wrapper = _single_top_level_folder(infolist)

        # Second pass: Extract files with filtering
        for member in infolist:
            if member.is_dir():
                continue

            # Archives built on Windows may use backslash separators.
            rel_path = Path(member.filename.replace("\\", "/"))
            if wrapper and rel_path.parts and rel_path.parts[0] == wrapper:
                rel_path = Path(*rel_path.parts[1:]) if len(rel_path.parts) > 1 else rel_path
            # Skip entries inside ignored directories (.git, node_modules, venv, etc.)
            if any(part in DEFAULT_IGNORED_DIRS for part in rel_path.parts):
                skip_report["ignored_dirs"].append(member.filename)
                file_counts["skipped"] += 1
                continue

            # Check individual file size limit (1MB max per source file)
            if member.file_size > 1_000_000:
                skip_report["oversize_files"].append(member.filename)
                file_counts["skipped"] += 1
                continue

            ext = rel_path.suffix.lower()
            if ext in BINARY_EXTENSIONS:
                skip_report["binary_files"].append(member.filename)
                file_counts["skipped"] += 1
                continue

            # Extract content
            target_path = (resolved_dest / rel_path).resolve()
            # Double check path boundary
            try:
                target_path.relative_to(resolved_dest)
            except ValueError:
                shutil.rmtree(dest_repo.parent, ignore_errors=True)
                raise UploadSecurityError(f"Path escape attempt detected: '{member.filename}'")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            content = zf.read(member)
            target_path.write_bytes(content)

            # Secret check
            if is_secret_filename(rel_path):
                secret_warnings.append(f"Excluded secret config file: {member.filename}")
                skip_report["secrets"].append(member.filename)
                file_counts["skipped"] += 1
                continue

            if ext == ".py":
                file_counts["python"] += 1
            else:
                file_counts["text_only"] += 1

            # Content secret check
            warns = scan_file_for_secrets(rel_path, content)
            if warns:
                secret_warnings.extend(warns)

        ws_name = name or Path(filename).stem or "Uploaded Workspace"
        record = self.storage.create_workspace(
            name=ws_name,
            source="upload",
            working_path=resolved_dest,
            original_path=None,
            edit_in_place=False,
            file_counts=file_counts,
            workspace_id=ws_id,
        )

        return record, secret_warnings, skip_report

    def create_from_loose_files(
        self,
        files: list[tuple[str, bytes]],
        name: str | None = None,
    ) -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
        """Create a workspace by copying loose files (e.g. single .py or multiple uploaded files/folder)."""
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        total_size = sum(len(b) for _, b in files)
        if total_size > max_bytes:
            raise UploadSecurityError(
                f"Uploaded files exceed maximum limit of {self.settings.max_upload_mb} MB."
            )
        if len(files) > self.settings.max_files:
            raise UploadSecurityError(
                f"Uploaded {len(files):,} files; maximum allowed is {self.settings.max_files:,}."
            )

        ws_id = self._generate_workspace_id()
        dest_repo = self._get_workspace_repo_path(ws_id)
        dest_repo.mkdir(parents=True, exist_ok=True)
        resolved_dest = dest_repo.resolve()

        file_counts = {"python": 0, "text_only": 0, "skipped": 0}
        secret_warnings: list[str] = []
        skip_report: dict[str, list[str]] = {
            "ignored_dirs": [],
            "binary_files": [],
            "oversize_files": [],
            "secrets": [],
        }

        first_filename = None
        for rel_name, content in files:
            clean_name = rel_name.replace("\\", "/").lstrip("/")
            if clean_name.startswith("/") or re.match(r"^[a-zA-Z]:", clean_name):
                shutil.rmtree(dest_repo.parent, ignore_errors=True)
                raise UploadSecurityError(f"Upload contains unsafe absolute path entry: '{rel_name}'")

            parts = Path(clean_name).parts
            if ".." in parts:
                shutil.rmtree(dest_repo.parent, ignore_errors=True)
                raise UploadSecurityError(f"Upload contains unsafe path traversal ('..') entry: '{rel_name}'")

            rel_path = Path(clean_name)
            if not first_filename:
                first_filename = rel_path.name

            if any(part in DEFAULT_IGNORED_DIRS for part in rel_path.parts):
                skip_report["ignored_dirs"].append(clean_name)
                file_counts["skipped"] += 1
                continue

            if len(content) > 1_000_000:
                skip_report["oversize_files"].append(clean_name)
                file_counts["skipped"] += 1
                continue

            ext = rel_path.suffix.lower()
            if ext in BINARY_EXTENSIONS:
                skip_report["binary_files"].append(clean_name)
                file_counts["skipped"] += 1
                continue

            target_path = (resolved_dest / rel_path).resolve()
            try:
                target_path.relative_to(resolved_dest)
            except ValueError:
                shutil.rmtree(dest_repo.parent, ignore_errors=True)
                raise UploadSecurityError(f"Path escape attempt detected: '{clean_name}'")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(content)

            if is_secret_filename(rel_path):
                secret_warnings.append(f"Excluded secret config file: {clean_name}")
                skip_report["secrets"].append(clean_name)
                file_counts["skipped"] += 1
                continue

            if ext == ".py":
                file_counts["python"] += 1
            else:
                file_counts["text_only"] += 1

            warns = scan_file_for_secrets(rel_path, content)
            if warns:
                secret_warnings.extend(warns)

        ws_name = name or (Path(first_filename).stem if first_filename else "Uploaded Workspace")
        record = self.storage.create_workspace(
            name=ws_name,
            source="upload",
            working_path=resolved_dest,
            original_path=None,
            edit_in_place=False,
            file_counts=file_counts,
            workspace_id=ws_id,
        )

        return record, secret_warnings, skip_report

    def add_files_to_workspace(
        self,
        workspace_id: str,
        files: list[tuple[str, bytes]],
    ) -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
        """Add files to an existing workspace and incrementally update index and graph."""
        ws = self.storage.get_workspace(workspace_id)
        if not ws:
            raise ValueError(f"Workspace '{workspace_id}' not found.")

        dest_repo = Path(ws["working_path"]).resolve()
        file_counts = dict(ws.get("file_counts") or {"python": 0, "text_only": 0, "skipped": 0})
        secret_warnings: list[str] = []
        skip_report: dict[str, list[str]] = {
            "ignored_dirs": [],
            "binary_files": [],
            "oversize_files": [],
            "secrets": [],
        }

        for rel_name, content in files:
            clean_name = rel_name.replace("\\", "/").lstrip("/")
            if clean_name.startswith("/") or re.match(r"^[a-zA-Z]:", clean_name):
                raise UploadSecurityError(f"Upload contains unsafe absolute path entry: '{rel_name}'")

            parts = Path(clean_name).parts
            if ".." in parts:
                raise UploadSecurityError(f"Upload contains unsafe path traversal ('..') entry: '{rel_name}'")

            rel_path = Path(clean_name)
            if any(part in DEFAULT_IGNORED_DIRS for part in rel_path.parts):
                skip_report["ignored_dirs"].append(clean_name)
                file_counts["skipped"] += 1
                continue

            if len(content) > 1_000_000:
                skip_report["oversize_files"].append(clean_name)
                file_counts["skipped"] += 1
                continue

            ext = rel_path.suffix.lower()
            if ext in BINARY_EXTENSIONS:
                skip_report["binary_files"].append(clean_name)
                file_counts["skipped"] += 1
                continue

            target_path = (dest_repo / rel_path).resolve()
            try:
                target_path.relative_to(dest_repo)
            except ValueError:
                raise UploadSecurityError(f"Path escape attempt detected: '{clean_name}'")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(content)

            if is_secret_filename(rel_path):
                secret_warnings.append(f"Excluded secret config file: {clean_name}")
                skip_report["secrets"].append(clean_name)
                file_counts["skipped"] += 1
                continue

            if ext == ".py":
                file_counts["python"] = file_counts.get("python", 0) + 1
            else:
                file_counts["text_only"] = file_counts.get("text_only", 0) + 1

            warns = scan_file_for_secrets(rel_path, content)
            if warns:
                secret_warnings.extend(warns)

        # Run incremental indexing on workspace repo
        report_data = None
        try:
            from app.services.incremental import update_index
            report = update_index(dest_repo, verify=False)
            report_data = report.model_dump()
        except Exception as exc:
            logger.warning(f"Incremental update failed after adding files: {exc}")

        self.storage.update_workspace_indexed(workspace_id, file_counts)
        updated_ws = self.storage.get_workspace(workspace_id)
        return updated_ws or ws, secret_warnings, skip_report, report_data

    # =========================================================================
    # Workspace Creation: Empty (Chat Only)
    # =========================================================================

    def create_empty(self, name: str = "Empty Workspace") -> dict[str, Any]:
        """Create a workspace with no repository (chat only)."""
        ws_id = self._generate_workspace_id()
        dest_repo = self._get_workspace_repo_path(ws_id)
        dest_repo.mkdir(parents=True, exist_ok=True)

        return self.storage.create_workspace(
            name=name,
            source="empty",
            working_path=dest_repo,
            original_path=None,
            edit_in_place=False,
            file_counts={"python": 0, "text_only": 0, "skipped": 0},
            workspace_id=ws_id,
        )

    # =========================================================================
    # Deletion & Cleanup
    # =========================================================================

    def delete_workspace(self, workspace_id: str) -> None:
        """Completely remove a workspace, disk files, indices, and database records."""
        ws = self.storage.get_workspace(workspace_id)
        if not ws:
            return

        working_path = Path(ws["working_path"])
        edit_in_place = ws.get("edit_in_place", False)

        # 1. Clean up workspace directory on disk if it is our owned copy
        if not edit_in_place:
            ws_folder = self.workspaces_base / workspace_id
            if ws_folder.exists():
                shutil.rmtree(ws_folder, ignore_errors=True)

        # 2. Clean up indexed artifacts
        try:
            from app.embeddings.vector_store import VectorStore
            from app.graph.graph_io import delete_graph
            from app.storage.metadata_store import MetadataStore

            MetadataStore.drop(working_path)
            delete_graph(working_path)
            VectorStore.drop(working_path)
        except Exception as exc:
            logger.warning(f"Error cleaning indexed artifacts for {workspace_id}: {exc}")

        # 3. Delete database records
        self.storage.delete_workspace(workspace_id)
