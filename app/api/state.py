"""Application state management: repository service cache and background jobs."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any
from fastapi import HTTPException

from app.config import get_settings
from app.services.pipeline import RepositoryService

_SERVICES_LOCK = threading.Lock()
_SERVICES: dict[Path, RepositoryService] = {}

_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def validate_and_resolve_repo_path(repo_path: str | Path) -> Path:
    """Validate repository path exists, is a directory, and complies with ALLOWED_REPO_ROOTS."""
    if not repo_path or not str(repo_path).strip():
        raise HTTPException(status_code=400, detail="Repository path cannot be empty.")

    try:
        resolved = Path(repo_path).resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid repository path format: {e}")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Repository path does not exist: {resolved}")
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail=f"Repository path is not a directory: {resolved}")

    # Check ALLOWED_REPO_ROOTS if configured
    settings = get_settings()
    allowed_roots = settings.get_allowed_roots()
    if allowed_roots:
        is_allowed = False
        for root in allowed_roots:
            try:
                resolved.relative_to(root)
                is_allowed = True
                break
            except ValueError:
                continue
        if not is_allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Access denied: Repository path '{resolved}' is outside allowed roots.",
            )

    return resolved


def resolve_repo_or_workspace(
    repo_path: str | Path | None = None,
    workspace_id: str | None = None,
) -> Path:
    """Resolve repository filesystem path from either workspace_id or raw repo_path.

    Uploaded workspaces MUST be resolved server-side via workspace_id rather than
    trusting client-supplied filesystem paths.
    """
    if workspace_id:
        from app.workspace.manager import WorkspaceManager
        mgr = WorkspaceManager()
        ws = mgr.get_workspace(workspace_id)
        if not ws:
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")
        target_path = Path(ws["working_path"]).resolve()
        if not target_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Workspace repository path not found: {target_path}",
            )
        return target_path

    if repo_path:
        return validate_and_resolve_repo_path(repo_path)

    raise HTTPException(
        status_code=400,
        detail="Either 'workspace_id' or 'repo_path' must be provided.",
    )


def get_service_for_target(
    repo_path: str | Path | None = None,
    workspace_id: str | None = None,
) -> RepositoryService:
    """Retrieve or create a cached RepositoryService instance for workspace or path."""
    resolved = resolve_repo_or_workspace(repo_path=repo_path, workspace_id=workspace_id)
    with _SERVICES_LOCK:
        if resolved not in _SERVICES:
            _SERVICES[resolved] = RepositoryService(resolved)
        return _SERVICES[resolved]


def get_service_for_repo(repo_path: str | Path) -> RepositoryService:
    """Retrieve or create a cached RepositoryService instance for a validated path."""
    resolved = validate_and_resolve_repo_path(repo_path)
    with _SERVICES_LOCK:
        if resolved not in _SERVICES:
            _SERVICES[resolved] = RepositoryService(resolved)
        return _SERVICES[resolved]


def create_background_job(repo_path: str) -> str:
    """Initialize a tracked background job."""
    job_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "repo_path": repo_path,
            "status": "queued",
            "result": None,
            "error": None,
        }
    return job_id


def update_job_status(
    job_id: str,
    status: str,
    result: Any = None,
    error: str | None = None,
) -> None:
    """Update status, output, or error of a background job."""
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = status
            if result is not None:
                _JOBS[job_id]["result"] = result
            if error is not None:
                _JOBS[job_id]["error"] = error


def get_background_job(job_id: str) -> dict[str, Any] | None:
    """Retrieve state of a tracked background job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None
