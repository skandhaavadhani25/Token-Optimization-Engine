"""API endpoints for Workspaces, file browsing, AI diffs, and undo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from app.api.state import get_service_for_repo
from app.chat.storage import ChatStorage
from app.security import upload_rate_limiter
from app.storage.mongo import StorageUnavailableError
from app.workspace.manager import UploadSecurityError, WorkspaceManager
from app.workspace.patcher import WorkspacePatcher

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Workspaces"])


class ApplyEditRequest(BaseModel):
    change_type: str = Field(default="modify", description="'modify' or 'new_file'")
    target_symbol: str = Field(description="Target symbol qualified name")
    file_path: str = Field(description="Relative path of file inside workspace")
    new_content: str = Field(description="Complete replacement content")
    allow_syntax_error: bool = Field(default=False, description="Override syntax validation errors")
    message_id: str | None = Field(default=None, description="Associated assistant message ID")


@router.post("/workspaces")
async def create_workspace(request: Request) -> dict[str, Any]:
    """Create a new workspace via local directory path copy or archive upload."""
    client_ip = getattr(getattr(request, "client", None), "host", "unknown")
    if not upload_rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many upload requests. Please wait a minute.")

    try:
        mgr = WorkspaceManager()
    except StorageUnavailableError:
        raise
    except Exception as e:
        logger.exception("Workspace manager failed to start")
        raise HTTPException(status_code=500, detail=f"Could not prepare the workspace folder: {e.__class__.__name__}: {e}")
    content_type = request.headers.get("content-type", "")

    local_path = None
    name = None
    edit_in_place = False

    uploaded_files_data = []
    if "multipart/form-data" in content_type:
        form = await request.form()
        raw_files = form.getlist("files") or form.getlist("file")
        for uf in raw_files:
            if hasattr(uf, "read"):
                data = await uf.read()
                fname = getattr(uf, "filename", "file.py")
                uploaded_files_data.append((fname, data))
        local_path = form.get("local_path")
        name = form.get("name")
        edit_in_place = str(form.get("edit_in_place", "false")).lower() in ("true", "1")
    elif "application/json" in content_type:
        try:
            body = await request.json()
            local_path = body.get("local_path")
            name = body.get("name")
            edit_in_place = bool(body.get("edit_in_place", False))
        except Exception:
            pass

    # 1. File Upload Case (ZIP archive, single code file, or multiple files)
    if uploaded_files_data:
        try:
            if len(uploaded_files_data) == 1 and uploaded_files_data[0][0].lower().endswith(".zip"):
                record, secrets, skip_rep = mgr.create_from_upload(
                    file_bytes=uploaded_files_data[0][1],
                    filename=uploaded_files_data[0][0],
                    name=name,
                )
            else:
                record, secrets, skip_rep = mgr.create_from_loose_files(
                    files=uploaded_files_data,
                    name=name,
                )

            # Trigger indexing
            ws_repo = Path(record["working_path"])
            try:
                svc = get_service_for_repo(ws_repo)
                svc.index(force=True)
            except Exception as exc:
                logger.warning(f"Initial indexing failed: {exc}")

            return {
                "workspace": record,
                "secret_warnings": secrets,
                "skip_report": skip_rep,
            }
        except UploadSecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except StorageUnavailableError:
            raise
        except Exception as e:
            logger.exception("Upload processing failed")
            raise HTTPException(status_code=500, detail=f"Upload processing failed: {e.__class__.__name__}: {e}")

    # 2. Local Path Copy / In-Place Case
    if local_path:
        try:
            record, secrets, skip_rep = mgr.create_from_local_path(
                local_path=local_path,
                name=name,
                edit_in_place=edit_in_place,
            )
            # Index workspace
            ws_repo = Path(record["working_path"])
            indexed_count = 0
            text_only_count = 0
            skipped_count = 0
            try:
                svc = get_service_for_repo(ws_repo)
                svc.index(force=False)
                stats = svc.stats()
                indexed_count = stats.get("database", {}).get("files", 0)
                text_only_count = len(skip_rep.get("text_only_files", []))
                skipped_count = len(skip_rep.get("skipped_files", []))
            except Exception as exc:
                logger.warning(f"Initial indexing failed: {exc}")

            return {
                "workspace": record,
                "secret_warnings": secrets,
                "skip_report": skip_rep,
                "indexed_files_count": indexed_count,
                "text_only_files_count": text_only_count,
                "skipped_files_count": skipped_count,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 3. Empty Workspace (Chat only)
    record = mgr.create_empty(name=name or "Empty Chat Workspace")
    return {
        "workspace": record,
        "secret_warnings": [],
        "skip_report": {"ignored_dirs": [], "binary_files": [], "secrets": []},
    }


@router.get("/workspaces")
def list_workspaces() -> list[dict[str, Any]]:
    """List all created workspaces."""
    mgr = WorkspaceManager()
    return mgr.list_workspaces()


@router.get("/workspaces/{workspace_id}")
def get_workspace(workspace_id: str) -> dict[str, Any]:
    """Retrieve details for a specific workspace."""
    mgr = WorkspaceManager()
    ws = mgr.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")
    return ws


@router.delete("/workspaces/{workspace_id}")
def delete_workspace(workspace_id: str) -> dict[str, Any]:
    """Delete workspace copy, indexed data, and database records."""
    mgr = WorkspaceManager()
    ws = mgr.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")
    mgr.delete_workspace(workspace_id)
    return {"status": "deleted", "workspace_id": workspace_id}


@router.post("/workspaces/{workspace_id}/reindex")
def reindex_workspace(workspace_id: str) -> dict[str, Any]:
    """Reindex an existing workspace."""
    mgr = WorkspaceManager()
    ws = mgr.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")
    repo_path = Path(ws["working_path"])
    svc = get_service_for_repo(repo_path)
    report = svc.index(force=True)
    return {"status": "completed", "report": report.model_dump()}


@router.post("/workspaces/{workspace_id}/files")
async def add_files_to_workspace_endpoint(workspace_id: str, request: Request) -> dict[str, Any]:
    """Incrementally add files to an existing workspace and update index/graph."""
    client_ip = getattr(getattr(request, "client", None), "host", "unknown")
    if not upload_rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many upload requests. Please wait a minute.")

    mgr = WorkspaceManager()
    ws = mgr.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")

    form = await request.form()
    raw_files = form.getlist("files") or form.getlist("file")
    if not raw_files:
        raise HTTPException(status_code=400, detail="No files provided in form.")

    files_data = []
    for f in raw_files:
        if hasattr(f, "read"):
            data = await f.read()
            fname = getattr(f, "filename", "file.py")
            files_data.append((fname, data))

    try:
        updated_ws, secrets, skip_rep, report = mgr.add_files_to_workspace(workspace_id, files_data)
        return {
            "status": "updated",
            "workspace": updated_ws,
            "secret_warnings": secrets,
            "skip_report": skip_rep,
            "report": report,
        }
    except UploadSecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add files to workspace: {e}")


@router.get("/workspaces/{workspace_id}/file")
def get_workspace_file_content(
    workspace_id: str,
    path: str = Query(..., description="Relative file path"),
    start_line: int | None = Query(default=None, ge=1),
    end_line: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    """Read-only secure file viewer for code citations (path-traversal safe)."""
    mgr = WorkspaceManager()
    ws = mgr.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")

    repo_path = Path(ws["working_path"]).resolve()
    target_path = (repo_path / path).resolve()

    try:
        target_path.relative_to(repo_path)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal outside workspace root is blocked.")

    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{path}' does not exist.")

    try:
        content = target_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}")

    lines = content.splitlines()
    total_lines = len(lines)

    slice_content = content
    if start_line is not None and end_line is not None:
        s_idx = max(0, start_line - 1)
        e_idx = min(total_lines, end_line)
        slice_content = "\n".join(lines[s_idx:e_idx])

    return {
        "file": path,
        "content": slice_content,
        "total_lines": total_lines,
        "start_line": start_line or 1,
        "end_line": end_line or total_lines,
    }


@router.get("/workspaces/{workspace_id}/download")
def download_workspace_zip(workspace_id: str) -> Response:
    """Download the current state of the workspace as a ZIP archive."""
    mgr = WorkspaceManager()
    ws = mgr.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")

    patcher = WorkspacePatcher(ws["working_path"])
    zip_bytes = patcher.export_as_zip()

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{ws["name"]}_workspace.zip"'},
    )


@router.get("/workspaces/{workspace_id}/patch")
def download_workspace_patch(workspace_id: str) -> Response:
    """Download all applied changes as a unified .patch file."""
    mgr = WorkspaceManager()
    ws = mgr.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")

    patcher = WorkspacePatcher(ws["working_path"])
    patch_text = patcher.generate_patch(workspace_id)

    return Response(
        content=patch_text,
        media_type="text/x-diff",
        headers={"Content-Disposition": f'attachment; filename="{ws["name"]}_changes.patch"'},
    )


@router.post("/workspaces/{workspace_id}/edits/apply")
def apply_edit(
    workspace_id: str,
    body: ApplyEditRequest,
) -> dict[str, Any]:
    """Apply an AI-proposed edit to the workspace copy."""
    mgr = WorkspaceManager()
    ws = mgr.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")

    patcher = WorkspacePatcher(ws["working_path"])
    proposals = patcher.parse_proposals(
        f"### CHANGE: {body.target_symbol} ({body.file_path})\n```python\n{body.new_content}\n```"
        if body.change_type == "modify"
        else f"### NEW FILE: {body.file_path}\n```python\n{body.new_content}\n```"
    )

    if not proposals:
        raise HTTPException(status_code=400, detail="Could not parse proposal into valid change structure.")

    proposal = proposals[0]
    try:
        result = patcher.apply_proposal(
            workspace_id=workspace_id,
            proposal=proposal,
            allow_syntax_error=body.allow_syntax_error,
            message_id=body.message_id,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/workspaces/{workspace_id}/edits/undo")
def undo_last_edit(workspace_id: str) -> dict[str, Any]:
    """Revert the most recent change in the workspace to its exact prior content."""
    mgr = WorkspaceManager()
    ws = mgr.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")

    patcher = WorkspacePatcher(ws["working_path"])
    try:
        return patcher.undo_last_edit(workspace_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/workspaces/{workspace_id}/edits/history")
def get_edit_history(workspace_id: str) -> list[dict[str, Any]]:
    """Retrieve audit log of all applied changes in this workspace."""
    storage = ChatStorage()
    return storage.list_edits(workspace_id)
