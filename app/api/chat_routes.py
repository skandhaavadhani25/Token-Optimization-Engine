"""API routes for conversational chat with memory, attachments, and token savings metrics."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from fastapi import APIRouter, File, Header, HTTPException, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field, SecretStr

from app.ai.base_provider import (
    ApiKeyRequiredError,
    CompletionResult,
)
from app.ai.provider_factory import get_provider
from app.api.state import get_service_for_repo
from app.chat.attachments import (
    extract_attachment_text,
    is_image_file,
    model_supports_vision,
    register_conversation_attachment,
)
from app.chat.storage import ChatStorage
from app.chat.turn_builder import TurnBuilder
from app.config import get_settings
from app.security import message_rate_limiter, upload_rate_limiter
from app.workspace.patcher import WorkspacePatcher
from app.workspace.secret_scanner import scan_file_for_secrets

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


class CreateConversationRequest(BaseModel):
    workspace_id: str | None = Field(default=None, description="Associated workspace ID")
    title: str = Field(default="New Chat", description="Conversation title")
    mode: str = Field(default="balanced", description="'economy', 'balanced', or 'quality'")


class PostMessageRequest(BaseModel):
    content: str = Field(description="User prompt or task question")
    provider: str | None = Field(default=None, description="Provider override")
    model: str | None = Field(default=None, description="Model ID override")
    mode: str | None = Field(default=None, description="Mode preset override")
    budget: int | None = Field(default=None, description="Token budget override")
    dry_run: bool = Field(default=False, description="Simulate response without invoking external LLM")


class FeedbackRequest(BaseModel):
    message_id: str = Field(description="Message ID being rated")
    rating: str = Field(description="'up' or 'down'")


@router.post("/conversations")
def create_conversation(body: CreateConversationRequest) -> dict[str, Any]:
    """Create a new persistent conversation."""
    storage = ChatStorage()
    conv = storage.create_conversation(
        title=body.title,
        workspace_id=body.workspace_id,
        mode=body.mode,
    )
    return conv


@router.get("/conversations")
def list_conversations(
    workspace_id: str | None = Query(default=None, description="Filter by workspace ID"),
) -> list[dict[str, Any]]:
    """List recent conversations."""
    storage = ChatStorage()
    return storage.list_conversations(workspace_id=workspace_id)


@router.get("/conversations/{conversation_id}")
def get_conversation_details(conversation_id: str) -> dict[str, Any]:
    """Retrieve full conversation record, messages, and attachments."""
    storage = ChatStorage()
    conv = storage.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    messages = storage.get_messages(conversation_id)
    attachments = storage.get_attachments(conversation_id)
    return {
        "conversation": conv,
        "messages": messages,
        "attachments": attachments,
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, Any]:
    """Delete a conversation and its messages/attachments."""
    storage = ChatStorage()
    conv = storage.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    storage.delete_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}


@router.post("/conversations/{conversation_id}/messages")
def send_message(
    conversation_id: str,
    body: PostMessageRequest,
    request: Request,
    response: Response,
    x_provider_key: str | None = Header(default=None, alias="X-Provider-Key"),
) -> dict[str, Any]:
    """Process a chat turn with conversational memory, context budgeting, and proposed edits."""
    response.headers["Cache-Control"] = "no-store"
    settings = get_settings()

    # Rate limiting
    client_ip = getattr(getattr(request, "client", None), "host", "unknown")
    if not message_rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many chat messages. Please wait a minute.")

    storage = ChatStorage()
    conv = storage.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    # Determine provider and model
    target_provider = (body.provider or settings.ai_provider).lower()
    canonical_provider = (
        "openai_compat" if target_provider == "openai" else ("ollama" if target_provider == "local" else target_provider)
    )

    # Key precedence: Header key > server key
    header_key = x_provider_key.strip() if x_provider_key else None
    server_key = settings.get_key_for_provider(canonical_provider)
    raw_server_key = (
        server_key.get_secret_value() if isinstance(server_key, SecretStr) else (str(server_key) if server_key else None)
    )
    api_key = header_key or raw_server_key

    # 1. Save User Message
    user_msg = storage.add_message(
        conversation_id=conversation_id,
        role="user",
        content=body.content,
    )

    # 2. Check workspace and attachments
    has_ws = False
    repo_svc = None
    repo_path = None
    if conv.get("workspace_id"):
        ws = storage.get_workspace(conv["workspace_id"])
        if ws:
            counts = ws.get("file_counts") or {}
            if (counts.get("python", 0) + counts.get("text_only", 0)) > 0:
                has_ws = True
            repo_path = Path(ws["working_path"])
            try:
                repo_svc = get_service_for_repo(repo_path)
            except Exception:
                repo_svc = None

    attachments = storage.get_attachments(conversation_id)
    has_attachments = bool(attachments)

    if not has_ws and not has_attachments:
        completion_text = "Upload code first to ask questions about a repository or inspect files."
        asst_msg = storage.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=completion_text,
            provider="system",
            model="cce-core",
        )
        return {
            "user_message": user_msg,
            "assistant_message": asst_msg,
            "turn_tokens": {
                "actual_context_tokens": 0,
                "actual_total_sent": 0,
                "whole_files_baseline": 0,
                "full_repo_baseline": 0,
                "conversation_baseline": 0,
                "files_in_context": 0,
                "total_repo_files": 0,
                "per_turn_savings_pct": 0.0,
                "cumulative_savings_pct": 0.0,
                "retrieved_items": [],
            },
        }

    # A real completion needs a key; checked after the "upload code first" path so
    # key-less users still get that guidance instead of a 401.
    if canonical_provider != "ollama" and not api_key and not body.dry_run:
        raise ApiKeyRequiredError(
            f"API key required for provider '{canonical_provider}'. Please enter your API key in the Model settings."
        )

    # 3. Assemble Turn Context
    turn_builder = TurnBuilder(repo_service=repo_svc, storage=storage)
    mode = body.mode or conv.get("mode") or "balanced"
    turn_ctx = turn_builder.assemble_turn(
        conversation_id=conversation_id,
        user_message=body.content,
        mode_name=mode,
        budget_override=body.budget,
    )

    # 4. Invoke AI Completion
    completion_text = ""
    usage_info = {
        "input_tokens": turn_ctx.raw_tokens["actual_total_sent"],
        "output_tokens": 120,
        "total_tokens": turn_ctx.raw_tokens["actual_total_sent"] + 120,
        "is_estimated": True,
    }

    if body.dry_run:
        completion_text = (
            f"[Dry Run Simulation]\n"
            f"I analyzed your question regarding '{body.content}'.\n\n"
            f"Here is an example structure based on the provided context:\n"
            f"```python\n"
            f"# Analysis for {turn_ctx.active_entities[-1] if turn_ctx.active_entities else 'target'}\n"
            f"def example_implementation():\n"
            f"    return True\n"
            f"```\n"
        )
    else:
        try:
            target_model = body.model or settings.get_model_for_provider(canonical_provider, raise_if_missing=False)
            provider_inst = get_provider(
                provider_name=canonical_provider,
                model=target_model,
                api_key=api_key,
                raise_if_model_missing=False,
            )
            res: CompletionResult = provider_inst.complete(
                system=turn_ctx.system_prompt,
                messages=turn_ctx.messages,
            )
            completion_text = res.text
            if res.usage and res.usage.total_tokens:
                usage_info = {
                    "input_tokens": res.usage.input_tokens,
                    "output_tokens": res.usage.output_tokens,
                    "total_tokens": res.usage.total_tokens,
                    "is_estimated": res.usage.usage_estimated,
                }
        except Exception as exc:
            logger.error(f"Error during AI completion in conversation {conversation_id}: {exc}")
            completion_text = f"Error during model completion: {exc}"

    # 5. Parse Proposals & Needs Context
    parsed_proposals: list[dict[str, Any]] = []
    if repo_path and repo_path.exists():
        patcher = WorkspacePatcher(repo_path, storage=storage)
        props = patcher.parse_proposals(completion_text)
        for p in props:
            parsed_proposals.append({
                "change_type": p.change_type,
                "target_symbol": p.target_symbol,
                "file_path": p.file_path,
                "new_content": p.new_content,
                "syntax_valid": p.syntax_valid,
                "syntax_error": p.syntax_error,
                "diff": p.diff,
                "target_exists": p.target_exists,
                "is_applicable": p.is_applicable,
                "reason": p.reason,
            })

    # Needs Context items
    needs_context_matches = re.findall(r"NEEDS_CONTEXT:\s*([^\n\r]+)", completion_text)
    needs_context_items = [item.strip() for item in needs_context_matches if item.strip()]

    # 6. Save Assistant Message
    assistant_msg = storage.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=completion_text,
        provider=canonical_provider,
        model=body.model or settings.get_model_for_provider(canonical_provider, raise_if_missing=False),
        usage=usage_info,
        context_manifest=turn_ctx.context_manifest,
        raw_tokens=turn_ctx.raw_tokens,
        needs_context_items=needs_context_items,
    )

    return {
        "user_message": user_msg,
        "assistant_message": assistant_msg,
        "parsed_proposals": parsed_proposals,
        "needs_context": needs_context_items,
        "turn_tokens": turn_ctx.raw_tokens,
        "manifest": turn_ctx.context_manifest,
        "trace": turn_ctx.trace,
    }


@router.post("/conversations/{conversation_id}/attachments")
async def upload_attachment(
    conversation_id: str,
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Query(default=None),
    provider: str | None = Query(default=None),
) -> dict[str, Any]:
    """Upload a file attachment scoped to the conversation."""
    settings = get_settings()
    client_ip = getattr(getattr(request, "client", None), "host", "unknown")
    if not upload_rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many uploads. Please wait a minute.")

    storage = ChatStorage()
    conv = storage.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    # Check count limit
    existing = storage.get_attachments(conversation_id)
    if len(existing) >= settings.max_attachments_per_message:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum of {settings.max_attachments_per_message} attachments per conversation reached.",
        )

    content_bytes = await file.read()
    max_att_bytes = settings.max_attachment_mb * 1024 * 1024
    if len(content_bytes) > max_att_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Attachment '{file.filename}' exceeds {settings.max_attachment_mb} MB limit.",
        )

    # Vision check for images
    if is_image_file(file.filename or ""):
        if not model_supports_vision(provider or settings.ai_provider, model or settings.ai_model):
            raise HTTPException(
                status_code=400,
                detail=f"Selected model '{model or 'current model'}' does not support image attachments. Please switch to a vision model (e.g. Gemini 2.5 Flash, Claude 3.5 Sonnet, GPT-4o).",
            )

    # Extract text
    text_content, file_type = extract_attachment_text(file.filename or "file.txt", content_bytes)

    # Secret scan
    secret_warnings = scan_file_for_secrets(file.filename or "file.txt", text_content)

    # Persist the attachment
    record = storage.add_attachment(
        conversation_id=conversation_id,
        filename=file.filename or "attached_file.txt",
        file_type=file_type,
        size_bytes=len(content_bytes),
        content_text=text_content,
        secret_warnings=secret_warnings,
    )

    # Register in memory for retrieval
    register_conversation_attachment(
        conversation_id=conversation_id,
        attachment_id=record["id"],
        filename=record["filename"],
        text=text_content,
    )

    return {
        "attachment": record,
        "secret_warnings": secret_warnings,
    }


@router.get("/conversations/{conversation_id}/attachments")
def list_attachments(conversation_id: str) -> list[dict[str, Any]]:
    """List attachments belonging to a conversation."""
    storage = ChatStorage()
    return storage.get_attachments(conversation_id)


@router.delete("/conversations/{conversation_id}/attachments/{attachment_id}")
def delete_attachment(conversation_id: str, attachment_id: str) -> dict[str, Any]:
    """Remove an attachment from a conversation."""
    storage = ChatStorage()
    storage.delete_attachment(attachment_id)
    return {"status": "deleted", "attachment_id": attachment_id}


@router.post("/conversations/{conversation_id}/feedback")
def submit_feedback(conversation_id: str, body: FeedbackRequest) -> dict[str, Any]:
    """Store thumbs up/down user feedback on an answer."""
    storage = ChatStorage()
    storage.set_feedback(conversation_id, body.message_id, body.rating)
    return {"status": "recorded"}
