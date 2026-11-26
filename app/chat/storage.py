"""MongoDB persistence layer for workspaces, conversations, messages, attachments, and edit history."""

from __future__ import annotations

import hashlib
import time
import uuid
import zlib
from pathlib import Path
from typing import Any

from bson.binary import Binary

from app.storage.mongo import get_database


def _new_id() -> str:
    return str(uuid.uuid4())[:12]


def _public(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a stored document to the API shape (id instead of _id)."""
    if doc is None:
        return None
    out = dict(doc)
    out["id"] = out.pop("_id")
    return out


class ChatStorage:
    """CRUD access for chat and workspace entities backed by MongoDB."""

    def __init__(self) -> None:
        self.db = get_database()

    # ==========================================
    # Workspaces
    # ==========================================

    def create_workspace(
        self,
        name: str,
        source: str,
        working_path: str | Path,
        original_path: str | Path | None = None,
        edit_in_place: bool = False,
        file_counts: dict[str, int] | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        ws_id = workspace_id or _new_id()
        self.db.workspaces.insert_one(
            {
                "_id": ws_id,
                "name": name,
                "source": source,
                "working_path": str(working_path),
                "original_path": str(original_path) if original_path else None,
                "edit_in_place": bool(edit_in_place),
                "created_at": time.time(),
                "last_indexed_at": None,
                "file_counts": file_counts or {"python": 0, "text_only": 0, "skipped": 0},
            }
        )
        return self.get_workspace(ws_id)  # type: ignore[return-value]

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        return _public(self.db.workspaces.find_one({"_id": workspace_id}))

    def list_workspaces(self) -> list[dict[str, Any]]:
        return [_public(d) for d in self.db.workspaces.find().sort("created_at", -1)]  # type: ignore[misc]

    def update_workspace_indexed(
        self,
        workspace_id: str,
        file_counts: dict[str, int] | None = None,
    ) -> None:
        fields: dict[str, Any] = {"last_indexed_at": time.time()}
        if file_counts is not None:
            fields["file_counts"] = file_counts
        self.db.workspaces.update_one({"_id": workspace_id}, {"$set": fields})

    def delete_workspace(self, workspace_id: str) -> None:
        conv_ids = [d["_id"] for d in self.db.conversations.find({"workspace_id": workspace_id}, {"_id": 1})]
        for cid in conv_ids:
            self._delete_conversation_children(cid)
        self.db.conversations.delete_many({"workspace_id": workspace_id})
        self.db.edit_history.delete_many({"workspace_id": workspace_id})
        self.db.workspaces.delete_one({"_id": workspace_id})

    # ==========================================
    # Conversations
    # ==========================================

    def create_conversation(
        self,
        title: str,
        workspace_id: str | None = None,
        mode: str = "balanced",
    ) -> dict[str, Any]:
        cid = _new_id()
        now = time.time()
        self.db.conversations.insert_one(
            {
                "_id": cid,
                "workspace_id": workspace_id,
                "title": title,
                "mode": mode,
                "created_at": now,
                "updated_at": now,
                "summary": "",
                "active_entities": [],
            }
        )
        return self.get_conversation(cid)  # type: ignore[return-value]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        return _public(self.db.conversations.find_one({"_id": conversation_id}))

    def list_conversations(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        query = {"workspace_id": workspace_id} if workspace_id else {}
        return [_public(d) for d in self.db.conversations.find(query).sort("updated_at", -1)]  # type: ignore[misc]

    def update_conversation(
        self,
        conversation_id: str,
        title: str | None = None,
        summary: str | None = None,
        mode: str | None = None,
        active_entities: list[str] | None = None,
    ) -> None:
        fields: dict[str, Any] = {"updated_at": time.time()}
        if title is not None:
            fields["title"] = title
        if summary is not None:
            fields["summary"] = summary
        if mode is not None:
            fields["mode"] = mode
        if active_entities is not None:
            fields["active_entities"] = active_entities
        self.db.conversations.update_one({"_id": conversation_id}, {"$set": fields})

    def _delete_conversation_children(self, conversation_id: str) -> None:
        self.db.messages.delete_many({"conversation_id": conversation_id})
        self.db.attachments.delete_many({"conversation_id": conversation_id})
        self.db.conversation_feedback.delete_many({"conversation_id": conversation_id})

    def delete_conversation(self, conversation_id: str) -> None:
        self._delete_conversation_children(conversation_id)
        self.db.conversations.delete_one({"_id": conversation_id})

    # ==========================================
    # Messages
    # ==========================================

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        provider: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
        context_manifest: dict[str, Any] | None = None,
        attachments: list[str] | None = None,
        raw_tokens: dict[str, Any] | None = None,
        needs_context_items: list[str] | None = None,
    ) -> dict[str, Any]:
        msg_id = _new_id()
        now = time.time()
        self.db.messages.insert_one(
            {
                "_id": msg_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "created_at": now,
                "provider": provider,
                "model": model,
                "usage": usage or None,
                "context_manifest": context_manifest or None,
                "attachments": attachments or [],
                "raw_tokens": raw_tokens or None,
                "needs_context_items": needs_context_items or [],
            }
        )
        self.db.conversations.update_one({"_id": conversation_id}, {"$set": {"updated_at": now}})
        return self.get_message(msg_id)  # type: ignore[return-value]

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        return _public(self.db.messages.find_one({"_id": message_id}))

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        cursor = self.db.messages.find({"conversation_id": conversation_id}).sort("created_at", 1)
        return [_public(d) for d in cursor]  # type: ignore[misc]

    # ==========================================
    # Attachments
    # ==========================================

    def add_attachment(
        self,
        conversation_id: str,
        filename: str,
        file_type: str,
        size_bytes: int,
        content_text: str,
        secret_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        att_id = _new_id()
        self.db.attachments.insert_one(
            {
                "_id": att_id,
                "conversation_id": conversation_id,
                "filename": filename,
                "file_type": file_type,
                "size_bytes": size_bytes,
                "content_text": content_text,
                "created_at": time.time(),
                "secret_warnings": secret_warnings or [],
            }
        )
        return self.get_attachment(att_id)  # type: ignore[return-value]

    def get_attachment(self, attachment_id: str) -> dict[str, Any] | None:
        return _public(self.db.attachments.find_one({"_id": attachment_id}))

    def get_attachments(self, conversation_id: str) -> list[dict[str, Any]]:
        cursor = self.db.attachments.find({"conversation_id": conversation_id}).sort("created_at", 1)
        return [_public(d) for d in cursor]  # type: ignore[misc]

    def delete_attachment(self, attachment_id: str) -> None:
        self.db.attachments.delete_one({"_id": attachment_id})

    # ==========================================
    # Edit History & Undo
    # ==========================================

    def record_edit(
        self,
        workspace_id: str,
        file_path: str,
        before_content: str,
        after_content_hash: str,
        target_symbol: str | None = None,
        diff_text: str | None = None,
        message_id: str | None = None,
    ) -> str:
        edit_id = _new_id()
        raw = before_content.encode("utf-8")
        self.db.edit_history.insert_one(
            {
                "_id": edit_id,
                "workspace_id": workspace_id,
                "file_path": file_path,
                "before_content_hash": hashlib.sha256(raw).hexdigest(),
                "after_content_hash": after_content_hash,
                "before_content_compressed": Binary(zlib.compress(raw)),
                "applied_at": time.time(),
                "message_id": message_id,
                "target_symbol": target_symbol,
                "diff_text": diff_text,
            }
        )
        return edit_id

    @staticmethod
    def _edit_with_content(doc: dict[str, Any] | None) -> dict[str, Any] | None:
        out = _public(doc)
        if out is None:
            return None
        out["before_content"] = zlib.decompress(bytes(out["before_content_compressed"])).decode("utf-8")
        return out

    def get_last_edit(self, workspace_id: str) -> dict[str, Any] | None:
        doc = self.db.edit_history.find_one({"workspace_id": workspace_id}, sort=[("applied_at", -1)])
        return self._edit_with_content(doc)

    def list_edits(self, workspace_id: str) -> list[dict[str, Any]]:
        cursor = self.db.edit_history.find(
            {"workspace_id": workspace_id}, {"before_content_compressed": 0}
        ).sort("applied_at", -1)
        return [_public(d) for d in cursor]  # type: ignore[misc]

    def remove_edit(self, edit_id: str) -> None:
        self.db.edit_history.delete_one({"_id": edit_id})

    # ==========================================
    # Feedback
    # ==========================================

    def set_feedback(self, conversation_id: str, message_id: str, rating: str) -> None:
        self.db.conversation_feedback.update_one(
            {"conversation_id": conversation_id, "message_id": message_id},
            {
                "$set": {"rating": rating, "created_at": time.time()},
                "$setOnInsert": {"_id": _new_id()},
            },
            upsert=True,
        )
