"""MongoDB connection, repository namespacing, and chunked blob storage.

All persistent state (code index, graph, embeddings, chats, workspaces, edit
history) lives in one MongoDB database selected by MONGODB_URI / MONGODB_DB.
When MONGODB_URI is unset, an in-process mongomock database is used so the
app and tests run without a server; that data is lost when the process exits.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

from bson.binary import Binary

from app.config import get_settings

logger = logging.getLogger(__name__)

BLOB_CHUNK_BYTES = 4 * 1024 * 1024  # stays well under MongoDB's 16 MB document limit

_LOCK = threading.RLock()
_client: Any | None = None
_db: Any | None = None
_in_memory = False


class StorageUnavailableError(RuntimeError):
    """Raised when the configured MongoDB server cannot be reached."""


def _create_indexes(db: Any) -> None:
    db.workspaces.create_index("created_at")
    db.conversations.create_index([("workspace_id", 1), ("updated_at", -1)])
    db.messages.create_index([("conversation_id", 1), ("created_at", 1)])
    db.attachments.create_index([("conversation_id", 1), ("created_at", 1)])
    db.edit_history.create_index([("workspace_id", 1), ("applied_at", -1)])
    db.index_files.create_index([("repo", 1), ("path", 1)], unique=True)
    db.index_symbols.create_index([("repo", 1), ("qualified_name", 1)])
    db.index_symbols.create_index([("repo", 1), ("name", 1)])
    db.index_symbols.create_index([("repo", 1), ("file_path", 1)])
    db.index_symbols.create_index([("repo", 1), ("call_names", 1)])
    db.blobs.create_index([("ns", 1), ("key", 1), ("seq", 1)])


def get_database() -> Any:
    """Return the shared database handle, connecting on first use."""
    global _client, _db, _in_memory
    with _LOCK:
        if _db is not None:
            return _db

        settings = get_settings()
        if settings.mongodb_uri:
            from pymongo import MongoClient
            from pymongo.errors import PyMongoError

            client = MongoClient(settings.mongodb_uri.get_secret_value(), serverSelectionTimeoutMS=5000, tz_aware=True)
            try:
                client.admin.command("ping")
            except PyMongoError as exc:
                client.close()
                raise StorageUnavailableError(
                    f"Cannot reach MongoDB at the configured MONGODB_URI: {exc.__class__.__name__}. "
                    "Check that the server is running and the URI is correct."
                ) from exc
            _in_memory = False
        else:
            import mongomock

            logger.warning(
                "MONGODB_URI is not set; using an in-memory MongoDB substitute. "
                "Data will be lost when the process exits. Set MONGODB_URI to persist."
            )
            client = mongomock.MongoClient()
            _in_memory = True

        _client = client
        _db = client[settings.mongodb_db]
        _create_indexes(_db)
        return _db


def reset_database() -> None:
    """Drop all data and forget the connection (used to isolate tests)."""
    global _client, _db
    with _LOCK:
        if _db is not None:
            for name in _db.list_collection_names():
                _db[name].drop()
        if _client is not None and not _in_memory:
            _client.close()
        _client = None
        _db = None


def repo_key(repo_path: Path | str, namespace: str | None = None) -> str:
    """Stable key identifying a repository's data, e.g. 'myrepo_1a2b3c4d'.

    An explicit namespace overrides the path-derived key (used by tests and by
    the incremental-equivalence check to build a scratch copy of the index).
    """
    if namespace:
        return f"ns:{namespace}"
    repo = Path(repo_path).resolve()
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (repo.name or "root"))
    path_hash = hashlib.sha256(str(repo).encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}_{path_hash}"


# -----------------------------------------------------------------------------
# Chunked blob storage (graph JSON, FAISS indexes, vector matrices)
# -----------------------------------------------------------------------------

def put_blob(ns: str, key: str, data: bytes) -> None:
    """Store bytes under (ns, key), replacing any previous value."""
    blobs = get_database().blobs
    blobs.delete_many({"ns": ns, "key": key})
    chunks = [data[i:i + BLOB_CHUNK_BYTES] for i in range(0, len(data), BLOB_CHUNK_BYTES)] or [b""]
    blobs.insert_many(
        [
            {"ns": ns, "key": key, "seq": seq, "chunks": len(chunks), "data": Binary(chunk)}
            for seq, chunk in enumerate(chunks)
        ]
    )


def get_blob(ns: str, key: str) -> bytes | None:
    """Return the bytes stored under (ns, key), or None when absent or incomplete."""
    docs = list(get_database().blobs.find({"ns": ns, "key": key}).sort("seq", 1))
    if not docs or len(docs) != docs[0]["chunks"]:
        return None
    return b"".join(bytes(d["data"]) for d in docs)


def has_blob(ns: str, key: str) -> bool:
    return get_database().blobs.count_documents({"ns": ns, "key": key}, limit=1) > 0


def delete_blobs(ns: str, key_prefix: str | None = None) -> None:
    """Delete every blob in a namespace, optionally only keys starting with a prefix."""
    query: dict[str, Any] = {"ns": ns}
    if key_prefix:
        import re

        query["key"] = {"$regex": f"^{re.escape(key_prefix)}"}
    get_database().blobs.delete_many(query)
