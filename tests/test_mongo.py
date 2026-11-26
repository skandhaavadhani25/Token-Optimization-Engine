"""Tests for MongoDB connection handling and chunked blob storage."""

from __future__ import annotations

import pytest

from app.storage import mongo
from app.storage.mongo import StorageUnavailableError, delete_blobs, get_blob, has_blob, put_blob, repo_key


def test_blob_round_trip_across_chunks(monkeypatch):
    monkeypatch.setattr(mongo, "BLOB_CHUNK_BYTES", 10)
    data = bytes(range(256)) * 3
    put_blob("ns", "k", data)
    assert mongo.get_database().blobs.count_documents({"ns": "ns", "key": "k"}) == 77
    assert get_blob("ns", "k") == data


def test_blob_overwrite_and_delete():
    put_blob("ns", "k", b"first value that is long")
    put_blob("ns", "k", b"short")
    assert get_blob("ns", "k") == b"short"
    put_blob("ns", "other", b"x")
    delete_blobs("ns")
    assert not has_blob("ns", "k") and not has_blob("ns", "other")
    assert get_blob("ns", "k") is None


def test_incomplete_blob_is_treated_as_missing():
    put_blob("ns", "k", b"abcdef")
    mongo.get_database().blobs.update_one({"ns": "ns", "key": "k"}, {"$set": {"chunks": 3}})
    assert get_blob("ns", "k") is None


def test_repo_key_is_stable_and_namespaceable(tmp_path):
    assert repo_key(tmp_path) == repo_key(tmp_path)
    assert repo_key(tmp_path, "scratch") == "ns:scratch"
    assert repo_key(tmp_path) != repo_key(tmp_path / "other")


def test_unreachable_server_raises_clear_error(monkeypatch):
    from pydantic import SecretStr

    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "mongodb_uri", SecretStr("mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=200"))
    mongo.reset_database()
    with pytest.raises(StorageUnavailableError) as exc:
        mongo.get_database()
    assert "MONGODB_URI" in str(exc.value)
    assert "127.0.0.1" not in str(exc.value)  # never echo the connection string
