"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from app.storage.mongo import reset_database


@pytest.fixture(autouse=True)
def _fresh_database(monkeypatch):
    """Give every test an empty in-memory MongoDB, ignoring any developer MONGODB_URI."""
    from app.config import get_settings

    monkeypatch.delenv("MONGODB_URI", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "mongodb_uri", None)
    reset_database()
    yield
    reset_database()
    get_settings.cache_clear()
