"""Contract tests validating that API response JSON strictly contains all paths expected by the UI.

Uses the shared contract specification in web/contract_paths.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.embeddings.embedding_generator import FakeEmbeddingGenerator
from tests.test_ai import FakeProvider


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "contract_mini_repo"
    repo.mkdir()

    auth_dir = repo / "auth"
    auth_dir.mkdir()
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "session.py").write_text(
        "def optimize_session(token: str) -> bool:\n"
        "    '''Optimize and cache authenticated user session.'''\n"
        "    return len(token) > 0\n",
        encoding="utf-8",
    )
    return repo


def _assert_has_path(data: dict, path_str: str) -> None:
    parts = path_str.split(".")
    curr = data
    for p in parts:
        assert isinstance(curr, dict), f"Expected dict at '{p}' while traversing '{path_str}', got {type(curr)}"
        assert p in curr, f"Path '{path_str}' missing field '{p}'. Available keys: {list(curr.keys())}"
        curr = curr[p]


def test_api_contract_retrieve(client: TestClient, mini_repo: Path) -> None:
    """Validate /api/retrieve returns every required field from contract_paths.json."""
    contract_file = Path("web/contract_paths.json")
    assert contract_file.exists(), "web/contract_paths.json must exist as single source of truth"
    spec = json.loads(contract_file.read_text(encoding="utf-8"))

    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
        patch("app.services.pipeline.get_configured_embedding_model", return_value="test-fake"),
    ):
        # Index first
        idx_res = client.post("/api/index", json={"repo_path": str(mini_repo), "force": True})
        assert idx_res.status_code == 200

        # Retrieve
        resp = client.post(
            "/api/retrieve",
            json={
                "repo_path": str(mini_repo),
                "query": "optimize session",
                "max_tokens": 4000,
                "use_semantic": True,
                "use_keyword": True,
                "use_graph": True,
                "explain": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Check all required top-level and nested paths
        for p in spec["retrieve"]:
            _assert_has_path(data, p)

        # Check candidate-level paths
        candidates = data["retrieval"]["candidates"]
        assert len(candidates) > 0
        for cand in candidates:
            for p in spec["candidate"]:
                assert p in cand, f"Candidate missing required field '{p}'. Cand: {cand}"


def test_api_contract_ask_dry_run(client: TestClient, mini_repo: Path) -> None:
    """Validate /api/ask with dry_run=True returns every required field from contract_paths.json."""
    contract_file = Path("web/contract_paths.json")
    spec = json.loads(contract_file.read_text(encoding="utf-8"))

    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
        patch("app.services.pipeline.get_configured_embedding_model", return_value="test-fake"),
    ):
        idx_res = client.post("/api/index", json={"repo_path": str(mini_repo), "force": True})
        assert idx_res.status_code == 200

        resp = client.post(
            "/api/ask",
            json={
                "repo_path": str(mini_repo),
                "query": "optimize session",
                "max_tokens": 4000,
                "dry_run": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        for p in spec["ask_dry_run"]:
            if "." in p:
                _assert_has_path(data, p)
            else:
                assert p in data, f"Missing required field '{p}' in dry_run ask response"


def test_api_contract_error_response(client: TestClient, mini_repo: Path) -> None:
    """Validate structured error responses return every required field from contract_paths.json."""
    contract_file = Path("web/contract_paths.json")
    spec = json.loads(contract_file.read_text(encoding="utf-8"))

    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    fake_provider = FakeProvider(configured=False)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
        patch("app.services.pipeline.get_configured_embedding_model", return_value="test-fake"),
        patch("app.services.pipeline.get_provider", return_value=fake_provider),
    ):
        idx_res = client.post("/api/index", json={"repo_path": str(mini_repo), "force": True})
        assert idx_res.status_code == 200

        resp = client.post(
            "/api/ask",
            json={
                "repo_path": str(mini_repo),
                "query": "optimize session",
                "max_tokens": 4000,
                "dry_run": False,
            },
        )
        assert resp.status_code in (401, 503)
        data = resp.json()

        for p in spec["error"]:
            if "." in p:
                _assert_has_path(data, p)
            else:
                assert p in data, f"Missing required error field '{p}'"
