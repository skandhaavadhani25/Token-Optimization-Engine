"""Tests for static web UI serving, /api endpoint contract, and redirects."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.embeddings.embedding_generator import FakeEmbeddingGenerator


@pytest.fixture
def client() -> TestClient:
    """TestClient fixture with follow_redirects=False to verify redirect headers."""
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def redirect_client() -> TestClient:
    """TestClient fixture with follow_redirects=True to test transparent redirects."""
    return TestClient(app, follow_redirects=True)


@pytest.fixture
def web_test_repo(tmp_path: Path) -> Path:
    """Mini repo for API endpoint testing."""
    repo = tmp_path / "web_test_repo"
    repo.mkdir()

    # auth/login.py
    a_dir = repo / "auth"
    a_dir.mkdir()
    (a_dir / "__init__.py").write_text("", encoding="utf-8")
    (a_dir / "login.py").write_text(
        "def login_user(username: str, password: str) -> bool:\n"
        "    return password == 'secret'\n",
        encoding="utf-8",
    )
    return repo


def test_root_serves_index_html(client: TestClient):
    """GET / serves index.html with HTML content-type."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "<title>ContextForge | Graph-Based AI Coding Agent &amp; Token Optimization Engine</title>" in res.text
    assert 'id="pane-ask"' in res.text
    assert 'id="cy-container"' in res.text


def test_static_assets_served(client: TestClient):
    """Static assets in web/ (styles.css, app.js, api.js, graph.js, components) are served."""
    for asset in ["/styles.css", "/app.js", "/api.js", "/graph.js"]:
        res = client.get(asset)
        assert res.status_code == 200, f"Failed to serve {asset}"
        assert len(res.text) > 0

    comp_res = client.get("/components/results_table.js")
    assert comp_res.status_code == 200
    assert "renderResultsTable" in comp_res.text


def test_unprefixed_endpoints_redirect_to_api(client: TestClient):
    """Old endpoints (e.g. /health, /stats, /retrieve) return HTTP 307 redirecting to /api/*."""
    # GET /health -> /api/health
    res_health = client.get("/health")
    assert res_health.status_code == 307
    assert res_health.headers["location"] == "/api/health"

    # GET /stats?repo_path=test -> /api/stats?repo_path=test
    res_stats = client.get("/stats?repo_path=test_repo")
    assert res_stats.status_code == 307
    assert res_stats.headers["location"] == "/api/stats?repo_path=test_repo"

    # POST /retrieve -> /api/retrieve
    res_ret = client.post("/retrieve", json={"repo_path": "test", "query": "test"})
    assert res_ret.status_code == 307
    assert res_ret.headers["location"] == "/api/retrieve"


def test_new_endpoints_shape(client: TestClient, redirect_client: TestClient, web_test_repo: Path):
    """Verify responses and payload shapes for /api/index/update, /api/benchmark/querysets, and /api/benchmark-one."""
    repo_str = str(web_test_repo)
    fake_gen = FakeEmbeddingGenerator("fake-test", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_gen),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_gen),
    ):
        # 1. First index repo
        res_idx = redirect_client.post("/api/index", json={"repo_path": repo_str})
        assert res_idx.status_code == 200

        # 2. Test POST /api/index/update
        res_up = redirect_client.post("/api/index/update", json={"repo_path": repo_str, "verify": False})
        assert res_up.status_code == 200
        up_data = res_up.json()
        assert "added" in up_data
        assert "modified" in up_data
        assert "deleted" in up_data
        assert "files_reparsed" in up_data
        assert "graph_nodes_changed" in up_data
        assert "items_reembedded" in up_data
        assert "total_seconds" in up_data

        # 3. Test GET /api/benchmark/querysets
        res_qs = redirect_client.get("/api/benchmark/querysets")
        assert res_qs.status_code == 200
        qs_data = res_qs.json()
        assert isinstance(qs_data, list)
        if len(qs_data) > 0:
            assert "filename" in qs_data[0]
            assert "query_count" in qs_data[0]

        # 4. Test POST /api/benchmark-one
        res_one = redirect_client.post("/api/benchmark-one", json={"repo_path": repo_str, "query": "login_user", "budget": 2000})
        assert res_one.status_code == 200
        one_data = res_one.json()
        assert one_data["query"] == "login_user"
        assert "whole_file_tokens" in one_data
        assert "graph_tokens" in one_data
        assert "reduction_pct" in one_data
        assert isinstance(one_data["files_inspected"], list)

        # 5. Test POST /api/benchmark (with mini query array)
        mini_queries = [
            {"query": "login_user", "task": "Check login", "must_see": ["login_user"]}
        ]
        res_bm = redirect_client.post("/api/benchmark", json={
            "repo_path": repo_str,
            "queries": mini_queries,
            "budget": 2000,
            "call_ai": False,
            "strategies": ["whole_file", "graph"],
        })
        assert res_bm.status_code == 200
        bm_data = res_bm.json()
        assert "repo_file_count" in bm_data
        assert "summaries" in bm_data
        assert len(bm_data["summaries"]) == 2


def test_no_secrets_leak_in_web_or_api(client: TestClient, redirect_client: TestClient):
    """Confirm no API keys or secrets ever leak in served static files or API responses."""
    # Check index.html, JS files, and health response
    for path in ["/", "/styles.css", "/app.js", "/api.js", "/graph.js"]:
        text = client.get(path).text.lower()
        assert "sk-ant" not in text
        assert "api_key" not in text or "api_key" in text and "without api key" in text or "no api key" in text

    health_res = redirect_client.get("/api/health")
    assert health_res.status_code == 200
    health_text = health_res.text.lower()
    assert "sk-ant" not in health_text
    assert "secret" not in health_text
