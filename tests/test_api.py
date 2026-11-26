"""Integration and unit tests for FastAPI endpoints, pipeline service, and background jobs."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.ai.base_provider import ProviderAuthError, Usage
from app.api.server import app
from app.config import Settings
from app.embeddings.embedding_generator import FakeEmbeddingGenerator
from tests.test_ai import FakeProvider


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Create a realistic mini repository on disk with multiple connected modules."""
    repo_root = tmp_path / "mini_service_repo"
    repo_root.mkdir()

    # auth/login.py
    auth_dir = repo_root / "auth"
    auth_dir.mkdir()
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "login.py").write_text(
        "from utils.hashing import verify_password\n"
        "from database.user import get_user\n\n"
        "def login_user(username: str, password: str) -> bool:\n"
        "    '''Authenticate user against database record.'''\n"
        "    user = get_user(username)\n"
        "    if not user:\n"
        "        return False\n"
        "    return verify_password(password, user.get('hash', ''))\n",
        encoding="utf-8",
    )

    # utils/hashing.py
    utils_dir = repo_root / "utils"
    utils_dir.mkdir()
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "hashing.py").write_text(
        "def verify_password(raw: str, hashed: str) -> bool:\n"
        "    '''Verify raw string matches hashed string.'''\n"
        "    return raw == hashed\n\n"
        "def hash_password(raw: str) -> str:\n"
        "    '''Hash raw string.'''\n"
        "    return 'hashed_' + raw\n",
        encoding="utf-8",
    )

    # database/user.py
    db_dir = repo_root / "database"
    db_dir.mkdir()
    (db_dir / "__init__.py").write_text("", encoding="utf-8")
    (db_dir / "user.py").write_text(
        "def get_user(username: str) -> dict | None:\n"
        "    '''Fetch user dictionary from database.'''\n"
        "    if username == 'alice':\n"
        "        return {'username': 'alice', 'hash': 'secret123'}\n"
        "    return None\n",
        encoding="utf-8",
    )

    return repo_root


def test_health_endpoint(client: TestClient):
    """Verify /health returns service readiness and configured metadata."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.8.0"
    assert "provider_configured" in data
    assert data["provider_name"] in ("claude", "ollama", "gemini", "openai_compat")
    assert "model_name" in data
    assert "embedding_model" in data


def test_full_pipeline_flow(client: TestClient, mini_repo: Path):
    """Test full flow: POST /index -> /search -> /retrieve -> /ask -> /graph -> /stats."""
    repo_str = str(mini_repo)

    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    fake_prov = FakeProvider(
        responses=[
            "### ANALYSIS\nChecking authentication flow.\n\n### CHANGES\n### CHANGE: auth.login.login_user (auth/login.py)\n```python\ndef login_user(username: str, password: str) -> bool:\n    return True\n```\n\n### RISKS\n- Hardcoded return value"
        ],
        usage=Usage(
            input_tokens=150,
            output_tokens=75,
            total_tokens=225,
            cache_read_tokens=None,
            cache_creation_tokens=None,
            estimated_cost_usd=None,
        ),
    )

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
        patch("app.services.pipeline.get_provider", return_value=fake_prov),
    ):
        # 1. POST /index
        res_index = client.post("/index", json={"repo_path": repo_str, "force": True})
        assert res_index.status_code == 200
        idx_data = res_index.json()
        assert idx_data["scan_and_store"]["files_scanned"] == 6
        assert idx_data["graph_build"]["total_nodes"] > 0
        assert idx_data["embeddings"]["items_total"] > 0

        # 2. POST /search
        res_search = client.post("/search", json={"repo_path": repo_str, "query": "password hashing"})
        assert res_search.status_code == 200
        search_data = res_search.json()
        assert search_data["total_hits"] > 0
        assert any("password" in h["ref"].lower() for h in search_data["hits"])

        # 3. POST /retrieve
        res_retrieve = client.post("/retrieve", json={"repo_path": repo_str, "query": "login_user", "max_tokens": 1500})
        assert res_retrieve.status_code == 200
        ret_data = res_retrieve.json()
        assert "retrieval" in ret_data
        assert "context" in ret_data
        assert ret_data["context"]["tokens"] <= 1500
        assert ret_data["context"]["target"] is not None

        # 4. POST /ask (dry_run=True)
        res_ask_dry = client.post("/ask", json={"repo_path": repo_str, "query": "Optimize login_user", "dry_run": True})
        assert res_ask_dry.status_code == 200
        dry_data = res_ask_dry.json()
        assert dry_data["dry_run"] is True
        assert dry_data["completion"] is None
        assert "TASK:\nOptimize login_user" in dry_data["messages"][0]["content"]
        assert dry_data["token_report"]["provider_input_tokens"] is None

        # 5. POST /ask (dry_run=False)
        res_ask = client.post(
            "/ask",
            json={"repo_path": repo_str, "query": "Optimize login_user", "dry_run": False},
            headers={"X-Provider-Key": "test-key"},
        )
        assert res_ask.status_code == 200
        ask_data = res_ask.json()
        assert ask_data["dry_run"] is False
        assert ask_data["completion"] is not None
        assert ask_data["answer"] is not None
        assert len(ask_data["answer"]["changes"]) == 1
        assert ask_data["answer"]["changes"][0]["qualified_name"] == "auth.login.login_user"
        assert ask_data["token_report"]["provider_input_tokens"] == 150
        assert ask_data["token_report"]["provider_output_tokens"] == 75

        # 6. GET /graph (overview)
        res_graph = client.get(f"/graph?repo_path={repo_str}&max_nodes=100")
        assert res_graph.status_code == 200
        graph_data = res_graph.json()
        assert len(graph_data["nodes"]) > 0
        assert len(graph_data["edges"]) > 0

        # 7. GET /graph/node
        res_node = client.get(f"/graph/node?repo_path={repo_str}&id=login_user")
        assert res_node.status_code == 200
        node_data = res_node.json()
        assert "login_user" in node_data["id"]
        assert "callees" in node_data
        assert len(node_data["callees"]) > 0

        # 8. GET /stats
        res_stats = client.get(f"/stats?repo_path={repo_str}")
        assert res_stats.status_code == 200
        stats_data = res_stats.json()
        assert stats_data["database"]["files"] == 6
        assert stats_data["database"]["functions"] >= 4
        assert stats_data["is_stale"] is False


def test_repo_path_validation(client: TestClient, tmp_path: Path):
    """Verify validation rules: nonexistent path (404), file instead of dir (400), outside allowed roots (400)."""
    # Nonexistent path
    res = client.post("/index", json={"repo_path": str(tmp_path / "does_not_exist")})
    assert res.status_code == 404
    err = res.json()["error"]
    assert err["code"] == "http_error"
    assert "does not exist" in err["message"]

    # File path instead of directory
    file_p = tmp_path / "test_file.py"
    file_p.write_text("print(1)", encoding="utf-8")
    res_file = client.post("/index", json={"repo_path": str(file_p)})
    assert res_file.status_code == 400
    assert "not a directory" in res_file.json()["error"]["message"]

    # Outside allowed roots
    allowed_dir = tmp_path / "allowed_zone"
    allowed_dir.mkdir()
    outside_dir = tmp_path / "forbidden_zone"
    outside_dir.mkdir()

    with patch("app.api.state.get_settings") as mock_settings:
        mock_settings.return_value = Settings(allowed_repo_roots=str(allowed_dir))
        res_forbidden = client.post("/index", json={"repo_path": str(outside_dir)})
        assert res_forbidden.status_code == 400
        assert "outside allowed roots" in res_forbidden.json()["error"]["message"]


def test_missing_prerequisites_error_code(client: TestClient, tmp_path: Path):
    """Verify unindexed repository returns 424 PrerequisiteMissingError naming the fix."""
    empty_dir = tmp_path / "empty_unindexed_repo"
    empty_dir.mkdir()
    res = client.post("/retrieve", json={"repo_path": str(empty_dir), "query": "test query"})
    assert res.status_code == 424
    err = res.json()["error"]
    assert err["code"] == "prerequisite_missing"
    assert "POST /index" in err["message"]


def test_missing_api_key_returns_401(client: TestClient, mini_repo: Path):
    """Verify calling /ask with unconfigured provider returns HTTP 401 api_key_required."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    unconfigured_provider = FakeProvider(configured=False)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.services.pipeline.get_provider", return_value=unconfigured_provider),
    ):
        # Index repo first
        client.post("/index", json={"repo_path": repo_str})

        # Ask should fail with 401 api_key_required
        res = client.post("/ask", json={"repo_path": repo_str, "query": "Fix bug in login", "dry_run": False})
        assert res.status_code in (401, 503)
        err = res.json()["error"]
        assert err["code"] in ("api_key_required", "provider_not_configured")
        assert "API key" in err["message"] or "key" in err["message"].lower()


def test_concurrent_index_returns_409(client: TestClient, mini_repo: Path):
    """Verify two concurrent indexing calls on the same repository return 409 conflict."""
    repo_str = str(mini_repo)
    from app.api.state import get_service_for_repo

    svc = get_service_for_repo(repo_str)
    # Simulate an active index lock
    assert svc._index_lock.acquire(blocking=False)
    svc._is_indexing = True

    try:
        res = client.post("/index", json={"repo_path": repo_str})
        assert res.status_code == 409
        err = res.json()["error"]
        assert err["code"] == "indexing_in_progress"
        assert "already in progress" in err["message"]
    finally:
        svc._is_indexing = False
        svc._index_lock.release()


def test_background_indexing_job_lifecycle(client: TestClient, mini_repo: Path):
    """Verify background indexing starts as queued, completes, and status endpoint works."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        res = client.post(f"/index?background=true", json={"repo_path": repo_str, "force": True})
        assert res.status_code == 200
        job_data = res.json()
        assert "job_id" in job_data
        job_id = job_data["job_id"]

        # Check job status endpoint
        res_status = client.get(f"/index/status/{job_id}")
        assert res_status.status_code == 200
        status_data = res_status.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ("queued", "running", "completed")

        # Invalid job ID
        res_404 = client.get("/index/status/nonexistent-job-id")
        assert res_404.status_code == 404


def test_graph_truncation_max_nodes(client: TestClient, mini_repo: Path):
    """Verify /graph respects max_nodes and sets truncated=True when exceeded."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        client.post("/index", json={"repo_path": repo_str})

        res = client.get(f"/graph?repo_path={repo_str}&max_nodes=2")
        assert res.status_code == 200
        data = res.json()
        assert len(data["nodes"]) <= 2
        assert data["truncated"] is True
        assert data["total_nodes"] > 2


@pytest.mark.integration
def test_real_provider_integration(mini_repo: Path):
    """Integration test running real Claude API completion if ANTHROPIC_API_KEY is configured."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set in environment.")

    from app.services.pipeline import RepositoryService
    from app.ai.claude_provider import ClaudeProvider
    from app.ai.base_provider import ProviderConfig

    svc = RepositoryService(mini_repo)
    svc.index(force=True)

    config = ProviderConfig(
        provider="claude",
        model="claude-3-5-sonnet-20241022",
        max_output_tokens=100,
        temperature=0.0,
    )
    provider = ClaudeProvider(config=config, api_key=api_key)

    ask_res = svc.ask(
        query="Explain how login_user works in auth/login.py",
        budget=1000,
        provider=provider,
        max_output_tokens_override=100,
    )

    assert ask_res.completion is not None
    assert len(ask_res.completion.text) > 0
    assert ask_res.token_report.provider_input_tokens is not None
    assert ask_res.token_report.provider_input_tokens > 0
    assert ask_res.token_report.provider_output_tokens is not None
    assert ask_res.token_report.provider_output_tokens > 0


def test_retrieve_overlong_query_returns_400(client: TestClient, mini_repo: Path):
    """POST /retrieve with query > MAX_QUERY_CHARS returns structured 400 QUERY_TOO_LONG with request id."""
    repo_str = str(mini_repo)
    long_query = "analyze " + ("x" * 8100)

    res = client.post("/api/retrieve", json={"repo_path": repo_str, "query": long_query})
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    err = data["error"]
    assert err["code"] == "QUERY_TOO_LONG"
    assert "Query is 8,108 characters; the maximum is 4,000" in err["message"]
    assert err["request_id"] is not None
    assert res.headers.get("X-Request-ID") == err["request_id"]


def test_retrieve_long_query_under_max_chars_succeeds(client: TestClient, mini_repo: Path):
    """POST /retrieve with a long query under MAX_QUERY_CHARS succeeds without crash."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        client.post("/api/index", json={"repo_path": repo_str, "force": True})

        # Query of ~3,000 characters
        query = "Authenticate user and verify hashing " + ("details " * 350)
        assert len(query) < 4000

        res = client.post("/api/retrieve", json={"repo_path": repo_str, "query": query, "max_tokens": 4000})
        assert res.status_code == 200
        data = res.json()
        assert "retrieval" in data
        assert "context" in data
        assert data["context"]["tokens"] > 0


def test_retrieve_signals_disabled_individually(client: TestClient, mini_repo: Path):
    """POST /retrieve with each signal disabled individually succeeds."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        client.post("/api/index", json={"repo_path": repo_str, "force": True})

        # 1. semantic=False
        res_no_sem = client.post(
            "/api/retrieve",
            json={"repo_path": repo_str, "query": "login authentication", "use_semantic": False},
        )
        assert res_no_sem.status_code == 200

        # 2. keyword=False
        res_no_kw = client.post(
            "/api/retrieve",
            json={"repo_path": repo_str, "query": "login authentication", "use_keyword": False},
        )
        assert res_no_kw.status_code == 200

        # 3. graph=False
        res_no_graph = client.post(
            "/api/retrieve",
            json={"repo_path": repo_str, "query": "login authentication", "use_graph": False},
        )
        assert res_no_graph.status_code == 200


def test_retrieve_missing_embeddings_graceful_degrade(client: TestClient, mini_repo: Path):
    """POST /retrieve when embeddings are missing degrades gracefully with warning, not 500."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        client.post("/api/index", json={"repo_path": repo_str, "force": True})

        # Remove stored embeddings
        from app.embeddings.vector_store import VectorStore
        VectorStore.drop(mini_repo)

        # Clear pipeline cache
        from app.api.state import get_service_for_repo
        get_service_for_repo(repo_str).invalidate_caches()

        res = client.post(
            "/api/retrieve",
            json={"repo_path": repo_str, "query": "login user", "use_semantic": True},
        )
        assert res.status_code == 200
        data = res.json()
        assert "context" in data
        assert any("embeddings index not found" in w.lower() for w in data["context"].get("warnings", []))


def test_retrieve_embedding_model_mismatch_returns_424(client: TestClient, mini_repo: Path):
    """POST /retrieve when embedding model in manifest differs from config returns 424 naming fix."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        client.post("/api/index", json={"repo_path": repo_str, "force": True})

        # Modify manifest to different model name
        from app.storage.mongo import get_database, repo_key
        get_database().embedding_manifests.update_one(
            {"_id": repo_key(mini_repo)}, {"$set": {"model_name": "mismatched-model-name-v1"}}
        )

        from app.api.state import get_service_for_repo
        get_service_for_repo(repo_str).invalidate_caches()

        res = client.post(
            "/api/retrieve",
            json={"repo_path": repo_str, "query": "login user", "use_semantic": True},
        )
        assert res.status_code == 424
        err = res.json()["error"]
        assert err["code"] == "prerequisite_missing"
        assert "mismatched-model-name-v1" in err["message"]
        assert "POST /index" in err["message"]


def test_unhandled_exception_returns_request_id_and_keeps_traceback_out_of_body(
    client: TestClient, mini_repo: Path, caplog: pytest.LogCaptureFixture
):
    """Unhandled exception returns 500 with request_id, logs traceback, but never puts traceback in body."""
    repo_str = str(mini_repo)

    with patch("app.services.pipeline.RepositoryService.retrieve", side_effect=RuntimeError("unexpected bug")):
        res = client.post("/api/retrieve", json={"repo_path": repo_str, "query": "test query"})
        assert res.status_code == 500
        data = res.json()
        assert "error" in data
        err = data["error"]
        assert err["code"] == "internal_error"
        assert err["message"] == "An internal server error occurred."
        assert err["request_id"] is not None
        assert "Traceback" not in res.text
        assert "unexpected bug" not in res.text

        # Verify traceback appeared in logs with the request ID
        req_id = err["request_id"]
        assert any(req_id in record.message for record in caplog.records)
        assert any("RuntimeError: unexpected bug" in record.message for record in caplog.records)


def test_error_json_never_contains_api_key(client: TestClient, mini_repo: Path):
    """Errors must never leak sensitive API keys in the response body."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        client.post("/api/index", json={"repo_path": repo_str, "force": True})

        secret_key = "sk-ant-api03-SECRET-TEST-KEY-XYZ12345"
        with patch(
            "app.ai.claude_provider.ClaudeProvider.complete",
            side_effect=ProviderAuthError(f"Unauthorized with key {secret_key}"),
        ):
            res = client.post(
                "/api/ask",
                json={
                    "repo_path": repo_str,
                    "query": "login function",
                    "dry_run": False,
                    "provider": "claude",
                },
            )
            assert res.status_code in (401, 500, 503)
            # Response must NOT leak secret key
            assert secret_key not in res.text


def test_config_and_health_endpoints_without_keys(client: TestClient):
    """Verify /config and /health return configured model without leaking credentials."""
    # /config
    res_cfg = client.get("/api/config")
    assert res_cfg.status_code == 200
    cfg = res_cfg.json()
    assert "ai_model" in cfg
    assert "provider_configured" in cfg
    assert "max_query_chars" in cfg
    assert "api_key" not in cfg
    assert "secret" not in str(cfg).lower()

    # /health
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    health = res_health.json()
    assert "model_name" in health
    assert "provider_configured" in health
    assert "api_key" not in health
    assert "secret" not in str(health).lower()

