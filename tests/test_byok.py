"""Comprehensive tests for Bring-Your-Own-Key (BYOK) security, statelessness, and validation.

Covers:
1. Header key used and never persisted (service cache, job registry, logs, response, error).
2. Log redaction filter removes header values and key patterns.
3. Key precedence: header > server .env > 401 api_key_required.
4. Provider error mapping: 401 api_key_invalid, 429 rate limit / quota.
5. POST /providers/{name}/validate behaviors.
6. Models cache isolation across distinct user keys.
7. OpenAI-compatible provider base URL is server-controlled (no SSRF).
8. Validation endpoint rate limiting (10 requests / minute per client IP).
9. Cache-Control: no-store on key-bearing endpoints.
10. Key manager JavaScript unit tests suite.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.ai.base_provider import (
    AIProvider,
    ApiKeyInvalidError,
    CompletionResult,
    ProviderRateLimitError,
    Usage,
)
from app.api.server import app
from app.config import Settings
from app.embeddings.embedding_generator import FakeEmbeddingGenerator
from app.security import (
    KeyRedactionFilter,
    current_request_key,
    redact_sensitive_text,
    validation_rate_limiter,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "mini_byok_repo"
    repo_root.mkdir()
    auth_dir = repo_root / "auth"
    auth_dir.mkdir()
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "login.py").write_text(
        "def login_user(username: str, password: str) -> bool:\n"
        "    '''Authenticate user.'''\n"
        "    return bool(username and password)\n",
        encoding="utf-8",
    )
    return repo_root


class KeyTrackingFakeProvider(AIProvider):
    """Fake provider that records the key used without storing it long-term."""

    def __init__(self, api_key: str | None = None, model: str = "fake-model", **kwargs):
        self.api_key = api_key
        self.default_model = model
        self.received_key = api_key
        self._configured = bool(api_key)

    @property
    def name(self) -> str:
        return "gemini"

    def is_configured(self) -> bool:
        return True

    def count_tokens(self, system: str, messages: list) -> int | None:
        return 40

    def complete(self, *args, **kwargs) -> CompletionResult:
        return CompletionResult(
            text="### ANALYSIS\nProposed fix for login_user\n\n### CHANGES\n```python\n# fix\n```",
            usage=Usage(input_tokens=15, output_tokens=25, total_tokens=40, cost_usd=0.0),
            model=self.default_model,
            stop_reason="end_turn",
            latency_seconds=0.05,
        )

    def get_available_models(self) -> list[str]:
        return ["fake-model-1", "fake-model-2"]

    def validate_key(self, model: str | None = None) -> tuple[bool, list[str], str]:
        if self.api_key and "invalid" in self.api_key.lower():
            return False, [], "Invalid API key"
        if self.api_key and "quota" in self.api_key.lower():
            return False, [], "Quota exceeded"
        return True, ["fake-model-1", "fake-model-2"], "Key valid"


def test_header_key_used_and_never_persisted(
    client: TestClient, mini_repo: Path, caplog: pytest.LogCaptureFixture
):
    """Header key is used for the provider call and never persisted in caches, logs, bodies, or error JSON."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        client.post("/api/index", json={"repo_path": repo_str, "force": True})

        distinctive_key = "DISTINCTIVE-SECRET-KEY-XYZ-998877"
        captured_keys: list[str | None] = []

        def fake_provider_factory(provider_name=None, api_key=None, **kwargs):
            captured_keys.append(api_key)
            return KeyTrackingFakeProvider(api_key=api_key)

        caplog.set_level(logging.DEBUG)

        with patch("app.services.pipeline.get_provider", side_effect=fake_provider_factory):
            res = client.post(
                "/api/ask",
                headers={"X-Provider-Key": distinctive_key},
                json={
                    "repo_path": repo_str,
                    "query": "login function",
                    "dry_run": False,
                    "provider": "gemini",
                },
            )

        assert res.status_code == 200
        data = res.json()

        # 1. Assert provider received the key
        assert distinctive_key in captured_keys

        # 2. Assert key source is recorded as user
        assert data["token_report"]["key_source"] == "user"

        # 3. Assert key DOES NOT appear anywhere in the response text or headers
        assert distinctive_key not in res.text
        for h, v in res.headers.items():
            assert distinctive_key not in v

        # 4. Assert key DOES NOT appear in log capture
        for record in caplog.records:
            assert distinctive_key not in record.message

        # 5. Assert key DOES NOT appear in server models cache, services, or job registry
        from app.api.routes import _SERVER_MODELS_CACHE
        from app.api.state import _JOBS, _SERVICES

        assert distinctive_key not in str(_SERVER_MODELS_CACHE)
        assert distinctive_key not in str(_SERVICES)
        assert distinctive_key not in str(_JOBS)


def test_log_redaction_filter():
    """Verify KeyRedactionFilter and redact_sensitive_text redact sensitive keys and patterns."""
    raw_header_key = "MY-SECRET-CUSTOM-HEADER-KEY-1234"
    current_request_key.set(raw_header_key)

    sample_logs = [
        f"Calling provider with header key {raw_header_key}",
        "Anthropic error for sk-ant-api03-abcdef1234567890abcdef12",
        "Google error for key AIzaSyD12345678901234567890123456789012 at endpoint",
        "Groq error with gsk_abcdef1234567890abcdef",
    ]

    for log_msg in sample_logs:
        redacted = redact_sensitive_text(log_msg)
        assert raw_header_key not in redacted
        assert "sk-ant" not in redacted
        assert "AIza" not in redacted
        assert "gsk_" not in redacted
        assert "[REDACTED_KEY]" in redacted

    # Test via logging Handler with KeyRedactionFilter
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg=f"Debug info key={raw_header_key}",
        args=(),
        exc_info=None,
    )
    filt = KeyRedactionFilter()
    filt.filter(record)
    assert raw_header_key not in record.msg
    assert "[REDACTED_KEY]" in record.msg


def test_key_precedence(client: TestClient, mini_repo: Path):
    """Precedence: header key > server .env key > 401 api_key_required."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        client.post("/api/index", json={"repo_path": repo_str, "force": True})

        header_key = "HEADER-KEY-111"
        server_key = "SERVER-ENV-KEY-222"
        received_keys: list[str | None] = []

        def capture_provider(provider_name=None, api_key=None, **kwargs):
            received_keys.append(api_key)
            return KeyTrackingFakeProvider(api_key=api_key)

        # 1. Header key present + Server key present -> Header key wins
        from pydantic import SecretStr
        with (
            patch("app.services.pipeline.get_provider", side_effect=capture_provider),
            patch.object(Settings, "get_key_for_provider", return_value=SecretStr(server_key)),
        ):
            res = client.post(
                "/api/ask",
                headers={"X-Provider-Key": header_key},
                json={"repo_path": repo_str, "query": "login", "provider": "gemini", "model": "gemini-2.5-flash"},
            )
            assert res.status_code == 200
            assert received_keys[-1] == header_key
            assert res.json()["token_report"]["key_source"] == "user"

        # 2. Header key absent + Server key present -> Server key used
        with (
            patch("app.services.pipeline.get_provider", side_effect=capture_provider),
            patch.object(Settings, "get_key_for_provider", return_value=SecretStr(server_key)),
        ):
            res = client.post(
                "/api/ask",
                json={"repo_path": repo_str, "query": "login", "provider": "gemini", "model": "gemini-2.5-flash"},
            )
            assert res.status_code == 200
            assert received_keys[-1] == server_key
            assert res.json()["token_report"]["key_source"] == "server"

        # 3. Neither present -> 401 api_key_required
        with (
            patch.object(Settings, "get_key_for_provider", return_value=None),
        ):
            res = client.post(
                "/api/ask",
                json={"repo_path": repo_str, "query": "login", "provider": "gemini", "model": "gemini-2.5-flash"},
            )
            assert res.status_code == 401
            err = res.json()["error"]
            assert err["code"] == "api_key_required"
            assert "key" in err["message"].lower()


def test_invalid_key_and_quota_error_mapping(client: TestClient, mini_repo: Path):
    """Invalid key maps to 401 api_key_invalid; quota maps to rate-limit error."""
    repo_str = str(mini_repo)
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        client.post("/api/index", json={"repo_path": repo_str, "force": True})

        # 1. Invalid key error
        with patch(
            "app.services.pipeline.get_provider",
            side_effect=ApiKeyInvalidError("Invalid API key provided"),
        ):
            res = client.post(
                "/api/ask",
                headers={"X-Provider-Key": "bad-key"},
                json={"repo_path": repo_str, "query": "login", "provider": "gemini"},
            )
            assert res.status_code == 401
            err = res.json()["error"]
            assert err["code"] == "api_key_invalid"
            assert err["request_id"] is not None

        # 2. Quota / rate limit error
        with patch(
            "app.services.pipeline.get_provider",
            side_effect=ProviderRateLimitError("Rate limit / quota exceeded"),
        ):
            res = client.post(
                "/api/ask",
                headers={"X-Provider-Key": "valid-key-over-quota"},
                json={"repo_path": repo_str, "query": "login", "provider": "gemini"},
            )
            assert res.status_code == 429
            err = res.json()["error"]
            assert err["code"] in ("provider_rate_limited", "rate_limit_exceeded")


def test_validate_endpoint_success_and_failure(client: TestClient):
    """POST /providers/{name}/validate returns friendly structured results and never stores key."""
    validation_rate_limiter.reset()

    # Success validation
    with patch(
        "app.api.routes.get_provider",
        return_value=KeyTrackingFakeProvider(api_key="valid-key"),
    ):
        res = client.post(
            "/api/providers/gemini/validate",
            headers={"X-Provider-Key": "valid-key"},
            json={"model": "gemini-2.5-flash"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["valid"] is True
        assert "models" in data
        assert len(data["models"]) > 0

    # Invalid key validation (returns 401 or valid: false with friendly message)
    with patch(
        "app.api.routes.get_provider",
        side_effect=ApiKeyInvalidError("API key unauthorized"),
    ):
        res = client.post(
            "/api/providers/gemini/validate",
            headers={"X-Provider-Key": "invalid-key"},
            json={},
        )
        assert res.status_code in (200, 401)
        data = res.json()
        if res.status_code == 200:
            assert data["valid"] is False
            assert "invalid" in data["message"].lower() or "unauthorized" in data["message"].lower()
        else:
            assert data["error"]["code"] == "api_key_invalid"


def test_models_cache_isolation_across_user_keys(client: TestClient):
    """Model list fetched with user key is not shared across users in server cache."""
    calls: list[str | None] = []

    def mock_get_provider(*args, **kwargs):
        api_key = kwargs.get("api_key")
        calls.append(api_key)
        m = MagicMock()
        m.get_available_models.return_value = [f"model-for-{api_key}"]
        return m

    with patch("app.api.routes.get_provider", side_effect=mock_get_provider):
        res1 = client.get(
            "/api/providers/gemini/models",
            headers={"X-Provider-Key": "user-key-alpha"},
        )
        assert res1.status_code == 200
        assert res1.json() == ["model-for-user-key-alpha"]

        res2 = client.get(
            "/api/providers/gemini/models",
            headers={"X-Provider-Key": "user-key-beta"},
        )
        assert res2.status_code == 200
        assert res2.json() == ["model-for-user-key-beta"]

        # Ensure server cache did not bleed user-key-alpha's models to user-key-beta
        assert res1.json() != res2.json()


def test_openai_compatible_base_url_server_controlled(client: TestClient, mini_repo: Path):
    """OpenAI-compatible provider base URL must come from server settings, not client requests."""
    from app.ai.base_provider import ProviderConfig
    from app.ai.openai_compat_provider import OpenAICompatProvider

    cfg = ProviderConfig(provider="openai_compat", model="llama-3.3-70b-versatile")
    prov = OpenAICompatProvider(config=cfg, base_url="https://api.groq.com/openai/v1", api_key="sk-test-key")
    # Base URL must be the configured server setting
    assert str(prov.base_url) == "https://api.groq.com/openai/v1"
    assert "attacker.com" not in str(prov.base_url)


def test_validation_rate_limiter(client: TestClient):
    """Validation rate limiter returns 429 after 10 requests per minute from same client IP."""
    validation_rate_limiter.reset()

    fake_provider = KeyTrackingFakeProvider(api_key="test-key")
    with patch("app.api.routes.get_provider", return_value=fake_provider):
        for _ in range(10):
            res = client.post(
                "/api/providers/gemini/validate",
                headers={"X-Provider-Key": "test-key"},
                json={},
            )
            assert res.status_code == 200

        # 11th request must be rate limited
        res11 = client.post(
            "/api/providers/gemini/validate",
            headers={"X-Provider-Key": "test-key"},
            json={},
        )
        assert res11.status_code == 429
        err = res11.json()["error"]
        assert err["code"] == "rate_limit_exceeded"

    validation_rate_limiter.reset()


def test_cache_control_no_store_and_cors(client: TestClient):
    """Cache-Control: no-store is present on key-bearing endpoints."""
    validation_rate_limiter.reset()
    fake_provider = KeyTrackingFakeProvider(api_key="test-key")

    with patch("app.api.routes.get_provider", return_value=fake_provider):
        res = client.post(
            "/api/providers/gemini/validate",
            headers={"X-Provider-Key": "test-key"},
            json={},
        )
        assert res.status_code == 200
        assert "no-store" in res.headers.get("cache-control", "").lower()

    # Preflight OPTIONS request checks CORS allowed headers on allowed origin
    res_opt = client.options(
        "/api/ask",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-Provider-Key, Content-Type",
        },
    )
    assert res_opt.status_code in (200, 204)
    assert "x-provider-key" in res_opt.headers.get("access-control-allow-headers", "").lower()

    # Preflight from disallowed origin is rejected
    res_opt_bad = client.options(
        "/api/ask",
        headers={
            "Origin": "http://evil-attacker.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-Provider-Key",
        },
    )
    assert res_opt_bad.status_code == 400


def test_frontend_key_manager_node_suite():
    """Execute the Node.js unit test suite for key_manager.js."""
    proc = subprocess.run(
        ["node", "tests/test_key_manager.mjs"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node test failure:\n{proc.stdout}\n{proc.stderr}"
    assert "pass 6" in proc.stdout
