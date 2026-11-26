"""Comprehensive unit and offline tests for multi-provider AI support.

Covers:
- GeminiProvider (google-genai SDK, role mapping, usage parsing, safety blocks, quota 429)
- OpenAICompatProvider (OpenAI SDK, error mappings, Retry-After, usage parsing)
- OllamaProvider (reachability, model pulled check, num_ctx explicit, warning)
- Factory / Config (missing model messages, AI_MODEL alias, supports_system_role folding)
- /providers and /ask API endpoints (configured status, secret privacy, provider override)
- Parser tolerance variants (## CHANGE, no lang tag, missing risks)
- Retrieval zero-candidate regression test (low confidence warning)
- Optional integration tests (@pytest.mark.integration)
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, PropertyMock, patch
import pytest
from fastapi.testclient import TestClient

from app.ai.base_provider import (
    ChatMessage,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConfig,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from app.ai.gemini_provider import GeminiProvider
from app.ai.openai_compat_provider import OpenAICompatProvider
from app.ai.ollama_provider import OllamaProvider
from app.ai.pricing import get_cost_info
from app.ai.provider_factory import get_provider
from app.ai.response_parser import parse_response
from app.api.server import app
from app.config import Settings
from app.services.pipeline import RepositoryService
from tests.test_ai import FakeProvider


# ==============================================================================
# 1. GeminiProvider Tests (Mocked google-genai SDK)
# ==============================================================================

def test_gemini_provider_success_and_real_usage():
    """Verify GeminiProvider maps text and real usage metadata correctly."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "### ANALYSIS\nOptimized.\n\n### CHANGES\n### CHANGE: foo.bar (foo/bar.py)\n```python\ndef bar(): pass\n```"
    mock_response.model_version = "gemini-2.5-flash-preview"
    mock_response.response_id = "res-gemini-123"

    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_response.candidates = [mock_candidate]

    # Usage metadata
    mock_response.usage_metadata.prompt_token_count = 150
    mock_response.usage_metadata.candidates_token_count = 75
    mock_response.usage_metadata.total_token_count = 225

    mock_client.models.generate_content.return_value = mock_response

    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash", max_output_tokens=1000)
    provider = GeminiProvider(config=config, client=mock_client)

    result = provider.complete("System instruction text", [ChatMessage(role="user", content="hello")])

    assert "Optimized." in result.text
    assert result.usage.input_tokens == 150
    assert result.usage.output_tokens == 75
    assert result.usage.total_tokens == 225
    assert result.usage.usage_estimated is False
    assert result.model == "gemini-2.5-flash-preview"
    assert result.stop_reason == "stop"

    # Verify generate_content call args
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["config"].system_instruction == "System instruction text"
    assert len(call_kwargs["contents"]) == 1
    assert call_kwargs["contents"][0].role == "user"


def test_gemini_provider_missing_usage_marks_estimated():
    """Verify missing usage_metadata sets tokens to None and flags usage_estimated=True."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello from Gemini"
    mock_response.candidates = [MagicMock(finish_reason="STOP")]
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash")
    provider = GeminiProvider(config=config, client=mock_client)

    result = provider.complete("sys", [ChatMessage(role="user", content="hi")])

    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.usage_estimated is True


def test_gemini_provider_safety_blocked_raises_clear_error():
    """Verify safety-blocked response raises ProviderBadRequestError with finish_reason."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "SAFETY"
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash")
    provider = GeminiProvider(config=config, client=mock_client)

    with pytest.raises(ProviderBadRequestError, match="blocked by safety policy.*SAFETY"):
        provider.complete("sys", [ChatMessage(role="user", content="test")])


def test_gemini_provider_empty_candidates_raises_clear_error():
    """Verify empty candidates response raises descriptive error instead of empty string."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.candidates = []
    mock_response.prompt_feedback = "BLOCK_LOW_AND_ABOVE"
    mock_client.models.generate_content.return_value = mock_response

    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash")
    provider = GeminiProvider(config=config, client=mock_client)

    with pytest.raises(ProviderBadRequestError, match="returned no candidates"):
        provider.complete("sys", [ChatMessage(role="user", content="test")])


def test_gemini_provider_free_tier_quota_error_message():
    """Verify free-tier 429 / RESOURCE_EXHAUSTED includes actionable message."""
    from google.genai import errors

    mock_client = MagicMock()
    # Simulate APIError 429
    api_err = errors.APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED: Quota exceeded for quota metric 'GenerateContent'", "code": 429}})
    mock_client.models.generate_content.side_effect = api_err

    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash", max_retries=1)
    provider = GeminiProvider(config=config, client=mock_client)

    with patch("time.sleep", return_value=None):
        with pytest.raises(ProviderRateLimitError) as exc_info:
            provider.complete("sys", [ChatMessage(role="user", content="test")])

    err_msg = str(exc_info.value)
    assert "Rate limit or free-tier quota reached for Gemini" in err_msg
    assert "free-tier rate or daily quota limit" in err_msg


def test_gemini_provider_count_tokens():
    """Verify count_tokens calls SDK endpoint or returns None on failure."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.total_tokens = 320
    mock_client.models.count_tokens.return_value = mock_res

    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash")
    provider = GeminiProvider(config=config, client=mock_client)

    assert provider.count_tokens("sys", [ChatMessage(role="user", content="hi")]) == 320

    # Failure returns None
    mock_client.models.count_tokens.side_effect = Exception("Service error")
    assert provider.count_tokens("sys", [ChatMessage(role="user", content="hi")]) is None


def test_gemini_provider_key_sanitization():
    """Verify API keys are masked in string representations and errors."""
    key = "AIzaSyD-fake-key-1234567890123456789"
    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash")
    provider = GeminiProvider(config=config, api_key=key)

    rep = repr(provider)
    assert key not in rep
    assert "configured=True" in rep


def test_gemini_provider_unsupported_model_folds_system_instructions():
    """Verify models that don't support developer instructions fold them into the user prompt."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Answer from gemini-1.0-pro"
    mock_response.candidates = [MagicMock(finish_reason="STOP")]
    mock_response.usage_metadata.prompt_token_count = 100
    mock_response.usage_metadata.candidates_token_count = 50
    mock_response.usage_metadata.total_token_count = 150
    mock_client.models.generate_content.return_value = mock_response

    config = ProviderConfig(provider="gemini", model="gemini-1.0-pro")
    provider = GeminiProvider(config=config, client=mock_client)
    assert provider.supports_system_role is False

    result = provider.complete("System prompt for instructions", [ChatMessage(role="user", content="User task")])

    assert result.text == "Answer from gemini-1.0-pro"
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-1.0-pro"
    # system_instruction must be None in config
    assert call_kwargs["config"].system_instruction is None
    # system prompt must be incorporated into the first user message
    contents = call_kwargs["contents"]
    assert len(contents) == 1
    assert "System prompt for instructions" in contents[0].parts[0].text
    assert "User task" in contents[0].parts[0].text


def test_gemini_provider_developer_instruction_error_fallback():
    """Verify fallback occurs when API raises 'Developer instruction is not enabled for this model'."""
    from google.genai import errors

    mock_client = MagicMock()
    # First call with system_instruction fails with 400 INVALID_ARGUMENT
    err_payload = {
        "error": {
            "code": 400,
            "message": "Developer instruction is not enabled for this model",
            "status": "INVALID_ARGUMENT",
        }
    }
    dev_inst_error = errors.APIError(400, err_payload)

    # Second call succeeds
    mock_response = MagicMock()
    mock_response.text = "Fallback solution generated successfully"
    mock_response.candidates = [MagicMock(finish_reason="STOP")]
    mock_response.usage_metadata.prompt_token_count = 120
    mock_response.usage_metadata.candidates_token_count = 60
    mock_response.usage_metadata.total_token_count = 180

    mock_client.models.generate_content.side_effect = [dev_inst_error, mock_response]

    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash")
    provider = GeminiProvider(config=config, client=mock_client)

    result = provider.complete("System instruction text", [ChatMessage(role="user", content="Analyze repository")])

    assert result.text == "Fallback solution generated successfully"
    assert mock_client.models.generate_content.call_count == 2

    # Second call must have folded instructions and passed system_instruction=None
    second_call_kwargs = mock_client.models.generate_content.call_args_list[1].kwargs
    assert second_call_kwargs["config"].system_instruction is None
    assert "System instruction text" in second_call_kwargs["contents"][0].parts[0].text
    assert "Analyze repository" in second_call_kwargs["contents"][0].parts[0].text


def test_gemini_provider_extracts_text_from_parts_fallback():
    """Verify text is extracted from candidate parts if response.text raises or is None."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Simulate response.text raising ValueError or being None
    type(mock_response).text = PropertyMock(side_effect=ValueError("No text property"))

    mock_part = MagicMock()
    mock_part.text = "Text from candidate part"
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_candidate.content.parts = [mock_part]
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash")
    provider = GeminiProvider(config=config, client=mock_client)

    result = provider.complete("sys", [ChatMessage(role="user", content="hello")])
    assert result.text == "Text from candidate part"


def test_gemini_provider_empty_text_fallback():
    """Verify empty text produces descriptive fallback message."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "   "
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_candidate.content.parts = []
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    config = ProviderConfig(provider="gemini", model="gemini-2.5-flash")
    provider = GeminiProvider(config=config, client=mock_client)

    result = provider.complete("sys", [ChatMessage(role="user", content="hello")])
    assert "empty text" in result.text


# ==============================================================================
# 2. OpenAICompatProvider Tests (Mocked openai SDK)
# ==============================================================================

def test_openai_compat_success_and_real_usage():
    """Verify OpenAICompatProvider parses completions and returns real tokens."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Here is the OpenAI compatible response"
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.model = "meta-llama/llama-3.3-70b-instruct"
    mock_response.id = "chatcmpl-groq-999"
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.completion_tokens = 40
    mock_response.usage.total_tokens = 120

    mock_client.chat.completions.create.return_value = mock_response

    config = ProviderConfig(provider="openai_compat", model="llama-3.3-70b")
    provider = OpenAICompatProvider(config=config, client=mock_client, base_url="https://api.groq.com/openai/v1")

    result = provider.complete("sys", [ChatMessage(role="user", content="hello")])

    assert result.text == "Here is the OpenAI compatible response"
    assert result.usage.input_tokens == 80
    assert result.usage.output_tokens == 40
    assert result.usage.total_tokens == 120
    assert result.usage.usage_estimated is False
    assert result.stop_reason == "stop"


def test_openai_compat_missing_usage():
    """Verify missing usage sets usage_estimated=True."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "No usage response"
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_response.id = "req-test-1"
    mock_response.model = "llama-3.3-70b"
    mock_client.chat.completions.create.return_value = mock_response

    config = ProviderConfig(provider="openai_compat", model="llama-3.3-70b")
    provider = OpenAICompatProvider(config=config, client=mock_client, api_key="sk-test-key")

    result = provider.complete("sys", [ChatMessage(role="user", content="hi")])
    assert result.usage.usage_estimated is True
    assert result.usage.input_tokens is None


def test_openai_compat_error_mapping_401_404_429_500():
    """Verify 401, 404 (with model id), 429, and 500 error mapping."""
    import openai

    config = ProviderConfig(provider="openai_compat", model="requested-model-x", max_retries=0)
    mock_client = MagicMock()
    provider = OpenAICompatProvider(config=config, client=mock_client, api_key="sk-test-key")

    # 401 AuthenticationError
    mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
        message="Invalid API Key", response=MagicMock(status_code=401), body={}
    )
    with pytest.raises(ProviderAuthError):
        provider.complete("sys", [ChatMessage(role="user", content="hi")])

    # 404 NotFoundError (must mention requested model)
    mock_client.chat.completions.create.side_effect = openai.NotFoundError(
        message="The model requested-model-x does not exist",
        response=MagicMock(status_code=404),
        body={},
    )
    with pytest.raises(ProviderBadRequestError) as exc_404:
        provider.complete("sys", [ChatMessage(role="user", content="hi")])
    assert "requested-model-x" in str(exc_404.value)

    # 429 RateLimitError
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit reached", response=MagicMock(status_code=429), body={}
    )
    with pytest.raises(ProviderRateLimitError):
        provider.complete("sys", [ChatMessage(role="user", content="hi")])

    # 500 InternalServerError
    mock_client.chat.completions.create.side_effect = openai.InternalServerError(
        message="Backend crash", response=MagicMock(status_code=500), body={}
    )
    with pytest.raises(ProviderUnavailableError):
        provider.complete("sys", [ChatMessage(role="user", content="hi")])


def test_openai_compat_retry_after_extraction():
    """Verify retry-after extraction from response header and error text."""
    config = ProviderConfig(provider="openai_compat", model="test-model")
    provider = OpenAICompatProvider(config=config)

    # From HTTP response headers
    exc_with_headers = Exception("Rate limited")
    mock_resp = MagicMock()
    mock_resp.headers = {"retry-after": "4.5"}
    exc_with_headers.response = mock_resp
    assert provider._extract_retry_after(exc_with_headers) == 4.5

    # From string message
    exc_with_msg = Exception("Please retry after 12.0 seconds")
    assert provider._extract_retry_after(exc_with_msg) == 12.0


def test_openai_compat_count_tokens_returns_none():
    """OpenAICompatProvider count_tokens must always return None."""
    provider = OpenAICompatProvider(config=ProviderConfig(provider="openai_compat", model="test"))
    assert provider.count_tokens("sys", [ChatMessage(role="user", content="hi")]) is None


# ==============================================================================
# 3. OllamaProvider Tests (Local Ollama /v1 & reachability)
# ==============================================================================

def test_ollama_not_running_error():
    """Verify clear error message when Ollama server is unreachable."""
    mock_http = MagicMock()
    mock_http.get.side_effect = Exception("Connection refused")

    config = ProviderConfig(provider="ollama", model="llama3.2")
    provider = OllamaProvider(config=config, http_client=mock_http)

    assert provider.is_configured() is False
    assert "Ollama server is not running" in (provider.get_setup_hint() or "")

    with pytest.raises(ProviderUnavailableError, match="Ollama server is not running.*ollama serve"):
        provider.complete("sys", [ChatMessage(role="user", content="hi")])


def test_ollama_model_not_pulled_error():
    """Verify clear error message specifying exact command to run when model not pulled."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "qwen2.5-coder:latest"}, {"name": "mistral:latest"}]
    }
    mock_http.get.return_value = mock_resp

    config = ProviderConfig(provider="ollama", model="deepseek-coder")
    provider = OllamaProvider(config=config, http_client=mock_http)

    assert provider.is_configured() is False
    hint = provider.get_setup_hint()
    assert "ollama pull deepseek-coder" in hint

    with pytest.raises(ProviderBadRequestError, match="Run 'ollama pull deepseek-coder'"):
        provider.complete("sys", [ChatMessage(role="user", content="hi")])


def test_ollama_num_ctx_passed_explicitly_and_context_warning(caplog):
    """Verify num_ctx is explicitly passed in extra_body options and context warning fires."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
    mock_http.get.return_value = mock_resp

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Local response"
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.model = "llama3.2"
    mock_response.id = "req-ollama-1"
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 30
    mock_response.usage.total_tokens = 80
    mock_client.chat.completions.create.return_value = mock_response

    # Small num_ctx (2000) with budget (2500) + max_output (1000) > num_ctx
    config = ProviderConfig(provider="ollama", model="llama3.2", max_output_tokens=1000)
    provider = OllamaProvider(
        config=config,
        num_ctx=2000,
        context_token_budget=2500,
        client=mock_client,
        http_client=mock_http,
    )

    with caplog.at_level(logging.WARNING):
        result = provider.complete("sys", [ChatMessage(role="user", content="hi")])

    assert result.text == "Local response"
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 30
    assert result.usage.total_tokens == 80

    # Verify num_ctx was passed explicitly in extra_body options
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {"options": {"num_ctx": 2000}}

    # Verify context window warning was logged
    assert any("exceeds Ollama num_ctx" in r.message for r in caplog.records)


# ==============================================================================
# 4. Factory & Config Tests
# ==============================================================================

def test_missing_model_raises_actionable_error():
    """Verify factory raises helpful message when model is missing for a provider."""
    settings = Settings(ai_model=None, claude_model=None)
    with pytest.raises(ValueError, match="set CLAUDE_MODEL in .env"):
        get_provider("claude", settings=settings)

    settings_gemini = Settings(gemini_model=None, ai_model=None)
    with pytest.raises(ValueError, match="set GEMINI_MODEL in .env"):
        get_provider("gemini", settings=settings_gemini)


def test_ai_model_backward_compatible_alias():
    """Verify AI_MODEL functions as a fallback model for the active provider."""
    settings = Settings(ai_model="claude-3-5-sonnet-alias", claude_model=None)
    prov = get_provider("claude", settings=settings)
    assert prov.config.model == "claude-3-5-sonnet-alias"


def test_supports_system_role_false_folds_into_user_message():
    """Verify fold_system_into_messages folds system text when supports_system_role=False."""
    prov = FakeProvider()
    prov.supports_system_role = False

    sys_text = "You are a code reviewer."
    messages = [ChatMessage(role="user", content="Check this code.")]
    system_out, folded = prov.fold_system_into_messages(sys_text, messages)

    assert system_out == ""
    assert len(folded) == 1
    assert "You are a code reviewer." in folded[0].content
    assert "Check this code." in folded[0].content


# ==============================================================================
# 5. API Endpoints: /providers and /ask overrides
# ==============================================================================

def test_api_get_providers_endpoint():
    """Verify GET /providers returns configured status and setup hints, never API keys."""
    client = TestClient(app)
    response = client.get("/providers")
    assert response.status_code == 200
    providers = response.json()
    assert isinstance(providers, list)

    names = [p["name"] for p in providers]
    assert "claude" in names
    assert "gemini" in names
    assert "openai_compat" in names
    assert "ollama" in names

    # Ensure no secret keys are present in any field
    for p in providers:
        assert "api_key" not in p
        assert "key" not in p
        assert "configured" in p
        assert "is_local" in p


def test_api_ask_with_provider_override(tmp_path):
    """Verify /ask accepts provider and model overrides, rejecting unknown or unconfigured."""
    client = TestClient(app)
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    repo_dir = str(tmp_path)
    assert client.post("/index", json={"repo_path": repo_dir, "force": True}).status_code == 200

    # 1. Unknown provider returns 400
    res_unknown = client.post("/ask", json={"repo_path": repo_dir, "query": "test query", "provider": "nonexistent_provider"})
    assert res_unknown.status_code == 400
    assert "Unknown AI provider" in str(res_unknown.json())

    # 2. Unconfigured provider returns 503 with setup hint
    # Patch provider as unconfigured in pipeline
    with patch("app.services.pipeline.get_provider") as mock_get:
        mock_gem = MagicMock()
        mock_gem.name = "gemini"
        mock_gem.is_configured.return_value = False
        mock_gem.get_setup_hint.return_value = "Set GEMINI_API_KEY in .env"
        mock_get.return_value = mock_gem

        res_unconf = client.post(
            "/ask",
            json={"repo_path": repo_dir, "query": "test query", "provider": "gemini"},
            headers={"X-Provider-Key": "dummy-key"},
        )
        assert res_unconf.status_code == 503
        assert "Set GEMINI_API_KEY" in str(res_unconf.json())


# ==============================================================================
# 6. Parser Tolerance Variants
# ==============================================================================

def test_parser_tolerant_of_hash_levels_and_missing_risks():
    """Verify parser handles ## CHANGE, missing risks, and code fences without language tags."""
    tolerant_text = """### ANALYSIS
Minor cleanup.

## CHANGE: utils.clean (utils/clean.py)
```
def clean(s):
    return s.strip()
```

### NEEDS_CONTEXT
- utils.other
"""
    parsed = parse_response(tolerant_text, known_targets={"utils.clean"})
    assert "Minor cleanup" in parsed.analysis
    assert len(parsed.changes) == 1
    assert parsed.changes[0].qualified_name == "utils.clean"
    assert "def clean" in parsed.changes[0].code
    assert parsed.changes[0].syntax_valid is True
    # Risks was omitted but parsing should succeed without raising
    assert len(parsed.risks) == 0
    assert len(parsed.needs_context) == 1


# ==============================================================================
# 7. Pricing Table (Free Tier Support)
# ==============================================================================

def test_pricing_free_tier():
    """Verify free tier models display as free tier with $0 cost."""
    cost, is_free, display = get_cost_info(
        "gemini-2.5-flash", 1000, 500, pricing_table={"gemini-2.5-flash": {"free": True}}
    )
    assert is_free is True
    assert cost == 0.0
    assert display == "free tier"

    # Ollama local model
    cost_oll, is_free_oll, display_oll = get_cost_info("llama3.2", 2000, 500, provider="ollama")
    assert is_free_oll is True
    assert cost_oll == 0.0
    assert display_oll == "free tier"


# ==============================================================================
# 8. Section 9 Regression: Low-confidence Query Returns Candidates With Warning
# ==============================================================================

def test_low_confidence_query_returns_candidates_with_warning():
    """Verify a query with low semantic confidence returns candidates with warning, not an empty list."""
    from pathlib import Path
    repo = Path("sample_repos/requests")
    if not repo.exists():
        pytest.skip("sample_repos/requests not found on disk")

    # Misspelled query "retrivew login page and optimize it"
    service = RepositoryService(repo)
    retrieval_res, built_ctx = service.retrieve("retrivew login page and optimize it", top_k=5)

    assert len(retrieval_res.candidates) > 0, "Low-confidence query must return best-effort candidates, not empty"
    assert retrieval_res.target_confidence == "low"
    # Warning should be present when confidence is low
    assert any("Low confidence" in w for w in built_ctx.warnings)


# ==============================================================================
# 9. Optional Live Integration Tests (Auto-skipped if not configured)
# ==============================================================================

@pytest.mark.integration
def test_integration_gemini_live():
    """Live smoke test for GeminiProvider if GEMINI_API_KEY is configured in environment."""
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set in environment")

    prov = get_provider("gemini", model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    if not prov.is_configured():
        pytest.skip("Gemini provider not configured")

    res = prov.complete("You are a concise tester.", [ChatMessage(role="user", content="Reply 'PONG'.")], max_output_tokens=10)
    assert len(res.text) > 0
    assert res.usage.total_tokens is not None


@pytest.mark.integration
def test_integration_ollama_live():
    """Live smoke test for OllamaProvider if Ollama server is running locally."""
    prov = get_provider("ollama", model="llama3.2")
    if not prov.is_configured():
        pytest.skip("Ollama is not running or llama3.2 is not pulled locally")

    res = prov.complete("Be brief.", [ChatMessage(role="user", content="Reply 'PONG'.")], max_output_tokens=10)
    assert len(res.text) > 0
