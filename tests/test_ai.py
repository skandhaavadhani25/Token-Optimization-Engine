"""Unit tests for AI provider layer, retry mechanics, prompts, and response parser."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import anthropic

from app.ai.base_provider import (
    AIProvider,
    ChatMessage,
    CompletionResult,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConfig,
    ProviderTimeoutError,
    Usage,
)
from app.ai.claude_provider import ClaudeProvider
from app.ai.pricing import estimate_cost
from app.ai.prompts import build_messages
from app.ai.provider_factory import get_provider
from app.ai.response_parser import parse_response


class FakeProvider(AIProvider):
    """Mock provider for unit testing without API access."""

    def __init__(
        self,
        responses: list[str] | None = None,
        usage: Usage | None = None,
        configured: bool = True,
        token_count: int | None = None,
        config: ProviderConfig | None = None,
    ) -> None:
        self.responses = responses or [
            "### ANALYSIS\nRefactored helper function.\n\n### CHANGES\n### CHANGE: utils.hash (utils/hashing.py)\n```python\ndef hash_val(x: str) -> str:\n    return x.strip()\n```\n\n### RISKS\n- Stripping whitespace changes semantics\n\n### NEEDS_CONTEXT\n- utils.types"
        ]
        self.usage = usage or Usage(
            input_tokens=120,
            output_tokens=60,
            total_tokens=180,
            cache_read_tokens=None,
            cache_creation_tokens=None,
            estimated_cost_usd=None,
        )
        self._configured = configured
        self.token_count = token_count
        self.config = config or ProviderConfig(
            provider="fake",
            model="fake-claude",
            max_output_tokens=2000,
            temperature=0.2,
            timeout_seconds=60,
            max_retries=3,
        )
        self.call_count = 0
        self.last_system: str | None = None
        self.last_messages: list[ChatMessage] | None = None

    @property
    def name(self) -> str:
        return "fake"

    def is_configured(self) -> bool:
        return self._configured

    def count_tokens(self, system: str, messages: list[ChatMessage]) -> int | None:
        return self.token_count if self.token_count is not None else 150

    def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        if not self._configured:
            raise ProviderAuthError("Fake provider is not configured.")
        self.call_count += 1
        self.last_system = system
        self.last_messages = messages
        idx = (self.call_count - 1) % len(self.responses)
        return CompletionResult(
            text=self.responses[idx],
            usage=self.usage,
            model="fake-model-v1",
            stop_reason="end_turn",
            latency_seconds=0.01,
            request_id="req-fake-999",
        )


def test_provider_factory_and_stubs():
    """Verify factory returns ClaudeProvider, GeminiProvider, OpenAICompatProvider, OllamaProvider."""
    from app.ai.gemini_provider import GeminiProvider
    from app.ai.openai_compat_provider import OpenAICompatProvider
    from app.ai.ollama_provider import OllamaProvider

    # Claude provider
    prov = get_provider("claude", model="claude-3-5-sonnet-20241022")
    assert isinstance(prov, ClaudeProvider)
    assert prov.name == "claude"

    # OpenAI-compatible provider and alias
    prov_oai = get_provider("openai_compat", model="llama-3.3-70b-versatile")
    assert isinstance(prov_oai, OpenAICompatProvider)
    assert prov_oai.name == "openai_compat"

    prov_oai_alias = get_provider("openai", model="llama-3.3-70b-versatile")
    assert isinstance(prov_oai_alias, OpenAICompatProvider)

    # Gemini provider
    prov_gem = get_provider("gemini", model="gemini-2.5-flash")
    assert isinstance(prov_gem, GeminiProvider)
    assert prov_gem.name == "gemini"

    # Ollama provider and alias
    prov_oll = get_provider("ollama", model="llama3.2")
    assert isinstance(prov_oll, OllamaProvider)
    assert prov_oll.name == "ollama"

    prov_local_alias = get_provider("local", model="llama3.2")
    assert isinstance(prov_local_alias, OllamaProvider)

    # Unknown provider
    with pytest.raises(ValueError, match="Unknown AI provider 'unknown'"):
        get_provider("unknown")


def test_api_key_not_leaked_in_repr_or_errors():
    """Verify secret API keys are never exposed in string representations or errors."""
    secret = "sk-ant-api03-very-secret-key-12345"
    config = ProviderConfig(provider="claude", model="claude-3-5-sonnet-20241022")
    provider = ClaudeProvider(config=config, api_key=secret)

    rep = repr(provider)
    assert secret not in rep
    assert "configured=True" in rep

    unconfigured = ClaudeProvider(config=config, api_key=None)
    assert "configured=False" in repr(unconfigured)

    # Calling complete on unconfigured provider raises clean error without secrets
    with pytest.raises(ProviderAuthError) as exc_info:
        unconfigured.complete("system", [ChatMessage(role="user", content="hello")])
    assert secret not in str(exc_info.value)
    assert "API key is not configured" in str(exc_info.value)


@patch("time.sleep", return_value=None)
def test_claude_provider_retry_transient_failures(mock_sleep):
    """Verify transient errors (rate limit, timeout, 5xx) are retried with backoff."""
    mock_client = MagicMock()
    # Fail 2 times with rate limit, then succeed
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Hello from Claude")]
    mock_response.model = "claude-3-5-sonnet-20241022"
    mock_response.stop_reason = "end_turn"
    mock_response.id = "msg_123"
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 20
    mock_response.usage.cache_read_input_tokens = None
    mock_response.usage.cache_creation_input_tokens = None

    rate_limit_err = anthropic.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429, headers={}),
        body={},
    )
    mock_client.messages.create.side_effect = [rate_limit_err, rate_limit_err, mock_response]

    config = ProviderConfig(provider="claude", model="claude-3-5-sonnet-20241022", max_retries=3)
    provider = ClaudeProvider(config=config, client=mock_client)

    result = provider.complete("sys", [ChatMessage(role="user", content="hi")])
    assert result.text == "Hello from Claude"
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 20
    assert mock_client.messages.create.call_count == 3
    assert mock_sleep.call_count == 2


@patch("time.sleep", return_value=None)
def test_claude_provider_does_not_retry_auth_or_bad_request(mock_sleep):
    """Verify non-transient errors (auth, bad request) fail immediately without retries."""
    mock_client = MagicMock()
    auth_err = anthropic.AuthenticationError(
        message="Invalid API Key",
        response=MagicMock(status_code=401, headers={}),
        body={},
    )
    mock_client.messages.create.side_effect = auth_err

    config = ProviderConfig(provider="claude", model="claude-3-5-sonnet-20241022", max_retries=3)
    provider = ClaudeProvider(config=config, client=mock_client)

    with pytest.raises(ProviderAuthError):
        provider.complete("sys", [ChatMessage(role="user", content="hi")])

    assert mock_client.messages.create.call_count == 1
    assert mock_sleep.call_count == 0

    # Bad request error
    bad_req_err = anthropic.BadRequestError(
        message="Invalid request format",
        response=MagicMock(status_code=400, headers={}),
        body={},
    )
    mock_client.messages.create.side_effect = bad_req_err
    with pytest.raises(ProviderBadRequestError):
        provider.complete("sys", [ChatMessage(role="user", content="hi")])

    assert mock_client.messages.create.call_count == 2
    assert mock_sleep.call_count == 0


@patch("time.sleep", return_value=None)
def test_claude_provider_exceeds_max_retries(mock_sleep):
    """Verify when transient failure persists past max_retries, it raises mapped error."""
    mock_client = MagicMock()
    timeout_err = anthropic.APITimeoutError(request=MagicMock())
    mock_client.messages.create.side_effect = timeout_err

    config = ProviderConfig(provider="claude", model="claude-3-5-sonnet-20241022", max_retries=2)
    provider = ClaudeProvider(config=config, client=mock_client)

    with pytest.raises(ProviderTimeoutError):
        provider.complete("sys", [ChatMessage(role="user", content="hi")])

    assert mock_client.messages.create.call_count == 3  # Initial + 2 retries
    assert mock_sleep.call_count == 2


def test_pricing_estimation():
    """Verify pricing cost calculation and missing model behavior."""
    # Unknown model with empty default pricing returns None
    assert estimate_cost("nonexistent-model-xyz", 1000, 500) is None

    # Custom pricing table
    custom_pricing = {
        "claude-3-5-sonnet-test": {
            "input_cost_per_million": 3.00,
            "output_cost_per_million": 15.00,
        }
    }
    # 1,000,000 input ($3.00) + 1,000,000 output ($15.00) = $18.00
    cost = estimate_cost("claude-3-5-sonnet-test", 1_000_000, 1_000_000, custom_pricing)
    assert cost == 18.0

    # Half million input ($1.50) + 100,000 output ($1.50) = $3.00
    cost2 = estimate_cost("claude-3-5-sonnet-test", 500_000, 100_000, custom_pricing)
    assert cost2 == 3.0


def test_prompt_builder():
    """Verify system prompt and user template construction."""
    sys_prompt, messages = build_messages(
        context_text="TARGET:\nfunc_a()",
        task_text="Refactor func_a for performance",
    )
    assert "senior software engineer" in sys_prompt
    assert "### CHANGE:" in sys_prompt
    assert len(messages) == 1
    user_msg = messages[0]
    assert user_msg.role == "user"
    assert "TARGET:\nfunc_a()" in user_msg.content
    assert "TASK:\nRefactor func_a for performance" in user_msg.content
    assert user_msg.content.strip().endswith("TASK:\nRefactor func_a for performance")


def test_response_parser_well_formed():
    """Verify parser extracts analysis, changes, risks, and needs_context correctly."""
    response_text = """### ANALYSIS
The function can be optimized by caching regex compilation.

### CHANGES
### CHANGE: utils.text.clean (utils/text.py)
```python
import re

_RE = re.compile(r"\\s+")

def clean(text: str) -> str:
    return _RE.sub(" ", text.strip())
```

### RISKS
- Pre-compiling consumes a small amount of module-level memory.
- Thread-safety is preserved.

### NEEDS_CONTEXT
- utils.types.TextConfig
"""
    parsed = parse_response(response_text, known_targets={"utils.text.clean"})
    assert "optimized by caching regex" in parsed.analysis
    assert len(parsed.changes) == 1
    ch = parsed.changes[0]
    assert ch.qualified_name == "utils.text.clean"
    assert ch.file == "utils/text.py"
    assert "def clean" in ch.code
    assert ch.syntax_valid is True
    assert ch.unknown_target is False

    assert len(parsed.risks) == 2
    assert "Pre-compiling consumes" in parsed.risks[0]

    assert len(parsed.needs_context) == 1
    assert parsed.needs_context[0] == "utils.types.TextConfig"
    assert len(parsed.parse_warnings) == 0


def test_response_parser_syntax_error_and_unknown_target():
    """Verify parser flags invalid python syntax and unknown symbol targets."""
    response_text = """### ANALYSIS
Fixing syntax error.

### CHANGES
### CHANGE: app.broken.func (app/broken.py)
```python
def invalid_code(
    # unclosed parenthesis
```
"""
    parsed = parse_response(response_text, known_targets={"app.other.func"})
    assert len(parsed.changes) == 1
    ch = parsed.changes[0]
    assert ch.qualified_name == "app.broken.func"
    assert ch.syntax_valid is False
    assert ch.unknown_target is True
    assert any("Syntax error" in w for w in parsed.parse_warnings)


def test_response_parser_malformed_and_empty():
    """Verify parser never raises on malformed or empty output."""
    empty_parsed = parse_response("")
    assert empty_parsed.analysis == ""
    assert len(empty_parsed.changes) == 0
    assert len(empty_parsed.parse_warnings) > 0

    prose_parsed = parse_response("Here is the answer without any markdown sections.")
    assert prose_parsed.analysis == "Here is the answer without any markdown sections."
    assert len(prose_parsed.changes) == 0
    assert len(prose_parsed.parse_warnings) > 0
