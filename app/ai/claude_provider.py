"""Anthropic Claude provider implementation using official anthropic Python SDK."""

from __future__ import annotations

import logging
import random
import time
from typing import Any
import anthropic
from pydantic import SecretStr

from app.ai.base_provider import (
    AIProvider,
    ApiKeyInvalidError,
    ApiKeyRequiredError,
    ChatMessage,
    CompletionResult,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConfig,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from app.ai.pricing import estimate_cost

logger = logging.getLogger(__name__)


class ClaudeProvider(AIProvider):
    """Claude provider wrapping official Anthropic Messages API."""

    def __init__(
        self,
        config: ProviderConfig,
        api_key: str | SecretStr | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.config = config
        self._raw_key: str | None = None
        if api_key is not None:
            if isinstance(api_key, SecretStr):
                self._raw_key = api_key.get_secret_value()
            else:
                self._raw_key = str(api_key)

        # Allow passing an injected client for mock testing
        if client is not None:
            self._client: anthropic.Anthropic | None = client
        elif self._raw_key:
            self._client = anthropic.Anthropic(
                api_key=self._raw_key,
                timeout=self.config.timeout_seconds,
                max_retries=0,  # We handle retries explicitly with backoff & logging
            )
        else:
            self._client = None

    @property
    def name(self) -> str:
        return "claude"

    def is_configured(self) -> bool:
        """Return True if an API key is set and client is available."""
        return self._client is not None

    def get_setup_hint(self) -> str | None:
        """Return setup hint if unconfigured."""
        if not self.is_configured():
            return "Set ANTHROPIC_API_KEY in .env"
        return None

    def __repr__(self) -> str:
        # Never leak the secret API key in representations
        return f"<ClaudeProvider model='{self.config.model}' configured={self.is_configured()}>"

    def __str__(self) -> str:
        return self.__repr__()

    def _map_exception(self, err: Exception) -> ProviderError:
        """Map Anthropic SDK exceptions to unified ProviderError subclasses."""
        if isinstance(err, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return ProviderAuthError(f"Anthropic authentication failed: {err}")
        if isinstance(err, anthropic.RateLimitError):
            return ProviderRateLimitError(f"Anthropic rate limit exceeded: {err}")
        if isinstance(err, anthropic.APITimeoutError):
            return ProviderTimeoutError(f"Anthropic request timed out: {err}")
        if isinstance(err, anthropic.BadRequestError):
            return ProviderBadRequestError(f"Anthropic bad request: {err}")
        if isinstance(err, (anthropic.APIConnectionError, anthropic.InternalServerError)):
            return ProviderUnavailableError(f"Anthropic service unavailable: {err}")
        if isinstance(err, anthropic.APIStatusError):
            if err.status_code >= 500:
                return ProviderUnavailableError(f"Anthropic server error ({err.status_code}): {err}")
            if err.status_code == 429:
                return ProviderRateLimitError(f"Anthropic rate limit: {err}")
            if err.status_code in (401, 403):
                return ApiKeyInvalidError(f"Anthropic authentication error ({err.status_code}): {err}")
            return ProviderBadRequestError(f"Anthropic API error ({err.status_code}): {err}")
        return ProviderError(f"Unexpected provider error: {err}")

    def get_available_models(self) -> list[str]:
        """Return available models for Claude."""
        default_list = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
        if not self.is_configured():
            return default_list
        try:
            if hasattr(self._client, "models") and hasattr(self._client.models, "list"):
                res = self._client.models.list()
                models = [m.id for m in getattr(res, "data", []) if getattr(m, "id", None)]
                if models:
                    return models
        except Exception:
            pass
        return default_list

    def validate_key(self, model: str | None = None) -> tuple[bool, list[str], str]:
        """Validate Anthropic API key with cheapest possible call."""
        if not self.is_configured():
            return False, [], "No API key provided. Please enter your Anthropic API key."
        target_model = model or self.config.model or "claude-3-5-sonnet-20241022"
        try:
            if hasattr(self._client, "models") and hasattr(self._client.models, "list"):
                res = self._client.models.list()
                models = [m.id for m in getattr(res, "data", []) if getattr(m, "id", None)]
                if models:
                    return True, models, "Key is valid."
        except anthropic.AuthenticationError:
            return False, [], "Invalid API key. Please check your Anthropic credentials."
        except anthropic.RateLimitError:
            return False, [], "API key is valid, but rate limit or quota has been reached."
        except Exception:
            pass

        # Fallback to 1-token dummy message create
        try:
            self._client.messages.create(
                model=target_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True, self.get_available_models(), "Key is valid."
        except anthropic.AuthenticationError:
            return False, [], "Invalid API key. Please check your Anthropic credentials."
        except anthropic.RateLimitError:
            return False, [], "API key is valid, but rate limit or quota has been reached."
        except Exception as exc:
            err_msg = str(exc)
            if "auth" in err_msg.lower() or "401" in err_msg or "403" in err_msg:
                return False, [], "Invalid API key. Please check your Anthropic credentials."
            if "rate" in err_msg.lower() or "429" in err_msg or "credit" in err_msg.lower():
                return False, [], "API key is valid, but rate limit or credit quota was reached."
            return False, [], f"Could not validate Anthropic key: {err_msg}"

    def count_tokens(self, system: str, messages: list[ChatMessage]) -> int | None:
        """Return exact token count via Anthropic counting API, or None if unsupported/fails."""
        if not self.is_configured():
            return None
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        try:
            res = self._client.messages.count_tokens(
                model=self.config.model,
                system=system,
                messages=formatted_messages,
            )
            return getattr(res, "input_tokens", None)
        except Exception as e:
            logger.debug(f"Anthropic token count check failed: {e}")
            return None

    def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Execute a message completion with exponential backoff on transient errors."""
        if not self.is_configured():
            raise ApiKeyRequiredError(
                "Anthropic API key is not configured. Please enter your API key in the web page."
            )

        max_tokens = max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens
        temp = temperature if temperature is not None else self.config.temperature
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        max_retries = self.config.max_retries
        attempt = 0
        last_error: ProviderError | None = None

        while attempt <= max_retries:
            t0 = time.perf_counter()
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "system": system,
                    "messages": formatted_messages,
                    "max_tokens": max_tokens,
                }
                if temp is not None:
                    create_kwargs["extra_body"] = {"temperature": temp}

                response = self._client.messages.create(**create_kwargs)
                latency = time.perf_counter() - t0

                # Extract content
                text_parts: list[str] = []
                for block in response.content:
                    if getattr(block, "type", "") == "text":
                        text_parts.append(getattr(block, "text", ""))
                full_text = "".join(text_parts)

                # Extract usage metrics
                inp_tokens = getattr(response.usage, "input_tokens", 0)
                out_tokens = getattr(response.usage, "output_tokens", 0)
                cache_read = getattr(response.usage, "cache_read_input_tokens", None)
                cache_create = getattr(response.usage, "cache_creation_input_tokens", None)
                cost = estimate_cost(self.config.model, inp_tokens, out_tokens)

                usage = Usage(
                    input_tokens=inp_tokens,
                    output_tokens=out_tokens,
                    total_tokens=inp_tokens + out_tokens,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_create,
                    estimated_cost_usd=cost,
                )

                return CompletionResult(
                    text=full_text,
                    usage=usage,
                    model=response.model,
                    stop_reason=str(response.stop_reason or "unknown"),
                    latency_seconds=latency,
                    request_id=getattr(response, "id", None),
                )

            except Exception as exc:
                latency = time.perf_counter() - t0
                mapped_error = self._map_exception(exc)
                last_error = mapped_error

                # Only retry transient errors: RateLimit, Timeout, Unavailable
                is_transient = isinstance(
                    mapped_error,
                    (ProviderRateLimitError, ProviderTimeoutError, ProviderUnavailableError),
                )
                if not is_transient or attempt >= max_retries:
                    raise mapped_error from exc

                attempt += 1
                backoff = (0.5 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.25)
                logger.warning(
                    f"Transient failure calling Claude ({mapped_error.__class__.__name__}). "
                    f"Retrying {attempt}/{max_retries} in {backoff:.2f}s..."
                )
                time.sleep(backoff)

        raise last_error or ProviderError("Failed to complete request after retries.")
