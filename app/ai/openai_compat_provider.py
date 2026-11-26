"""OpenAI-compatible AI provider implementation using the official openai SDK."""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any
from pydantic import SecretStr

import openai
from openai import OpenAI

from app.ai.base_provider import (
    AIProvider,
    ApiKeyInvalidError,
    ApiKeyRequiredError,
    ChatMessage,
    CompletionResult,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from app.ai.pricing import estimate_cost

logger = logging.getLogger(__name__)


def _sanitize_msg(msg: str) -> str:
    """Mask potential secret tokens and authorization headers from error messages."""
    cleaned = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]{10,}", "Bearer ***", str(msg), flags=re.IGNORECASE)
    cleaned = re.sub(r"sk-[A-Za-z0-9_\-\.]{15,}", "sk-***", cleaned)
    return cleaned


class OpenAICompatProvider(AIProvider):
    """Provider for any OpenAI-compatible completions endpoint (Groq, OpenRouter, Together, etc.)."""

    def __init__(
        self,
        config: Any,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.base_url = base_url.rstrip("/") if base_url else None

        self._raw_key: str | None = None
        if api_key is not None:
            if isinstance(api_key, SecretStr):
                self._raw_key = api_key.get_secret_value()
            else:
                self._raw_key = str(api_key)

        if client is not None:
            self._client = client
        elif self._raw_key or self.base_url:
            self._client = OpenAI(
                api_key=self._raw_key or "dummy-key",
                base_url=self.base_url,
                timeout=self.config.timeout_seconds,
                max_retries=0,  # We manage retries and backoff explicitly
            )
        else:
            self._client = None

    @property
    def name(self) -> str:
        return "openai_compat"

    @property
    def supports_system_role(self) -> bool:
        return True

    @property
    def is_local(self) -> bool:
        return False

    def is_configured(self) -> bool:
        return self._client is not None and bool(self._raw_key or self.base_url)

    def get_setup_hint(self) -> str | None:
        if not self.is_configured():
            return "Set OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_BASE_URL in .env"
        return None

    def __repr__(self) -> str:
        return f"<OpenAICompatProvider model='{self.config.model}' base_url='{self.base_url}' configured={self.is_configured()}>"

    def __str__(self) -> str:
        return self.__repr__()

    def _map_exception(self, err: Exception) -> ProviderError:
        """Map openai SDK exceptions to standardized ProviderError subclasses."""
        err_str = _sanitize_msg(str(err))

        if isinstance(err, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return ApiKeyInvalidError(f"Authentication failed on {self.name}: {err_str}")

        if isinstance(err, openai.NotFoundError):
            return ProviderBadRequestError(
                f"Model '{self.config.model}' not found on provider endpoint ({self.base_url or 'default'}): {err_str}"
            )

        if isinstance(err, openai.RateLimitError):
            return ProviderRateLimitError(
                f"Rate limit exceeded on {self.name} ({self.config.model}): {err_str}"
            )

        if isinstance(err, openai.APITimeoutError):
            return ProviderTimeoutError(f"Request timed out on {self.name}: {err_str}")

        if isinstance(err, openai.BadRequestError):
            return ProviderBadRequestError(f"Bad request on {self.name}: {err_str}")

        if isinstance(err, (openai.APIConnectionError, openai.InternalServerError)):
            return ProviderUnavailableError(f"Service unavailable on {self.name}: {err_str}")

        if isinstance(err, openai.APIStatusError):
            status = err.status_code
            if status in (401, 403):
                return ApiKeyInvalidError(f"Authentication failed ({status}): {err_str}")
            if status == 404:
                return ProviderBadRequestError(
                    f"Model '{self.config.model}' not found on endpoint ({self.base_url or 'default'}): {err_str}"
                )
            if status == 429:
                return ProviderRateLimitError(f"Rate limit exceeded ({status}): {err_str}")
            if status >= 500:
                return ProviderUnavailableError(f"Server error ({status}) on {self.name}: {err_str}")
            return ProviderBadRequestError(f"Provider API error ({status}): {err_str}")

        return ProviderError(f"Unexpected provider error on {self.name}: {err_str}")

    def count_tokens(self, system: str, messages: list[ChatMessage]) -> int | None:
        """Return None because standard OpenAI endpoints do not provide a count-tokens endpoint."""
        return None

    def _extract_retry_after(self, exc: Exception) -> float | None:
        """Extract Retry-After header or delay hint if present in response or error."""
        # 1. Check HTTP response headers if available
        resp = getattr(exc, "response", None)
        if resp is not None and hasattr(resp, "headers"):
            ra_hdr = resp.headers.get("retry-after")
            if ra_hdr:
                try:
                    return float(ra_hdr)
                except (ValueError, TypeError):
                    pass
        # 2. Check regex in error message
        m = re.search(r"retry after (\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    def get_available_models(self) -> list[str]:
        """Return available models for this endpoint if supported."""
        if not self.is_configured():
            return [self.config.model] if self.config.model else []
        try:
            if hasattr(self._client, "models") and hasattr(self._client.models, "list"):
                res = self._client.models.list()
                models = [m.id for m in getattr(res, "data", []) if getattr(m, "id", None)]
                if models:
                    return models
        except Exception:
            pass
        return [self.config.model] if self.config.model else []

    def validate_key(self, model: str | None = None) -> tuple[bool, list[str], str]:
        """Validate key using cheapest call (models.list or 1-token completion)."""
        if not self.is_configured():
            return False, [], f"No API key provided. Please enter your {self.name} API key."
        target_model = model or self.config.model
        try:
            if hasattr(self._client, "models") and hasattr(self._client.models, "list"):
                res = self._client.models.list()
                models = [m.id for m in getattr(res, "data", []) if getattr(m, "id", None)]
                if models:
                    return True, models, "Key is valid."
        except (openai.AuthenticationError, openai.PermissionDeniedError):
            return False, [], "Invalid API key. Please check your credentials."
        except openai.RateLimitError:
            return False, [], "API key is valid, but rate limit or quota has been reached."
        except Exception:
            pass

        if target_model:
            try:
                self._client.chat.completions.create(
                    model=target_model,
                    max_tokens=1,
                    messages=[{"role": "user", "content": "hi"}],
                )
                return True, [target_model], "Key is valid."
            except (openai.AuthenticationError, openai.PermissionDeniedError):
                return False, [], "Invalid API key. Please check your credentials."
            except openai.RateLimitError:
                return False, [], "API key is valid, but rate limit or quota has been reached."
            except Exception as exc:
                err_msg = _sanitize_msg(str(exc))
                if "auth" in err_msg.lower() or "401" in err_msg or "403" in err_msg:
                    return False, [], "Invalid API key. Please check your credentials."
                if "rate" in err_msg.lower() or "429" in err_msg:
                    return False, [], "API key is valid, but rate limit or quota has been reached."
                return False, [], f"Could not validate key on endpoint: {err_msg}"

        return True, self.get_available_models(), "Key configured."

    def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Execute chat completion request with exponential backoff and usage parsing."""
        if not self.is_configured():
            raise ApiKeyRequiredError(
                f"API key required for {self.name}. Please enter your API key in the web page."
            )

        # Assemble messages payload
        payload_messages: list[dict[str, str]] = []
        if self.supports_system_role:
            if system and system.strip():
                payload_messages.append({"role": "system", "content": system.strip()})
            for m in messages:
                payload_messages.append({"role": m.role, "content": m.content})
        else:
            _, folded_messages = self.fold_system_into_messages(system, messages)
            for m in folded_messages:
                payload_messages.append({"role": m.role, "content": m.content})

        max_tokens = max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens
        temp = temperature if temperature is not None else self.config.temperature

        retries = 0
        backoff = 1.0

        extra_args: dict[str, Any] = {}

        while True:
            t0 = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=payload_messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    **extra_args,
                )
                latency = time.perf_counter() - t0
                break
            except Exception as exc:
                latency = time.perf_counter() - t0
                mapped_err = self._map_exception(exc)

                is_transient = isinstance(mapped_err, (ProviderRateLimitError, ProviderUnavailableError, ProviderTimeoutError))
                if is_transient and retries < self.config.max_retries:
                    retry_after = self._extract_retry_after(exc)
                    sleep_time = retry_after if retry_after is not None else (backoff + random.uniform(0.1, 0.5))

                    logger.warning(
                        f"Transient error on {self.name}: {mapped_err}. Retrying in {sleep_time:.2f}s (attempt {retries + 1}/{self.config.max_retries})"
                    )
                    time.sleep(sleep_time)
                    retries += 1
                    backoff *= 2.0
                    continue

                logger.error(f"{self.name} completion failed after {retries} retries: {mapped_err}")
                raise mapped_err

        # Extract text & finish reason
        choice = response.choices[0] if getattr(response, "choices", None) else None
        if not choice:
            raise ProviderBadRequestError(f"No choices returned by {self.name}.")

        msg_obj = getattr(choice, "message", None)
        text = getattr(msg_obj, "content", "") or ""
        stop_reason = str(getattr(choice, "finish_reason", "stop") or "stop")

        # Extract usage metadata
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None and getattr(raw_usage, "prompt_tokens", None) is not None:
            in_tok = raw_usage.prompt_tokens
            out_tok = getattr(raw_usage, "completion_tokens", 0) or 0
            tot_tok = getattr(raw_usage, "total_tokens", in_tok + out_tok) or (in_tok + out_tok)
            usage = Usage(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=tot_tok,
                usage_estimated=False,
                estimated_cost_usd=estimate_cost(self.config.model, in_tok, out_tok),
            )
        else:
            usage = Usage(
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                usage_estimated=True,
                estimated_cost_usd=None,
            )

        raw_model = getattr(response, "model", None)
        model_str = raw_model if isinstance(raw_model, str) and raw_model else self.config.model
        raw_req_id = getattr(response, "id", None)
        req_id = raw_req_id if isinstance(raw_req_id, str) else None

        return CompletionResult(
            text=text,
            usage=usage,
            model=model_str,
            stop_reason=stop_reason.lower(),
            latency_seconds=round(latency, 4),
            request_id=req_id,
            raw_stop=stop_reason,
        )
