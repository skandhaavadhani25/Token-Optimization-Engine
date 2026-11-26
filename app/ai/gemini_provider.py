"""Google Gemini provider implementation using the official google-genai SDK."""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any
from pydantic import SecretStr

from google import genai
from google.genai import errors, types

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
    """Mask any potential API key tokens from error messages."""
    return re.sub(r"AIza[0-9A-Za-z-_]{35}", "AIza***", str(msg))


def _model_supports_system_instruction(model_name: str | None) -> bool:
    """Check if model is known to support native developer/system instructions."""
    if not model_name:
        return True
    m = model_name.lower().replace("models/", "")
    # Known models that do NOT support developer instruction in Gemini API:
    if "gemini-1.0" in m or m in ("gemini-pro", "gemini-pro-vision"):
        return False
    if "gemma" in m:
        return False
    return True


class GeminiProvider(AIProvider):
    """Google Gemini completion provider using current official google-genai SDK."""

    def __init__(
        self,
        config: Any,
        api_key: str | SecretStr | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._raw_key: str | None = None
        if api_key is not None:
            if isinstance(api_key, SecretStr):
                self._raw_key = api_key.get_secret_value()
            else:
                self._raw_key = str(api_key)

        if client is not None:
            self._client = client
        elif self._raw_key:
            self._client = genai.Client(
                api_key=self._raw_key,
                http_options=types.HttpOptions(timeout=int(self.config.timeout_seconds * 1000)),
            )
        else:
            self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def supports_system_role(self) -> bool:
        return _model_supports_system_instruction(self.config.model)

    @property
    def is_local(self) -> bool:
        return False

    def is_configured(self) -> bool:
        return self._client is not None

    def get_setup_hint(self) -> str | None:
        if not self.is_configured():
            return "Set GEMINI_API_KEY in .env (get a free key at https://aistudio.google.com/)"
        return None

    def __repr__(self) -> str:
        return f"<GeminiProvider model='{self.config.model}' configured={self.is_configured()}>"

    def __str__(self) -> str:
        return self.__repr__()

    def _convert_contents(self, messages: list[ChatMessage]) -> list[types.Content]:
        """Map generic ChatMessage list to Gemini Content objects."""
        contents: list[types.Content] = []
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m.content)],
                )
            )
        return contents

    def _map_exception(self, err: Exception) -> ProviderError:
        """Map google-genai exceptions to standardized ProviderError subclasses."""
        err_str = _sanitize_msg(str(err))
        status_code = getattr(err, "code", None)

        if isinstance(err, errors.APIError):
            status_code = err.code

        # Rate limits & free tier quota exhaustion (HTTP 429 or RESOURCE_EXHAUSTED)
        if status_code == 429 or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            return ProviderRateLimitError(
                f"Rate limit or free-tier quota reached for Gemini ({self.config.model}). "
                f"This is likely a free-tier rate or daily quota limit. Please wait and retry, or switch provider: {err_str}"
            )

        if status_code in (401, 403) or "API_KEY_INVALID" in err_str or "PERMISSION_DENIED" in err_str or "UNAUTHENTICATED" in err_str:
            return ApiKeyInvalidError(f"Gemini authentication failed: {err_str}")

        if status_code == 404 or "NOT_FOUND" in err_str:
            return ProviderBadRequestError(f"Gemini model '{self.config.model}' not found: {err_str}")

        if status_code == 400 or "INVALID_ARGUMENT" in err_str:
            return ProviderBadRequestError(f"Gemini bad request: {err_str}")

        if status_code and status_code >= 500:
            return ProviderUnavailableError(f"Gemini service unavailable ({status_code}): {err_str}")

        if "timeout" in err_str.lower() or "deadline" in err_str.lower():
            return ProviderTimeoutError(f"Gemini request timed out: {err_str}")

        return ProviderError(f"Gemini provider error: {err_str}")

    def get_available_models(self) -> list[str]:
        """Return available models for Gemini."""
        default_models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
        if self.config.model and self.config.model not in default_models:
            default_models.insert(0, self.config.model)
        if not self.is_configured():
            return default_models
        try:
            if hasattr(self._client, "models") and hasattr(self._client.models, "list"):
                results = []
                for m in self._client.models.list():
                    m_name = getattr(m, "name", "")
                    if "gemini" in m_name.lower():
                        clean_name = m_name.replace("models/", "")
                        results.append(clean_name)
                if results:
                    return results
        except Exception:
            pass
        return default_models

    def validate_key(self, model: str | None = None) -> tuple[bool, list[str], str]:
        """Validate Gemini API key using cheapest possible call."""
        if not self.is_configured():
            return False, [], "No API key provided. Please enter your Gemini API key."
        try:
            models = self.get_available_models()
            return True, models, "Key is valid."
        except errors.ClientError as exc:
            code = getattr(exc, "code", 400)
            status = getattr(exc, "status", "")
            msg = str(exc)
            if code in (401, 403) or status in ("PERMISSION_DENIED", "UNAUTHENTICATED") or "API_KEY_INVALID" in msg:
                return False, [], "Invalid API key. Please check your Google Gemini credentials."
            if code == 429 or status == "RESOURCE_EXHAUSTED":
                return False, [], "API key is valid, but rate limit or quota has been reached."
            return False, [], f"Gemini validation error: {_sanitize_msg(msg)}"
        except Exception as exc:
            err_msg = str(exc)
            if "auth" in err_msg.lower() or "permission" in err_msg.lower() or "401" in err_msg or "403" in err_msg:
                return False, [], "Invalid API key. Please check your Google Gemini credentials."
            if "quota" in err_msg.lower() or "rate" in err_msg.lower() or "429" in err_msg:
                return False, [], "API key is valid, but rate limit or quota has been reached."
            return False, [], f"Could not validate Gemini key: {_sanitize_msg(err_msg)}"

    def count_tokens(self, system: str, messages: list[ChatMessage]) -> int | None:
        """Count tokens using Gemini's SDK count-tokens endpoint."""
        if not self.is_configured():
            return None
        try:
            # Safely fold system instructions into messages for token counting
            # to remain universally compatible across models with or without developer instructions
            if system and system.strip():
                _, folded = self.fold_system_into_messages(system, messages)
                contents = self._convert_contents(folded)
            else:
                contents = self._convert_contents(messages)

            res = self._client.models.count_tokens(
                model=self.config.model,
                contents=contents,
            )
            return getattr(res, "total_tokens", None)
        except Exception:
            return None

    def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Invoke Gemini generate_content with retry, error mapping, and usage parsing."""
        if not self.is_configured():
            raise ApiKeyRequiredError(
                "Gemini API key is not configured. Please enter your API key in the web page."
            )

        max_tokens = max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens
        temp = temperature if temperature is not None else self.config.temperature

        # Determine whether the model supports developer instructions natively
        use_system_instruction = bool(system and system.strip() and _model_supports_system_instruction(self.config.model))

        if use_system_instruction:
            contents = self._convert_contents(messages)
            gen_config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=temp,
                max_output_tokens=max_tokens,
            )
        else:
            # Safely incorporate application instructions directly into the user prompt
            _, folded_messages = self.fold_system_into_messages(system, messages)
            contents = self._convert_contents(folded_messages)
            gen_config = types.GenerateContentConfig(
                system_instruction=None,
                temperature=temp,
                max_output_tokens=max_tokens,
            )

        retries = 0
        backoff = 1.0

        while True:
            t0 = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=gen_config,
                )
                latency = time.perf_counter() - t0
                break
            except Exception as exc:
                latency = time.perf_counter() - t0
                err_str = str(exc)

                # If developer instructions are rejected by the API for this model,
                # immediately retry with instructions safely folded into the user prompt
                if use_system_instruction and "developer instruction" in err_str.lower():
                    logger.warning(
                        f"Gemini model '{self.config.model}' does not support developer instructions ({_sanitize_msg(err_str)}). "
                        "Safely incorporating application instructions into user prompt and retrying."
                    )
                    use_system_instruction = False
                    _, folded_messages = self.fold_system_into_messages(system, messages)
                    contents = self._convert_contents(folded_messages)
                    gen_config = types.GenerateContentConfig(
                        system_instruction=None,
                        temperature=temp,
                        max_output_tokens=max_tokens,
                    )
                    continue

                mapped_err = self._map_exception(exc)

                # Retry transient failures (rate limits and 5xx)
                is_transient = isinstance(mapped_err, (ProviderRateLimitError, ProviderUnavailableError, ProviderTimeoutError))
                if is_transient and retries < self.config.max_retries:
                    # Check for explicit retry-after hint if mentioned in string
                    retry_match = re.search(r"retry after (\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
                    if retry_match:
                        sleep_time = float(retry_match.group(1))
                    else:
                        sleep_time = backoff + random.uniform(0.1, 0.5)

                    logger.warning(
                        f"Gemini call failed with transient error: {mapped_err}. Retrying in {sleep_time:.2f}s (attempt {retries + 1}/{self.config.max_retries})"
                    )
                    time.sleep(sleep_time)
                    retries += 1
                    backoff *= 2.0
                    continue

                logger.error(f"Gemini completion failed after {retries} retries ({latency:.2f}s): {mapped_err}")
                raise mapped_err

        # Check for empty response or safety blocks
        candidates = getattr(response, "candidates", None)
        if not candidates or len(candidates) == 0:
            prompt_fb = getattr(response, "prompt_feedback", None)
            raise ProviderBadRequestError(
                f"Gemini returned no candidates. Prompt feedback: {prompt_fb}"
            )

        candidate = candidates[0]
        raw_finish = getattr(candidate, "finish_reason", "STOP")
        finish_reason = str(raw_finish)

        if finish_reason in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
            raise ProviderBadRequestError(
                f"Gemini generation was blocked by safety policy (finish_reason={finish_reason})."
            )

        # Extract text safely
        text = None
        try:
            text = getattr(response, "text", None)
        except Exception:
            text = None

        if text is None:
            # Fallback to part inspection
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
            text_parts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
            text = "".join(text_parts)

        if not text or not text.strip():
            text = f"[Gemini completed with finish_reason={finish_reason} but produced empty text]"

        # Extract usage metadata
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None and getattr(usage_meta, "prompt_token_count", None) is not None:
            in_tok = usage_meta.prompt_token_count
            out_tok = usage_meta.candidates_token_count or 0
            tot_tok = usage_meta.total_token_count or (in_tok + out_tok)
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

        raw_model = getattr(response, "model_version", None)
        model_str = raw_model if isinstance(raw_model, str) and raw_model else self.config.model
        raw_req_id = getattr(response, "response_id", None)
        req_id = raw_req_id if isinstance(raw_req_id, str) else None

        return CompletionResult(
            text=text,
            usage=usage,
            model=model_str,
            stop_reason=finish_reason.lower(),
            latency_seconds=round(latency, 4),
            request_id=req_id,
            raw_stop=finish_reason,
        )
