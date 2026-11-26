"""Ollama local AI provider implementation via Ollama's OpenAI-compatible /v1 endpoint."""

from __future__ import annotations

import logging
import time
from typing import Any
import httpx
from openai import OpenAI

from app.ai.base_provider import (
    AIProvider,
    ChatMessage,
    CompletionResult,
    ProviderBadRequestError,
    ProviderError,
    ProviderUnavailableError,
    Usage,
)

logger = logging.getLogger(__name__)


class OllamaProvider(AIProvider):
    """Local model provider for Ollama instances with reachability check and explicit num_ctx."""

    def __init__(
        self,
        config: Any,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        context_token_budget: int = 4000,
        client: Any | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.config = config
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.num_ctx = num_ctx
        self.context_token_budget = context_token_budget
        self._http_client = http_client

        if client is not None:
            self._client = client
        else:
            self._client = OpenAI(
                api_key="ollama",  # Dummy key required by OpenAI client
                base_url=f"{self.base_url}/v1",
                timeout=self.config.timeout_seconds,
                max_retries=0,
            )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def supports_system_role(self) -> bool:
        return True

    @property
    def is_local(self) -> bool:
        return True

    def check_status(self) -> tuple[bool, str | None, list[str]]:
        """Check server reachability and whether target model has been pulled."""
        # Brief 2s cache to prevent repeated slow timeouts within a single request
        now = time.time()
        cached = getattr(self, "_cached_status", None)
        if cached and (now - cached[0] < 2.0):
            return cached[1]

        tags_url = f"{self.base_url}/api/tags"
        try:
            if self._http_client is not None:
                resp = self._http_client.get(tags_url)
            else:
                resp = httpx.get(tags_url, timeout=0.8)
            if resp.status_code != 200:
                res = (
                    False,
                    f"Ollama server at {self.base_url} returned HTTP {resp.status_code}. Ensure Ollama is running.",
                    [],
                )
                self._cached_status = (now, res)
                return res
            data = resp.json()
            models = [
                m.get("name")
                for m in data.get("models", [])
                if isinstance(m, dict) and m.get("name")
            ]
        except Exception:
            res = (
                False,
                f"Ollama server is not running at {self.base_url}. Start Ollama (e.g. 'ollama serve') and retry.",
                [],
            )
            self._cached_status = (now, res)
            return res

        if not self.config.model:
            res = (False, "Set OLLAMA_MODEL in .env, for example 'llama3.2'", models)
            self._cached_status = (now, res)
            return res

        target = self.config.model.lower()
        # Match e.g. "llama3.2" against "llama3.2:latest" or exact match
        found = any(
            target == m.lower() or f"{target}:latest" == m.lower() or m.lower().startswith(f"{target}:")
            for m in models
        )
        if not found:
            res = (
                False,
                f"Model '{self.config.model}' is not pulled in Ollama. Run 'ollama pull {self.config.model}' to download it.",
                models,
            )
            self._cached_status = (now, res)
            return res

        res = (True, None, models)
        self._cached_status = (now, res)
        return res

    def is_configured(self) -> bool:
        """Return True only if Ollama server is reachable and requested model is available."""
        configured, _, _ = self.check_status()
        return configured

    def get_setup_hint(self) -> str | None:
        """Return actionable setup instructions if server or model is not ready."""
        _, hint, _ = self.check_status()
        return hint

    def get_available_models(self) -> list[str]:
        """Return list of locally downloaded models if server is reachable."""
        _, _, models = self.check_status()
        return models

    def validate_key(self, model: str | None = None) -> tuple[bool, list[str], str]:
        """Validate Ollama reachability and available models (no API key needed)."""
        ok, hint, models = self.check_status()
        if ok:
            return True, models, "Ollama service is reachable."
        return False, models, hint or "Ollama service is unreachable."

    def __repr__(self) -> str:
        return f"<OllamaProvider model='{self.config.model}' base_url='{self.base_url}' num_ctx={self.num_ctx}>"

    def __str__(self) -> str:
        return self.__repr__()

    def count_tokens(self, system: str, messages: list[ChatMessage]) -> int | None:
        """Token counting not supported natively via /v1 endpoint."""
        return None

    def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Execute chat completion against Ollama /v1 passing explicit num_ctx."""
        # 1. Verify reachability and model availability
        ok, hint, _ = self.check_status()
        if not ok:
            if "not running" in (hint or ""):
                raise ProviderUnavailableError(hint or "Ollama is not running.")
            raise ProviderBadRequestError(hint or f"Model '{self.config.model}' not available.")

        # 2. Context window check
        max_tokens = max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens
        if (self.context_token_budget + max_tokens) > self.num_ctx:
            logger.warning(
                f"Context budget ({self.context_token_budget}) plus output limit ({max_tokens}) "
                f"exceeds Ollama num_ctx ({self.num_ctx}). Input may be truncated by Ollama."
            )

        # 3. Format messages
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

        temp = temperature if temperature is not None else self.config.temperature

        t0 = time.perf_counter()
        try:
            # Pass num_ctx explicitly in options to prevent Ollama from silently truncating context
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=payload_messages,
                temperature=temp,
                max_tokens=max_tokens,
                extra_body={"options": {"num_ctx": self.num_ctx}},
            )
            latency = time.perf_counter() - t0
        except Exception as exc:
            latency = time.perf_counter() - t0
            err_str = str(exc)
            if "connection" in err_str.lower() or "connect" in err_str.lower():
                raise ProviderUnavailableError(f"Ollama server disconnected: {err_str}")
            if "not found" in err_str.lower():
                raise ProviderBadRequestError(
                    f"Model '{self.config.model}' not found in Ollama: {err_str}"
                )
            raise ProviderError(f"Ollama completion error: {err_str}")

        choice = response.choices[0] if getattr(response, "choices", None) else None
        if not choice:
            raise ProviderBadRequestError("No completion choices returned by Ollama.")

        msg_obj = getattr(choice, "message", None)
        text = getattr(msg_obj, "content", "") or ""
        stop_reason = str(getattr(choice, "finish_reason", "stop") or "stop")

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
                estimated_cost_usd=0.0,  # Local models are free
            )
        else:
            usage = Usage(
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                usage_estimated=True,
                estimated_cost_usd=0.0,
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
