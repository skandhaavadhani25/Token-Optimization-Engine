"""Base abstractions, data models, and exceptions for AI model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ProviderError(Exception):
    """Base exception for all AI provider communication failures."""


class ProviderAuthError(ProviderError):
    """Raised when authentication fails (invalid or missing API key)."""


class ApiKeyRequiredError(ProviderAuthError):
    """Raised when an API key is missing for a cloud provider."""


class ApiKeyInvalidError(ProviderAuthError):
    """Raised when the provided API key is rejected by the provider."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider rate limit is exceeded (HTTP 429)."""


class ProviderTimeoutError(ProviderError):
    """Raised when a request to the provider times out."""


class ProviderBadRequestError(ProviderError):
    """Raised when the request payload is rejected by the provider (HTTP 400)."""


class ProviderUnavailableError(ProviderError):
    """Raised when provider servers are unavailable or return internal errors (HTTP 5xx)."""


class ProviderConfig(BaseModel):
    """Configuration settings for an AI provider instance."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="claude", description="Provider identifier (claude, openai, etc.)")
    model: str = Field(..., description="Target model name/id")
    max_output_tokens: int = Field(default=2000, ge=1, description="Maximum tokens generated")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    timeout_seconds: float = Field(default=60.0, gt=0.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Max retries for transient failures")


class ChatMessage(BaseModel):
    """A single chat message in a conversation."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"] = Field(description="Message author role")
    content: str = Field(description="Message body text")


class Usage(BaseModel):
    """Actual token consumption metrics reported by the provider API."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = Field(default=None, ge=0, description="Prompt / context tokens processed")
    output_tokens: int | None = Field(default=None, ge=0, description="Generated completion tokens")
    total_tokens: int | None = Field(default=None, ge=0, description="Sum of input and output tokens")
    usage_estimated: bool = Field(default=False, description="Whether token usage is estimated rather than reported by API")
    cache_read_tokens: int | None = Field(default=None, description="Tokens read from prompt cache")
    cache_creation_tokens: int | None = Field(default=None, description="Tokens written to prompt cache")
    estimated_cost_usd: float | None = Field(default=None, description="Estimated total cost in USD, if known")


class CompletionResult(BaseModel):
    """Standardized response from an AI model completion call."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Generated model response text")
    usage: Usage = Field(description="Actual token consumption report")
    model: str = Field(description="Exact model identifier that produced the response")
    stop_reason: str = Field(description="Reason completion stopped (e.g. end_turn, max_tokens)")
    latency_seconds: float = Field(ge=0.0, description="Round-trip request latency in seconds")
    request_id: str | None = Field(default=None, description="Provider request correlation ID")
    raw_stop: str | None = Field(default=None, description="Raw stop sequence if applicable")


class AIProvider(ABC):
    """Abstract interface for AI model completion providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return provider identifier name (e.g. 'claude')."""

    @property
    def supports_system_role(self) -> bool:
        """Return True if model natively supports a separate system role/instruction."""
        return getattr(self, "_supports_system_role", True)

    @supports_system_role.setter
    def supports_system_role(self, value: bool) -> None:
        self._supports_system_role = bool(value)

    @property
    def is_local(self) -> bool:
        """Return True if provider runs locally without external cloud API."""
        return False

    def get_setup_hint(self) -> str | None:
        """Return user-friendly configuration hint if provider is unconfigured."""
        return None

    def fold_system_into_messages(self, system: str, messages: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
        """Fold system prompt into the first user message when system role is unsupported."""
        if not system or not system.strip():
            return "", messages
        if not messages:
            return "", [ChatMessage(role="user", content=system)]
        first = messages[0]
        folded_first = ChatMessage(
            role=first.role,
            content=f"{system.strip()}\n\n---\n\n{first.content}",
        )
        return "", [folded_first] + list(messages[1:])

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Execute a completion call against the provider."""

    @abstractmethod
    def count_tokens(self, system: str, messages: list[ChatMessage]) -> int | None:
        """Return exact token count via provider counting API, or None if unsupported."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return whether provider credentials and requirements are validly configured."""

    def get_available_models(self) -> list[str]:
        """Return list of available model identifiers for this provider."""
        return []

    def validate_key(self, model: str | None = None) -> tuple[bool, list[str], str]:
        """Validate credentials using the cheapest possible call.

        Returns:
            (valid, models_list, user_friendly_message)
        """
        if not self.is_configured():
            return False, [], self.get_setup_hint() or "Provider credentials not configured."
        return True, self.get_available_models(), "Key is valid."
