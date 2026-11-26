"""AI provider, prompts, and completion parsing package."""

from app.ai.base_provider import (
    AIProvider,
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
from app.ai.claude_provider import ClaudeProvider
from app.ai.pricing import estimate_cost, load_pricing_table
from app.ai.prompts import build_messages, load_prompt_templates
from app.ai.provider_factory import get_provider, register_provider
from app.ai.response_parser import ParsedAnswer, ProposedChange, parse_response

__all__ = [
    "AIProvider",
    "ChatMessage",
    "CompletionResult",
    "ProviderAuthError",
    "ProviderBadRequestError",
    "ProviderConfig",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "Usage",
    "ClaudeProvider",
    "estimate_cost",
    "load_pricing_table",
    "build_messages",
    "load_prompt_templates",
    "get_provider",
    "register_provider",
    "ParsedAnswer",
    "ProposedChange",
    "parse_response",
]
