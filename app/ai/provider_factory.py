"""Factory for creating and registering AI provider instances."""

from __future__ import annotations

from typing import Type
from pydantic import SecretStr

from app.ai.base_provider import AIProvider, ProviderConfig
from app.ai.claude_provider import ClaudeProvider
from app.ai.gemini_provider import GeminiProvider
from app.ai.openai_compat_provider import OpenAICompatProvider
from app.ai.ollama_provider import OllamaProvider
from app.config import Settings, get_settings


_REGISTRY: dict[str, Type[AIProvider]] = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "openai_compat": OpenAICompatProvider,
    "ollama": OllamaProvider,
    # Aliases
    "openai": OpenAICompatProvider,
    "local": OllamaProvider,
}


def register_provider(name: str, provider_cls: Type[AIProvider]) -> None:
    """Register a new AIProvider class under a given name identifier."""
    _REGISTRY[name.lower()] = provider_cls


def get_registered_providers() -> list[str]:
    """Return distinct canonical provider names."""
    return ["claude", "gemini", "openai_compat", "ollama"]


def get_provider(
    provider_name: str | None = None,
    model: str | None = None,
    settings: Settings | None = None,
    config: ProviderConfig | None = None,
    api_key: str | SecretStr | None = None,
    raise_if_model_missing: bool = True,
) -> AIProvider:
    """Instantiate and configure an AIProvider based on settings or explicit overrides."""
    cfg_settings = settings or get_settings()
    raw_name = (provider_name or cfg_settings.ai_provider).lower()

    if raw_name not in _REGISTRY:
        raise ValueError(
            f"Unknown AI provider '{raw_name}'. Registered providers: {sorted(list(_REGISTRY.keys()))}"
        )

    provider_cls = _REGISTRY[raw_name]
    canonical_name = "openai_compat" if raw_name == "openai" else ("ollama" if raw_name == "local" else raw_name)

    if config is not None:
        provider_config = config
    else:
        resolved_model = model or cfg_settings.get_model_for_provider(
            canonical_name,
            raise_if_missing=raise_if_model_missing,
        )
        provider_config = ProviderConfig(
            provider=canonical_name,
            model=resolved_model or "",
            max_output_tokens=cfg_settings.ai_max_output_tokens,
            temperature=cfg_settings.ai_temperature,
            timeout_seconds=cfg_settings.ai_timeout_seconds,
            max_retries=cfg_settings.ai_max_retries,
        )

    if provider_cls is ClaudeProvider:
        key = api_key if api_key is not None else cfg_settings.anthropic_api_key
        return ClaudeProvider(config=provider_config, api_key=key)

    if provider_cls is GeminiProvider:
        key = api_key if api_key is not None else cfg_settings.gemini_api_key
        return GeminiProvider(config=provider_config, api_key=key)

    if provider_cls is OpenAICompatProvider:
        key = api_key if api_key is not None else cfg_settings.openai_compat_api_key
        return OpenAICompatProvider(
            config=provider_config,
            api_key=key,
            base_url=cfg_settings.openai_compat_base_url,
        )

    if provider_cls is OllamaProvider:
        return OllamaProvider(
            config=provider_config,
            base_url=cfg_settings.ollama_base_url,
            num_ctx=cfg_settings.ollama_num_ctx,
            context_token_budget=cfg_settings.context_token_budget,
        )

    return provider_cls(config=provider_config)
