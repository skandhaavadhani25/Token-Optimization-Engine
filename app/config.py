"""Configuration management for code-context-engine using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Anthropic / Claude Provider
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Anthropic Claude. Kept secret and never exposed or logged.",
    )

    # Google Gemini Provider
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Google Gemini. Kept secret and never exposed or logged.",
    )

    # OpenAI-Compatible Provider (Groq, OpenRouter, Together, etc.)
    openai_compat_api_key: SecretStr | None = Field(
        default=None,
        description="API key for OpenAI-compatible endpoint. Kept secret and never exposed or logged.",
    )
    openai_compat_base_url: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible endpoint (e.g. https://api.groq.com/openai/v1).",
    )

    # Ollama Provider (Local)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama local service.",
    )
    ollama_num_ctx: int = Field(
        default=4096,
        description="Context window token limit (num_ctx) passed explicitly to Ollama.",
    )

    # AI Provider Selection & Per-Provider Default Models
    ai_provider: str = Field(
        default="claude",
        description="Default AI provider ('claude', 'gemini', 'openai_compat', 'ollama').",
    )
    claude_model: str | None = Field(
        default=None,
        description="Model ID for Claude (e.g. 'claude-3-5-sonnet-20241022').",
    )
    gemini_model: str | None = Field(
        default=None,
        description="Model ID for Gemini (e.g. 'gemini-2.5-flash').",
    )
    openai_compat_model: str | None = Field(
        default=None,
        description="Model ID for OpenAI-compatible endpoint (e.g. 'llama-3.3-70b-versatile').",
    )
    ollama_model: str | None = Field(
        default=None,
        description="Model ID for Ollama local model (e.g. 'llama3.2').",
    )
    ai_model: str | None = Field(
        default=None,
        description="Backward-compatible alias for the active provider's model ID.",
    )

    # Prompt style: 'full' (default) or 'simple' (concise prompt for smaller/local models)
    prompt_style: str = Field(
        default="full",
        description="Prompt style: 'full' (default) or 'simple' for small models.",
    )
    ai_max_output_tokens: int = Field(
        default=2000,
        description="Maximum output tokens for AI completion response.",
    )
    ai_temperature: float = Field(
        default=0.2,
        description="Sampling temperature for the AI completion model.",
    )
    ai_timeout_seconds: float = Field(
        default=60.0,
        description="HTTP request timeout in seconds for AI completions.",
    )
    ai_max_retries: int = Field(
        default=3,
        description="Maximum exponential backoff retries for transient provider failures.",
    )

    # Retrieval and Context Budget
    context_token_budget: int = Field(
        default=4000,
        description="Default token budget for context assembly.",
    )
    max_query_chars: int = Field(
        default=4000,
        description="Maximum allowed character length for natural language user queries.",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for semantic embeddings.",
    )

    # Workspaces & Upload Limits
    max_upload_mb: int = Field(
        default=50,
        description="Maximum allowed upload archive size in megabytes.",
    )
    max_files: int = Field(
        default=20000,
        description="Maximum allowed files in an uploaded workspace archive.",
    )
    max_attachment_mb: int = Field(
        default=5,
        description="Maximum allowed size per individual chat attachment in megabytes.",
    )
    max_attachments_per_message: int = Field(
        default=5,
        description="Maximum attachments permitted in a single message turn.",
    )
    workspaces_dir: Path = Field(
        default=Path("data/workspaces"),
        description="Local directory where working copies of workspaces reside.",
    )

    # Persistence (MongoDB)
    mongodb_uri: SecretStr | None = Field(
        default=None,
        description="MongoDB connection URI (e.g. mongodb://localhost:27017). "
        "When unset, an in-memory MongoDB substitute is used and data is not persisted.",
    )
    mongodb_db: str = Field(
        default="code_context_engine",
        description="MongoDB database name holding the index, graph, embeddings, chats, and workspaces.",
    )

    # Security & Path Restrictions
    allowed_repo_roots: str | None = Field(
        default=None,
        description="Comma-separated list of allowed repository roots. If set, paths outside are rejected.",
    )

    # Server configuration
    api_host: str = Field(
        default="127.0.0.1",
        description="FastAPI service bind host.",
    )
    api_port: int = Field(
        default=8000,
        description="FastAPI service bind port.",
    )
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
        description="Comma-separated allowed CORS origins.",
    )

    def get_allowed_roots(self) -> list[Path] | None:
        """Parse allowed_repo_roots into resolved Path objects if configured."""
        if not self.allowed_repo_roots:
            return None
        roots = []
        for r in self.allowed_repo_roots.split(","):
            cleaned = r.strip()
            if cleaned:
                roots.append(Path(cleaned).resolve())
        return roots or None

    def get_cors_origins_list(self) -> list[str]:
        """Parse cors_origins into a list of strings."""
        if not self.cors_origins:
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def get_model_for_provider(self, provider: str, raise_if_missing: bool = True) -> str | None:
        """Resolve the model ID for a specific provider, or fail with a clear message."""
        p = provider.lower()
        model: str | None = None
        if p == "claude":
            model = self.claude_model or self.ai_model
            if not model and raise_if_missing:
                raise ValueError(
                    "Model not configured for provider 'claude'. Please set CLAUDE_MODEL in .env, for example 'claude-3-5-sonnet-20241022'"
                )
        elif p == "gemini":
            model = self.gemini_model or (self.ai_model if self.ai_provider == "gemini" else None)
            if not model and raise_if_missing:
                raise ValueError(
                    "Model not configured for provider 'gemini'. Please set GEMINI_MODEL in .env, for example 'gemini-2.5-flash'"
                )
        elif p in ("openai_compat", "openai"):
            model = self.openai_compat_model or (self.ai_model if self.ai_provider in ("openai_compat", "openai") else None)
            if not model and raise_if_missing:
                raise ValueError(
                    "Model not configured for provider 'openai_compat'. Please set OPENAI_COMPAT_MODEL in .env, for example 'llama-3.3-70b-versatile'"
                )
        elif p in ("ollama", "local"):
            model = self.ollama_model or (self.ai_model if self.ai_provider in ("ollama", "local") else None)
            if not model and raise_if_missing:
                raise ValueError(
                    "Model not configured for provider 'ollama'. Please set OLLAMA_MODEL in .env, for example 'llama3.2'"
                )
        else:
            model = self.ai_model
            if not model and raise_if_missing:
                raise ValueError(
                    f"Model not configured for provider '{provider}'. Please set AI_MODEL in .env"
                )
        return model

    def get_active_model(self, raise_if_missing: bool = False) -> str | None:
        """Return the active model ID for the currently selected ai_provider."""
        return self.get_model_for_provider(self.ai_provider, raise_if_missing=raise_if_missing)

    def get_key_for_provider(self, provider: str) -> SecretStr | None:
        """Return the server-configured API key for a specific provider if set."""
        p = provider.lower()
        if p == "claude":
            return self.anthropic_api_key
        elif p == "gemini":
            return self.gemini_api_key
        elif p in ("openai_compat", "openai"):
            return self.openai_compat_api_key
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
