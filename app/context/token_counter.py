"""Token counting abstractions with tiktoken and heuristic fallbacks."""

from __future__ import annotations

import hashlib
import math
import warnings
from abc import ABC, abstractmethod
from typing import Any


class TokenCounter(ABC):
    """Abstract interface for counting tokens in text strings."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifying the tokenizer/counter."""
        ...

    @abstractmethod
    def count(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text: Input string to tokenize.

        Returns:
            Non-negative integer token count.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Alias for count(text)."""
        return self.count(text)


class TiktokenCounter(TokenCounter):
    """Token counter using OpenAI's tiktoken library with cl100k_base encoding.

    Note on Tokenizer Accuracy:
    Tiktoken's `cl100k_base` encoding is used as a fast, robust local approximation
    for token budgeting and comparison. Note that Anthropic Claude models use a different
    byte-pair encoding vocabulary; tiktoken counts serve as an effective proxy (typically
    within 5-10% of Claude counts) for budgeting and compression decisions. Step 9
    will introduce exact provider-specific token counts via official API measurement.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._encoding: Any | None = None
        self._cache: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "tiktoken"

    def _ensure_encoding(self) -> Any:
        if self._encoding is None:
            import tiktoken

            try:
                self._encoding = tiktoken.get_encoding(self._encoding_name)
            except Exception as exc:  # noqa: BLE001 - offline / blocked BPE download
                # tiktoken downloads the BPE ranks on first use. When that fails
                # (offline machine, firewalled CI) fall back to the char heuristic
                # instead of crashing every request that needs a token count.
                warnings.warn(
                    f"tiktoken encoding '{self._encoding_name}' unavailable ({type(exc).__name__}); "
                    "falling back to the len/4 char estimate.",
                    UserWarning,
                    stacklevel=2,
                )
                self._encoding = _CharEncoding()
                self._fallback = True
        return self._encoding

    @property
    def is_estimate(self) -> bool:
        """True when counts come from the char heuristic instead of real BPE."""
        self._ensure_encoding()
        return getattr(self, "_fallback", False)

    def count(self, text: str) -> int:
        """Count tokens with internal MD5 hash caching for fast duplicate lookups."""
        if not text:
            return 0

        # Fast cache lookup by hash to avoid storing huge duplicate strings in RAM
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash]

        enc = self._ensure_encoding()
        # Tiktoken encode without special tokens handling
        token_count = len(enc.encode(text, disallowed_special=()))
        self._cache[text_hash] = token_count
        return token_count


class _CharEncoding:
    """Minimal stand-in for a tiktoken Encoding used when BPE ranks can't be loaded."""

    def encode(self, text: str, disallowed_special: Any = ()) -> range:
        return range(math.ceil(len(text) / 4))


class CharEstimateCounter(TokenCounter):
    """Fallback heuristic token counter estimating 4 characters per token.

    Emits a warning upon instantiation or use to notify that counts are estimates.
    """

    def __init__(self, warn: bool = True) -> None:
        self._cache: dict[str, int] = {}
        if warn:
            warnings.warn(
                "Using CharEstimateCounter (len/4 approximation). "
                "Token counts may deviate from real tokenizer values.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def name(self) -> str:
        return "char"

    def count(self, text: str) -> int:
        if not text:
            return 0

        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash]

        # Standard industry approximation: 4 characters per token, rounded up
        token_count = math.ceil(len(text) / 4)
        self._cache[text_hash] = token_count
        return token_count


def get_counter(name: str = "tiktoken") -> TokenCounter:
    """Factory creating a TokenCounter instance by name.

    Args:
        name: Name of counter ('tiktoken' or 'char').

    Returns:
        Configured TokenCounter instance.

    Raises:
        ValueError: If unknown counter name is provided.
    """
    normalized = name.strip().lower()
    if normalized in ("tiktoken", "cl100k", "cl100k_base"):
        try:
            return TiktokenCounter()
        except ImportError:
            warnings.warn(
                "tiktoken is not installed. Falling back to CharEstimateCounter.",
                UserWarning,
                stacklevel=2,
            )
            return CharEstimateCounter(warn=False)
    elif normalized in ("char", "char_estimate", "heuristic"):
        return CharEstimateCounter(warn=True)
    else:
        raise ValueError(
            f"Unknown token counter '{name}'. Supported counters: 'tiktoken', 'char'."
        )


get_token_counter = get_counter
