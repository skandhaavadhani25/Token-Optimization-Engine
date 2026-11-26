"""Security utilities: API key redaction, log filters, and in-memory rate limiting."""

from __future__ import annotations

import contextvars
import logging
import re
import threading
import time

# Tracks the raw X-Provider-Key for the active HTTP request in the current async context
current_request_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_request_key", default=None
)

_REDACTION_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{14,}", re.IGNORECASE),
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    re.compile(r"AQ\.[0-9A-Za-z\-_]{20,}"),
    re.compile(r"gsk_[a-zA-Z0-9_\-]{14,}"),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{14,}", re.IGNORECASE),
    re.compile(r"x-provider-key:\s*[^\s,;]+", re.IGNORECASE),
    re.compile(r"api[_-]?key[:=]\s*['\"]?[a-zA-Z0-9_\-\.]+['\"]?", re.IGNORECASE),
]


def redact_sensitive_text(text: str) -> str:
    """Redact any API key tokens, header values, or key-shaped patterns from text."""
    if not text:
        return text

    # 1. Exact match against the current request's key if set
    req_key = current_request_key.get()
    if req_key and len(req_key) >= 4 and req_key in text:
        text = text.replace(req_key, "[REDACTED_KEY]")

    # 2. General pattern matching for common provider key structures
    for pat in _REDACTION_PATTERNS:
        text = pat.sub("[REDACTED_KEY]", text)

    return text


class KeyRedactionFilter(logging.Filter):
    """Logging filter that strips API keys and patterns from all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_text(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact_sensitive_text(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: redact_sensitive_text(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        return True


_filter_installed = False
_filter_lock = threading.Lock()


def setup_key_redaction_logging() -> None:
    """Attach KeyRedactionFilter to root and relevant loggers globally."""
    global _filter_installed
    with _filter_lock:
        if _filter_installed:
            return
        redaction_filter = KeyRedactionFilter()

        # Attach to root logger
        root_logger = logging.getLogger()
        root_logger.addFilter(redaction_filter)
        for h in root_logger.handlers:
            h.addFilter(redaction_filter)

        # Attach to app logger
        app_logger = logging.getLogger("code_context_engine")
        app_logger.addFilter(redaction_filter)
        for h in app_logger.handlers:
            h.addFilter(redaction_filter)

        _filter_installed = True


class InMemoryRateLimiter:
    """Sliding-window in-memory rate limiter per client IP.

    Default: 10 validation attempts per 60-second window.
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        """Return True if the request is allowed within rate limits; False otherwise."""
        now = time.monotonic()
        with self._lock:
            history = self._requests.get(client_ip, [])
            valid_history = [t for t in history if now - t < self.window_seconds]
            if len(valid_history) >= self.max_requests:
                self._requests[client_ip] = valid_history
                return False
            valid_history.append(now)
            self._requests[client_ip] = valid_history
            return True

    def reset(self) -> None:
        """Clear all rate limit history."""
        with self._lock:
            self._requests.clear()


# Default singleton instances for rate limiting
validation_rate_limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60.0)
upload_rate_limiter = InMemoryRateLimiter(max_requests=30, window_seconds=60.0)
message_rate_limiter = InMemoryRateLimiter(max_requests=60, window_seconds=60.0)
