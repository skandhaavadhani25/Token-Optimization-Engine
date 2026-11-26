"""Secret scanner for uploads, attachments, and files in workspaces.

Detects API key patterns, private key headers, .env files, and credentials.
Automatically flags them for exclusion from indexing and LLM prompt context.
"""

from __future__ import annotations

import re
from pathlib import Path

# Regex patterns for well-known secrets and keys
SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Anthropic API Key", re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}")),
    ("Google API Key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("OpenAI/Compatible API Key", re.compile(r"sk-[a-zA-Z0-9_\-]{20,}")),
    ("GitHub Personal Access Token", re.compile(r"gh[pousr]_[a-zA-Z0-9]{36}")),
    ("AWS Access Key ID", re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b")),
    ("Private Key Header", re.compile(r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----")),
    ("Generic Bearer Secret", re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{25,}")),
]

# Sensitive file name patterns that should be excluded by default
SENSITIVE_FILE_NAMES: set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials.json",
    "service_account.json",
    "secret.key",
}

SENSITIVE_EXTENSIONS: set[str] = {
    ".pem",
    ".key",
    ".pkcs12",
    ".pfx",
    ".p12",
    ".kdbx",
}


def is_secret_filename(filename: str | Path) -> bool:
    """Return True if the file name or extension indicates sensitive credentials."""
    name = Path(filename).name.lower()
    if name in SENSITIVE_FILE_NAMES:
        return True
    if name.startswith(".env"):
        return True
    ext = Path(filename).suffix.lower()
    if ext in SENSITIVE_EXTENSIONS:
        return True
    return False


def scan_file_for_secrets(
    filename: str | Path,
    content: str | bytes,
) -> list[str]:
    """Inspect file path and content for credential leaks.

    Returns:
        List of human-readable warnings describing detected secrets.
    """
    warnings: list[str] = []
    str(filename)

    if is_secret_filename(filename):
        warnings.append(
            f"File '{Path(filename).name}' is a sensitive configuration/credential file (.env or key)."
        )

    text_to_scan: str
    if isinstance(content, bytes):
        try:
            text_to_scan = content.decode("utf-8", errors="ignore")
        except Exception:
            return warnings
    else:
        text_to_scan = content

    # Scan content against regex patterns (limit to first 100KB for speed)
    sample = text_to_scan[:100_000]
    for desc, pat in SECRET_PATTERNS:
        if pat.search(sample):
            warnings.append(f"Detected potential {desc} inside '{Path(filename).name}'.")

    return warnings
