"""Chat attachment processing, chunking, and conversation-scoped hybrid search.

Handles text extraction, AST/line chunking, vision validation, and per-conversation indexing.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

from app.context.token_counter import get_counter

# Models with verified multimodal vision support
VISION_SUPPORTED_MODELS = (
    "claude-3-5-sonnet",
    "claude-3-opus",
    "claude-3-haiku",
    "claude-3-5-haiku",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "llava",
    "bakllava",
    "llama3.2-vision",
)

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def model_supports_vision(provider: str | None, model: str | None) -> bool:
    """Return True if the current provider and model configuration supports image inputs."""
    if not model:
        return False
    m_lower = model.lower()
    return any(vm in m_lower for vm in VISION_SUPPORTED_MODELS)


def extract_attachment_text(filename: str, data: bytes) -> tuple[str, str]:
    """Extract plain text from an attachment.

    Returns:
        Tuple of (text_content, file_type_label).
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text, "pdf"
        except ImportError:
            return (
                "[PDF text extraction is optional and requires 'pypdf' installed on server]",
                "pdf",
            )
        except Exception as exc:
            return f"[Error extracting PDF: {exc}]", "pdf"

    if ext in IMAGE_EXTENSIONS:
        return "[Image attachment]", "image"

    # Standard code / text files
    try:
        text = data.decode("utf-8")
        return text, ext.lstrip(".") or "text"
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
            return text, ext.lstrip(".") or "text"
        except Exception:
            return "[Binary non-text content]", "binary"


@dataclass
class AttachmentChunk:
    chunk_id: str
    attachment_id: str
    filename: str
    start_line: int
    end_line: int
    content: str
    token_count: int
    is_whole_file: bool = False


# In-memory registry of conversation-scoped attachment chunk indexes
_CONVERSATION_CHUNKS: dict[str, list[AttachmentChunk]] = {}


def chunk_attachment(
    attachment_id: str,
    filename: str,
    text: str,
    whole_file_token_threshold: int = 1500,
) -> list[AttachmentChunk]:
    """Chunk attachment text into searchable segments with exact line numbers."""
    counter = get_counter("cl100k_base")
    total_tokens = counter.count_tokens(text)

    # Small attachment: keep as single whole item
    if total_tokens <= whole_file_token_threshold:
        line_count = len(text.splitlines()) or 1
        return [
            AttachmentChunk(
                chunk_id=f"{attachment_id}:1-{line_count}",
                attachment_id=attachment_id,
                filename=filename,
                start_line=1,
                end_line=line_count,
                content=text,
                token_count=total_tokens,
                is_whole_file=True,
            )
        ]

    lines = text.splitlines(keepends=True)
    chunks: list[AttachmentChunk] = []
    chunk_size_lines = 60
    overlap_lines = 10

    i = 0
    while i < len(lines):
        end_idx = min(i + chunk_size_lines, len(lines))
        chunk_text = "".join(lines[i:end_idx])
        tok = counter.count_tokens(chunk_text)
        chunks.append(
            AttachmentChunk(
                chunk_id=f"{attachment_id}:{i+1}-{end_idx}",
                attachment_id=attachment_id,
                filename=filename,
                start_line=i + 1,
                end_line=end_idx,
                content=chunk_text,
                token_count=tok,
                is_whole_file=False,
            )
        )
        if end_idx >= len(lines):
            break
        i += chunk_size_lines - overlap_lines

    return chunks


def register_conversation_attachment(
    conversation_id: str,
    attachment_id: str,
    filename: str,
    text: str,
) -> list[AttachmentChunk]:
    """Index an attachment for retrieval within a specific conversation."""
    chunks = chunk_attachment(attachment_id, filename, text)
    if conversation_id not in _CONVERSATION_CHUNKS:
        _CONVERSATION_CHUNKS[conversation_id] = []
    _CONVERSATION_CHUNKS[conversation_id].extend(chunks)
    return chunks


def search_conversation_attachments(
    conversation_id: str,
    query: str,
    top_k: int = 3,
) -> list[AttachmentChunk]:
    """Retrieve top relevant attachment chunks for the given conversation query using BM25 scoring."""
    chunks = _CONVERSATION_CHUNKS.get(conversation_id, [])
    if not chunks:
        return []

    # If all chunks fit in top_k, return all
    if len(chunks) <= top_k:
        return list(chunks)

    # Simple BM25 / token overlap scoring over attachment chunks
    terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
    if not terms:
        return chunks[:top_k]

    scored: list[tuple[float, AttachmentChunk]] = []
    for c in chunks:
        content_lower = c.content.lower()
        score = 0.0
        for t in terms:
            cnt = content_lower.count(t)
            if cnt > 0:
                score += math.log(1.0 + cnt)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:top_k] if s > 0.0] or chunks[:top_k]
