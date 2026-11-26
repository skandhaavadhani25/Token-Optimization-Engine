"""Text assembly routines converting code entities into embeddable text."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from app.models.schemas import (
    ClassInfo,
    EmbeddedItem,
    FunctionInfo,
    ParsedFile,
)


def split_identifier(name: str) -> list[str]:
    """Split snake_case, camelCase, PascalCase, or dotted identifier into words.

    Examples:
        'verify_password' -> ['verify', 'password']
        'checkPasswordHash' -> ['check', 'password', 'hash']
        'HTTPClient' -> ['http', 'client']
        '_internal_call' -> ['internal', 'call']
    """
    cleaned = re.sub(r"[_\-.]+", " ", name).strip()
    # Transition from lower/digit to upper: fooBar -> foo Bar
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", cleaned)
    # Transition from multiple upper to upper+lower: HTTPServer -> HTTP Server
    cleaned = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", cleaned)
    return [word.lower() for word in cleaned.split() if word]


def build_function_desc(fn: FunctionInfo) -> str:
    """Build natural-language style description for a function or method.

    Format:
        "Function name (words) in file. Arguments: arg1, arg2. Calls: call1, call2. Docstring: ..."
    """
    words = split_identifier(fn.name)
    readable = " ".join(words)

    if fn.is_method and fn.parent_class:
        header = f"Method {fn.parent_class}.{fn.name} ({readable}) in {fn.file_path}."
    else:
        header = f"Function {fn.name} ({readable}) in {fn.file_path}."

    parts = [header]

    # Arguments
    arg_names: list[str] = []
    for arg in fn.arguments:
        if arg.name in ("self", "cls"):
            continue
        if arg.annotation:
            arg_names.append(f"{arg.name}: {arg.annotation}")
        else:
            arg_names.append(arg.name)
    if arg_names:
        parts.append(f"Arguments: {', '.join(arg_names)}.")

    # Calls
    call_words: list[str] = []
    for call in fn.calls:
        cw = split_identifier(call.name)
        if cw:
            call_words.append(" ".join(cw))
    unique_calls = list(dict.fromkeys(call_words))[:15]
    if unique_calls:
        parts.append(f"Calls: {', '.join(unique_calls)}.")

    # Docstring
    if fn.docstring:
        clean_doc = " ".join(fn.docstring.strip().split())[:300]
        parts.append(f"Docstring: {clean_doc}")

    return " ".join(parts).strip()


def truncate_code_at_line_boundary(
    qualified_name: str,
    source: str,
    max_tokens: int = 256,
    tokenizer: Any | None = None,
) -> str:
    """Truncate function source code at a line boundary up to max_tokens.

    Prefixes the source with '# <qualified_name>\\n'.
    """
    header = f"# {qualified_name}\n"
    lines = source.splitlines(keepends=True)
    selected: list[str] = []

    if tokenizer is not None and hasattr(tokenizer, "encode"):
        curr_text = header
        for line in lines:
            test_text = curr_text + line
            tokens = tokenizer.encode(test_text, add_special_tokens=False)
            if len(tokens) > max_tokens and selected:
                break
            selected.append(line)
            curr_text = test_text
    else:
        # Approximate: ~4 chars per token
        curr_chars = len(header)
        max_chars = max_tokens * 4
        for line in lines:
            if curr_chars + len(line) > max_chars and selected:
                break
            selected.append(line)
            curr_chars += len(line)

    if not selected and lines:
        selected.append(lines[0])

    return header + "".join(selected).rstrip()


def build_function_code(
    fn: FunctionInfo,
    max_tokens: int = 256,
    tokenizer: Any | None = None,
) -> str:
    """Build function source code text prefixed with qualified name and truncated at line boundary."""
    return truncate_code_at_line_boundary(
        qualified_name=fn.qualified_name,
        source=fn.source,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
    )


def build_class_desc(cls: ClassInfo) -> str:
    """Build descriptive text for a class definition."""
    words = split_identifier(cls.name)
    readable = " ".join(words)
    parts = [f"Class {cls.name} ({readable}) in {cls.file_path}."]

    if cls.bases:
        parts.append(f"Bases: {', '.join(cls.bases)}.")

    method_names = [
        m.name for m in cls.methods
        if not m.name.startswith("__") or m.name == "__init__"
    ]
    if method_names:
        parts.append(f"Methods: {', '.join(method_names[:25])}.")

    if cls.docstring:
        clean_doc = " ".join(cls.docstring.strip().split())[:300]
        parts.append(f"Docstring: {clean_doc}")

    return " ".join(parts).strip()


def build_file_summary(pf: ParsedFile, max_definitions: int = 30) -> str:
    """Build summary text for a file and module."""
    parts = [f"File {pf.file_path} (module {pf.module_name})."]

    if pf.module_docstring:
        clean_doc = " ".join(pf.module_docstring.strip().split())[:300]
        parts.append(f"Docstring: {clean_doc}.")

    defs = [f"class {c.name}" for c in pf.classes] + [f"function {f.name}" for f in pf.functions]
    if defs:
        parts.append(f"Defines: {', '.join(defs[:max_definitions])}.")

    imp_names = [imp.module.split(".")[0] for imp in pf.imports if imp.module]
    unique_imps = list(dict.fromkeys(imp_names))[:15]
    if unique_imps:
        parts.append(f"Imports: {', '.join(unique_imps)}.")

    return " ".join(parts).strip()


def is_trivial_function(fn: FunctionInfo, min_body_lines: int = 2) -> bool:
    """Check if a function is trivially small (no docstring and < min_body_lines body)."""
    if fn.docstring:
        return False
    body_lines = [
        line.strip()
        for line in fn.source.splitlines()
        if line.strip()
        and not line.strip().startswith("@")
        and not line.strip().startswith("def ")
        and not line.strip().startswith("async def ")
    ]
    return len(body_lines) < min_body_lines


def build_items_for_file(
    pf: ParsedFile,
    max_code_tokens: int = 256,
    tokenizer: Any | None = None,
    min_body_lines: int = 2,
) -> tuple[list[EmbeddedItem], int]:
    """Generate all embeddable items (file, classes, functions) for a parsed file.

    Returns:
        Tuple of (list of EmbeddedItem instances, count of skipped trivial items).
    """
    items: list[EmbeddedItem] = []
    skipped_count = 0

    # 1. file_summary
    f_text = build_file_summary(pf)
    f_hash = hashlib.sha256(f_text.encode("utf-8")).hexdigest()
    items.append(
        EmbeddedItem(
            item_id=f"file_summary:{pf.file_path}",
            kind="file_summary",
            ref=pf.file_path,
            file=pf.file_path,
            text_hash=f_hash,
            text=f_text,
        )
    )

    # 2. class_desc
    for cls in pf.classes:
        c_text = build_class_desc(cls)
        c_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
        items.append(
            EmbeddedItem(
                item_id=f"class_desc:{cls.qualified_name}",
                kind="class_desc",
                ref=cls.qualified_name,
                file=pf.file_path,
                text_hash=c_hash,
                text=c_text,
            )
        )

    # 3. function_desc and function_code
    all_funcs = list(pf.functions) + [m for c in pf.classes for m in c.methods]
    for fn in all_funcs:
        # function_desc
        fn_desc_text = build_function_desc(fn)
        fn_desc_hash = hashlib.sha256(fn_desc_text.encode("utf-8")).hexdigest()
        items.append(
            EmbeddedItem(
                item_id=f"function_desc:{fn.qualified_name}",
                kind="function_desc",
                ref=fn.qualified_name,
                file=pf.file_path,
                text_hash=fn_desc_hash,
                text=fn_desc_text,
            )
        )

        # function_code
        if is_trivial_function(fn, min_body_lines=min_body_lines):
            skipped_count += 1
        else:
            fn_code_text = truncate_code_at_line_boundary(
                fn.qualified_name,
                fn.source,
                max_tokens=max_code_tokens,
                tokenizer=tokenizer,
            )
            fn_code_hash = hashlib.sha256(fn_code_text.encode("utf-8")).hexdigest()
            items.append(
                EmbeddedItem(
                    item_id=f"function_code:{fn.qualified_name}",
                    kind="function_code",
                    ref=fn.qualified_name,
                    file=pf.file_path,
                    text_hash=fn_code_hash,
                    text=fn_code_text,
                )
            )

    return items, skipped_count
