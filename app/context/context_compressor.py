"""Code compression engine generating progressive detail levels for functions and classes."""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Any

from app.context.token_counter import TokenCounter, get_counter
from app.models.schemas import ClassInfo, CompressedSnippet, DetailLevel, FunctionInfo


class ASTLiteralTrimmer(ast.NodeTransformer):
    """AST transformer collapsing long literals into placeholders."""

    def __init__(self, max_lines: int = 3) -> None:
        super().__init__()
        self.max_lines = max_lines
        self.is_first_stmt = True
        self.docstring_node: ast.AST | None = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        # Protect docstring from being collapsed as a regular string literal
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            self.docstring_node = node.body[0].value
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            self.docstring_node = node.body[0].value
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            self.docstring_node = node.body[0].value
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if node is self.docstring_node:
            return node

        if isinstance(node.value, str):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is not None and end is not None and (end - start) >= self.max_lines:
                lines_omitted = end - start + 1
                return ast.copy_location(
                    ast.Constant(value=f"...  # {lines_omitted} lines omitted"),
                    node,
                )
        return node

    def visit_List(self, node: ast.List) -> Any:
        self.generic_visit(node)
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is not None and end is not None and (end - start) >= self.max_lines:
            lines_omitted = end - start + 1
            # Replace elements with single ellipsis placeholder
            return ast.copy_location(
                ast.List(
                    elts=[ast.Constant(value=f"...  # {lines_omitted} items omitted")],
                    ctx=node.ctx,
                ),
                node,
            )
        return node

    def visit_Dict(self, node: ast.Dict) -> Any:
        self.generic_visit(node)
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is not None and end is not None and (end - start) >= self.max_lines:
            lines_omitted = end - start + 1
            return ast.copy_location(
                ast.Dict(
                    keys=[ast.Constant(value="...")],
                    values=[ast.Constant(value=f"...  # {lines_omitted} entries omitted")],
                ),
                node,
            )
        return node


def _build_args_string(arguments: list[Any]) -> str:
    """Reconstruct function arguments string from ArgumentInfo list."""
    arg_parts: list[str] = []
    for arg in arguments:
        part = arg.name
        if arg.kind == "vararg":
            part = f"*{part}"
        elif arg.kind == "kwarg":
            part = f"**{part}"

        if getattr(arg, "annotation", None):
            part = f"{part}: {arg.annotation}"
        if getattr(arg, "default", None):
            part = f"{part} = {arg.default}"
        arg_parts.append(part)
    return ", ".join(arg_parts)


def _line_based_trim_fallback(source: str) -> str:
    """Heuristic line-based cleaner removing comments and extra blank lines."""
    lines = source.splitlines()
    cleaned: list[str] = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()
        # Remove comment lines (but keep docstrings)
        if stripped.startswith("#"):
            continue
        if not stripped:
            if not prev_blank and cleaned:
                cleaned.append("")
                prev_blank = True
            continue
        prev_blank = False
        # Remove trailing comments
        cleaned.append(re.sub(r"\s+#.*$", "", line))

    return "\n".join(cleaned)


def _compress_function(
    fn: FunctionInfo,
    level: DetailLevel,
    counter: TokenCounter,
    max_docstring_chars: int = 200,
    max_literal_lines: int = 3,
) -> CompressedSnippet:
    """Compress a FunctionInfo entity into the requested DetailLevel snippet."""
    orig_tokens = counter.count(fn.source)
    lines_str = f"{fn.file_path}:{fn.start_line}-{fn.end_line}"
    ref = fn.qualified_name

    # 1. FULL LEVEL
    if level == DetailLevel.FULL:
        text = f"# {lines_str} [full]\n{fn.source.rstrip()}"
        return CompressedSnippet(
            text=text,
            level=DetailLevel.FULL,
            tokens=counter.count(text),
            original_tokens=orig_tokens,
            ref=ref,
            file=fn.file_path,
            start_line=fn.start_line,
            end_line=fn.end_line,
            degraded=False,
        )

    # 2. TRIMMED LEVEL
    if level == DetailLevel.TRIMMED:
        degraded = False
        try:
            tree = ast.parse(fn.source)
        except SyntaxError:
            try:
                tree = ast.parse(textwrap.dedent(fn.source))
            except Exception:
                tree = None

        if tree is not None:
            try:
                transformer = ASTLiteralTrimmer(max_lines=max_literal_lines)
                transformed = transformer.visit(tree)
                ast.fix_missing_locations(transformed)
                unparsed = ast.unparse(transformed)
                # Verify that unparsed code is valid syntax
                ast.parse(unparsed)
                body_code = unparsed
            except Exception:
                degraded = True
                body_code = _line_based_trim_fallback(fn.source)
        else:
            degraded = True
            body_code = _line_based_trim_fallback(fn.source)

        text = f"# {lines_str} [trimmed]\n{body_code.rstrip()}"
        return CompressedSnippet(
            text=text,
            level=DetailLevel.TRIMMED,
            tokens=counter.count(text),
            original_tokens=orig_tokens,
            ref=ref,
            file=fn.file_path,
            start_line=fn.start_line,
            end_line=fn.end_line,
            degraded=degraded,
        )

    # 3. SIGNATURE LEVEL
    if level == DetailLevel.SIGNATURE:
        parts: list[str] = [f"# {lines_str} [signature]"]
        for dec in fn.decorators:
            parts.append(dec if dec.startswith("@") else f"@{dec}")

        prefix = "async def " if fn.is_async else "def "
        args_str = _build_args_string(fn.arguments)
        ret_str = f" -> {fn.return_annotation}" if fn.return_annotation else ""
        header = f"{prefix}{fn.name}({args_str}){ret_str}:"
        parts.append(header)

        indent = "    "
        if fn.docstring:
            clean_doc = " ".join(fn.docstring.strip().split())
            if len(clean_doc) > max_docstring_chars:
                clean_doc = clean_doc[:max_docstring_chars].rstrip() + "..."
            parts.append(f'{indent}"""{clean_doc}"""')

        call_names = [c.name for c in fn.calls if c.name]
        unique_calls = list(dict.fromkeys(call_names))[:10]
        if unique_calls:
            parts.append(f"{indent}# calls: {', '.join(unique_calls)}")

        parts.append(f"{indent}...")
        text = "\n".join(parts)
        return CompressedSnippet(
            text=text,
            level=DetailLevel.SIGNATURE,
            tokens=counter.count(text),
            original_tokens=orig_tokens,
            ref=ref,
            file=fn.file_path,
            start_line=fn.start_line,
            end_line=fn.end_line,
            degraded=False,
        )

    # 4. STUB LEVEL
    if level == DetailLevel.STUB:
        prefix = "async def " if fn.is_async else "def "
        args_str = _build_args_string(fn.arguments)
        ret_str = f" -> {fn.return_annotation}" if fn.return_annotation else ""
        text = f"{prefix}{fn.name}({args_str}){ret_str}: ...  # {lines_str}"
        return CompressedSnippet(
            text=text,
            level=DetailLevel.STUB,
            tokens=counter.count(text),
            original_tokens=orig_tokens,
            ref=ref,
            file=fn.file_path,
            start_line=fn.start_line,
            end_line=fn.end_line,
            degraded=False,
        )

    # 5. REFERENCE LEVEL
    text = f"{ref} ({lines_str})"
    return CompressedSnippet(
        text=text,
        level=DetailLevel.REFERENCE,
        tokens=counter.count(text),
        original_tokens=orig_tokens,
        ref=ref,
        file=fn.file_path,
        start_line=fn.start_line,
        end_line=fn.end_line,
        degraded=False,
    )


def _compress_class(
    cls: ClassInfo,
    level: DetailLevel,
    counter: TokenCounter,
    max_docstring_chars: int = 200,
    max_literal_lines: int = 3,
) -> CompressedSnippet:
    """Compress a ClassInfo entity into the requested DetailLevel snippet."""
    orig_tokens = counter.count(cls.source)
    lines_str = f"{cls.file_path}:{cls.start_line}-{cls.end_line}"
    ref = cls.qualified_name

    # 1. FULL LEVEL
    if level == DetailLevel.FULL:
        text = f"# {lines_str} [full]\n{cls.source.rstrip()}"
        return CompressedSnippet(
            text=text,
            level=DetailLevel.FULL,
            tokens=counter.count(text),
            original_tokens=orig_tokens,
            ref=ref,
            file=cls.file_path,
            start_line=cls.start_line,
            end_line=cls.end_line,
            degraded=False,
        )

    # 2. TRIMMED LEVEL
    if level == DetailLevel.TRIMMED:
        degraded = False
        try:
            tree = ast.parse(cls.source)
        except SyntaxError:
            try:
                tree = ast.parse(textwrap.dedent(cls.source))
            except Exception:
                tree = None

        if tree is not None:
            try:
                transformer = ASTLiteralTrimmer(max_lines=max_literal_lines)
                transformed = transformer.visit(tree)
                ast.fix_missing_locations(transformed)
                unparsed = ast.unparse(transformed)
                ast.parse(unparsed)
                body_code = unparsed
            except Exception:
                degraded = True
                body_code = _line_based_trim_fallback(cls.source)
        else:
            degraded = True
            body_code = _line_based_trim_fallback(cls.source)

        text = f"# {lines_str} [trimmed]\n{body_code.rstrip()}"
        return CompressedSnippet(
            text=text,
            level=DetailLevel.TRIMMED,
            tokens=counter.count(text),
            original_tokens=orig_tokens,
            ref=ref,
            file=cls.file_path,
            start_line=cls.start_line,
            end_line=cls.end_line,
            degraded=degraded,
        )

    bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""

    # 3. SIGNATURE LEVEL
    if level == DetailLevel.SIGNATURE:
        parts: list[str] = [f"# {lines_str} [signature]"]
        for dec in cls.decorators:
            parts.append(dec if dec.startswith("@") else f"@{dec}")
        parts.append(f"class {cls.name}{bases_str}:")
        indent = "    "
        if cls.docstring:
            clean_doc = " ".join(cls.docstring.strip().split())
            if len(clean_doc) > max_docstring_chars:
                clean_doc = clean_doc[:max_docstring_chars].rstrip() + "..."
            parts.append(f'{indent}"""{clean_doc}"""')

        # List method stubs
        if cls.methods:
            for m in cls.methods[:20]:
                prefix = "async def " if m.is_async else "def "
                args_str = _build_args_string(m.arguments)
                ret_str = f" -> {m.return_annotation}" if m.return_annotation else ""
                parts.append(f"{indent}{prefix}{m.name}({args_str}){ret_str}: ...")
        else:
            parts.append(f"{indent}...")

        text = "\n".join(parts)
        return CompressedSnippet(
            text=text,
            level=DetailLevel.SIGNATURE,
            tokens=counter.count(text),
            original_tokens=orig_tokens,
            ref=ref,
            file=cls.file_path,
            start_line=cls.start_line,
            end_line=cls.end_line,
            degraded=False,
        )

    # 4. STUB LEVEL
    if level == DetailLevel.STUB:
        method_names = [m.name for m in cls.methods[:15]]
        methods_annotation = f"  # methods: {', '.join(method_names)}" if method_names else ""
        text = f"class {cls.name}{bases_str}: ...{methods_annotation}  # {lines_str}"
        return CompressedSnippet(
            text=text,
            level=DetailLevel.STUB,
            tokens=counter.count(text),
            original_tokens=orig_tokens,
            ref=ref,
            file=cls.file_path,
            start_line=cls.start_line,
            end_line=cls.end_line,
            degraded=False,
        )

    # 5. REFERENCE LEVEL
    text = f"{ref} ({lines_str})"
    return CompressedSnippet(
        text=text,
        level=DetailLevel.REFERENCE,
        tokens=counter.count(text),
        original_tokens=orig_tokens,
        ref=ref,
        file=cls.file_path,
        start_line=cls.start_line,
        end_line=cls.end_line,
        degraded=False,
    )


def compress(
    entity: FunctionInfo | ClassInfo,
    level: DetailLevel | str,
    counter: TokenCounter | None = None,
    max_docstring_chars: int = 200,
    max_literal_lines: int = 3,
) -> CompressedSnippet:
    """Compress a function or class into a structured CompressedSnippet.

    Args:
        entity: FunctionInfo or ClassInfo instance from AST parser/store.
        level: Target DetailLevel enum or string ("full", "trimmed", "signature", "stub", "reference").
        counter: Optional TokenCounter instance; defaults to TiktokenCounter.
        max_docstring_chars: Maximum characters to preserve in docstring preview for SIGNATURE.
        max_literal_lines: Threshold beyond which literal collections/strings are collapsed in TRIMMED.

    Returns:
        CompressedSnippet holding snippet text, token count, and metadata.
    """
    if counter is None:
        counter = get_counter()

    if isinstance(level, str):
        level = DetailLevel(level.lower())

    if isinstance(entity, FunctionInfo):
        return _compress_function(
            fn=entity,
            level=level,
            counter=counter,
            max_docstring_chars=max_docstring_chars,
            max_literal_lines=max_literal_lines,
        )
    elif isinstance(entity, ClassInfo):
        return _compress_class(
            cls=entity,
            level=level,
            counter=counter,
            max_docstring_chars=max_docstring_chars,
            max_literal_lines=max_literal_lines,
        )
    else:
        raise TypeError(f"Unsupported entity type for compression: {type(entity)}")


def compress_to_fit(
    entity: FunctionInfo | ClassInfo,
    max_tokens: int,
    counter: TokenCounter | None = None,
    max_docstring_chars: int = 200,
    max_literal_lines: int = 3,
) -> CompressedSnippet | None:
    """Find the most detailed compression level that fits within a token budget.

    Tries levels in descending order: FULL -> TRIMMED -> SIGNATURE -> STUB -> REFERENCE.

    Args:
        entity: FunctionInfo or ClassInfo instance.
        max_tokens: Maximum allowable token count.
        counter: TokenCounter instance.
        max_docstring_chars: Docstring preview character limit.
        max_literal_lines: Literal collapsing line threshold.

    Returns:
        The highest-detail CompressedSnippet whose tokens <= max_tokens, or None if even REFERENCE fails.
    """
    if max_tokens <= 0:
        return None

    if counter is None:
        counter = get_counter()

    order = [
        DetailLevel.FULL,
        DetailLevel.TRIMMED,
        DetailLevel.SIGNATURE,
        DetailLevel.STUB,
        DetailLevel.REFERENCE,
    ]

    for lvl in order:
        snippet = compress(
            entity=entity,
            level=lvl,
            counter=counter,
            max_docstring_chars=max_docstring_chars,
            max_literal_lines=max_literal_lines,
        )
        if snippet.tokens <= max_tokens:
            return snippet

    return None
