"""AST-based code extraction logic for Python source files."""

from __future__ import annotations

import ast
from typing import Sequence

from app.models.schemas import (
    ArgumentInfo,
    CallInfo,
    ClassInfo,
    FunctionInfo,
    ImportInfo,
)


def resolve_call_name(node: ast.AST) -> str | None:
    """Resolve a call expression to its dotted callee name representation.

    Handles:
        foo() -> 'foo'
        self.foo() -> 'self.foo'
        module.foo() -> 'module.foo'
        a.b.c() -> 'a.b.c'
        foo().bar() -> 'bar'
        x[0].run() -> 'run'
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = resolve_call_name(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
        return node.attr
    return None


class CallExtractor(ast.NodeVisitor):
    """Walks an AST subtree to extract and de-duplicate call expressions."""

    def __init__(self) -> None:
        self.seen_names: set[str] = set()
        self.calls: list[CallInfo] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = resolve_call_name(node.func)
        if name and name not in self.seen_names:
            self.seen_names.add(name)
            self.calls.append(CallInfo(name=name, lineno=node.lineno))
        self.generic_visit(node)


def extract_calls(node: ast.AST) -> list[CallInfo]:
    """Extract all function/method invocations within an AST node's body, including nested functions."""
    extractor = CallExtractor()
    body = getattr(node, "body", [])
    if isinstance(body, list):
        for stmt in body:
            extractor.visit(stmt)
    else:
        extractor.visit(node)
    return extractor.calls


def extract_arguments(args_node: ast.arguments) -> list[ArgumentInfo]:
    """Extract structured argument metadata from an ast.arguments node."""
    arguments: list[ArgumentInfo] = []

    pos_args = list(args_node.posonlyargs) + list(args_node.args)
    num_pos = len(pos_args)
    num_defaults = len(args_node.defaults)
    offset = num_pos - num_defaults

    for idx, arg in enumerate(pos_args):
        annotation = ast.unparse(arg.annotation) if arg.annotation is not None else None
        default = None
        if idx >= offset:
            default = ast.unparse(args_node.defaults[idx - offset])
        arguments.append(
            ArgumentInfo(
                name=arg.arg,
                annotation=annotation,
                default=default,
                kind="positional",
            )
        )

    if args_node.vararg is not None:
        v = args_node.vararg
        arguments.append(
            ArgumentInfo(
                name=v.arg,
                annotation=ast.unparse(v.annotation) if v.annotation is not None else None,
                default=None,
                kind="vararg",
            )
        )

    for idx, kwarg in enumerate(args_node.kwonlyargs):
        annotation = ast.unparse(kwarg.annotation) if kwarg.annotation is not None else None
        kw_def = args_node.kw_defaults[idx]
        default = ast.unparse(kw_def) if kw_def is not None else None
        arguments.append(
            ArgumentInfo(
                name=kwarg.arg,
                annotation=annotation,
                default=default,
                kind="keyword_only",
            )
        )

    if args_node.kwarg is not None:
        k = args_node.kwarg
        arguments.append(
            ArgumentInfo(
                name=k.arg,
                annotation=ast.unparse(k.annotation) if k.annotation is not None else None,
                default=None,
                kind="kwarg",
            )
        )

    return arguments


def get_node_range(node: ast.AST, lines: Sequence[str]) -> tuple[int, int, str]:
    """Return 1-indexed (start_line, end_line, source_text) including any decorators."""
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start_line = decorators[0].lineno
    else:
        start_line = getattr(node, "lineno", 1)

    end_line = getattr(node, "end_lineno", start_line)
    if end_line is None:
        end_line = start_line

    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    source = "".join(lines[start_idx:end_idx])
    return start_line, end_line, source


class AstExtractor(ast.NodeVisitor):
    """Extracts imports, classes, methods, and functions from an AST."""

    def __init__(
        self,
        source_text: str,
        lines: Sequence[str],
        file_path: str,
        module_name: str,
    ) -> None:
        self.source_text = source_text
        self.lines = lines
        self.file_path = file_path
        self.module_name = module_name

        self.imports: list[ImportInfo] = []
        self.classes: list[ClassInfo] = []
        self.functions: list[FunctionInfo] = []
        self.module_docstring: str | None = None

        self._in_class: str | None = None
        self._function_depth: int = 0

    def extract(self, tree: ast.Module) -> None:
        """Run extraction across the parsed module."""
        self.module_docstring = ast.get_docstring(tree)
        self.visit(tree)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=alias.name,
                    name=None,
                    alias=alias.asname,
                    is_from_import=False,
                    level=0,
                    lineno=node.lineno,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        level = node.level if node.level is not None else 0
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=module,
                    name=alias.name,
                    alias=alias.asname,
                    is_from_import=True,
                    level=level,
                    lineno=node.lineno,
                )
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # If class definition is inside a function, do not register as top-level class
        if self._function_depth > 0:
            self.generic_visit(node)
            return

        start_line, end_line, source = get_node_range(node, self.lines)
        bases = [ast.unparse(b) for b in node.bases]
        decorators = [ast.unparse(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)

        prev_class = self._in_class
        self._in_class = node.name

        methods: list[FunctionInfo] = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item, is_method=True, parent_class=node.name))
            else:
                self.visit(item)

        cls_info = ClassInfo(
            name=node.name,
            qualified_name=f"{self.module_name}.{node.name}",
            file_path=self.file_path,
            start_line=start_line,
            end_line=end_line,
            bases=bases,
            decorators=decorators,
            docstring=docstring,
            methods=methods,
            source=source,
        )
        self.classes.append(cls_info)
        self._in_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function_node(node)

    def _handle_function_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # If we are inside an outer function, this is a nested function:
        # Do NOT record it separately as a top-level function or method.
        # However, visit its body so any internal imports are collected.
        if self._function_depth > 0:
            self._function_depth += 1
            self.generic_visit(node)
            self._function_depth -= 1
            return

        # Top-level module function
        fn_info = self._extract_function(node, is_method=False, parent_class=None)
        self.functions.append(fn_info)

        # Visit inner statements to capture any imports inside the function
        self._function_depth += 1
        for item in node.body:
            self.visit(item)
        self._function_depth -= 1

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_method: bool,
        parent_class: str | None,
    ) -> FunctionInfo:
        start_line, end_line, source = get_node_range(node, self.lines)
        arguments = extract_arguments(node.args)
        return_annotation = ast.unparse(node.returns) if node.returns is not None else None
        decorators = [ast.unparse(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)
        calls = extract_calls(node)

        if is_method and parent_class:
            qualified_name = f"{self.module_name}.{parent_class}.{node.name}"
        else:
            qualified_name = f"{self.module_name}.{node.name}"

        return FunctionInfo(
            name=node.name,
            qualified_name=qualified_name,
            file_path=self.file_path,
            start_line=start_line,
            end_line=end_line,
            arguments=arguments,
            return_annotation=return_annotation,
            decorators=decorators,
            docstring=docstring,
            calls=calls,
            source=source,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            parent_class=parent_class,
        )
