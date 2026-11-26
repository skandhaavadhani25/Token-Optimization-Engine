"""Call resolution engine mapping raw AST call strings to concrete definitions."""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.models.schemas import (
    CallInfo,
    ClassInfo,
    FunctionInfo,
    ParsedFile,
)

# Set of standard Python built-in names and exceptions
BUILTIN_NAMES: frozenset[str] = frozenset(
    {name for name in dir(builtins) if not name.startswith("_")}
    | {
        "print",
        "len",
        "isinstance",
        "issubclass",
        "range",
        "int",
        "str",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "bytes",
        "bytearray",
        "open",
        "type",
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "super",
        "enumerate",
        "zip",
        "sum",
        "min",
        "max",
        "any",
        "all",
        "id",
        "repr",
        "iter",
        "next",
        "callable",
        "filter",
        "map",
        "sorted",
        "reversed",
        "vars",
        "abs",
        "round",
        "pow",
        "divmod",
        "hash",
        "dir",
        "eval",
        "exec",
        "format",
        "input",
        "bin",
        "hex",
        "oct",
        "ord",
        "chr",
        "slice",
        "help",
        "Exception",
        "BaseException",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "StopIteration",
        "NotImplementedError",
        "IOError",
        "OSError",
        "FileNotFoundError",
        "PermissionError",
        "ZeroDivisionError",
        "OverflowError",
    }
)


def resolve_relative_module(
    file_path: str,
    module_name: str,
    level: int,
    import_module: str,
) -> str:
    """Resolve a relative import (level >= 1) to an absolute dotted module name.

    Args:
        file_path: Relative POSIX path to the importing file.
        module_name: Dotted module name of the importing file.
        level: Relative import level (1 for ., 2 for .., etc.).
        import_module: Suffix module string in the import statement.

    Returns:
        Dotted module name string.
    """
    is_init = Path(file_path).name == "__init__.py"
    parts = module_name.split(".") if module_name else []
    pkg_parts = parts if is_init else parts[:-1]

    up = level - 1
    if up > len(pkg_parts):
        return import_module if import_module else ""

    base_parts = pkg_parts[: len(pkg_parts) - up] if up > 0 else list(pkg_parts)
    if import_module:
        base_parts.extend(import_module.split("."))
    return ".".join(base_parts)


@dataclass(frozen=True)
class ResolvedImport:
    """Resolved symbol or module brought in by an import statement."""

    local_name: str
    is_module: bool
    target_module: str
    target_symbol: str | None
    is_internal: bool
    target_file: str | None


@dataclass(frozen=True)
class CallTarget:
    """A successfully resolved call destination."""

    node_id: str
    confidence: str  # "exact", "high", "medium", "low"
    resolution: str  # "same_module", "import", "self_method", "class_method", "unique_name", "ambiguous"


@dataclass
class ResolutionResult:
    """Complete outcome of resolving a single CallInfo invocation."""

    targets: list[CallTarget] = field(default_factory=list)
    rule_name: str | None = None
    is_external: bool = False
    is_builtin: bool = False
    is_ambiguous_skipped: bool = False
    is_unresolved: bool = False


class CallResolver:
    """Resolves function calls to definition nodes using ordered heuristics."""

    def __init__(self, parsed_files: Sequence[ParsedFile]) -> None:
        self.files_by_path: dict[str, ParsedFile] = {f.file_path: f for f in parsed_files}
        self.module_to_file: dict[str, str] = {f.module_name: f.file_path for f in parsed_files}

        # Index classes and functions
        self.classes_by_qname: dict[str, ClassInfo] = {}
        self.classes_by_short_name_in_file: dict[str, dict[str, ClassInfo]] = {}
        self.functions_by_qname: dict[str, FunctionInfo] = {}
        self.functions_by_short_name_in_file: dict[str, dict[str, FunctionInfo]] = {}
        self.all_callables_by_short_name: dict[str, list[str]] = {}

        for pf in parsed_files:
            self.classes_by_short_name_in_file[pf.file_path] = {}
            self.functions_by_short_name_in_file[pf.file_path] = {}

            for cls in pf.classes:
                self.classes_by_qname[cls.qualified_name] = cls
                self.classes_by_short_name_in_file[pf.file_path][cls.name] = cls

                for method in cls.methods:
                    self.functions_by_qname[method.qualified_name] = method
                    m_node_id = f"method:{method.qualified_name}"
                    self.all_callables_by_short_name.setdefault(method.name, []).append(m_node_id)

            for fn in pf.functions:
                self.functions_by_qname[fn.qualified_name] = fn
                self.functions_by_short_name_in_file[pf.file_path][fn.name] = fn
                f_node_id = f"func:{fn.qualified_name}"
                self.all_callables_by_short_name.setdefault(fn.name, []).append(f_node_id)

        # Build imports table per file
        self.file_imports: dict[str, dict[str, ResolvedImport]] = {}
        self._build_import_tables(parsed_files)

        # Build inheritance hierarchy
        self.class_ancestors: dict[str, list[str]] = {}
        self._build_class_hierarchy()

    def _build_import_tables(self, parsed_files: Sequence[ParsedFile]) -> None:
        """Construct dictionary of resolved imports for each source file."""
        for pf in parsed_files:
            file_table: dict[str, ResolvedImport] = {}
            for imp in pf.imports:
                if imp.level > 0:
                    abs_module = resolve_relative_module(
                        pf.file_path, pf.module_name, imp.level, imp.module
                    )
                else:
                    abs_module = imp.module

                is_internal_mod = (
                    abs_module in self.module_to_file
                    or any(m.startswith(f"{abs_module}.") for m in self.module_to_file)
                )

                if not imp.is_from_import:
                    # 'import x' or 'import x as y'
                    local_name = imp.alias if imp.alias else imp.module.split(".")[0]
                    file_table[local_name] = ResolvedImport(
                        local_name=local_name,
                        is_module=True,
                        target_module=abs_module,
                        target_symbol=None,
                        is_internal=is_internal_mod,
                        target_file=self.module_to_file.get(abs_module),
                    )
                    # Also register full module alias if dotted import without alias
                    if not imp.alias and "." in imp.module:
                        file_table[imp.module] = ResolvedImport(
                            local_name=imp.module,
                            is_module=True,
                            target_module=abs_module,
                            target_symbol=None,
                            is_internal=is_internal_mod,
                            target_file=self.module_to_file.get(abs_module),
                        )
                else:
                    # 'from x import y'
                    if imp.name == "*":
                        # Wildcard import: pull in all symbols defined in target module
                        target_file_path = self.module_to_file.get(abs_module)
                        if target_file_path and target_file_path in self.files_by_path:
                            target_pf = self.files_by_path[target_file_path]
                            for fn in target_pf.functions:
                                file_table[fn.name] = ResolvedImport(
                                    local_name=fn.name,
                                    is_module=False,
                                    target_module=abs_module,
                                    target_symbol=fn.name,
                                    is_internal=True,
                                    target_file=target_file_path,
                                )
                            for cls in target_pf.classes:
                                file_table[cls.name] = ResolvedImport(
                                    local_name=cls.name,
                                    is_module=False,
                                    target_module=abs_module,
                                    target_symbol=cls.name,
                                    is_internal=True,
                                    target_file=target_file_path,
                                )
                        continue

                    local_name = imp.alias if imp.alias else (imp.name or "")
                    submodule_candidate = f"{abs_module}.{imp.name}" if abs_module else (imp.name or "")

                    if submodule_candidate in self.module_to_file:
                        # Imported symbol is actually a submodule
                        file_table[local_name] = ResolvedImport(
                            local_name=local_name,
                            is_module=True,
                            target_module=submodule_candidate,
                            target_symbol=None,
                            is_internal=True,
                            target_file=self.module_to_file.get(submodule_candidate),
                        )
                    else:
                        file_table[local_name] = ResolvedImport(
                            local_name=local_name,
                            is_module=False,
                            target_module=abs_module,
                            target_symbol=imp.name,
                            is_internal=is_internal_mod,
                            target_file=self.module_to_file.get(abs_module),
                        )

            self.file_imports[pf.file_path] = file_table

        # Second pass: follow re-exports one level through __init__.py (Rule 2c)
        for pf in parsed_files:
            table = self.file_imports[pf.file_path]
            for local_name, r_imp in list(table.items()):
                if (
                    r_imp.is_internal
                    and r_imp.target_symbol
                    and r_imp.target_file
                    and Path(r_imp.target_file).name == "__init__.py"
                ):
                    init_pf = self.files_by_path.get(r_imp.target_file)
                    if init_pf:
                        # Check if target_symbol is defined directly in __init__.py
                        has_fn = r_imp.target_symbol in self.functions_by_short_name_in_file.get(
                            init_pf.file_path, {}
                        )
                        has_cls = r_imp.target_symbol in self.classes_by_short_name_in_file.get(
                            init_pf.file_path, {}
                        )
                        if not (has_fn or has_cls):
                            # Not defined directly; check if imported in __init__.py
                            init_table = self.file_imports.get(init_pf.file_path, {})
                            if r_imp.target_symbol in init_table:
                                reexported = init_table[r_imp.target_symbol]
                                table[local_name] = ResolvedImport(
                                    local_name=local_name,
                                    is_module=reexported.is_module,
                                    target_module=reexported.target_module,
                                    target_symbol=reexported.target_symbol or r_imp.target_symbol,
                                    is_internal=reexported.is_internal,
                                    target_file=reexported.target_file,
                                )

    def _build_class_hierarchy(self) -> None:
        """Resolve base classes and precompute MRO inheritance chains."""
        direct_bases: dict[str, list[str]] = {}

        for cls_qname, cls in self.classes_by_qname.items():
            class_id = f"class:{cls_qname}"
            resolved_base_ids: list[str] = []
            self.files_by_path[cls.file_path]
            table = self.file_imports.get(cls.file_path, {})

            for base_expr in cls.bases:
                base_expr = base_expr.strip()
                if not base_expr:
                    continue

                if "." not in base_expr:
                    # Bare base name: check same file first
                    if base_expr in self.classes_by_short_name_in_file.get(cls.file_path, {}):
                        cand_cls = self.classes_by_short_name_in_file[cls.file_path][base_expr]
                        resolved_base_ids.append(f"class:{cand_cls.qualified_name}")
                    elif base_expr in table:
                        r_imp = table[base_expr]
                        if r_imp.is_internal:
                            target_qname = f"{r_imp.target_module}.{r_imp.target_symbol or base_expr}"
                            if target_qname in self.classes_by_qname:
                                resolved_base_ids.append(f"class:{target_qname}")
                            else:
                                resolved_base_ids.append(f"external:{base_expr}")
                        else:
                            resolved_base_ids.append(f"external:{r_imp.target_module}.{base_expr}")
                    else:
                        resolved_base_ids.append(f"external:{base_expr}")
                else:
                    # Dotted base expression: e.g. base.Model
                    parts = base_expr.split(".")
                    prefix = parts[0]
                    suffix = ".".join(parts[1:])
                    if prefix in table:
                        r_imp = table[prefix]
                        if r_imp.is_internal:
                            target_qname = f"{r_imp.target_module}.{suffix}"
                            if target_qname in self.classes_by_qname:
                                resolved_base_ids.append(f"class:{target_qname}")
                            else:
                                resolved_base_ids.append(f"external:{base_expr}")
                        else:
                            resolved_base_ids.append(f"external:{base_expr}")
                    else:
                        resolved_base_ids.append(f"external:{base_expr}")

            direct_bases[class_id] = resolved_base_ids

        # Walk ancestors for each class using BFS to prevent cycles
        for class_id in direct_bases:
            ancestors: list[str] = []
            visited: set[str] = {class_id}
            queue = list(direct_bases.get(class_id, []))

            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                if curr.startswith("class:"):
                    ancestors.append(curr)
                    queue.extend(direct_bases.get(curr, []))

            self.class_ancestors[class_id] = ancestors

    def get_class_ancestors(self, class_id: str) -> list[str]:
        """Return list of ancestor class node IDs in BFS hierarchy order."""
        return self.class_ancestors.get(class_id, [])

    def resolve_call(
        self,
        caller: FunctionInfo,
        call: CallInfo,
        file_path: str,
        include_ambiguous: bool = False,
    ) -> ResolutionResult:
        """Resolve a single AST call expression using the 7 ordered strategy rules.

        Args:
            caller: Enclosing function or method where invocation occurred.
            call: Extracted call information.
            file_path: File path of caller.
            include_ambiguous: Whether to create edges for ambiguous candidates.

        Returns:
            ResolutionResult describing matched targets and rule type.
        """
        pf = self.files_by_path.get(file_path)
        if not pf:
            return ResolutionResult(is_unresolved=True)

        call_name = call.name.strip()
        import_table = self.file_imports.get(file_path, {})

        # ---------------------------------------------------------------------
        # Rule 1: self.name() / cls.name()
        # ---------------------------------------------------------------------
        if caller.is_method and caller.parent_class:
            if call_name.startswith("self.") or call_name.startswith("cls."):
                parts = call_name.split(".")
                if len(parts) == 2:
                    method_name = parts[1]
                    caller_class_qname = f"{pf.module_name}.{caller.parent_class}"
                    caller_class_id = f"class:{caller_class_qname}"

                    # 1a: Look for method in the same class
                    cand_method_qname = f"{caller_class_qname}.{method_name}"
                    if cand_method_qname in self.functions_by_qname:
                        return ResolutionResult(
                            targets=[
                                CallTarget(
                                    node_id=f"method:{cand_method_qname}",
                                    confidence="exact",
                                    resolution="self_method",
                                )
                            ],
                            rule_name="self_method",
                        )

                    # 1b: Walk up INHERITS parents
                    for ancestor_id in self.get_class_ancestors(caller_class_id):
                        ancestor_qname = ancestor_id.split(":", 1)[1]
                        cand_ancestor_method = f"{ancestor_qname}.{method_name}"
                        if cand_ancestor_method in self.functions_by_qname:
                            return ResolutionResult(
                                targets=[
                                    CallTarget(
                                        node_id=f"method:{cand_ancestor_method}",
                                        confidence="high",
                                        resolution="self_method",
                                    )
                                ],
                                rule_name="self_method",
                            )

        # ---------------------------------------------------------------------
        # Rule 2: Bare name foo()
        # ---------------------------------------------------------------------
        if "." not in call_name:
            # 2a: Defined in the same module
            same_mod_qname = f"{pf.module_name}.{call_name}"
            if same_mod_qname in self.functions_by_qname:
                cand_fn = self.functions_by_qname[same_mod_qname]
                if not cand_fn.is_method:
                    return ResolutionResult(
                        targets=[
                            CallTarget(
                                node_id=f"func:{same_mod_qname}",
                                confidence="exact",
                                resolution="same_module",
                            )
                        ],
                        rule_name="same_module",
                    )
            # Or same module class with __init__
            if call_name in self.classes_by_short_name_in_file.get(file_path, {}):
                cand_cls = self.classes_by_short_name_in_file[file_path][call_name]
                init_qname = f"{cand_cls.qualified_name}.__init__"
                if init_qname in self.functions_by_qname:
                    return ResolutionResult(
                        targets=[
                            CallTarget(
                                node_id=f"method:{init_qname}",
                                confidence="exact",
                                resolution="same_module",
                            )
                        ],
                        rule_name="same_module",
                    )

            # 2b & 2c: Name brought in via import (including re-export)
            if call_name in import_table:
                r_imp = import_table[call_name]
                if r_imp.is_internal:
                    target_qname = f"{r_imp.target_module}.{r_imp.target_symbol or call_name}"
                    if target_qname in self.functions_by_qname:
                        cand = self.functions_by_qname[target_qname]
                        pfx = "method" if cand.is_method else "func"
                        return ResolutionResult(
                            targets=[
                                CallTarget(
                                    node_id=f"{pfx}:{target_qname}",
                                    confidence="exact",
                                    resolution="import",
                                )
                            ],
                            rule_name="import",
                        )
                    # Check class __init__
                    if target_qname in self.classes_by_qname:
                        init_qname = f"{target_qname}.__init__"
                        if init_qname in self.functions_by_qname:
                            return ResolutionResult(
                                targets=[
                                    CallTarget(
                                        node_id=f"method:{init_qname}",
                                        confidence="exact",
                                        resolution="import",
                                    )
                                ],
                                rule_name="import",
                            )
                else:
                    return ResolutionResult(is_external=True)

            # Check if builtin
            if call_name in BUILTIN_NAMES:
                return ResolutionResult(is_builtin=True)

        # ---------------------------------------------------------------------
        # Rule 3: Dotted module.foo() or pkg.module.foo()
        # ---------------------------------------------------------------------
        if "." in call_name:
            parts = call_name.split(".")
            for i in range(len(parts) - 1, 0, -1):
                prefix = ".".join(parts[:i])
                suffix = ".".join(parts[i:])
                if prefix in import_table:
                    r_imp = import_table[prefix]
                    if r_imp.is_internal and r_imp.is_module:
                        target_qname = f"{r_imp.target_module}.{suffix}"
                        if target_qname in self.functions_by_qname:
                            cand = self.functions_by_qname[target_qname]
                            pfx = "method" if cand.is_method else "func"
                            return ResolutionResult(
                                targets=[
                                    CallTarget(
                                        node_id=f"{pfx}:{target_qname}",
                                        confidence="exact",
                                        resolution="import",
                                    )
                                ],
                                rule_name="import",
                            )
                        # Check class __init__
                        if target_qname in self.classes_by_qname:
                            init_qname = f"{target_qname}.__init__"
                            if init_qname in self.functions_by_qname:
                                return ResolutionResult(
                                    targets=[
                                        CallTarget(
                                            node_id=f"method:{init_qname}",
                                            confidence="exact",
                                            resolution="import",
                                        )
                                    ],
                                    rule_name="import",
                                )
                    elif not r_imp.is_internal:
                        return ResolutionResult(is_external=True)

        # ---------------------------------------------------------------------
        # Rule 4: ClassName.method() where ClassName is internal visible class
        # ---------------------------------------------------------------------
        if "." in call_name:
            parts = call_name.split(".")
            if len(parts) == 2:
                class_name, method_name = parts[0], parts[1]
                target_cls: ClassInfo | None = None

                if class_name in self.classes_by_short_name_in_file.get(file_path, {}):
                    target_cls = self.classes_by_short_name_in_file[file_path][class_name]
                elif class_name in import_table:
                    r_imp = import_table[class_name]
                    if r_imp.is_internal:
                        cls_qname = f"{r_imp.target_module}.{r_imp.target_symbol or class_name}"
                        if cls_qname in self.classes_by_qname:
                            target_cls = self.classes_by_qname[cls_qname]

                if target_cls:
                    cand_method_qname = f"{target_cls.qualified_name}.{method_name}"
                    if cand_method_qname in self.functions_by_qname:
                        return ResolutionResult(
                            targets=[
                                CallTarget(
                                    node_id=f"method:{cand_method_qname}",
                                    confidence="high",
                                    resolution="class_method",
                                )
                            ],
                            rule_name="class_method",
                        )
                    # Check base classes of target_cls
                    for anc_id in self.get_class_ancestors(f"class:{target_cls.qualified_name}"):
                        anc_qname = anc_id.split(":", 1)[1]
                        anc_method_qname = f"{anc_qname}.{method_name}"
                        if anc_method_qname in self.functions_by_qname:
                            return ResolutionResult(
                                targets=[
                                    CallTarget(
                                        node_id=f"method:{anc_method_qname}",
                                        confidence="high",
                                        resolution="class_method",
                                    )
                                ],
                                rule_name="class_method",
                            )

        # Dotted call on known external import (e.g. os.path.join, logging.getLogger)
        if "." in call_name:
            top_pkg = call_name.split(".")[0]
            if top_pkg in import_table and not import_table[top_pkg].is_internal:
                return ResolutionResult(is_external=True)

        # ---------------------------------------------------------------------
        # Fallback Rule 5 & 6: Unique name match or Ambiguous
        # ---------------------------------------------------------------------
        leaf_name = call_name.split(".")[-1]
        if leaf_name in BUILTIN_NAMES:
            return ResolutionResult(is_builtin=True)

        candidates = self.all_callables_by_short_name.get(leaf_name, [])

        # Rule 5: Exactly one match across entire repository
        if len(candidates) == 1:
            return ResolutionResult(
                targets=[
                    CallTarget(
                        node_id=candidates[0],
                        confidence="low",
                        resolution="unique_name",
                    )
                ],
                rule_name="unique_name",
            )

        # Rule 6: Multiple candidates (ambiguous)
        if len(candidates) > 1:
            if include_ambiguous:
                return ResolutionResult(
                    targets=[
                        CallTarget(
                            node_id=cand_id,
                            confidence="low",
                            resolution="ambiguous",
                        )
                        for cand_id in candidates
                    ],
                    rule_name="ambiguous",
                )
            else:
                return ResolutionResult(is_ambiguous_skipped=True)

        # Rule 7: Unresolved
        return ResolutionResult(is_unresolved=True)
