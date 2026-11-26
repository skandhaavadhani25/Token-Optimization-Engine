"""Context formatter rendering structured output in text, markdown, or JSON formats."""

from __future__ import annotations

from typing import Sequence

from app.models.schemas import BuiltContext, IncludedItem


def render_context(
    task: str,
    target_ref: str | None = None,
    target_location: str | None = None,
    dependency_paths: Sequence[str] | None = None,
    related_items: Sequence[IncludedItem] | None = None,
    relevant_imports: dict[str, list[str]] | None = None,
    code_snippets_by_file: dict[str, list[str]] | None = None,
    omitted_items: Sequence[dict[str, str]] | None = None,
    format_type: str = "text",
) -> str:
    """Render structured prompt context according to specification.

    Sections with no content are omitted to preserve token budget.

    Args:
        task: User task description or query string.
        target_ref: Qualified name of the primary target entity.
        target_location: String of form 'file:start-end'.
        dependency_paths: List of shortest dependency paths, e.g. 'a -> b -> c'.
        related_items: List of IncludedItem objects for related code.
        relevant_imports: Mapping of file path -> list of used imported symbols.
        code_snippets_by_file: Mapping of file path -> list of source snippets.
        omitted_items: List of dicts with 'ref' and 'reason' / 'lines'.
        format_type: 'text' (default) or 'markdown'.

    Returns:
        Structured context string formatted to exact layout.
    """
    is_markdown = format_type.strip().lower() in ("markdown", "md")
    sections: list[str] = []

    # 1. TASK section
    if is_markdown:
        sections.append(f"## TASK\n{task.strip()}")
    else:
        sections.append(f"TASK:\n{task.strip()}")

    # 2. TARGET section
    if target_ref:
        loc_str = f"  ({target_location})" if target_location else ""
        if is_markdown:
            sections.append(f"## TARGET\n`{target_ref}`{loc_str}")
        else:
            sections.append(f"TARGET:\n{target_ref}{loc_str}")

    # 3. DEPENDENCY MAP section
    if dependency_paths:
        paths_text = "\n".join(dependency_paths)
        if is_markdown:
            sections.append(f"## DEPENDENCY MAP\n```text\n{paths_text}\n```")
        else:
            sections.append(f"DEPENDENCY MAP:\n{paths_text}")

    # 4. RELATED FUNCTIONS / ENTITIES section
    if related_items:
        lines: list[str] = []
        for item in related_items:
            loc = f", {item.file}:{item.lines}" if item.lines else f", {item.file}"
            lines.append(f"- {item.ref} ({item.role}{loc}) [{item.level}]")
        rel_text = "\n".join(lines)
        if is_markdown:
            sections.append(f"## RELATED FUNCTIONS\n{rel_text}")
        else:
            sections.append(f"RELATED FUNCTIONS:\n{rel_text}")

    # 5. RELEVANT IMPORTS section
    if relevant_imports:
        imp_lines: list[str] = []
        for fpath, imps in sorted(relevant_imports.items()):
            if imps:
                unique_imps = sorted(list(dict.fromkeys(imps)))
                if is_markdown:
                    imp_lines.append(f"- **{fpath}**: {', '.join(unique_imps)}")
                else:
                    imp_lines.append(f"{fpath}: {', '.join(unique_imps)}")
        if imp_lines:
            imp_text = "\n".join(imp_lines)
            if is_markdown:
                sections.append(f"## RELEVANT IMPORTS\n{imp_text}")
            else:
                sections.append(f"RELEVANT IMPORTS:\n{imp_text}")

    # 6. SOURCE CODE section
    if code_snippets_by_file:
        file_blocks: list[str] = []
        for fpath, snippets in code_snippets_by_file.items():
            if not snippets:
                continue
            snip_joined = "\n\n".join(s.rstrip() for s in snippets if s.strip())
            if is_markdown:
                file_blocks.append(f"### {fpath}\n```python\n{snip_joined}\n```")
            else:
                file_blocks.append(f"# {fpath}\n{snip_joined}")

        if file_blocks:
            src_header = "## SOURCE CODE" if is_markdown else "SOURCE CODE:"
            sections.append(f"{src_header}\n" + "\n\n".join(file_blocks))

    # 7. OMITTED section
    if omitted_items:
        omit_lines: list[str] = []
        for it in omitted_items:
            ref = it.get("ref", "")
            loc = it.get("lines") or it.get("file") or ""
            it.get("reason", "")
            detail = f" ({loc})" if loc else ""
            omit_lines.append(f"- {ref}{detail}")
        if omit_lines:
            header = "## OMITTED (lower relevance, available on request)" if is_markdown else "OMITTED (lower relevance, available on request):"
            sections.append(f"{header}\n" + "\n".join(omit_lines))

    return "\n\n".join(sections).strip() + "\n"


def _text_to_markdown(text: str) -> str:
    """Transform text-formatted context into markdown format with headings and code fences."""
    if text.startswith("## ") or "\n## " in text:
        return text

    lines = text.splitlines()
    out: list[str] = []
    in_code_block = False
    in_dep_map = False

    for line in lines:
        if line == "TASK:":
            out.append("## TASK")
        elif line == "TARGET:":
            out.append("## TARGET")
        elif line == "DEPENDENCY MAP:":
            out.append("## DEPENDENCY MAP\n```text")
            in_dep_map = True
        elif line == "RELATED FUNCTIONS:":
            if in_dep_map:
                out.append("```")
                in_dep_map = False
            out.append("## RELATED FUNCTIONS")
        elif line == "RELEVANT IMPORTS:":
            if in_dep_map:
                out.append("```")
                in_dep_map = False
            out.append("## RELEVANT IMPORTS")
        elif line == "SOURCE CODE:":
            if in_dep_map:
                out.append("```")
                in_dep_map = False
            out.append("## SOURCE CODE")
        elif line.startswith("OMITTED"):
            if in_code_block:
                out.append("```")
                in_code_block = False
            if in_dep_map:
                out.append("```")
                in_dep_map = False
            out.append("## OMITTED (lower relevance, available on request)")
        elif line.startswith("# ") and "." in line and ":" not in line and not line.endswith("]"):
            if in_code_block:
                out.append("```")
            out.append(f"### {line[2:].strip()}\n```python")
            in_code_block = True
        else:
            out.append(line)

    if in_code_block:
        out.append("```")
    if in_dep_map:
        out.append("```")

    return "\n".join(out).strip() + "\n"


def format_context(context: BuiltContext, fmt: str = "text") -> str:
    """Format a BuiltContext model into text, markdown, or JSON."""
    normalized_fmt = fmt.strip().lower()
    if normalized_fmt == "json":
        return context.model_dump_json(indent=2)
    elif normalized_fmt in ("markdown", "md"):
        return _text_to_markdown(context.text)
    return context.text
