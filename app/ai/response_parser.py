"""Structured parser for AI completion responses."""

from __future__ import annotations

import ast
import re
from typing import Container
from pydantic import BaseModel, ConfigDict, Field


class ProposedChange(BaseModel):
    """A proposed full replacement for a code entity."""

    model_config = ConfigDict(frozen=True)

    qualified_name: str = Field(description="Target entity qualified name")
    file: str = Field(description="Relative file path where the entity belongs")
    code: str = Field(description="Complete source code replacement")
    language: str = Field(default="python", description="Code language")
    syntax_valid: bool = Field(default=True, description="Whether code parses successfully with ast.parse")
    unknown_target: bool = Field(default=False, description="Whether target symbol is not found in the index")


class ParsedAnswer(BaseModel):
    """Structured decomposition of an AI coding response."""

    model_config = ConfigDict(frozen=True)

    analysis: str = Field(default="", description="Technical analysis and explanation")
    changes: list[ProposedChange] = Field(default_factory=list, description="List of proposed code replacements")
    risks: list[str] = Field(default_factory=list, description="Identified risks, trade-offs, or assumptions")
    needs_context: list[str] = Field(default_factory=list, description="Missing symbols or files requested by AI")
    raw_text: str = Field(description="Original unparsed completion text")
    parse_warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings encountered during parsing")


_CHANGE_HEADER_PATTERN = re.compile(
    r"^#{2,4}\s*CHANGE\b\s*:?\s*([^\s(]+)(?:\s*\(([^)]+)\))?",
    re.IGNORECASE | re.MULTILINE,
)

_CODE_BLOCK_PATTERN = re.compile(
    r"```([a-zA-Z0-9_-]*)[ \t]*\r?\n?(.*?)\r?\n?```",
    re.DOTALL,
)


def parse_response(
    raw_text: str,
    known_targets: Container[str] | None = None,
) -> ParsedAnswer:
    """Parse raw LLM response text into structured ParsedAnswer model."""
    warnings: list[str] = []
    if not raw_text or not raw_text.strip():
        return ParsedAnswer(
            analysis="",
            changes=[],
            risks=[],
            needs_context=[],
            raw_text=raw_text or "",
            parse_warnings=["Empty or blank response from AI model"],
        )

    # Section extraction using standard section markers (tolerant to ##, ###, ####)
    analysis_text = ""
    risks_list: list[str] = []
    needs_context_list: list[str] = []
    changes_list: list[ProposedChange] = []

    # 1. Parse sections using splitters
    sections = re.split(r"(?m)^#{2,4}\s+", raw_text)

    for sec in sections:
        sec_clean = sec.strip()
        if not sec_clean:
            continue

        first_line, _, rest = sec_clean.partition("\n")
        header = first_line.strip().upper()

        if header.startswith("ANALYSIS"):
            analysis_text = rest.strip()
        elif header.startswith("RISKS"):
            for line in rest.splitlines():
                line_str = line.strip()
                if line_str.startswith(("-", "*", "•")):
                    item = line_str.lstrip("-*• ").strip()
                    if item:
                        risks_list.append(item)
                elif re.match(r"^\d+\.\s+", line_str):
                    item = re.sub(r"^\d+\.\s+", "", line_str).strip()
                    if item:
                        risks_list.append(item)
        elif header.startswith("NEEDS_CONTEXT") or header.startswith("NEEDS CONTEXT"):
            for line in rest.splitlines():
                line_str = line.strip()
                if line_str.startswith(("-", "*", "•")):
                    item = line_str.lstrip("-*• ").strip()
                    if item:
                        needs_context_list.append(item)
                elif line_str and not line_str.startswith("#"):
                    needs_context_list.append(line_str)

    if not risks_list:
        warnings.append("No RISKS section provided in model response.")

    # 2. Extract proposed changes
    # Look for headers: ### CHANGE: <qualified_name> (<file>) or ## CHANGE
    matches = list(_CHANGE_HEADER_PATTERN.finditer(raw_text))
    if matches:
        for i, match in enumerate(matches):
            qual_name = match.group(1).strip()
            file_path = (match.group(2) or "").strip()

            # Find the block following this header up to the next change header or end
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            chunk = raw_text[start_pos:end_pos]

            code_match = _CODE_BLOCK_PATTERN.search(chunk)
            if code_match:
                lang = (code_match.group(1) or "python").strip()
                code_content = code_match.group(2).rstrip()
            else:
                lang = "python"
                code_content = chunk.strip()
                warnings.append(f"No fenced code block found for change: {qual_name}")

            # Validate syntax with AST if Python
            syntax_valid = True
            if lang in ("python", "py", ""):
                try:
                    ast.parse(code_content)
                except SyntaxError as syn_err:
                    syntax_valid = False
                    warnings.append(
                        f"Syntax error in proposed change for '{qual_name}' at line {syn_err.lineno}: {syn_err.msg}"
                    )

            # Check unknown target
            unknown = False
            if known_targets is not None:
                unknown = qual_name not in known_targets

            changes_list.append(
                ProposedChange(
                    qualified_name=qual_name,
                    file=file_path,
                    code=code_content,
                    language=lang or "python",
                    syntax_valid=syntax_valid,
                    unknown_target=unknown,
                )
            )

    # Fallback if no structured analysis was found
    if not analysis_text and not changes_list:
        analysis_text = raw_text.strip()
        warnings.append("Output did not follow standard section layout; full text retained in analysis.")

    return ParsedAnswer(
        analysis=analysis_text,
        changes=changes_list,
        risks=risks_list,
        needs_context=needs_context_list,
        raw_text=raw_text,
        parse_warnings=warnings,
    )
