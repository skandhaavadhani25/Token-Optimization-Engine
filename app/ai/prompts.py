"""Prompt assembly and template loader for coding agent completions."""

from __future__ import annotations

from pathlib import Path
from app.ai.base_provider import ChatMessage

DEFAULT_SYSTEM_PROMPT = """You are a senior software engineer working on a codebase repository.
You are provided with a CONTEXT block assembled by a retrieval engine. You may not see the entire codebase.

RULES:
1. Base your implementation and reasoning ONLY on the provided code and definitions.
2. If critical dependencies, types, or helpers are missing, do NOT invent or hallucinate them; explicitly list them in the ### NEEDS_CONTEXT section.
3. Keep technical explanations concise and direct.

OUTPUT CONTRACT:
Your response MUST strictly follow these sections:

### ANALYSIS
Brief explanation of the diagnosis, approach, and implementation strategy.

### CHANGES
For each changed function, method, or class, output a header line:
### CHANGE: <qualified_name> (<file_path>)
Followed immediately by a fenced code block containing the COMPLETE replacement for that entity (not diffs, not incomplete fragments):
```python
<full replacement code>
```

### RISKS
- Bullet points detailing potential regressions, concurrency/state impacts, performance considerations, or edge case assumptions.

### NEEDS_CONTEXT
(Include only if additional context is strictly required to complete or verify the task)
- qualified.name.or.file.path
"""

SIMPLE_SYSTEM_PROMPT = """You are a software engineer modifying code.
Follow this format strictly:

### ANALYSIS
Brief explanation of what you are changing.

### CHANGES
### CHANGE: qualified_name (path/to/file.py)
```python
def example():
    pass
```

### RISKS
- Potential issues or edge cases.

### NEEDS_CONTEXT
- List any missing dependencies or symbols if needed.
"""

DEFAULT_USER_TEMPLATE = """{context}

TASK:
{task}
"""


def load_prompt_templates(
    prompts_dir: Path | str | None = None,
    style: str = "full",
) -> tuple[str, str]:
    """Load system prompt and user template from disk if present, else fallback to defaults."""
    dir_path = Path(prompts_dir) if prompts_dir else Path("prompts")
    sys_file = "system_prompt_simple.txt" if style.lower() == "simple" else "system_prompt.txt"
    sys_path = dir_path / sys_file
    user_path = dir_path / "user_template.txt"

    default_sys = SIMPLE_SYSTEM_PROMPT if style.lower() == "simple" else DEFAULT_SYSTEM_PROMPT
    system_prompt = default_sys
    user_template = DEFAULT_USER_TEMPLATE

    if sys_path.is_file():
        try:
            content = sys_path.read_text(encoding="utf-8").strip()
            if content:
                system_prompt = content
        except Exception:
            pass

    if user_path.is_file():
        try:
            content = user_path.read_text(encoding="utf-8").strip()
            if content:
                user_template = content
        except Exception:
            pass

    return system_prompt, user_template


def build_messages(
    context_text: str,
    task_text: str,
    prompts_dir: Path | str | None = None,
    style: str = "full",
) -> tuple[str, list[ChatMessage]]:
    """Assemble the system prompt and conversation messages for the coding task."""
    system_prompt, user_template = load_prompt_templates(prompts_dir, style=style)
    user_content = user_template.format(context=context_text, task=task_text)

    messages = [
        ChatMessage(role="user", content=user_content),
    ]

    return system_prompt, messages


def load_system_prompt(prompts_dir: Path | str | None = None, style: str = "full") -> str:
    """Load system prompt from disk or default."""
    sys_prompt, _ = load_prompt_templates(prompts_dir, style=style)
    return sys_prompt
