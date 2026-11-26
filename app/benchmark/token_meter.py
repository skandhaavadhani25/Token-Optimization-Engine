"""Unified token measurement combining real provider counts and local tiktoken estimation."""

from __future__ import annotations


from pydantic import BaseModel

from app.ai.base_provider import AIProvider, ChatMessage
from app.ai.pricing import estimate_cost
from app.context.token_counter import get_token_counter


class TokenMeasurement(BaseModel):
    """Result of measuring token usage on a context or prompt."""

    tiktoken_tokens: int
    exact_tokens: int | None = None
    counter: str = "tiktoken"  # "provider" or "tiktoken"
    chars: int
    est_cost_usd: float | None = None

    @property
    def effective_tokens(self) -> int:
        """Effective token count (provider-exact if available, else tiktoken)."""
        return self.exact_tokens if self.exact_tokens is not None else self.tiktoken_tokens


def measure_tokens(
    context_text: str,
    system_prompt: str | None = None,
    task_text: str | None = None,
    provider: AIProvider | None = None,
    model: str | None = None,
) -> TokenMeasurement:
    """Measure token counts for a context text with fallback to cl100k_base.

    Never fabricates provider token counts or costs.
    """
    counter = get_token_counter("cl100k_base")

    # Combine text for holistic measurement
    parts = []
    if system_prompt:
        parts.append(system_prompt)
    parts.append(context_text)
    if task_text:
        parts.append(task_text)
    full_text = "\n\n".join(parts)

    tiktoken_count = counter.count_tokens(full_text)
    exact_count: int | None = None
    active_counter = "tiktoken"

    if provider is not None and provider.is_configured():
        user_content = context_text
        if task_text:
            user_content = f"{context_text}\n\nTASK:\n{task_text}"
        messages = [ChatMessage(role="user", content=user_content)]

        exact_val = provider.count_tokens(system=system_prompt or "", messages=messages)
        if exact_val is not None:
            exact_count = exact_val
            active_counter = "provider"

    # Cost estimation
    eval_model = model or (provider.config.model if provider else None)
    cost: float | None = None
    if eval_model:
        effective_tokens = exact_count if exact_count is not None else tiktoken_count
        cost = estimate_cost(eval_model, input_tokens=effective_tokens, output_tokens=0)

    return TokenMeasurement(
        tiktoken_tokens=tiktoken_count,
        exact_tokens=exact_count,
        counter=active_counter,
        chars=len(full_text),
        est_cost_usd=cost,
    )
