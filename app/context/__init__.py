"""Context builder package (Step 7)."""

from app.context.context_builder import ContextBuilder
from app.context.context_compressor import compress, compress_to_fit
from app.context.formatter import format_context, render_context
from app.context.token_budget import BudgetTooSmallError, TokenBudget
from app.context.token_counter import (
    CharEstimateCounter,
    TiktokenCounter,
    TokenCounter,
    get_counter,
)

__all__ = [
    "ContextBuilder",
    "TokenBudget",
    "BudgetTooSmallError",
    "compress",
    "compress_to_fit",
    "render_context",
    "format_context",
    "TokenCounter",
    "TiktokenCounter",
    "CharEstimateCounter",
    "get_counter",
]
