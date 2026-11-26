"""Token budget tracking and allocation logic for prompt context construction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BudgetTooSmallError(ValueError):
    """Raised when the token budget is too small to fit even minimal essential context."""


class TokenBudget(BaseModel):
    """Token budget configuration and state tracker.

    Guarantees that context assembly respects allocated token limits,
    tracks consumption, and reserves slices for overhead and user requests.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_budget: int = Field(default=4000, ge=0, description="Total maximum token budget")
    reserved_overhead: int = Field(default=300, ge=0, description="Reserved tokens for section labels & headers")
    reserved_for_task: int = Field(default=0, ge=0, description="Tokens reserved for user task query")
    target_share: float = Field(
        default=0.45,
        gt=0.0,
        le=1.0,
        description="Maximum fraction of total budget allocated to the target entity before downgrade",
    )
    min_item_tokens: int = Field(
        default=20,
        ge=1,
        description="Tokens threshold below which items are considered negligible for separate tracking",
    )
    consumed: int = Field(default=0, ge=0, description="Total tokens consumed so far")

    def remaining(self) -> int:
        """Calculate remaining unused token budget."""
        return max(0, self.total_budget - self.consumed)

    def can_afford(self, tokens: int) -> bool:
        """Check if an item costing `tokens` can fit in the remaining budget."""
        return self.remaining() >= tokens

    def consume(self, tokens: int) -> None:
        """Record consumption of tokens."""
        if tokens < 0:
            raise ValueError(f"Cannot consume negative tokens: {tokens}")
        self.consumed += tokens

    def utilization(self) -> float:
        """Compute current fraction of total budget consumed (0.0 to 1.0)."""
        if self.total_budget <= 0:
            return 0.0
        return min(1.0, self.consumed / self.total_budget)

    def target_max_tokens(self) -> int:
        """Calculate maximum tokens allowed for the target entity at FULL level."""
        return int(self.total_budget * self.target_share)

    def reset(self) -> None:
        """Reset consumed tokens counter back to 0."""
        self.consumed = 0
