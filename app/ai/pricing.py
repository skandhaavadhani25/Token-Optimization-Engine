"""Model pricing table and cost calculation utilities.

PRICING NOTICE:
By default, the pricing table ships empty so no arbitrary or outdated prices are invented.
Users can define actual pricing by creating a `pricing.json` file in the project root or
configuring model pricing rates. Models can also be marked as free tiers: `{"free": true}`.

Example `pricing.json` format:
{
  "claude-3-5-sonnet-20241022": {
    "input_cost_per_million": 3.00,
    "output_cost_per_million": 15.00
  },
  "gemini-2.5-flash": {
    "free": true
  },
  "ollama": {
    "free": true
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default pricing table is empty by default
DEFAULT_PRICING: dict[str, dict[str, Any]] = {}


def load_pricing_table(pricing_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Load pricing definitions from pricing.json or return default empty table."""
    table: dict[str, dict[str, Any]] = dict(DEFAULT_PRICING)
    path = Path(pricing_path) if pricing_path else Path("pricing.json")
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for model_id, rates in data.items():
                    if isinstance(rates, dict):
                        is_free = bool(rates.get("free", False))
                        in_cost = float(rates.get("input_cost_per_million", 0.0))
                        out_cost = float(rates.get("output_cost_per_million", 0.0))
                        table[str(model_id)] = {
                            "free": is_free,
                            "input_cost_per_million": in_cost,
                            "output_cost_per_million": out_cost,
                        }
        except Exception:
            pass
    return table


def is_model_free(
    model: str,
    provider: str | None = None,
    pricing_table: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Return True if model is known to be free tier or runs locally."""
    # Local providers have zero API cost
    if provider and provider.lower() in ("ollama", "local"):
        return True

    table = pricing_table if pricing_table is not None else load_pricing_table()
    if model in table and table[model].get("free") is True:
        return True

    # Free tag in model ID (e.g. openrouter :free models)
    if ":free" in model.lower():
        return True

    return False


def estimate_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    pricing_table: dict[str, dict[str, Any]] | None = None,
    provider: str | None = None,
) -> float | None:
    """Calculate estimated cost in USD for a completion, or None if model pricing is unknown."""
    table = pricing_table if pricing_table is not None else load_pricing_table()

    if is_model_free(model, provider=provider, pricing_table=table):
        return 0.0

    if model not in table:
        return None

    rates = table[model]
    in_tok = input_tokens or 0
    out_tok = output_tokens or 0
    input_cost = (in_tok / 1_000_000.0) * rates.get("input_cost_per_million", 0.0)
    output_cost = (out_tok / 1_000_000.0) * rates.get("output_cost_per_million", 0.0)
    return round(input_cost + output_cost, 6)


def get_cost_info(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    provider: str | None = None,
    pricing_table: dict[str, dict[str, Any]] | None = None,
) -> tuple[float | None, bool, str]:
    """Return tuple of (cost_usd, is_free, cost_display_str)."""
    table = pricing_table if pricing_table is not None else load_pricing_table()
    if is_model_free(model, provider=provider, pricing_table=table):
        return 0.0, True, "free tier"

    cost = estimate_cost(model, input_tokens, output_tokens, pricing_table=table, provider=provider)
    if cost is not None:
        return cost, False, f"${cost:.6f}"
    return None, False, "unknown (no pricing configured)"
