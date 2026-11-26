"""Benchmark runner executing queries across baseline and graph strategies."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.ai.base_provider import AIProvider, ChatMessage
from app.ai.prompts import load_system_prompt
from app.ai.provider_factory import get_provider
from app.ai.response_parser import parse_response
from app.benchmark.baselines import ALL_STRATEGIES, ContextStrategy
from app.benchmark.token_meter import measure_tokens
from app.config import get_settings
from app.context.token_counter import get_token_counter
from app.storage.metadata_store import MetadataStore


class BenchmarkQuery(BaseModel):
    """Schema for a single benchmark evaluation query."""

    query: str
    task: str = ""
    expected_target: str | None = None
    must_see: list[str] = Field(default_factory=list)
    notes: str = ""


class TaskOutcome(BaseModel):
    """Evaluation of an LLM completion for a strategy context."""

    answered: bool = False
    needs_context: list[str] = Field(default_factory=list)
    unknown_target_changes: list[str] = Field(default_factory=list)
    raw_response: str | None = None


class BenchmarkRow(BaseModel):
    """Single measurement row for a (query, strategy) pair."""

    query: str
    task: str
    strategy_name: str
    files_inspected: list[str] = Field(default_factory=list)
    file_count: int = 0
    context_chars: int = 0
    tiktoken_tokens: int = 0
    exact_tokens: int | None = None
    effective_tokens: int = 0
    counter: str = "tiktoken"
    build_latency_seconds: float = 0.0
    truncated: bool = False
    skipped_over_limit: bool = False
    notes: str = ""

    # Must-see coverage
    must_see_total: int = 0
    must_see_found: int = 0
    must_see_missing: list[str] = Field(default_factory=list)
    coverage_pct: float = 100.0

    # AI metrics (populated when call_ai=True)
    real_input_tokens: int | None = None
    real_output_tokens: int | None = None
    real_total_tokens: int | None = None
    ai_latency_seconds: float | None = None
    est_cost_usd: float | None = None
    outcome: TaskOutcome | None = None


class StrategySummary(BaseModel):
    """Aggregated metrics for a strategy across all benchmark queries."""

    strategy_name: str
    query_count: int
    mean_context_tokens: float
    median_context_tokens: float
    mean_files_inspected: float
    mean_build_latency_seconds: float
    over_limit_count: int
    mean_coverage_pct: float

    # Reductions vs graph
    mean_of_reductions_pct: float = 0.0
    reduction_of_means_pct: float = 0.0

    # AI stats (when call_ai=True)
    mean_real_input_tokens: float | None = None
    mean_real_output_tokens: float | None = None
    mean_ai_latency_seconds: float | None = None
    mean_cost_usd: float | None = None
    answered_rate: float | None = None
    needs_context_rate: float | None = None


class BenchmarkConfig(BaseModel):
    """Runtime configuration for benchmark execution."""

    model_context_limit: int = 200_000
    max_safety_chars: int = 2_000_000
    fixed_window_n: int = 10
    system_prompt_path: str | None = None


class BenchmarkResult(BaseModel):
    """Complete benchmark execution report."""

    repo_path: str
    repo_file_count: int
    repo_total_tokens: int
    model_name: str
    counter_used: str  # "provider" or "tiktoken"
    budget: int
    call_ai: bool
    repeat: int = 1
    rows: list[BenchmarkRow]
    summaries: dict[str, StrategySummary]
    reduction_explanations: dict[str, str] = Field(default_factory=dict)


def compute_repo_total_tokens(repo_path: Path | str) -> tuple[int, int]:
    """Calculate total tiktoken tokens and file count for all indexed source files."""
    repo_root = Path(repo_path).resolve()
    counter = get_token_counter("cl100k_base")
    total_tokens = 0
    file_count = 0

    with MetadataStore.open(repo_root) as store:
        paths = store.get_all_file_paths()
        file_count = len(paths)
        for rel_path in paths:
            p = repo_root / rel_path
            if p.is_file():
                try:
                    source = p.read_text(encoding="utf-8", errors="replace")
                    if source:
                        total_tokens += counter.count_tokens(source)
                except Exception:
                    pass

    return total_tokens, file_count


def _check_coverage(context_text: str, must_see: list[str]) -> tuple[int, list[str], float]:
    """Check how many must_see symbols appear in the provided context text."""
    if not must_see:
        return 0, [], 100.0

    found_count = 0
    missing: list[str] = []

    for item in must_see:
        # Match either exact qualified name or definition (def / class)
        short_name = item.split(".")[-1]
        patterns = [
            item,
            f"def {short_name}",
            f"class {short_name}",
        ]
        if any(pat in context_text for pat in patterns):
            found_count += 1
        else:
            missing.append(item)

    coverage = (found_count / len(must_see)) * 100.0 if must_see else 100.0
    return found_count, missing, coverage


def run_benchmark(
    repo_path: Path | str,
    queries: list[BenchmarkQuery] | list[dict[str, Any]],
    strategies: list[ContextStrategy] | None = None,
    budget: int = 4000,
    call_ai: bool = False,
    ai_output_tokens: int = 512,
    config: BenchmarkConfig | None = None,
    repeat: int = 1,
    provider: AIProvider | None = None,
) -> BenchmarkResult:
    """Execute benchmark across all strategies and queries."""
    cfg = config or BenchmarkConfig()
    settings = get_settings()
    repo_root = Path(repo_path).resolve()

    # Normalize queries
    norm_queries: list[BenchmarkQuery] = []
    for q in queries:
        if isinstance(q, BenchmarkQuery):
            norm_queries.append(q)
        else:
            norm_queries.append(BenchmarkQuery(**q))

    active_strategies = strategies if strategies is not None else ALL_STRATEGIES

    # Obtain provider if needed
    active_provider: AIProvider | None = provider
    if active_provider is None:
        try:
            active_provider = get_provider()
        except Exception:
            active_provider = None

    model_name = (
        (active_provider.config.model if active_provider else None)
        or settings.get_active_model()
        or settings.ai_model
        or "offline"
    )
    system_prompt = load_system_prompt()

    # Measure whole repo scale
    repo_total_tokens, repo_file_count = compute_repo_total_tokens(repo_root)

    rows: list[BenchmarkRow] = []
    counter_used = "tiktoken"

    for q in norm_queries:
        task_text = q.task or q.query

        for strat in active_strategies:
            t0 = time.perf_counter()
            strat_ctx = strat.build(repo_root, query=q.query, budget=budget)
            build_latency = time.perf_counter() - t0

            # Measure tokens
            t_meas = measure_tokens(
                context_text=strat_ctx.context_text,
                system_prompt=system_prompt,
                task_text=task_text,
                provider=active_provider,
                model=model_name,
            )
            if t_meas.counter == "provider":
                counter_used = "provider"

            effective_tok = t_meas.exact_tokens if t_meas.exact_tokens is not None else t_meas.tiktoken_tokens
            is_over_limit = effective_tok > cfg.model_context_limit

            # Coverage check
            found_cnt, missing_list, cov_pct = _check_coverage(strat_ctx.context_text, q.must_see)

            # AI Execution
            real_in: int | None = None
            real_out: int | None = None
            real_total: int | None = None
            ai_latency: float | None = None
            cost_val: float | None = None
            outcome: TaskOutcome | None = None

            if call_ai and active_provider and active_provider.is_configured():
                if is_over_limit:
                    outcome = TaskOutcome(
                        answered=False,
                        needs_context=[],
                        unknown_target_changes=[],
                        raw_response="SKIPPED: Exceeds model context limit",
                    )
                else:
                    # Run completion with repeats for latency
                    latencies: list[float] = []
                    last_res = None
                    runs_to_do = max(1, repeat)

                    for _ in range(runs_to_do):
                        user_content = f"{strat_ctx.context_text}\n\nTASK:\n{task_text}"
                        comp_res = active_provider.complete(
                            system=system_prompt,
                            messages=[ChatMessage(role="user", content=user_content)],
                            max_output_tokens=ai_output_tokens,
                        )
                        latencies.append(comp_res.latency_seconds)
                        last_res = comp_res

                    ai_latency = statistics.median(latencies) if latencies else 0.0

                    if last_res:
                        real_in = last_res.usage.input_tokens
                        real_out = last_res.usage.output_tokens
                        real_total = last_res.usage.total_tokens
                        cost_val = last_res.usage.estimated_cost_usd

                        # Parse model answer
                        parsed = parse_response(last_res.text)
                        unknown_in_context = [
                            c.qualified_name
                            for c in parsed.changes
                            if c.qualified_name not in strat_ctx.context_text
                        ]

                        outcome = TaskOutcome(
                            answered=bool(last_res.text.strip()),
                            needs_context=parsed.needs_context,
                            unknown_target_changes=unknown_in_context,
                            raw_response=last_res.text,
                        )

            row = BenchmarkRow(
                query=q.query,
                task=task_text,
                strategy_name=strat.name,
                files_inspected=strat_ctx.files_inspected,
                file_count=strat_ctx.file_count,
                context_chars=len(strat_ctx.context_text),
                tiktoken_tokens=t_meas.tiktoken_tokens,
                exact_tokens=t_meas.exact_tokens,
                effective_tokens=effective_tok,
                counter=t_meas.counter,
                build_latency_seconds=build_latency,
                truncated=strat_ctx.truncated,
                skipped_over_limit=is_over_limit,
                notes=strat_ctx.notes,
                must_see_total=len(q.must_see),
                must_see_found=found_cnt,
                must_see_missing=missing_list,
                coverage_pct=cov_pct,
                real_input_tokens=real_in,
                real_output_tokens=real_out,
                real_total_tokens=real_total,
                ai_latency_seconds=ai_latency,
                est_cost_usd=cost_val,
                outcome=outcome,
            )
            rows.append(row)

    # Compute Summaries per strategy
    summaries: dict[str, StrategySummary] = {}
    strategy_rows: dict[str, list[BenchmarkRow]] = {}
    for r in rows:
        strategy_rows.setdefault(r.strategy_name, []).append(r)

    # Find Graph rows for comparison
    graph_rows = strategy_rows.get("graph", [])
    graph_tokens_by_query = {r.query: r.effective_tokens for r in graph_rows}
    graph_mean_tokens = statistics.mean([r.effective_tokens for r in graph_rows]) if graph_rows else 0.0

    for s_name, s_list in strategy_rows.items():
        q_cnt = len(s_list)
        tok_vals = [r.effective_tokens for r in s_list]
        files_vals = [r.file_count for r in s_list]
        lat_vals = [r.build_latency_seconds for r in s_list]
        cov_vals = [r.coverage_pct for r in s_list]
        over_lim = sum(1 for r in s_list if r.skipped_over_limit)

        mean_tok = statistics.mean(tok_vals) if tok_vals else 0.0
        med_tok = statistics.median(tok_vals) if tok_vals else 0.0
        mean_files = statistics.mean(files_vals) if files_vals else 0.0
        mean_lat = statistics.mean(lat_vals) if lat_vals else 0.0
        mean_cov = statistics.mean(cov_vals) if cov_vals else 100.0

        # Reductions vs graph
        per_query_reductions = []
        for r in s_list:
            g_tok = graph_tokens_by_query.get(r.query, 0)
            if r.effective_tokens > 0:
                red = (r.effective_tokens - g_tok) / r.effective_tokens * 100.0
            else:
                red = 0.0
            per_query_reductions.append(red)

        mean_of_red = statistics.mean(per_query_reductions) if per_query_reductions else 0.0
        red_of_means = ((mean_tok - graph_mean_tokens) / mean_tok * 100.0) if mean_tok > 0 else 0.0

        # AI stats
        mean_rin: float | None = None
        mean_rout: float | None = None
        mean_ai_lat: float | None = None
        mean_c: float | None = None
        ans_rate: float | None = None
        needs_ctx_rate: float | None = None

        if call_ai:
            valid_ins = [r.real_input_tokens for r in s_list if r.real_input_tokens is not None]
            valid_outs = [r.real_output_tokens for r in s_list if r.real_output_tokens is not None]
            valid_ai_lats = [r.ai_latency_seconds for r in s_list if r.ai_latency_seconds is not None]
            valid_costs = [r.est_cost_usd for r in s_list if r.est_cost_usd is not None]
            valid_outcomes = [r.outcome for r in s_list if r.outcome is not None]

            if valid_ins:
                mean_rin = statistics.mean(valid_ins)
            if valid_outs:
                mean_rout = statistics.mean(valid_outs)
            if valid_ai_lats:
                mean_ai_lat = statistics.mean(valid_ai_lats)
            if valid_costs:
                mean_c = statistics.mean(valid_costs)
            if valid_outcomes:
                ans_rate = sum(1 for o in valid_outcomes if o.answered) / len(valid_outcomes) * 100.0
                needs_ctx_rate = sum(1 for o in valid_outcomes if o.needs_context) / len(valid_outcomes) * 100.0

        summaries[s_name] = StrategySummary(
            strategy_name=s_name,
            query_count=q_cnt,
            mean_context_tokens=round(mean_tok, 1),
            median_context_tokens=round(med_tok, 1),
            mean_files_inspected=round(mean_files, 1),
            mean_build_latency_seconds=round(mean_lat, 3),
            over_limit_count=over_lim,
            mean_coverage_pct=round(mean_cov, 1),
            mean_of_reductions_pct=round(mean_of_red, 1),
            reduction_of_means_pct=round(red_of_means, 1),
            mean_real_input_tokens=round(mean_rin, 1) if mean_rin is not None else None,
            mean_real_output_tokens=round(mean_rout, 1) if mean_rout is not None else None,
            mean_ai_latency_seconds=round(mean_ai_lat, 3) if mean_ai_lat is not None else None,
            mean_cost_usd=round(mean_c, 5) if mean_c is not None else None,
            answered_rate=round(ans_rate, 1) if ans_rate is not None else None,
            needs_context_rate=round(needs_ctx_rate, 1) if needs_ctx_rate is not None else None,
        )

    explanations = {
        "difference": (
            "The 'mean of reductions' weights each query equally (1/N), while the 'reduction of means' "
            "weights queries with larger baseline files more heavily."
        )
    }

    return BenchmarkResult(
        repo_path=str(repo_root),
        repo_file_count=repo_file_count,
        repo_total_tokens=repo_total_tokens,
        model_name=model_name,
        counter_used=counter_used,
        budget=budget,
        call_ai=call_ai,
        repeat=repeat,
        rows=rows,
        summaries=summaries,
        reduction_explanations=explanations,
    )
