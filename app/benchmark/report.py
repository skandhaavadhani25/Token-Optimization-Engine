"""Reporting utilities producing tables, JSON, and markdown benchmark summaries."""

from __future__ import annotations

import datetime
from pathlib import Path

from app.benchmark.runner import BenchmarkResult


def format_table(result: BenchmarkResult) -> str:
    """Produce a formatted plaintext comparison table."""
    counter_label = (
        "Exact Provider Token Counts"
        if result.counter_used == "provider"
        else "Tiktoken (cl100k_base) Token Counts"
    )

    lines: list[str] = [
        "=" * 100,
        f"BENCHMARK COMPARISON: {Path(result.repo_path).name.upper()} ({counter_label})",
        f"Repo Scale: {result.repo_file_count} files | {result.repo_total_tokens:,} total tokens | Budget: {result.budget}",
        "=" * 100,
    ]

    if not result.call_ai:
        headers = [
            f"{'Strategy':<18}",
            f"{'Files':>6}",
            f"{'Tokens (Mean)':>14}",
            f"{'Tokens (Med)':>13}",
            f"{'Latency':>9}",
            f"{'Coverage':>9}",
            f"{'Over-Lim':>9}",
            f"{'Reduct (Mean)':>14}",
            f"{'Reduct (Agg)':>13}",
        ]
        lines.append(" | ".join(headers))
        lines.append("-" * 100)

        for s_name in ["whole_repo", "naive_keyword", "fixed_window", "whole_file", "graph"]:
            s = result.summaries.get(s_name)
            if not s:
                continue
            red_mean = f"{s.mean_of_reductions_pct:+.1f}%" if s_name != "graph" else "BASELINE"
            red_agg = f"{s.reduction_of_means_pct:+.1f}%" if s_name != "graph" else "BASELINE"
            row = [
                f"{s.strategy_name:<18}",
                f"{s.mean_files_inspected:>6.1f}",
                f"{s.mean_context_tokens:>14,.1f}",
                f"{s.median_context_tokens:>13,.1f}",
                f"{s.mean_build_latency_seconds:>8.3f}s",
                f"{s.mean_coverage_pct:>8.1f}%",
                f"{s.over_limit_count:>9}",
                f"{red_mean:>14}",
                f"{red_agg:>13}",
            ]
            lines.append(" | ".join(row))

    else:
        headers = [
            f"{'Strategy':<16}",
            f"{'Files':>5}",
            f"{'In-Tok':>9}",
            f"{'Out-Tok':>8}",
            f"{'AI Lat':>8}",
            f"{'Ans%':>6}",
            f"{'Need%':>6}",
            f"{'Cov%':>6}",
            f"{'Red(M)':>8}",
            f"{'Red(A)':>8}",
        ]
        lines.append(" | ".join(headers))
        lines.append("-" * 100)

        for s_name in ["whole_repo", "naive_keyword", "fixed_window", "whole_file", "graph"]:
            s = result.summaries.get(s_name)
            if not s:
                continue
            in_tok = f"{s.mean_real_input_tokens:,.0f}" if s.mean_real_input_tokens is not None else "N/A"
            out_tok = f"{s.mean_real_output_tokens:,.0f}" if s.mean_real_output_tokens is not None else "N/A"
            ai_lat = f"{s.mean_ai_latency_seconds:.2f}s" if s.mean_ai_latency_seconds is not None else "N/A"
            ans = f"{s.answered_rate:.0f}%" if s.answered_rate is not None else "N/A"
            need = f"{s.needs_context_rate:.0f}%" if s.needs_context_rate is not None else "N/A"
            red_mean = f"{s.mean_of_reductions_pct:+.1f}%" if s_name != "graph" else "BASE"
            red_agg = f"{s.reduction_of_means_pct:+.1f}%" if s_name != "graph" else "BASE"

            row = [
                f"{s.strategy_name:<16}",
                f"{s.mean_files_inspected:>5.1f}",
                f"{in_tok:>9}",
                f"{out_tok:>8}",
                f"{ai_lat:>8}",
                f"{ans:>6}",
                f"{need:>6}",
                f"{s.mean_coverage_pct:>5.1f}%",
                f"{red_mean:>8}",
                f"{red_agg:>8}",
            ]
            lines.append(" | ".join(row))

    lines.append("=" * 100)
    lines.append("Note: Reduct (Mean) is the mean of per-query reductions; Reduct (Agg) is the reduction of means.")
    lines.append("=" * 100)
    return "\n".join(lines)


def generate_markdown(result: BenchmarkResult) -> str:
    """Generate a clean markdown report with honest prose conclusions."""
    repo_name = Path(result.repo_path).name
    counter_name = "provider exact token counting" if result.counter_used == "provider" else "tiktoken cl100k_base"

    whole_file = result.summaries.get("whole_file")
    fixed_win = result.summaries.get("fixed_window")
    graph = result.summaries.get("graph")
    whole_repo = result.summaries.get("whole_repo")

    # Honest prose comparison
    prose_parts: list[str] = []
    prose_parts.append(
        f"This benchmark evaluated **{len(result.rows) // max(1, len(result.summaries))} queries** on the "
        f"`{repo_name}` repository ({result.repo_file_count} files, {result.repo_total_tokens:,} total tokens), "
        f"measuring context size using **{counter_name}**."
    )

    if graph and whole_file:
        wf_red_mean = whole_file.mean_of_reductions_pct
        wf_red_agg = whole_file.reduction_of_means_pct
        prose_parts.append(
            f"Against the fairest single-file baseline (`whole_file`), graph-based retrieval achieved a "
            f"**{wf_red_mean:+.1f}% mean reduction** per query (**{wf_red_agg:+.1f}%** reduction of aggregate means), "
            f"reducing average context from {whole_file.mean_context_tokens:,.1f} to {graph.mean_context_tokens:,.1f} tokens."
        )

    if graph and fixed_win:
        fw_red_mean = fixed_win.mean_of_reductions_pct
        prose_parts.append(
            f"Against the multi-file directory baseline (`fixed_window`), graph retrieval achieved a "
            f"**{fw_red_mean:+.1f}% mean reduction** (mean context dropped from {fixed_win.mean_context_tokens:,.1f} to {graph.mean_context_tokens:,.1f} tokens)."
        )

    if whole_repo:
        prose_parts.append(
            f"The naive `whole_repo` baseline required {whole_repo.mean_context_tokens:,.1f} tokens on average "
            f"and exceeded model context limits in {whole_repo.over_limit_count} test cases."
        )

    if graph:
        prose_parts.append(
            f"In terms of coverage of necessary symbols (`must_see`), graph retrieval maintained "
            f"**{graph.mean_coverage_pct:.1f}% coverage** while inspecting an average of {graph.mean_files_inspected:.1f} files per task."
        )

    if not result.call_ai:
        prose_parts.append(
            "*Note on AI execution*: This benchmark was run offline without active LLM completion calls; "
            "metrics reflect measured context sizes and symbol coverage, not automated correctness grades."
        )
    else:
        ans = graph.answered_rate if graph else 0.0
        prose_parts.append(
            f"*Live AI Completion*: Model completion answered {ans:.0f}% of graph tasks with measured API usage."
        )

    prose_text = " ".join(prose_parts)

    # Markdown table
    md_lines: list[str] = [
        f"# Benchmark Report: {repo_name}",
        "",
        f"- **Repository**: `{result.repo_path}`",
        f"- **Files**: {result.repo_file_count}",
        f"- **Total Source Tokens**: {result.repo_total_tokens:,}",
        f"- **Token Counter**: `{counter_name}`",
        f"- **Context Budget**: {result.budget}",
        f"- **Live AI Execution**: `{result.call_ai}`",
        "",
        "## Summary Verdict",
        "",
        prose_text,
        "",
        "## Comparison Table",
        "",
    ]

    if not result.call_ai:
        md_lines.extend([
            "| Strategy | Files Inspected | Mean Tokens | Median Tokens | Latency | Coverage | Over-Limit | Reduction (Mean) | Reduction (Agg) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for s_name in ["whole_repo", "naive_keyword", "fixed_window", "whole_file", "graph"]:
            s = result.summaries.get(s_name)
            if not s:
                continue
            red_mean = f"{s.mean_of_reductions_pct:+.1f}%" if s_name != "graph" else "BASELINE"
            red_agg = f"{s.reduction_of_means_pct:+.1f}%" if s_name != "graph" else "BASELINE"
            md_lines.append(
                f"| `{s.strategy_name}` | {s.mean_files_inspected:.1f} | {s.mean_context_tokens:,.1f} | "
                f"{s.median_context_tokens:,.1f} | {s.mean_build_latency_seconds:.3f}s | {s.mean_coverage_pct:.1f}% | "
                f"{s.over_limit_count} | {red_mean} | {red_agg} |"
            )
    else:
        md_lines.extend([
            "| Strategy | Files | In-Tokens | Out-Tokens | AI Latency | Answered | Needs Ctx | Coverage | Reduct (Mean) | Reduct (Agg) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for s_name in ["whole_repo", "naive_keyword", "fixed_window", "whole_file", "graph"]:
            s = result.summaries.get(s_name)
            if not s:
                continue
            in_tok = f"{s.mean_real_input_tokens:,.0f}" if s.mean_real_input_tokens is not None else "N/A"
            out_tok = f"{s.mean_real_output_tokens:,.0f}" if s.mean_real_output_tokens is not None else "N/A"
            ai_lat = f"{s.mean_ai_latency_seconds:.2f}s" if s.mean_ai_latency_seconds is not None else "N/A"
            ans = f"{s.answered_rate:.0f}%" if s.answered_rate is not None else "N/A"
            need = f"{s.needs_context_rate:.0f}%" if s.needs_context_rate is not None else "N/A"
            red_mean = f"{s.mean_of_reductions_pct:+.1f}%" if s_name != "graph" else "BASELINE"
            red_agg = f"{s.reduction_of_means_pct:+.1f}%" if s_name != "graph" else "BASELINE"
            md_lines.append(
                f"| `{s.strategy_name}` | {s.mean_files_inspected:.1f} | {in_tok} | {out_tok} | {ai_lat} | "
                f"{ans} | {need} | {s.mean_coverage_pct:.1f}% | {red_mean} | {red_agg} |"
            )

    md_lines.extend([
        "",
        "## Mathematical Reductions Definition",
        "",
        "- **Reduction (Mean)**: $\\frac{1}{N} \\sum_{i=1}^N \\frac{\\text{baseline}_i - \\text{graph}_i}{\\text{baseline}_i} \\times 100\\%$ (equal weight per query).",
        "- **Reduction (Agg)**: $\\frac{\\overline{\\text{baseline}} - \\overline{\\text{graph}}}{\\overline{\\text{baseline}}} \\times 100\\%$ (weighted by baseline scale).",
    ])

    return "\n".join(md_lines)


def generate_reports(
    result: BenchmarkResult,
    json_path: Path | str | None = None,
    md_path: Path | str | None = None,
) -> tuple[str, str, str]:
    """Generate plaintext table, JSON, and markdown artifacts.

    Returns: (table_text, json_path_str, md_path_str)
    """
    repo_name = Path(result.repo_path).name or "repo"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    target_json = Path(json_path) if json_path else eval_dir / f"benchmark_{repo_name}_{timestamp}.json"
    target_md = Path(md_path) if md_path else eval_dir / f"benchmark_{repo_name}_{timestamp}.md"

    table_text = format_table(result)
    md_text = generate_markdown(result)
    json_text = result.model_dump_json(indent=2)

    target_json.write_text(json_text, encoding="utf-8")
    target_md.write_text(md_text, encoding="utf-8")

    return table_text, str(target_json), str(target_md)
