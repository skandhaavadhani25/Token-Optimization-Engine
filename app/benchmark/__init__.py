"""Benchmark module for measuring context reduction and AI coding performance."""

from app.benchmark.baselines import (
    ContextStrategy,
    FixedWindowStrategy,
    GraphRetrievalStrategy,
    NaiveKeywordFilesStrategy,
    StrategyContext,
    WholeFileOfTargetStrategy,
    WholeRepoStrategy,
)
from app.benchmark.report import generate_reports
from app.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkQuery,
    BenchmarkResult,
    BenchmarkRow,
    StrategySummary,
    run_benchmark,
)
from app.benchmark.token_meter import TokenMeasurement, measure_tokens

__all__ = [
    "ContextStrategy",
    "WholeRepoStrategy",
    "WholeFileOfTargetStrategy",
    "NaiveKeywordFilesStrategy",
    "FixedWindowStrategy",
    "GraphRetrievalStrategy",
    "StrategyContext",
    "TokenMeasurement",
    "measure_tokens",
    "BenchmarkConfig",
    "BenchmarkQuery",
    "BenchmarkRow",
    "StrategySummary",
    "BenchmarkResult",
    "run_benchmark",
    "generate_reports",
]
