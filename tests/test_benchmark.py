"""Tests for benchmark baseline strategies, token meter, runner, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.ai.base_provider import Usage
from app.benchmark.baselines import (
    FixedWindowStrategy,
    GraphRetrievalStrategy,
    NaiveKeywordFilesStrategy,
    WholeFileOfTargetStrategy,
    WholeRepoStrategy,
)
from app.benchmark.report import generate_reports
from app.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkQuery,
    BenchmarkRow,
    _check_coverage,
    run_benchmark,
)
from app.benchmark.token_meter import measure_tokens
from app.embeddings.embedding_generator import FakeEmbeddingGenerator
from app.services.pipeline import RepositoryService
from tests.test_ai import FakeProvider


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Create a realistic mini repository on disk with connected modules."""
    repo_root = tmp_path / "mini_bench_repo"
    repo_root.mkdir()

    # auth/login.py
    auth_dir = repo_root / "auth"
    auth_dir.mkdir()
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "login.py").write_text(
        "from utils.hashing import verify_password\n"
        "from database.user import get_user\n\n"
        "def login_user(username: str, password: str) -> bool:\n"
        "    '''Authenticate user against database record.'''\n"
        "    user = get_user(username)\n"
        "    if not user:\n"
        "        return False\n"
        "    return verify_password(password, user.get('hash', ''))\n",
        encoding="utf-8",
    )
    (auth_dir / "tokens.py").write_text(
        "def generate_token(user_id: int) -> str:\n"
        "    '''Generate session token for user.'''\n"
        "    return f'tok_{user_id}'\n",
        encoding="utf-8",
    )

    # utils/hashing.py
    utils_dir = repo_root / "utils"
    utils_dir.mkdir()
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "hashing.py").write_text(
        "def verify_password(raw: str, hashed: str) -> bool:\n"
        "    '''Verify raw string matches hashed string.'''\n"
        "    return raw == hashed\n\n"
        "def hash_password(raw: str) -> str:\n"
        "    '''Hash raw string.'''\n"
        "    return 'hashed_' + raw\n",
        encoding="utf-8",
    )

    # database/user.py
    db_dir = repo_root / "database"
    db_dir.mkdir()
    (db_dir / "__init__.py").write_text("", encoding="utf-8")
    (db_dir / "user.py").write_text(
        "def get_user(username: str) -> dict | None:\n"
        "    '''Fetch user dictionary from database.'''\n"
        "    if username == 'alice':\n"
        "        return {'username': 'alice', 'hash': 'secret123'}\n"
        "    return None\n",
        encoding="utf-8",
    )

    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        svc = RepositoryService(repo_root)
        svc.index(force=True)

    return repo_root


def test_baseline_strategies(mini_repo: Path):
    """Verify each baseline builds context and reports correct inspected files."""
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        # 1. WholeRepo
        wr = WholeRepoStrategy()
        ctx_wr = wr.build(mini_repo, "login_user")
        assert ctx_wr.file_count >= 5
        assert any("login.py" in f for f in ctx_wr.files_inspected)
        assert any("hashing.py" in f for f in ctx_wr.files_inspected)
        assert not ctx_wr.truncated

        # 2. WholeFile
        wf = WholeFileOfTargetStrategy()
        ctx_wf = wf.build(mini_repo, "login_user")
        assert ctx_wf.file_count == 1
        assert "login.py" in ctx_wf.files_inspected[0]
        assert "def login_user" in ctx_wf.context_text
        assert "def verify_password" not in ctx_wf.context_text  # cross-file dependency not in file

        # 3. NaiveKeyword
        nk = NaiveKeywordFilesStrategy()
        ctx_nk = nk.build(mini_repo, "password hashing verify")
        assert ctx_nk.file_count >= 1
        assert any("hashing.py" in f for f in ctx_nk.files_inspected)

        # 4. FixedWindow
        fw = FixedWindowStrategy(n=3, rule="sibling")
        ctx_fw = fw.build(mini_repo, "login_user")
        assert ctx_fw.file_count == 3
        # Should include login.py and its sibling tokens.py in auth/
        assert any("login.py" in f for f in ctx_fw.files_inspected)
        assert any("tokens.py" in f for f in ctx_fw.files_inspected)

        # 5. Graph
        gr = GraphRetrievalStrategy()
        ctx_gr = gr.build(mini_repo, "login_user", budget=2000)
        assert ctx_gr.file_count >= 1
        assert "login_user" in ctx_gr.context_text


def test_token_meter():
    """Verify tiktoken and provider token measurement."""
    text = "def hello():\n    return 'world'\n"
    meas_offline = measure_tokens(context_text=text)
    assert meas_offline.tiktoken_tokens > 0
    assert meas_offline.exact_tokens is None
    assert meas_offline.counter == "tiktoken"
    assert meas_offline.est_cost_usd is None

    fake_prov = FakeProvider(token_count=42)
    meas_provider = measure_tokens(context_text=text, provider=fake_prov)
    assert meas_provider.exact_tokens == 42
    assert meas_provider.counter == "provider"


def test_coverage_and_cross_file_advantage(mini_repo: Path):
    """Verify that graph retrieval covers cross-file dependencies that whole_file misses."""
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        wf = WholeFileOfTargetStrategy()
        gr = GraphRetrievalStrategy()

        must_see = ["login_user", "utils.hashing.verify_password"]

        ctx_wf = wf.build(mini_repo, "login_user")
        found_wf, missing_wf, cov_wf = _check_coverage(ctx_wf.context_text, must_see)
        assert "login_user" not in missing_wf
        assert "utils.hashing.verify_password" in missing_wf
        assert cov_wf == 50.0

        ctx_gr = gr.build(mini_repo, "login_user", budget=2000)
        found_gr, missing_gr, cov_gr = _check_coverage(ctx_gr.context_text, must_see)
        assert "login_user" not in missing_gr
        assert "utils.hashing.verify_password" not in missing_gr
        assert cov_gr == 100.0


def test_runner_offline_mode(mini_repo: Path):
    """Verify offline benchmark runner executes with zero AI calls."""
    queries = [
        BenchmarkQuery(
            query="login_user",
            task="Optimize login_user authentication flow",
            must_see=["login_user"],
        )
    ]
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        res = run_benchmark(mini_repo, queries=queries, call_ai=False, budget=2000)

        assert res.repo_file_count >= 5
        assert res.repo_total_tokens > 0
        assert not res.call_ai
        assert len(res.rows) == 5  # 5 strategies x 1 query
        assert "graph" in res.summaries
        assert "whole_file" in res.summaries


def test_over_limit_handling(mini_repo: Path):
    """Verify strategy exceeding model context limit is marked skipped_over_limit and not called."""
    queries = [BenchmarkQuery(query="login_user", task="Do something")]
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    fake_prov = FakeProvider(responses=["OK"])

    cfg = BenchmarkConfig(model_context_limit=15)  # Tiny limit: whole_repo will exceed

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        res = run_benchmark(
            mini_repo,
            queries=queries,
            call_ai=True,
            config=cfg,
            provider=fake_prov,
        )

        wr_row = next(r for r in res.rows if r.strategy_name == "whole_repo")
        assert wr_row.skipped_over_limit is True
        assert wr_row.outcome is not None
        assert "SKIPPED" in (wr_row.outcome.raw_response or "")


def test_reduction_math():
    """Verify mean of reductions vs reduction of means calculations."""
    row1_base = BenchmarkRow(
        query="q1",
        task="t1",
        strategy_name="base",
        effective_tokens=1000,
        tiktoken_tokens=1000,
    )
    row1_graph = BenchmarkRow(
        query="q1",
        task="t1",
        strategy_name="graph",
        effective_tokens=500,
        tiktoken_tokens=500,
    )
    row2_base = BenchmarkRow(
        query="q2",
        task="t2",
        strategy_name="base",
        effective_tokens=3000,
        tiktoken_tokens=3000,
    )
    row2_graph = BenchmarkRow(
        query="q2",
        task="t2",
        strategy_name="graph",
        effective_tokens=1000,
        tiktoken_tokens=1000,
    )

    # q1 reduction: (1000 - 500) / 1000 = 50%
    # q2 reduction: (3000 - 1000) / 3000 = 66.666%
    # Mean of reductions: (50 + 66.666%) / 2 = 58.33%
    # Mean baseline: 2000, Mean graph: 750
    # Reduction of means: (2000 - 750) / 2000 = 62.5%

    q_red1 = (row1_base.effective_tokens - row1_graph.effective_tokens) / row1_base.effective_tokens * 100
    q_red2 = (row2_base.effective_tokens - row2_graph.effective_tokens) / row2_base.effective_tokens * 100
    mean_of_red = (q_red1 + q_red2) / 2.0

    mean_base = (row1_base.effective_tokens + row2_base.effective_tokens) / 2.0
    mean_graph = (row1_graph.effective_tokens + row2_graph.effective_tokens) / 2.0
    red_of_means = (mean_base - mean_graph) / mean_base * 100

    assert round(mean_of_red, 2) == 58.33
    assert round(red_of_means, 2) == 62.50


def test_call_ai_with_fake_provider(mini_repo: Path):
    """Verify fake provider usage numbers flow into rows unchanged."""
    queries = [BenchmarkQuery(query="login_user", task="Refactor login_user")]
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    fake_prov = FakeProvider(
        responses=["### ANALYSIS\nRefactored.\n### NEEDS_CONTEXT\n- missing.module"],
        usage=Usage(
            input_tokens=180,
            output_tokens=45,
            total_tokens=225,
            estimated_cost_usd=None,
        ),
    )

    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        res = run_benchmark(
            mini_repo,
            queries=queries,
            strategies=[WholeFileOfTargetStrategy()],
            call_ai=True,
            provider=fake_prov,
        )

        row = res.rows[0]
        assert row.real_input_tokens == 180
        assert row.real_output_tokens == 45
        assert row.real_total_tokens == 225
        assert row.outcome is not None
        assert row.outcome.answered is True
        assert "missing.module" in row.outcome.needs_context


def test_report_generation(mini_repo: Path, tmp_path: Path):
    """Verify JSON and markdown reports are generated and formatted correctly."""
    queries = [BenchmarkQuery(query="login_user", task="Task")]
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        res = run_benchmark(mini_repo, queries=queries, call_ai=False)

        json_out = tmp_path / "out.json"
        md_out = tmp_path / "out.md"
        table_text, j_path, m_path = generate_reports(res, json_path=json_out, md_path=md_out)

        assert "BENCHMARK COMPARISON" in table_text
        assert json_out.is_file()
        assert md_out.is_file()

        # Check JSON schema
        loaded_json = json.loads(json_out.read_text(encoding="utf-8"))
        assert "rows" in loaded_json
        assert "summaries" in loaded_json

        # Check Markdown prose
        md_text = md_out.read_text(encoding="utf-8")
        assert "# Benchmark Report" in md_text
        assert "tiktoken cl100k_base" in md_text
        assert "without active LLM completion" in md_text


def test_offline_determinism(mini_repo: Path):
    """Verify offline benchmark run twice produces identical token numbers."""
    queries = [BenchmarkQuery(query="login_user", task="Task")]
    fake_emb = FakeEmbeddingGenerator(model_name="test-fake", dimension=32)
    with (
        patch("app.embeddings.indexer.EmbeddingGenerator", return_value=fake_emb),
        patch("app.embeddings.semantic_search.EmbeddingGenerator", return_value=fake_emb),
    ):
        res1 = run_benchmark(mini_repo, queries=queries, call_ai=False)
        res2 = run_benchmark(mini_repo, queries=queries, call_ai=False)

        toks1 = [r.effective_tokens for r in res1.rows]
        toks2 = [r.effective_tokens for r in res2.rows]
        assert toks1 == toks2
