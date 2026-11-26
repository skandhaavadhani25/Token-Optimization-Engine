"""Tests for the execution trace and token-savings report attached to retrieval results."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.context.token_counter import CharEstimateCounter
from app.services.pipeline import RepositoryService


@pytest.fixture
def traced_repo(tmp_path: Path) -> Path:
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / "login.py").write_text(
        "from utils import verify_password\n\n"
        "def login_user(username, password):\n"
        "    '''Authenticate user.'''\n"
        "    return verify_password(password, 'x')\n",
        encoding="utf-8",
    )
    (tmp_path / "utils.py").write_text(
        "def verify_password(raw, hashed):\n    return raw == hashed\n\n"
        "def unrelated_helper():\n    return " + " + ".join(str(i) for i in range(200)) + "\n",
        encoding="utf-8",
    )
    # Padding files make the whole-repo baseline meaningfully larger than the context.
    for i in range(5):
        (tmp_path / f"extra_{i}.py").write_text(
            "\n".join(f"def extra_{i}_{j}(x):\n    return x * {j}\n" for j in range(40)), encoding="utf-8"
        )
    return tmp_path


def test_retrieval_trace_reports_graph_walk_and_savings(traced_repo: Path):
    svc = RepositoryService(traced_repo)
    svc.index(force=True)
    retrieval, ctx = svc.retrieve("Optimize login_user", max_tokens=1500)

    trace = retrieval.trace
    assert trace is not None
    assert trace.target == "auth.login.login_user"
    assert trace.signals["graph"] == "used"
    assert any(
        e.direction == "callee" and e.target == "utils.verify_password" and e.rel == "CALLS"
        for e in trace.graph_edges
    )
    names = [op.name for op in trace.operations]
    assert names[:2] == ["intent_detection", "target_identification"]
    assert "graph_expansion" in names and "context_packing" in names and names[-1] == "token_accounting"
    assert [op.step for op in trace.operations] == list(range(1, len(names) + 1))

    assert trace.selection is not None
    assert {i.ref for i in trace.selection.included} == {i.ref for i in ctx.items}

    sav = trace.savings
    assert sav is not None
    assert sav.repo_files == 7
    assert sav.naive_full_repo_tokens > sav.prompt_tokens
    assert sav.saved_vs_full_repo_pct > 0
    assert sav.reduction_factor > 1


def test_disabled_signals_are_reported_as_skipped(traced_repo: Path):
    svc = RepositoryService(traced_repo)
    svc.index(force=True)
    retrieval, _ = svc.retrieve("login_user", use_graph=False, use_semantic=False)
    trace = retrieval.trace
    assert trace.signals["graph"].startswith("skipped")
    assert trace.signals["semantic"].startswith("skipped")
    assert not trace.graph_edges
    skipped = {op.name for op in trace.operations if op.status == "skipped"}
    assert {"graph_expansion", "semantic_search"} <= skipped


def test_ask_dry_run_returns_trace_with_prompt_savings(traced_repo: Path):
    client = TestClient(app)
    repo = str(traced_repo)
    assert client.post("/api/index", json={"repo_path": repo, "force": True}).status_code == 200
    res = client.post("/api/ask", json={"repo_path": repo, "query": "Optimize login_user", "dry_run": True})
    assert res.status_code == 200
    trace = res.json()["trace"]
    assert trace["savings"]["prompt_tokens"] >= trace["savings"]["context_tokens"]
    assert trace["selection"]["included"]
    assert trace["graph_edges"]


def test_char_counter_flagged_as_estimate(tmp_path: Path, monkeypatch):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    svc = RepositoryService(tmp_path)
    svc.index(force=True)
    monkeypatch.setattr(svc, "_counter", CharEstimateCounter(warn=False))
    retrieval, _ = svc.retrieve("f")
    assert retrieval.trace.savings.is_estimate is True


def test_chat_turn_reports_real_baselines_and_trace(traced_repo: Path):
    from app.chat.storage import ChatStorage
    from app.chat.turn_builder import TurnBuilder

    svc = RepositoryService(traced_repo)
    svc.index(force=True)
    storage = ChatStorage()
    conv = storage.create_conversation(title="t", mode="economy")

    turn = TurnBuilder(repo_service=svc, storage=storage).assemble_turn(
        conv["id"], "Optimize login_user", mode_name="economy"
    )

    assert turn.trace is not None and turn.trace["target"] == "auth.login.login_user"
    tokens = turn.raw_tokens
    assert tokens["full_repo_baseline"] > tokens["actual_total_sent"]
    assert tokens["full_repo_saved_pct"] > 0
    assert tokens["total_repo_files"] == 7
    assert turn.context_manifest["trace"]["savings"]["repo_files"] == 7
    assert not any(it["role"] == "caller" for it in turn.retrieved_items)
