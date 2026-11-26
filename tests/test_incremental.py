"""Unit and integration tests for production incremental indexing and watch mode."""

from __future__ import annotations

from pathlib import Path
import pytest

from app.embeddings.embedding_generator import FakeEmbeddingGenerator
from app.graph.graph_io import load_graph
from app.services.incremental import update_index
from app.services.watcher import RepositoryWatcher


@pytest.fixture
def incremental_repo(tmp_path: Path) -> Path:
    """Create a multi-file connected test repository."""
    repo = tmp_path / "test_incr_repo"
    repo.mkdir()

    # utils/hashing.py
    u_dir = repo / "utils"
    u_dir.mkdir()
    (u_dir / "__init__.py").write_text("", encoding="utf-8")
    (u_dir / "hashing.py").write_text(
        "def hash_password(plain: str) -> str:\n"
        "    return 'hashed_' + plain\n\n"
        "def verify_password(plain: str, hashed: str) -> bool:\n"
        "    return hash_password(plain) == hashed\n",
        encoding="utf-8",
    )

    # auth/login.py calling utils.hashing.verify_password
    a_dir = repo / "auth"
    a_dir.mkdir()
    (a_dir / "__init__.py").write_text("", encoding="utf-8")
    (a_dir / "login.py").write_text(
        "from utils.hashing import verify_password\n\n"
        "def login_user(username: str, password: str) -> bool:\n"
        "    return verify_password(password, 'hashed_admin')\n",
        encoding="utf-8",
    )

    return repo


def test_first_build_and_idempotent_second_update(incremental_repo: Path):
    """First update builds complete index; second update with no changes reparses 0 files."""
    fake_gen = FakeEmbeddingGenerator("fake-test", dimension=32)

    # First update: full build
    report1 = update_index(incremental_repo, generator=fake_gen)
    assert report1.files_reparsed >= 2
    assert report1.items_reembedded > 0
    assert report1.graph_nodes_changed > 0

    # Second update: no changes
    report2 = update_index(incremental_repo, generator=fake_gen)
    assert report2.files_reparsed == 0
    assert report2.items_reembedded == 0
    assert report2.graph_nodes_changed == 0
    assert report2.graph_edges_rechecked == 0
    assert len(report2.added) == 0
    assert len(report2.modified) == 0
    assert len(report2.deleted) == 0


def test_editing_function_body_and_cross_file_call_updates(incremental_repo: Path):
    """Editing a function body updates only that file; renaming a callee breaks/updates cross-file edges."""
    fake_gen = FakeEmbeddingGenerator("fake-test", dimension=32)
    update_index(incremental_repo, generator=fake_gen)

    # Verify initial CALLS edge from login_user to verify_password
    g_initial = load_graph(incremental_repo)
    calls_edges = [
        (u, v) for u, v, d in g_initial.edges(data=True)
        if d.get("rel") == "CALLS" and "login_user" in u and "verify_password" in v
    ]
    assert len(calls_edges) == 1

    # Rename verify_password -> check_password in utils/hashing.py
    hashing_file = incremental_repo / "utils" / "hashing.py"
    hashing_file.write_text(
        "def hash_password(plain: str) -> str:\n"
        "    return 'hashed_' + plain\n\n"
        "def check_password(plain: str, hashed: str) -> bool:\n"
        "    return hash_password(plain) == hashed\n",
        encoding="utf-8",
    )

    report = update_index(incremental_repo, generator=fake_gen)
    assert report.files_reparsed == 1
    assert "utils/hashing.py" in report.modified

    # Graph check: old CALLS edge to verify_password should be dropped from login_user
    g_updated = load_graph(incremental_repo)
    calls_old = [
        (u, v) for u, v, d in g_updated.edges(data=True)
        if d.get("rel") == "CALLS" and "login_user" in u and "verify_password" in v
    ]
    assert len(calls_old) == 0


def test_adding_new_file_calling_existing_function(incremental_repo: Path):
    """Adding a new file introduces new nodes and edges while preserving existing data."""
    fake_gen = FakeEmbeddingGenerator("fake-test", dimension=32)
    update_index(incremental_repo, generator=fake_gen)

    # Add api/endpoints.py calling login_user
    api_dir = incremental_repo / "api"
    api_dir.mkdir()
    (api_dir / "__init__.py").write_text("", encoding="utf-8")
    (api_dir / "endpoints.py").write_text(
        "from auth.login import login_user\n\n"
        "def handle_login_request(u: str, p: str):\n"
        "    return login_user(u, p)\n",
        encoding="utf-8",
    )

    report = update_index(incremental_repo, generator=fake_gen)
    assert "api/endpoints.py" in report.added
    assert report.files_reparsed >= 1

    g = load_graph(incremental_repo)
    assert any("handle_login_request" in n for n in g.nodes())

    calls_to_login = [
        (u, v) for u, v, d in g.edges(data=True)
        if d.get("rel") == "CALLS" and "handle_login_request" in u and "login_user" in v
    ]
    assert len(calls_to_login) == 1


def test_deleting_file_removes_nodes_and_edges(incremental_repo: Path):
    """Deleting a file removes its metadata, graph nodes, and embeddings."""
    fake_gen = FakeEmbeddingGenerator("fake-test", dimension=32)
    update_index(incremental_repo, generator=fake_gen)

    # Delete auth/login.py
    login_file = incremental_repo / "auth" / "login.py"
    login_file.unlink()

    report = update_index(incremental_repo, generator=fake_gen)
    assert "auth/login.py" in report.deleted

    g = load_graph(incremental_repo)
    assert not any("login_user" in n for n in g.nodes())


def test_key_verification_equivalence_across_edits(incremental_repo: Path):
    """THE KEY TEST: sequence of edits/adds/deletes applied incrementally matches a full rebuild."""
    fake_gen = FakeEmbeddingGenerator("fake-test", dimension=32)

    # Initial
    update_index(incremental_repo, generator=fake_gen)

    # Edit 1: modify utils/hashing.py
    (incremental_repo / "utils" / "hashing.py").write_text(
        "def hash_password(plain: str) -> str:\n"
        "    return 'sha256_' + plain\n\n"
        "def verify_password(plain: str, hashed: str) -> bool:\n"
        "    return plain == hashed\n",
        encoding="utf-8",
    )
    # Add 1: add models/user.py
    m_dir = incremental_repo / "models"
    m_dir.mkdir()
    (m_dir / "__init__.py").write_text("", encoding="utf-8")
    (m_dir / "user.py").write_text(
        "class UserModel:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n",
        encoding="utf-8",
    )

    # Run incremental update with verify=True
    report = update_index(incremental_repo, verify=True, generator=fake_gen)
    assert report.verified is True


def test_watcher_poll_and_debouncing(incremental_repo: Path):
    """Watcher poll detects modifications on disk and executes debounced incremental indexing."""
    fake_gen = FakeEmbeddingGenerator("fake-test", dimension=32)
    update_index(incremental_repo, generator=fake_gen)

    updated_reports = []
    watcher = RepositoryWatcher(
        repo_path=incremental_repo,
        interval=0.1,
        debounce_window=0.05,
        generator=fake_gen,
        on_update=lambda r: updated_reports.append(r),
    )

    # First poll: no changes
    assert watcher.poll_once() is None
    assert len(updated_reports) == 0

    # Modify file
    (incremental_repo / "utils" / "hashing.py").write_text(
        "def hash_password(plain: str) -> str:\n    return 'salt_' + plain\n",
        encoding="utf-8",
    )

    # Next poll should detect modification and update
    rep = watcher.poll_once()
    assert rep is not None
    assert "utils/hashing.py" in rep.modified
    assert len(updated_reports) == 1
