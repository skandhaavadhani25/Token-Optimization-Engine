"""Unit and integration tests for the code graph and call resolution (Step 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.graph.graph_builder import build_graph
from app.graph.graph_io import graph_exists, load_graph, save_graph
from app.graph.graph_search import CodeGraph
from app.indexer import build_index
from app.storage.metadata_store import MetadataStore


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Create a realistic mini repository on disk matching the architecture spec."""
    repo_root = tmp_path / "mini_repo"
    repo_root.mkdir()

    # 1. database package
    db_dir = repo_root / "database"
    db_dir.mkdir()
    (db_dir / "__init__.py").write_text(
        "from .user import get_user\n",
        encoding="utf-8",
    )
    (db_dir / "base.py").write_text(
        "class Model:\n"
        "    def save(self):\n"
        "        return True\n",
        encoding="utf-8",
    )
    (db_dir / "user.py").write_text(
        "from database.base import Model\n\n"
        "class User(Model):\n"
        "    def __init__(self, id, username, password_hash):\n"
        "        self.id = id\n"
        "        self.username = username\n"
        "        self.password_hash = password_hash\n\n"
        "    def check_user(self):\n"
        "        return self.save()\n\n"
        "def get_user(username):\n"
        "    return User(1, username, 'hashed_pw')\n",
        encoding="utf-8",
    )

    # 2. utils package
    utils_dir = repo_root / "utils"
    utils_dir.mkdir()
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "hashing.py").write_text(
        "def hash_password(password):\n"
        "    return f'hashed_{password}'\n\n"
        "def verify_password(password, hashed):\n"
        "    return hash_password(password) == hashed\n",
        encoding="utf-8",
    )

    # 3. auth package
    auth_dir = repo_root / "auth"
    auth_dir.mkdir()
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "session.py").write_text(
        "def create_session(user_id):\n"
        "    return f'session_{user_id}'\n\n"
        "def validate_session(token):\n"
        "    return True\n",
        encoding="utf-8",
    )
    (auth_dir / "login.py").write_text(
        "import os\n"
        "import bcrypt\n"
        "from database import get_user\n"
        "import utils.hashing as hasher\n"
        "from .session import create_session\n\n"
        "def login_user(username, password):\n"
        "    print('attempting login')\n"
        "    user = get_user(username)\n"
        "    if user and hasher.verify_password(password, user.password_hash):\n"
        "        return create_session(user.id)\n"
        "    return None\n",
        encoding="utf-8",
    )

    # 4. config package
    cfg_dir = repo_root / "config"
    cfg_dir.mkdir()
    (cfg_dir / "__init__.py").write_text("", encoding="utf-8")
    (cfg_dir / "settings.py").write_text(
        "import os\n"
        "from pydantic import BaseModel\n\n"
        "class Config(BaseModel):\n"
        "    env: str = 'prod'\n\n"
        "def get_setting(key):\n"
        "    print('reading', key)\n"
        "    n = len(key)\n"
        "    return os.getenv(key)\n",
        encoding="utf-8",
    )

    # 5. services package (cycles, ambiguity, syntax error, nested functions)
    srv_dir = repo_root / "services"
    srv_dir.mkdir()
    (srv_dir / "__init__.py").write_text("", encoding="utf-8")
    (srv_dir / "ambiguous_a.py").write_text(
        "def process_data(val):\n"
        "    return val + 1\n",
        encoding="utf-8",
    )
    (srv_dir / "ambiguous_b.py").write_text(
        "def process_data(val):\n"
        "    return val * 2\n\n"
        "def run_ambiguous(val):\n"
        "    return process_data(val)\n",
        encoding="utf-8",
    )
    (srv_dir / "caller_ambig.py").write_text(
        "def invoke_ambiguous(x):\n"
        "    return process_data(x)\n",
        encoding="utf-8",
    )
    (srv_dir / "cycles.py").write_text(
        "def func_a():\n"
        "    return func_b()\n\n"
        "def func_b():\n"
        "    return func_a()\n\n"
        "def recursive_fact(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * recursive_fact(n - 1)\n",
        encoding="utf-8",
    )
    (srv_dir / "syntax_err.py").write_text(
        "def broken_function(\n"
        "    this is not valid python!\n",
        encoding="utf-8",
    )
    (srv_dir / "nested.py").write_text(
        "from auth.session import validate_session\n\n"
        "def outer_service(token):\n"
        "    def inner_helper():\n"
        "        return validate_session(token)\n"
        "    return inner_helper()\n",
        encoding="utf-8",
    )

    return repo_root


def test_graph_building_and_call_resolution(mini_repo: Path) -> None:
    """Verify core graph structure and call resolution on login_user example."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, report = build_graph(store, repo_name="mini_repo", include_ambiguous=False)

    assert report.total_nodes > 0
    assert report.total_edges > 0
    assert report.call_resolution.total_calls > 0

    login_node = "func:auth.login.login_user"
    assert login_node in graph

    # Find outgoing CALLS edges from login_user
    out_calls = [
        (v, d)
        for _, v, d in graph.out_edges(login_node, data=True)
        if d.get("rel") == "CALLS"
    ]
    target_nodes = {v for v, _ in out_calls}

    # 1. login_user -> get_user (re-export from database/__init__.py)
    assert "func:database.user.get_user" in target_nodes
    # 2. login_user -> verify_password (aliased import hasher.verify_password)
    assert "func:utils.hashing.verify_password" in target_nodes
    # 3. login_user -> create_session (relative import from .session)
    assert "func:auth.session.create_session" in target_nodes

    # Check edge resolution details
    for target, data in out_calls:
        if target == "func:database.user.get_user":
            assert data["resolution"] == "import"
            assert data["confidence"] == "exact"
        elif target == "func:utils.hashing.verify_password":
            assert data["resolution"] == "import"
            assert data["confidence"] == "exact"
        elif target == "func:auth.session.create_session":
            assert data["resolution"] == "import"
            assert data["confidence"] == "exact"


def test_self_method_and_inheritance_resolution(mini_repo: Path) -> None:
    """Verify self.method() resolution within class and through inheritance."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    check_user_node = "method:database.user.User.check_user"
    assert check_user_node in graph

    calls = [
        (v, d)
        for _, v, d in graph.out_edges(check_user_node, data=True)
        if d.get("rel") == "CALLS"
    ]
    assert len(calls) == 1
    target, data = calls[0]
    assert target == "method:database.base.Model.save"
    assert data["resolution"] == "self_method"
    assert data["confidence"] == "high"


def test_inherits_edges_for_internal_and_external_bases(mini_repo: Path) -> None:
    """Verify INHERITS edges to internal classes and external nodes."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    # Internal inheritance: User -> Model
    user_inherits = [
        v for _, v, d in graph.out_edges("class:database.user.User", data=True)
        if d.get("rel") == "INHERITS"
    ]
    assert "class:database.base.Model" in user_inherits

    # External inheritance: Config -> external:BaseModel
    config_inherits = [
        v for _, v, d in graph.out_edges("class:config.settings.Config", data=True)
        if d.get("rel") == "INHERITS"
    ]
    assert any(v.startswith("external:") and "BaseModel" in v for v in config_inherits)


def test_imports_and_external_nodes(mini_repo: Path) -> None:
    """Verify IMPORTS edges for internal files and external nodes."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    login_file = "file:auth/login.py"
    imports_targets = {
        v for _, v, d in graph.out_edges(login_file, data=True)
        if d.get("rel") == "IMPORTS"
    }

    # Internal file imports
    assert "file:database/__init__.py" in imports_targets or "file:database/user.py" in imports_targets
    assert "file:utils/hashing.py" in imports_targets
    assert "file:auth/session.py" in imports_targets

    # External imports
    assert "external:bcrypt" in imports_targets
    assert "external:os" in imports_targets
    assert graph.nodes["external:bcrypt"]["type"] == "external"


def test_depends_on_weights_derived_correctly(mini_repo: Path) -> None:
    """Verify DEPENDS_ON file edges derive correct reference weights."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    # auth/login.py depends on auth/session.py (1 import + 1 call = weight 2)
    edges = [
        d for _, v, d in graph.out_edges("file:auth/login.py", data=True)
        if v == "file:auth/session.py" and d.get("rel") == "DEPENDS_ON"
    ]
    assert len(edges) == 1
    assert edges[0]["weight"] >= 2


def test_builtin_calls_not_added_to_graph_but_counted(mini_repo: Path) -> None:
    """Verify print and len create no edges but are reflected in stats."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, report = build_graph(store, repo_name="mini_repo")

    # Function get_setting calls print, len, and os.getenv
    get_setting_node = "func:config.settings.get_setting"
    calls = [
        v for _, v, d in graph.out_edges(get_setting_node, data=True)
        if d.get("rel") == "CALLS"
    ]
    # No CALLS edges created for builtins or external os.getenv
    assert len(calls) == 0
    assert report.call_resolution.builtin_calls >= 2
    assert report.call_resolution.external_calls >= 1


def test_ambiguous_calls_skipped_by_default_and_included_when_flagged(mini_repo: Path) -> None:
    """Verify ambiguous names are skipped by default and linked with confidence low when enabled."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        # Default: include_ambiguous=False
        graph_default, rep_default = build_graph(store, repo_name="mini_repo", include_ambiguous=False)
        # Flagged: include_ambiguous=True
        graph_ambig, rep_ambig = build_graph(store, repo_name="mini_repo", include_ambiguous=True)

    caller_node = "func:services.caller_ambig.invoke_ambiguous"
    assert caller_node in graph_default
    assert caller_node in graph_ambig

    # In default graph: ambiguous call to process_data should have NO CALLS edge
    default_calls = [
        v for _, v, d in graph_default.out_edges(caller_node, data=True)
        if d.get("rel") == "CALLS"
    ]
    assert len(default_calls) == 0
    assert rep_default.call_resolution.skipped_ambiguous >= 1

    # In ambiguous graph: edges to both ambiguous_a.process_data and ambiguous_b.process_data
    ambig_calls = [
        (v, d)
        for _, v, d in graph_ambig.out_edges(caller_node, data=True)
        if d.get("rel") == "CALLS"
    ]
    assert len(ambig_calls) == 2
    for target, d in ambig_calls:
        assert d["confidence"] == "low"
        assert d["resolution"] == "ambiguous"
        assert "process_data" in target


def test_callers_and_callees_transitive_depth(mini_repo: Path) -> None:
    """Verify callers() and callees() at depth 1 and 2."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    cg = CodeGraph(graph)

    # Callees of login_user depth 1
    callees_d1 = cg.callees("func:auth.login.login_user", depth=1)
    d1_qnames = {c.qualified_name for c in callees_d1}
    assert "database.user.get_user" in d1_qnames
    assert "utils.hashing.verify_password" in d1_qnames
    assert "auth.session.create_session" in d1_qnames

    # Callees of login_user depth 2 (should include hash_password called by verify_password)
    callees_d2 = cg.callees("func:auth.login.login_user", depth=2)
    d2_qnames = {c.qualified_name for c in callees_d2}
    assert "utils.hashing.hash_password" in d2_qnames

    # Callers of verify_password depth 1
    callers = cg.callers("func:utils.hashing.verify_password", depth=1)
    caller_qnames = {c.qualified_name for c in callers}
    assert "auth.login.login_user" in caller_qnames


def test_cycles_do_not_cause_infinite_loops(mini_repo: Path) -> None:
    """Verify mutual cycles (A calls B, B calls A) and recursion do not loop infinitely."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    cg = CodeGraph(graph)

    # Mutual cycle
    callees_a = cg.callees("func:services.cycles.func_a", depth=5)
    assert len(callees_a) > 0

    # Self recursion
    callees_fact = cg.callees("func:services.cycles.recursive_fact", depth=5)
    assert len(callees_fact) > 0

    # Neighborhood with cycle
    sub = cg.neighborhood("func:services.cycles.func_a", hops=3)
    assert sub.number_of_nodes() >= 2


def test_dependency_path_shortest_path_and_unreachable(mini_repo: Path) -> None:
    """Verify dependency_path returns correct CALLS path and None when unreachable."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    cg = CodeGraph(graph)

    # Path from login_user to hash_password
    path = cg.dependency_path(
        "func:auth.login.login_user",
        "func:utils.hashing.hash_password",
    )
    assert path == [
        "func:auth.login.login_user",
        "func:utils.hashing.verify_password",
        "func:utils.hashing.hash_password",
    ]

    # Unreachable path
    unreachable = cg.dependency_path(
        "func:auth.session.create_session",
        "func:config.settings.get_setting",
    )
    assert unreachable is None


def test_file_dependencies_and_related_files(mini_repo: Path) -> None:
    """Verify file_dependencies and related_files ranking."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    cg = CodeGraph(graph)

    # file_dependencies for auth/login.py
    deps = cg.file_dependencies("auth/login.py")
    assert "utils/hashing.py" in deps["depends_on"]
    assert "auth/session.py" in deps["depends_on"]

    # related_files for login_user
    rel_files = cg.related_files("func:auth.login.login_user", hops=2)
    assert len(rel_files) > 0
    # Files directly called should be at the top
    top_file_paths = [f for f, _ in rel_files]
    assert "auth/login.py" in top_file_paths or "utils/hashing.py" in top_file_paths


def test_save_graph_and_load_graph_round_trip(mini_repo: Path) -> None:
    """Verify save_graph and load_graph round trip preserves all nodes and edges."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    save_graph(graph, mini_repo)
    assert graph_exists(mini_repo)

    loaded_graph = load_graph(mini_repo)
    assert loaded_graph.number_of_nodes() == graph.number_of_nodes()
    assert loaded_graph.number_of_edges() == graph.number_of_edges()
    assert loaded_graph.graph.get("repo_name") == "mini_repo"


def test_syntax_error_file_handled_gracefully(mini_repo: Path) -> None:
    """Verify files with syntax errors do not crash graph construction."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, report = build_graph(store, repo_name="mini_repo")

    # File with syntax error is indexed and present as a node
    assert "file:services/syntax_err.py" in graph
    assert report.total_nodes > 0


def test_nested_functions_calls_attributed_to_enclosing_function(mini_repo: Path) -> None:
    """Verify calls inside nested functions roll up to the parent function node."""
    build_index(mini_repo)

    with MetadataStore.open(mini_repo) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")

    outer_node = "func:services.nested.outer_service"
    assert outer_node in graph

    calls = [
        v for _, v, d in graph.out_edges(outer_node, data=True)
        if d.get("rel") == "CALLS"
    ]
    # Call to validate_session inside inner_helper should roll up to outer_service
    assert "func:auth.session.validate_session" in calls
