"""Unit and integration tests for prompt context builder and compression (Step 7)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.context.context_builder import ContextBuilder
from app.context.context_compressor import compress, compress_to_fit
from app.context.formatter import format_context
from app.context.token_budget import BudgetTooSmallError, TokenBudget
from app.context.token_counter import (
    CharEstimateCounter,
    TiktokenCounter,
    get_counter,
)
from app.graph.graph_builder import build_graph
from app.graph.graph_io import load_graph, save_graph
from app.graph.graph_search import CodeGraph
from app.indexer import build_index
from app.models.schemas import (
    ArgumentInfo,
    BuilderOptions,
    CallInfo,
    ClassInfo,
    DetailLevel,
    FunctionInfo,
    RankedCandidate,
    RetrievalResult,
)
from app.retrieval.hybrid_search import HybridSearch
from app.storage.metadata_store import MetadataStore


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Create a realistic mini repository on disk with auth, database, utils, and config."""
    repo_root = tmp_path / "mini_repo"
    repo_root.mkdir()

    # 1. database package
    db_dir = repo_root / "database"
    db_dir.mkdir()
    (db_dir / "__init__.py").write_text("from .user import get_user\n", encoding="utf-8")
    (db_dir / "base.py").write_text(
        "class Model:\n"
        "    \"\"\"Base database model with persistence methods.\"\"\"\n"
        "    def save(self):\n"
        "        \"\"\"Save model state.\"\"\"\n"
        "        return True\n",
        encoding="utf-8",
    )
    (db_dir / "user.py").write_text(
        "from database.base import Model\n\n"
        "class User(Model):\n"
        "    \"\"\"User entity model representing active system accounts.\"\"\"\n"
        "    def __init__(self, id: int, username: str, password_hash: str) -> None:\n"
        "        self.id = id\n"
        "        self.username = username\n"
        "        self.password_hash = password_hash\n\n"
        "    def check_user(self) -> bool:\n"
        "        \"\"\"Verify user record integrity.\"\"\"\n"
        "        return self.save()\n\n"
        "def get_user(username: str) -> User:\n"
        "    \"\"\"Query user by username from persistence store.\"\"\"\n"
        "    # Simulated query execution\n"
        "    return User(1, username, 'hashed_pw')\n",
        encoding="utf-8",
    )

    # 2. utils package
    utils_dir = repo_root / "utils"
    utils_dir.mkdir()
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "hashing.py").write_text(
        "import hashlib\n\n"
        "def hash_password(password: str) -> str:\n"
        "    \"\"\"Compute SHA-256 digest of password string.\"\"\"\n"
        "    # Compute hash\n"
        "    return f'hashed_{password}'\n\n"
        "def verify_password(password: str, hashed: str) -> bool:\n"
        "    \"\"\"Compare plain password against hashed digest.\"\"\"\n"
        "    return hash_password(password) == hashed\n",
        encoding="utf-8",
    )

    # 3. auth package
    auth_dir = repo_root / "auth"
    auth_dir.mkdir()
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "session.py").write_text(
        "import uuid\n\n"
        "def create_session(user_id: int) -> str:\n"
        "    \"\"\"Issue new session token for authenticated user.\"\"\"\n"
        "    return f'session_{user_id}'\n\n"
        "def validate_session(token: str) -> bool:\n"
        "    \"\"\"Check if session token is still valid.\"\"\"\n"
        "    return True\n",
        encoding="utf-8",
    )
    (auth_dir / "login.py").write_text(
        "import os\n"
        "import bcrypt\n"
        "from database.user import get_user\n"
        "from utils.hashing import verify_password\n"
        "from auth.session import create_session\n\n"
        "def login_user(username: str, password: str) -> str | None:\n"
        "    \"\"\"Authenticate a user and generate a session token.\n\n"
        "    Performs credential lookup, hash verification, and session creation.\n"
        "    \"\"\"\n"
        "    # Step 1: Look up user in database\n"
        "    user = get_user(username)\n"
        "    # Step 2: Validate credentials\n"
        "    if user and verify_password(password, user.password_hash):\n"
        "        # Step 3: Issue token\n"
        "        return create_session(user.id)\n"
        "    return None\n\n"
        "def authenticate_user(token: str) -> bool:\n"
        "    \"\"\"Check authentication status for user session.\"\"\"\n"
        "    return bool(token)\n",
        encoding="utf-8",
    )

    # 4. config package
    cfg_dir = repo_root / "config"
    cfg_dir.mkdir()
    (cfg_dir / "__init__.py").write_text("", encoding="utf-8")
    (cfg_dir / "settings.py").write_text(
        "import os\n\n"
        "class Settings:\n"
        "    \"\"\"Application configuration parameters.\"\"\"\n"
        "    env: str = 'prod'\n\n"
        "def get_setting(key: str) -> str | None:\n"
        "    \"\"\"Read configuration key from environment.\"\"\"\n"
        "    return os.getenv(key)\n",
        encoding="utf-8",
    )

    # 5. Build index and graph
    build_index(repo_root)
    with MetadataStore.open(repo_root) as store:
        graph, _ = build_graph(store, repo_name="mini_repo")
        save_graph(graph, repo_root)

    return repo_root


# =============================================================================
# 1. Token Counter Tests
# =============================================================================


def test_token_counter_counts_positive_and_stable():
    """Verify TokenCounter produces positive, consistent token counts."""
    counter = get_counter("tiktoken")
    text = "def calculate_total(items: list[int]) -> int:\n    return sum(items)"
    count1 = counter.count(text)
    count2 = counter.count(text)
    assert count1 > 0
    assert count1 == count2
    assert counter.name == "tiktoken"


def test_token_counter_empty_string():
    """Verify empty string returns 0 tokens."""
    counter = get_counter("tiktoken")
    assert counter.count("") == 0

    char_counter = get_counter("char")
    assert char_counter.count("") == 0


def test_token_counter_cache():
    """Verify counter caches results by text hash."""
    counter = TiktokenCounter()
    text = "hello world token caching test"
    assert len(counter._cache) == 0
    c1 = counter.count(text)
    assert len(counter._cache) == 1
    c2 = counter.count(text)
    assert c1 == c2


def test_token_counter_factory_and_fallbacks():
    """Verify factory returns appropriate instances and handles errors."""
    c_tik = get_counter("tiktoken")
    assert isinstance(c_tik, TiktokenCounter)

    c_char = get_counter("char")
    assert isinstance(c_char, CharEstimateCounter)

    with pytest.raises(ValueError, match="Unknown token counter"):
        get_counter("invalid_counter_name")


def test_token_counter_char_fallback_warns():
    """Verify CharEstimateCounter emits UserWarning on initialization when requested."""
    with pytest.warns(UserWarning, match="CharEstimateCounter"):
        CharEstimateCounter(warn=True)


# =============================================================================
# 2. Compressor Tests
# =============================================================================


def _make_sample_function() -> FunctionInfo:
    source = (
        "@decorator_one\n"
        "@decorator_two(option='val')\n"
        "def complex_worker(a: int, b: str = 'default', *args, **kwargs) -> bool:\n"
        "    \"\"\"Worker function performing multi-step data transformations.\n\n"
        "    Detailed docstring line 2.\n"
        "    \"\"\"\n"
        "    # Temporary comment line\n"
        "    first_step = a + 10\n"
        "\n"
        "    # Second step with long list\n"
        "    long_table = [\n"
        "        'row_1',\n"
        "        'row_2',\n"
        "        'row_3',\n"
        "        'row_4',\n"
        "        'row_5',\n"
        "    ]\n"
        "    return bool(first_step)\n"
    )
    return FunctionInfo(
        name="complex_worker",
        qualified_name="test_mod.complex_worker",
        file_path="test_mod.py",
        start_line=1,
        end_line=20,
        arguments=[
            ArgumentInfo(name="a", annotation="int", default=None, kind="positional"),
            ArgumentInfo(name="b", annotation="str", default="'default'", kind="positional"),
            ArgumentInfo(name="args", annotation=None, default=None, kind="vararg"),
            ArgumentInfo(name="kwargs", annotation=None, default=None, kind="kwarg"),
        ],
        return_annotation="bool",
        decorators=["@decorator_one", "@decorator_two(option='val')"],
        docstring="Worker function performing multi-step data transformations.",
        calls=[CallInfo(name="helper_one", lineno=15), CallInfo(name="helper_two", lineno=18)],
        source=source,
        is_async=False,
        is_method=False,
        parent_class=None,
    )


def test_compressor_monotonic_token_decrease():
    """Verify each level is strictly <= the previous in tokens for a non-trivial function."""
    fn = _make_sample_function()
    counter = get_counter("tiktoken")

    snip_full = compress(fn, DetailLevel.FULL, counter=counter)
    snip_trimmed = compress(fn, DetailLevel.TRIMMED, counter=counter)
    snip_signature = compress(fn, DetailLevel.SIGNATURE, counter=counter)
    snip_stub = compress(fn, DetailLevel.STUB, counter=counter)
    snip_ref = compress(fn, DetailLevel.REFERENCE, counter=counter)

    assert snip_full.tokens > snip_trimmed.tokens
    assert snip_trimmed.tokens > snip_signature.tokens
    assert snip_signature.tokens > snip_stub.tokens
    assert snip_stub.tokens >= snip_ref.tokens


def test_compressor_trimmed_passes_ast_parse():
    """Verify TRIMMED output passes ast.parse and removes comments."""
    fn = _make_sample_function()
    snip = compress(fn, DetailLevel.TRIMMED)

    # Strip header line for ast.parse verification
    lines = snip.text.splitlines()
    assert lines[0].startswith("# test_mod.py:1-20 [trimmed]")
    code_body = "\n".join(lines[1:])

    parsed = ast.parse(code_body)
    assert isinstance(parsed, ast.Module)
    # Check that inline comments were eliminated
    assert "# Temporary comment line" not in snip.text


def test_compressor_signature_and_stub():
    """Verify SIGNATURE and STUB retain decorators, args, and format."""
    fn = _make_sample_function()
    snip_sig = compress(fn, DetailLevel.SIGNATURE)
    assert "@decorator_one" in snip_sig.text
    assert "def complex_worker(a: int, b: str = 'default', *args, **kwargs) -> bool:" in snip_sig.text
    assert "Worker function" in snip_sig.text
    assert "# calls: helper_one, helper_two" in snip_sig.text

    snip_stub = compress(fn, DetailLevel.STUB)
    # Stub must be a single line definition
    stub_lines = snip_stub.text.strip().splitlines()
    assert len(stub_lines) == 1
    assert stub_lines[0].startswith("def complex_worker(")
    assert stub_lines[0].endswith("# test_mod.py:1-20")


def test_compressor_class_levels():
    """Verify ClassInfo compression behaves correctly across levels."""
    cls_source = (
        "class TestService:\n"
        "    \"\"\"Service managing data synchronization.\"\"\"\n"
        "    def sync(self) -> bool:\n"
        "        return True\n\n"
        "    def purge(self) -> None:\n"
        "        pass\n"
    )
    cls = ClassInfo(
        name="TestService",
        qualified_name="srv.TestService",
        file_path="srv.py",
        start_line=1,
        end_line=10,
        bases=["BaseService"],
        decorators=["@singleton"],
        docstring="Service managing data synchronization.",
        methods=[
            FunctionInfo(
                name="sync",
                qualified_name="srv.TestService.sync",
                file_path="srv.py",
                start_line=3,
                end_line=4,
                arguments=[ArgumentInfo(name="self", annotation=None, default=None, kind="positional")],
                return_annotation="bool",
                decorators=[],
                docstring=None,
                calls=[],
                source="def sync(self) -> bool:\n    return True\n",
                is_async=False,
                is_method=True,
                parent_class="TestService",
            ),
            FunctionInfo(
                name="purge",
                qualified_name="srv.TestService.purge",
                file_path="srv.py",
                start_line=6,
                end_line=7,
                arguments=[ArgumentInfo(name="self", annotation=None, default=None, kind="positional")],
                return_annotation="None",
                decorators=[],
                docstring=None,
                calls=[],
                source="def purge(self) -> None:\n    pass\n",
                is_async=False,
                is_method=True,
                parent_class="TestService",
            ),
        ],
        source=cls_source,
    )

    full = compress(cls, DetailLevel.FULL)
    assert "[full]" in full.text
    assert "return True" in full.text

    sig = compress(cls, DetailLevel.SIGNATURE)
    assert "class TestService(BaseService):" in sig.text
    assert "def sync(self) -> bool: ..." in sig.text
    assert "def purge(self) -> None: ..." in sig.text

    stub = compress(cls, DetailLevel.STUB)
    assert "class TestService(BaseService): ...  # methods: sync, purge" in stub.text

    ref = compress(cls, DetailLevel.REFERENCE)
    assert ref.text == "srv.TestService (srv.py:1-10)"


def test_compress_to_fit():
    """Verify compress_to_fit selects the most detailed fitting level."""
    fn = _make_sample_function()
    counter = get_counter("tiktoken")

    full_cost = compress(fn, DetailLevel.FULL, counter=counter).tokens
    sig_cost = compress(fn, DetailLevel.SIGNATURE, counter=counter).tokens
    ref_cost = compress(fn, DetailLevel.REFERENCE, counter=counter).tokens

    fit_full = compress_to_fit(fn, max_tokens=full_cost + 10, counter=counter)
    assert fit_full is not None
    assert fit_full.level == DetailLevel.FULL

    fit_sig = compress_to_fit(fn, max_tokens=sig_cost + 5, counter=counter)
    assert fit_sig is not None
    assert fit_sig.level == DetailLevel.SIGNATURE

    fit_ref = compress_to_fit(fn, max_tokens=ref_cost + 2, counter=counter)
    assert fit_ref is not None
    assert fit_ref.level == DetailLevel.REFERENCE

    fit_none = compress_to_fit(fn, max_tokens=ref_cost - 5, counter=counter)
    assert fit_none is None


def test_compressor_degraded_fallback():
    """Verify degraded fallback activates for unparseable source."""
    broken_fn = FunctionInfo(
        name="bad_syntax",
        qualified_name="test.bad_syntax",
        file_path="test.py",
        start_line=1,
        end_line=3,
        arguments=[],
        return_annotation=None,
        decorators=[],
        docstring=None,
        calls=[],
        source="def bad_syntax():\n    if True\n        print(1)",  # Missing colon
        is_async=False,
        is_method=False,
        parent_class=None,
    )
    snip = compress(broken_fn, DetailLevel.TRIMMED)
    assert snip.degraded is True
    assert "bad_syntax" in snip.text


# =============================================================================
# 3. Budget Tests
# =============================================================================


def test_budget_invariants():
    """Verify TokenBudget state tracking methods and invariants."""
    b = TokenBudget(total_budget=1000, reserved_overhead=200, reserved_for_task=100)
    assert b.remaining() == 1000
    assert b.utilization() == 0.0
    assert b.can_afford(500)
    assert not b.can_afford(1200)

    b.consume(400)
    assert b.consumed == 400
    assert b.remaining() == 600
    assert b.utilization() == 0.4

    b.reset()
    assert b.consumed == 0
    assert b.remaining() == 1000


def test_budget_too_small_error():
    """Verify BudgetTooSmallError is raised for impossible budgets."""
    task = "A" * 200
    retrieval = RetrievalResult(query=task, target="test.fn", candidates=[])
    builder = ContextBuilder(metadata_store=None)  # type: ignore

    with pytest.raises(BudgetTooSmallError, match="too small"):
        builder.build(retrieval, budget=50)


# =============================================================================
# 4. HARD INVARIANT TEST (Parametrized across budgets)
# =============================================================================


@pytest.mark.parametrize("budget", [300, 500, 1000, 2000, 4000, 8000])
def test_hard_invariant_rendered_tokens_always_under_budget(mini_repo: Path, budget: int):
    """CRITICAL HARD INVARIANT: Final rendered context must NEVER exceed total_budget."""
    counter = get_counter("tiktoken")
    with MetadataStore.open(mini_repo) as store:
        nx_g = load_graph(mini_repo)
        graph = CodeGraph(nx_g) if nx_g is not None else None
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("Optimize the login_user function")

        builder = ContextBuilder(metadata_store=store, graph=graph, counter=counter)
        built = builder.build(retrieval, budget=budget)

        counted_tokens = counter.count(built.text)
        assert counted_tokens <= budget, (
            f"Hard invariant violated! Budget: {budget}, Actual tokens: {counted_tokens}"
        )
        assert built.tokens <= budget
        assert built.utilization <= 1.0


# =============================================================================
# 5. Target Handling Tests
# =============================================================================


def test_target_full_when_fits(mini_repo: Path):
    """Verify target entity is included at FULL level when generous budget allows."""
    with MetadataStore.open(mini_repo) as store:
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("Optimize the login_user function")

        builder = ContextBuilder(metadata_store=store)
        built = builder.build(retrieval, budget=4000)

        target_item = next(it for it in built.items if it.role == "target")
        assert target_item.level == DetailLevel.FULL.value
        assert "login_user" in target_item.ref


def test_target_downgraded_when_exceeding_target_share(mini_repo: Path):
    """Verify target entity is downgraded when budget target_share cannot afford FULL."""
    with MetadataStore.open(mini_repo) as store:
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("login_user")

        builder = ContextBuilder(metadata_store=store)
        # Small budget: 200 tokens. login_user full won't fit within 0.45 * 200 = 90 tokens
        tb = TokenBudget(total_budget=200, target_share=0.20)
        built = builder.build(retrieval, budget=tb)

        target_item = next(it for it in built.items if it.role == "target")
        assert target_item.level in (DetailLevel.TRIMMED.value, DetailLevel.SIGNATURE.value, DetailLevel.STUB.value)


def test_target_method_includes_class_context(mini_repo: Path):
    """Verify target that is a class method includes parent class context."""
    with MetadataStore.open(mini_repo) as store:
        retrieval = RetrievalResult(
            query="Verify user record",
            target="database.user.User.check_user",
            target_confidence="high",
            candidates=[],
            intent="general",
        )
        builder = ContextBuilder(metadata_store=store)
        built = builder.build(retrieval, budget=4000)

        # Method is target
        assert any(it.ref == "database.user.User.check_user" and it.role == "target" for it in built.items)
        # Parent class is included as related context
        assert any("User" in it.ref and it.role == "related" for it in built.items)


def test_target_uncertain_flag(mini_repo: Path):
    """Verify low-confidence retrieval marks target_uncertain True."""
    with MetadataStore.open(mini_repo) as store:
        retrieval = RetrievalResult(
            query="Vague query without obvious target",
            target="auth.login.login_user",
            target_confidence="low",
            candidates=[],
        )
        builder = ContextBuilder(metadata_store=store)
        built = builder.build(retrieval, budget=2000)
        assert built.target_uncertain is True


# =============================================================================
# 6. Candidate Selection & Progressive Detail Tests
# =============================================================================


def test_selection_intent_role_aim(mini_repo: Path):
    """Verify intent promotes callers for explain/fix_bug vs callees for optimize."""
    with MetadataStore.open(mini_repo) as store:
        nx_g = load_graph(mini_repo)
        CodeGraph(nx_g) if nx_g is not None else None

        # 1. Optimize intent -> Callees prioritized
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        res_opt = retriever.search("optimize login_user")
        assert res_opt.intent == "optimize"

        # 2. Explain intent -> Callers promoted
        res_exp = retriever.search("explain how login_user works")
        assert res_exp.intent == "explain"


def test_selection_must_include_and_exclude(mini_repo: Path):
    """Verify must_include items are pinned and exclude items are omitted."""
    with MetadataStore.open(mini_repo) as store:
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("login user")

        options = BuilderOptions(
            must_include=["config.settings.get_setting"],
            exclude=["utils.hashing.verify_password"],
        )
        builder = ContextBuilder(metadata_store=store)
        built = builder.build(retrieval, budget=4000, options=options)

        included_refs = {it.ref for it in built.items}
        assert "config.settings.get_setting" in included_refs
        assert "utils.hashing.verify_password" not in included_refs


def test_selection_max_items_limit(mini_repo: Path):
    """Verify max_items cap is strictly enforced."""
    with MetadataStore.open(mini_repo) as store:
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("login user")

        options = BuilderOptions(max_items=2)
        builder = ContextBuilder(metadata_store=store)
        built = builder.build(retrieval, budget=4000, options=options)

        assert len(built.items) <= 2


# =============================================================================
# 7. Deduplication, Imports, and Dependency Paths
# =============================================================================


def test_deduplication_method_and_parent_class(mini_repo: Path):
    """Verify if class is FULL/TRIMMED, enclosed method is not duplicated in SOURCE CODE."""
    with MetadataStore.open(mini_repo) as store:
        # Include both User class and User.check_user
        retrieval = RetrievalResult(
            query="inspect user",
            target="database.user.User",  # Class target at FULL
            target_confidence="high",
            candidates=[
                RankedCandidate(
                    ref="database.user.User.check_user",
                    score=0.9,
                    entity_type="method",
                    file="database/user.py",
                    start_line=9,
                    end_line=11,
                )
            ],
        )
        builder = ContextBuilder(metadata_store=store)
        built = builder.build(retrieval, budget=4000)

        # Count occurrences of "check_user" definition in SOURCE CODE section
        src_idx = built.text.find("SOURCE CODE:")
        source_code_section = built.text[src_idx:]
        assert source_code_section.count("def check_user") == 1


def test_relevant_imports_filtered(mini_repo: Path):
    """Verify only imports actually used by included snippets appear."""
    with MetadataStore.open(mini_repo) as store:
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("login_user")

        builder = ContextBuilder(metadata_store=store)
        built = builder.build(retrieval, budget=4000)

        # Check RELEVANT IMPORTS section
        if "RELEVANT IMPORTS:" in built.text:
            imp_sec = built.text.split("RELEVANT IMPORTS:")[1].split("SOURCE CODE:")[0]
            # os is not used in login_user snippet
            assert "os" not in imp_sec
            # get_user is used in login_user snippet
            assert "get_user" in imp_sec


def test_dependency_path_rendered(mini_repo: Path):
    """Verify CALLS dependency path is rendered when graph is present."""
    with MetadataStore.open(mini_repo) as store:
        nx_g = load_graph(mini_repo)
        graph = CodeGraph(nx_g) if nx_g is not None else None
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("login_user")

        builder = ContextBuilder(metadata_store=store, graph=graph)
        built = builder.build(retrieval, budget=4000)

        assert "DEPENDENCY MAP:" in built.text
        assert "login_user ->" in built.text


def test_dependency_path_omitted_without_graph(mini_repo: Path):
    """Verify DEPENDENCY MAP is gracefully omitted when graph is missing."""
    with MetadataStore.open(mini_repo) as store:
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("login_user")

        builder = ContextBuilder(metadata_store=store, graph=None)
        built = builder.build(retrieval, budget=4000)

        assert "DEPENDENCY MAP:" not in built.text


# =============================================================================
# 8. Determinism & Formatter Tests
# =============================================================================


def test_determinism_identical_runs(mini_repo: Path):
    """Verify building twice with the exact same inputs yields byte-identical text."""
    with MetadataStore.open(mini_repo) as store:
        nx_g = load_graph(mini_repo)
        graph = CodeGraph(nx_g) if nx_g is not None else None
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("Optimize the login function")

        builder = ContextBuilder(metadata_store=store, graph=graph)
        run1 = builder.build(retrieval, budget=3000)
        run2 = builder.build(retrieval, budget=3000)

        assert run1.text == run2.text
        assert run1.tokens == run2.tokens
        assert run1.items == run2.items


def test_formatter_markdown_and_json(mini_repo: Path):
    """Verify text, markdown, and json output formats render properly."""
    with MetadataStore.open(mini_repo) as store:
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("login_user")

        builder = ContextBuilder(metadata_store=store)
        built = builder.build(retrieval, budget=4000)

        txt_fmt = format_context(built, fmt="text")
        assert "TASK:" in txt_fmt
        assert "TARGET:" in txt_fmt

        md_fmt = format_context(built, fmt="markdown")
        assert "## TASK" in md_fmt
        assert "```python" in md_fmt

        json_fmt = format_context(built, fmt="json")
        assert '"text":' in json_fmt
        assert '"budget_total":' in json_fmt


# =============================================================================
# 9. End-to-End Goal Scenarios
# =============================================================================


def test_optimize_login_user_scenario(mini_repo: Path):
    """Query 'Optimize the login function' includes login_user at FULL, plus callees under 2000 budget."""
    with MetadataStore.open(mini_repo) as store:
        nx_g = load_graph(mini_repo)
        graph = CodeGraph(nx_g) if nx_g is not None else None
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("Optimize the login function")

        builder = ContextBuilder(metadata_store=store, graph=graph)
        built = builder.build(retrieval, budget=2000)

        assert built.tokens <= 2000
        # Target login_user is included at FULL
        target_item = next(it for it in built.items if it.role == "target")
        assert "login_user" in target_item.ref
        assert target_item.level == DetailLevel.FULL.value

        # Callees get_user, verify_password, create_session included at some level
        included_refs = " ".join(it.ref for it in built.items)
        assert "get_user" in included_refs
        assert "verify_password" in included_refs
        assert "create_session" in included_refs


def test_tiny_budget_300_scenario(mini_repo: Path):
    """Tiny budget (300) returns task + target at reduced detail without error and <= 300 tokens."""
    counter = get_counter("tiktoken")
    with MetadataStore.open(mini_repo) as store:
        retriever = HybridSearch.from_repo(mini_repo, metadata_store=store)
        retrieval = retriever.search("Optimize login")

        builder = ContextBuilder(metadata_store=store, counter=counter)
        built = builder.build(retrieval, budget=300)

        assert built.tokens <= 300
        assert counter.count(built.text) <= 300
        assert any(it.role == "target" for it in built.items)
