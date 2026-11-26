"""Unit and integration tests for semantic embeddings and vector search (Step 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.embeddings.embedding_generator import EmbeddingGenerator, FakeEmbeddingGenerator
from app.embeddings.indexer import build_embeddings
from app.embeddings.semantic_search import SemanticSearch
from app.embeddings.text_builder import (
    build_function_desc,
    is_trivial_function,
    split_identifier,
    truncate_code_at_line_boundary,
)
from app.embeddings.vector_store import VectorStore
from app.indexer import build_index
from app.models.schemas import ArgumentInfo, CallInfo, EmbeddedItem, FunctionInfo
from app.storage.metadata_store import MetadataStore


@pytest.fixture
def fake_generator() -> FakeEmbeddingGenerator:
    """Fixture providing a fast, deterministic offline embedding generator."""
    return FakeEmbeddingGenerator(model_name="test-fake-model", dimension=32)


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Create a realistic mini repository on disk with multiple packages."""
    repo_root = tmp_path / "mini_repo"
    repo_root.mkdir()

    # 1. auth package
    auth_dir = repo_root / "auth"
    auth_dir.mkdir()
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "security.py").write_text(
        "import hashlib\n\n"
        "def hash_password(password: str) -> str:\n"
        "    '''Hash a password using sha256 algorithm.'''\n"
        "    return hashlib.sha256(password.encode()).hexdigest()\n\n"
        "def verify_password(password: str, hashed: str) -> bool:\n"
        "    '''Verify that raw password matches the given hash.'''\n"
        "    return hash_password(password) == hashed\n\n"
        "def trivial_func():\n"
        "    pass\n",
        encoding="utf-8",
    )

    # 2. mail package
    mail_dir = repo_root / "mail"
    mail_dir.mkdir()
    (mail_dir / "__init__.py").write_text("", encoding="utf-8")
    (mail_dir / "notifier.py").write_text(
        "class EmailClient:\n"
        "    '''Client for dispatching electronic mail.'''\n"
        "    def __init__(self, host: str):\n"
        "        self.host = host\n\n"
        "    def send_email(self, recipient: str, body: str) -> bool:\n"
        "        '''Send an outgoing email message to the recipient.'''\n"
        "        return True\n",
        encoding="utf-8",
    )

    return repo_root


# ============================================================================
# Unit Tests (Fast & Offline)
# ============================================================================


def test_split_identifier() -> None:
    """Check that split_identifier correctly splits snake_case, camelCase, and PascalCase."""
    assert split_identifier("verify_password") == ["verify", "password"]
    assert split_identifier("checkPasswordHash") == ["check", "password", "hash"]
    assert split_identifier("HTTPClient") == ["http", "client"]
    assert split_identifier("parse_JSON_response") == ["parse", "json", "response"]
    assert split_identifier("_internal_helper_fn") == ["internal", "helper", "fn"]
    assert split_identifier("") == []


def test_text_builder_function_desc() -> None:
    """Check that function_desc incorporates split name, args, calls, and docstrings."""
    fn = FunctionInfo(
        name="verify_password",
        qualified_name="auth.security.verify_password",
        file_path="auth/security.py",
        start_line=10,
        end_line=20,
        source="def verify_password(password: str, hashed: str):\n    return hash_password(password) == hashed\n",
        arguments=[
            ArgumentInfo(name="self", annotation=None, default=None, kind="positional"),
            ArgumentInfo(name="password", annotation="str", default=None, kind="positional"),
            ArgumentInfo(name="hashed", annotation="str", default=None, kind="positional"),
        ],
        calls=[CallInfo(name="hash_password", lineno=11)],
        docstring="Verify a raw password against stored hash.",
        is_method=True,
        parent_class="AuthManager",
    )

    desc = build_function_desc(fn)
    assert "Method AuthManager.verify_password" in desc
    assert "(verify password)" in desc
    assert "Arguments: password: str, hashed: str" in desc
    # 'self' must be excluded
    assert "Arguments: self" not in desc
    assert "Calls: hash password" in desc
    assert "Docstring: Verify a raw password against stored hash." in desc


def test_text_builder_truncation_at_line_boundary() -> None:
    """Ensure source truncation cuts strictly at line boundaries and never mid-line."""
    source = (
        "def long_function():\n"
        "    line_one = 1\n"
        "    line_two = 2\n"
        "    line_three = 3\n"
        "    line_four = 4\n"
    )

    # Limit to small character/token budget
    truncated = truncate_code_at_line_boundary(
        qualified_name="long_function",
        source=source,
        max_tokens=8,  # Very small budget
        tokenizer=None,
    )

    assert truncated.startswith("# long_function\n")
    # All preserved lines must be complete lines ending with newline or end of line
    for line in truncated.splitlines()[1:]:
        assert line in ["def long_function():", "    line_one = 1", "    line_two = 2", "    line_three = 3"]


def test_text_builder_skips_trivial_functions() -> None:
    """Verify that trivially short functions without docstrings are detected and skipped."""
    trivial = FunctionInfo(
        name="noop",
        qualified_name="utils.noop",
        file_path="utils.py",
        start_line=1,
        end_line=2,
        source="def noop():\n    pass\n",
        docstring=None,
    )
    assert is_trivial_function(trivial, min_body_lines=2) is True

    with_doc = FunctionInfo(
        name="noop_doc",
        qualified_name="utils.noop_doc",
        file_path="utils.py",
        start_line=1,
        end_line=3,
        source="def noop_doc():\n    '''Documented noop.'''\n    pass\n",
        docstring="Documented noop.",
    )
    assert is_trivial_function(with_doc, min_body_lines=2) is False

    two_lines_body = FunctionInfo(
        name="calc",
        qualified_name="utils.calc",
        file_path="utils.py",
        start_line=1,
        end_line=3,
        source="def calc():\n    x = 1\n    return x\n",
        docstring=None,
    )
    assert is_trivial_function(two_lines_body, min_body_lines=2) is False


def test_vector_store_add_and_search(tmp_path: Path) -> None:
    """Test adding items, searching, and filtering by kind and file."""
    store = VectorStore(
        dimension=4,
        model_name="test-model",
        repo_path=tmp_path,
        namespace=str(tmp_path / "embeddings"),
    )

    items = [
        EmbeddedItem(
            item_id="fn_desc:one",
            kind="function_desc",
            ref="mod.func_one",
            file="mod.py",
            text_hash="h1",
            text="func one",
        ),
        EmbeddedItem(
            item_id="fn_code:one",
            kind="function_code",
            ref="mod.func_one",
            file="mod.py",
            text_hash="h2",
            text="code one",
        ),
        EmbeddedItem(
            item_id="cls_desc:two",
            kind="class_desc",
            ref="other.ClassTwo",
            file="other.py",
            text_hash="h3",
            text="class two",
        ),
    ]

    # Normalized vectors
    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.add(items, vecs)
    assert store.index.ntotal == 3

    # Query closest to [1.0, 0.0, 0.0, 0.0]
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    hits = store.search(q_vec, top_k=2)
    assert len(hits) == 2
    assert hits[0].item_id == "fn_desc:one"
    assert pytest.approx(hits[0].score, 1e-4) == 1.0

    # Filter by kind
    code_hits = store.search(q_vec, top_k=5, kinds=["function_code"])
    assert len(code_hits) == 1
    assert code_hits[0].item_id == "fn_code:one"

    # Filter by file
    other_hits = store.search(q_vec, top_k=5, files=["other.py"])
    assert len(other_hits) == 1
    assert other_hits[0].item_id == "cls_desc:two"


def test_vector_store_save_load_round_trip(tmp_path: Path) -> None:
    """Test saving VectorStore to MongoDB and reloading it; search results must match identically."""
    base_dir = tmp_path / "embeddings"
    store = VectorStore(
        dimension=3,
        model_name="round-trip-model",
        repo_path=tmp_path,
        namespace=str(base_dir),
    )

    items = [
        EmbeddedItem(
            item_id="item:1",
            kind="function_desc",
            ref="foo",
            file="foo.py",
            text_hash="h1",
            text="foo desc",
        ),
        EmbeddedItem(
            item_id="item:2",
            kind="class_desc",
            ref="Bar",
            file="bar.py",
            text_hash="h2",
            text="bar desc",
        ),
    ]
    vecs = np.array(
        [
            [0.6, 0.8, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    store.add(items, vecs)
    store.save()

    manifest = VectorStore.read_manifest(tmp_path, namespace=str(base_dir))
    assert manifest is not None
    assert manifest["item_count"] == 2
    assert manifest["model_name"] == "round-trip-model"
    assert manifest["index_size_bytes"] > 0

    # Reload in a new store instance
    loaded_store = VectorStore(
        dimension=3,
        model_name="round-trip-model",
        repo_path=tmp_path,
        namespace=str(base_dir),
    )
    loaded_store.load()

    assert loaded_store.index.ntotal == 2
    assert len(loaded_store.items) == 2
    assert loaded_store.items[0].item_id == "item:1"
    assert loaded_store.items[1].item_id == "item:2"

    q_vec = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    orig_results = store.search(q_vec, top_k=2)
    loaded_results = loaded_store.search(q_vec, top_k=2)

    assert len(orig_results) == len(loaded_results)
    for orig, loaded in zip(orig_results, loaded_results):
        assert orig.item_id == loaded.item_id
        assert pytest.approx(orig.score, 1e-4) == loaded.score


def test_vector_store_remove_by_file(tmp_path: Path) -> None:
    """Test removing items by file path correctly rebuilds the FAISS index."""
    store = VectorStore(
        dimension=2,
        model_name="test-model",
        repo_path=tmp_path,
        namespace=str(tmp_path / "embeddings"),
    )

    items = [
        EmbeddedItem(
            item_id="f1:a",
            kind="function_desc",
            ref="a",
            file="file1.py",
            text_hash="h1",
            text="a",
        ),
        EmbeddedItem(
            item_id="f2:b",
            kind="function_desc",
            ref="b",
            file="file2.py",
            text_hash="h2",
            text="b",
        ),
        EmbeddedItem(
            item_id="f1:c",
            kind="function_code",
            ref="c",
            file="file1.py",
            text_hash="h3",
            text="c",
        ),
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    store.add(items, vecs)
    assert store.index.ntotal == 3

    removed = store.remove_by_file("file1.py")
    assert removed == 2
    assert store.index.ntotal == 1
    assert len(store.items) == 1
    assert store.items[0].file == "file2.py"

    q_vec = np.array([0.0, 1.0], dtype=np.float32)
    hits = store.search(q_vec, top_k=5)
    assert len(hits) == 1
    assert hits[0].item_id == "f2:b"


def test_vector_store_model_dimension_mismatch(tmp_path: Path) -> None:
    """Verify that a model name or dimension mismatch on load raises a clear ValueError."""
    base_dir = tmp_path / "embeddings"
    store = VectorStore(
        dimension=32,
        model_name="model-alpha",
        repo_path=tmp_path,
        namespace=str(base_dir),
    )
    store.save()

    # Mismatched model name
    bad_model_store = VectorStore(
        dimension=32,
        model_name="model-beta",
        repo_path=tmp_path,
        namespace=str(base_dir),
    )
    with pytest.raises(ValueError, match="Model mismatch"):
        bad_model_store.load()

    # Mismatched dimension
    bad_dim_store = VectorStore(
        dimension=64,
        model_name="model-alpha",
        repo_path=tmp_path,
        namespace=str(base_dir),
    )
    with pytest.raises(ValueError, match="Dimension mismatch"):
        bad_dim_store.load()


def test_build_embeddings_incremental_idempotent(
    mini_repo: Path, fake_generator: FakeEmbeddingGenerator, tmp_path: Path
) -> None:
    """Running build_embeddings twice with no changes embeds 0 items on the second run."""
    emb_base = tmp_path / "emb_storage"
    build_index(mini_repo)

    # First run
    r1 = build_embeddings(
        repo_path=mini_repo,
        generator=fake_generator,
        namespace=str(emb_base),
    )
    assert r1.items_total > 0
    assert r1.items_new == r1.items_total
    assert r1.items_unchanged == 0
    assert r1.items_changed == 0

    # Second run without any code modification
    r2 = build_embeddings(
        repo_path=mini_repo,
        generator=fake_generator,
        namespace=str(emb_base),
    )
    assert r2.items_new == 0
    assert r2.items_changed == 0
    assert r2.items_unchanged == r1.items_total
    assert r2.items_total == r1.items_total


def test_build_embeddings_after_editing_function(
    mini_repo: Path, fake_generator: FakeEmbeddingGenerator, tmp_path: Path
) -> None:
    """Editing a function causes only that file's items to be re-embedded."""
    emb_base = tmp_path / "emb_storage"
    build_index(mini_repo)

    r1 = build_embeddings(
        repo_path=mini_repo,
        generator=fake_generator,
        namespace=str(emb_base),
    )
    assert r1.items_total > 0

    # Modify only security.py
    sec_file = mini_repo / "auth" / "security.py"
    sec_text = sec_file.read_text(encoding="utf-8")
    sec_file.write_text(
        sec_text + "\ndef new_auth_helper():\n    '''Helper.'''\n    return True\n",
        encoding="utf-8",
    )

    # Re-index metadata store
    build_index(mini_repo)

    # Re-run embedding build
    r2 = build_embeddings(
        repo_path=mini_repo,
        generator=fake_generator,
        namespace=str(emb_base),
    )
    # security.py had items changed/added, notifier.py was untouched
    assert r2.items_changed > 0 or r2.items_new > 0
    assert r2.items_unchanged > 0


def test_build_embeddings_after_deleting_file(
    mini_repo: Path, fake_generator: FakeEmbeddingGenerator, tmp_path: Path
) -> None:
    """Deleting a file removes its items from the vector index."""
    emb_base = tmp_path / "emb_storage"
    build_index(mini_repo)

    r1 = build_embeddings(
        repo_path=mini_repo,
        generator=fake_generator,
        namespace=str(emb_base),
    )
    assert r1.items_total > 0

    # Remove notifier.py
    notifier_file = mini_repo / "mail" / "notifier.py"
    notifier_file.unlink()

    # Re-index metadata store
    build_index(mini_repo)

    # Re-run embedding build
    r2 = build_embeddings(
        repo_path=mini_repo,
        generator=fake_generator,
        namespace=str(emb_base),
    )
    assert r2.items_removed > 0
    assert r2.items_total < r1.items_total


def test_semantic_search_deduplication(mini_repo: Path, fake_generator: FakeEmbeddingGenerator, tmp_path: Path) -> None:
    """Ensure search deduplicates multiple vectors for the same function into a single hit."""
    emb_base = tmp_path / "emb_storage"
    build_index(mini_repo)

    build_embeddings(
        repo_path=mini_repo,
        generator=fake_generator,
        namespace=str(emb_base),
    )

    with MetadataStore.open(mini_repo) as store:
        vector_store = VectorStore(
            dimension=fake_generator.dimension,
            model_name=fake_generator.model_name,
            repo_path=mini_repo,
            namespace=str(emb_base),
        )
        vector_store.load()

        searcher = SemanticSearch(
            vector_store=vector_store,
            metadata_store=store,
            generator=fake_generator,
        )

        hits = searcher.search(query="verify password hash", top_k=10)
        assert len(hits) > 0

        # Check unique refs
        refs = [h.ref for h in hits]
        assert len(refs) == len(set(refs)), "Hits must be deduplicated by ref"

        # Find verify_password hit if present
        v_hit = next((h for h in hits if "verify_password" in h.ref), None)
        if v_hit:
            assert isinstance(v_hit.matched_kinds, list)
            assert v_hit.entity_type in ("function", "method")
            assert v_hit.start_line is not None
            assert v_hit.end_line is not None


def test_empty_repository_and_no_index_errors(tmp_path: Path, fake_generator: FakeEmbeddingGenerator) -> None:
    """Unindexed repositories or empty stores must raise clear errors."""
    unindexed = tmp_path / "unindexed_repo"
    unindexed.mkdir()

    with pytest.raises(RuntimeError, match="Repository has not been indexed yet"):
        build_embeddings(repo_path=unindexed, generator=fake_generator)

    # Indexed but empty repository
    build_index(unindexed)
    with pytest.raises(RuntimeError, match="metadata store contains 0 files"):
        build_embeddings(repo_path=unindexed, generator=fake_generator)


# ============================================================================
# Integration Tests (Real Neural Model, marked with @pytest.mark.integration)
# ============================================================================


@pytest.mark.integration
def test_real_model_password_performance(tmp_path: Path) -> None:
    """Real model test: 'improve password performance' surfaces hash_password & verify_password."""
    repo_root = tmp_path / "auth_repo"
    repo_root.mkdir()

    (repo_root / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "service.py").write_text(
        "def hash_password(password: str) -> str:\n"
        "    '''Hash a user password using bcrypt with optimal work factor.'''\n"
        "    return 'hashed'\n\n"
        "def verify_password(password: str, hashed: str) -> bool:\n"
        "    '''Verify a plaintext password against the stored bcrypt hash.'''\n"
        "    return True\n\n"
        "def create_session(user_id: str) -> str:\n"
        "    '''Create a session identifier for an active browser connection.'''\n"
        "    return 'sess-123'\n\n"
        "def get_user(user_id: str) -> dict:\n"
        "    '''Fetch user account record from database.'''\n"
        "    return {'id': user_id}\n\n"
        "def render_template(template_name: str, context: dict) -> str:\n"
        "    '''Compile and render Jinja HTML template.'''\n"
        "    return '<html></html>'\n\n"
        "def send_email(to: str, subject: str, message: str) -> bool:\n"
        "    '''Deliver an outbound SMTP email notification.'''\n"
        "    return True\n",
        encoding="utf-8",
    )

    build_index(repo_root)
    build_embeddings(repo_root)

    with MetadataStore.open(repo_root) as store:
        manifest = VectorStore.read_manifest(repo_root)
        generator = EmbeddingGenerator(model_name=manifest["model_name"])

        vs = VectorStore(
            dimension=manifest["dimension"],
            model_name=manifest["model_name"],
            repo_path=repo_root,
        )
        vs.load()

        searcher = SemanticSearch(vector_store=vs, metadata_store=store, generator=generator)
        hits = searcher.search(query="improve password performance", top_k=10)

        refs = [h.ref for h in hits]
        # hash_password or verify_password should be at top ranks
        assert any("password" in r for r in refs[:2]), f"Top 2 results should contain password functions: {refs}"
        assert refs.index("service.hash_password") < refs.index("service.render_template")


@pytest.mark.integration
def test_real_model_notification_message(tmp_path: Path) -> None:
    """Real model test: 'send a notification message' ranks send_email near the top."""
    repo_root = tmp_path / "notify_repo"
    repo_root.mkdir()

    (repo_root / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "app.py").write_text(
        "def send_email(to_address: str, subject: str, body: str) -> bool:\n"
        "    '''Transmit electronic mail to an outbound recipient via SMTP protocol.'''\n"
        "    return True\n\n"
        "def compute_fibonacci(n: int) -> int:\n"
        "    '''Compute the nth Fibonacci number recursively.'''\n"
        "    return n if n <= 1 else compute_fibonacci(n-1) + compute_fibonacci(n-2)\n\n"
        "def parse_xml_payload(raw_xml: str) -> dict:\n"
        "    '''Parse XML document into Python dictionary tree.'''\n"
        "    return {}\n",
        encoding="utf-8",
    )

    build_index(repo_root)
    build_embeddings(repo_root)

    with MetadataStore.open(repo_root) as store:
        manifest = VectorStore.read_manifest(repo_root)
        generator = EmbeddingGenerator(model_name=manifest["model_name"])

        vs = VectorStore(
            dimension=manifest["dimension"],
            model_name=manifest["model_name"],
            repo_path=repo_root,
        )
        vs.load()

        searcher = SemanticSearch(vector_store=vs, metadata_store=store, generator=generator)
        hits = searcher.search(query="send a notification message", top_k=3)

        assert len(hits) > 0
        assert hits[0].ref == "app.send_email"
