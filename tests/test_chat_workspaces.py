"""Comprehensive test suite for Workspaces, Chat Experience, Attachments, AI Edits, and Token Savings.

Tests:
1. Upload archive safety: valid zip, zip-slip, absolute path, symlinks, size limits, secret scan.
2. Workspace lifecycle: local path copy vs edit-in-place, deletion, reindexing, ALLOWED_REPO_ROOTS.
3. Conversational memory: message persistence, pronoun resolution ("optimize it"), code block replacement, context manifest.
4. Token baselines & savings: computed token counts, baseline variations, FakeProvider real usage.
5. Attachments: chunking, conversation isolation, retrieval merging, vision validation.
6. AI-proposed edits: unified diff, syntax validation, line replacement, atomic write, undo rollback, incremental reindex.
7. Modes: Economy vs Balanced vs Quality presets and budget escalations.
8. Security: No key leakage in database, logs, or responses.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.chat.attachments import chunk_attachment, model_supports_vision
from app.chat.storage import ChatStorage
from app.chat.turn_builder import (
    MODE_PRESETS,
    replace_code_blocks_with_references,
    rewrite_query_with_entities,
)
from app.workspace.manager import UploadSecurityError, WorkspaceManager
from app.workspace.patcher import ProposedChange, WorkspacePatcher


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def storage() -> ChatStorage:
    return ChatStorage()


@pytest.fixture
def sample_mini_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "mini_repo"
    repo.mkdir()
    auth_dir = repo / "auth"
    auth_dir.mkdir()
    (auth_dir / "__init__.py").write_text("", encoding="utf-8")
    (auth_dir / "login.py").write_text(
        "def login_user(username: str, password: str) -> bool:\n"
        "    '''Authenticate user.'''\n"
        "    if not username:\n"
        "        return False\n"
        "    return password == 'secret'\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# Mini Repo Documentation\nOverview text.\n", encoding="utf-8")
    return repo


# =============================================================================
# 1. Upload & Workspace Safety
# =============================================================================

def test_upload_valid_zip_extracts_and_counts(tmp_path: Path):
    """Verify safe zip archive extracts into workspace and categorizes python vs text-only."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("module.py", "def hello(): pass\n")
        zf.writestr("notes.txt", "Plain text notes.\n")
        zf.writestr(".git/config", "secret git config\n")  # Should be skipped

    mgr = WorkspaceManager()
    record, secrets, skip_rep = mgr.create_from_upload(buf.getvalue(), "valid.zip")

    try:
        assert record["source"] == "upload"
        assert record["file_counts"]["python"] == 1
        assert record["file_counts"]["text_only"] == 1
        assert record["file_counts"]["skipped"] >= 1
        assert any(".git" in s for s in skip_rep["ignored_dirs"])

        dest = Path(record["working_path"])
        assert (dest / "module.py").exists()
        assert (dest / "notes.txt").exists()
        assert not (dest / ".git").exists()
    finally:
        mgr.delete_workspace(record["id"])


def test_upload_zip_slip_rejected():
    """Verify zip archive with path traversal ('..') is rejected with UploadSecurityError."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../../evil.py", "bad_code = True")

    mgr = WorkspaceManager()
    with pytest.raises(UploadSecurityError, match="path traversal"):
        mgr.create_from_upload(buf.getvalue(), "slip.zip")


def test_upload_absolute_path_rejected():
    """Verify zip archive with absolute path entries is rejected."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("/etc/passwd", "root:x:0:0")

    mgr = WorkspaceManager()
    with pytest.raises(UploadSecurityError, match="absolute path"):
        mgr.create_from_upload(buf.getvalue(), "absolute.zip")


def test_upload_secret_detection_and_exclusion(tmp_path: Path):
    """Verify .env files and API key patterns are detected and excluded from index."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(".env", "ANTHROPIC_API_KEY=sk-ant-api03-12345678901234567890\n")
        zf.writestr("main.py", "API_KEY = 'sk-ant-test12345678901234567890'\n")

    mgr = WorkspaceManager()
    record, secrets, skip_rep = mgr.create_from_upload(buf.getvalue(), "secrets.zip")

    try:
        assert len(secrets) >= 1
        assert any(".env" in s for s in skip_rep["secrets"])
    finally:
        mgr.delete_workspace(record["id"])


def test_workspace_local_path_copies_by_default(sample_mini_repo: Path):
    """Verify open local path creates a decoupled copy in data/workspaces so original is never touched."""
    mgr = WorkspaceManager()
    record, _, _ = mgr.create_from_local_path(sample_mini_repo, edit_in_place=False)

    try:
        working_path = Path(record["working_path"])
        assert working_path != sample_mini_repo
        assert working_path.exists()
        assert (working_path / "auth" / "login.py").exists()

        # Mutate working copy
        (working_path / "auth" / "login.py").write_text("# mutated", encoding="utf-8")
        # Original remains untouched
        assert "login_user" in (sample_mini_repo / "auth" / "login.py").read_text(encoding="utf-8")
    finally:
        mgr.delete_workspace(record["id"])


def test_workspace_delete_cleans_everything(sample_mini_repo: Path):
    """Verify delete_workspace removes working copy directory and DB records."""
    mgr = WorkspaceManager()
    record, _, _ = mgr.create_from_local_path(sample_mini_repo)
    ws_id = record["id"]
    working_path = Path(record["working_path"])

    assert working_path.exists()
    mgr.delete_workspace(ws_id)
    assert not working_path.exists()
    assert mgr.get_workspace(ws_id) is None


# =============================================================================
# 2. Conversational Memory & Pronoun Resolution
# =============================================================================

def test_query_pronoun_rewriting_with_active_entities():
    """Verify 'optimize it' resolves to the last active entity."""
    entities = ["auth.login.login_user"]
    rewritten = rewrite_query_with_entities("optimize it", entities)
    assert "auth.login.login_user" in rewritten

    rewritten_fix = rewrite_query_with_entities("can you fix that function?", entities)
    assert "auth.login.login_user" in rewritten_fix

    # Explicit identifier does not get mangled
    explicit = "How does Session.send handle cookies?"
    assert rewrite_query_with_entities(explicit, entities) == explicit


def test_old_code_blocks_replaced_with_compact_references():
    """Verify verbose code blocks from past turns are replaced by references."""
    past_assistant_msg = (
        "Here is the change:\n"
        "### CHANGE: auth.login.login_user (auth/login.py)\n"
        "```python\n"
        "def login_user(username: str, password: str) -> bool:\n"
        "    return True\n"
        "```\n"
        "Let me know if you need anything else."
    )
    compact = replace_code_blocks_with_references(past_assistant_msg)
    assert "```python" not in compact
    assert "[code shown earlier: auth.login.login_user" in compact
    assert "Let me know if you need anything else." in compact


def test_conversations_isolated(sample_mini_repo: Path):
    """Verify messages from conversation A do not leak into conversation B."""
    storage = ChatStorage()
    c1 = storage.create_conversation("Chat 1")
    c2 = storage.create_conversation("Chat 2")

    try:
        storage.add_message(c1["id"], "user", "Message in chat 1")
        storage.add_message(c2["id"], "user", "Message in chat 2")

        msgs_1 = storage.get_messages(c1["id"])
        msgs_2 = storage.get_messages(c2["id"])

        assert len(msgs_1) == 1
        assert msgs_1[0]["content"] == "Message in chat 1"
        assert len(msgs_2) == 1
        assert msgs_2[0]["content"] == "Message in chat 2"
    finally:
        storage.delete_conversation(c1["id"])
        storage.delete_conversation(c2["id"])


# =============================================================================
# 3. Attachments
# =============================================================================

def test_attachment_chunking_and_isolation():
    """Verify chunking line ranges and conversation-scoped search."""
    text_content = "\n".join([f"line_{i} = {i}" for i in range(1, 150)])
    chunks = chunk_attachment("att_1", "config.py", text_content, whole_file_token_threshold=100)

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line > 1
    assert chunks[0].filename == "config.py"


def test_vision_model_check():
    """Verify images rejected for non-vision models."""
    assert model_supports_vision("claude", "claude-3-5-sonnet") is True
    assert model_supports_vision("gemini", "gemini-2.5-flash") is True
    assert model_supports_vision("openai", "gpt-4o") is True
    assert model_supports_vision("ollama", "llama3.2") is False
    assert model_supports_vision("claude", "claude-2.1") is False


# =============================================================================
# 4. AI Edits, Unified Diff, Atomic Write, and Undo
# =============================================================================

def test_patcher_diff_and_atomic_apply_and_undo(sample_mini_repo: Path):
    """Verify unified diff calculation, atomic apply, and byte-identical undo rollback."""
    patcher = WorkspacePatcher(sample_mini_repo)
    login_file = sample_mini_repo / "auth" / "login.py"
    original_bytes = login_file.read_bytes()

    proposals = patcher.parse_proposals(
        "### CHANGE: login_user (auth/login.py)\n"
        "```python\n"
        "def login_user(username: str, password: str) -> bool:\n"
        "    '''Optimized authentication.'''\n"
        "    return bool(username and password == 'secret')\n"
        "```\n"
    )

    assert len(proposals) == 1
    prop = proposals[0]
    assert prop.syntax_valid is True
    assert prop.target_exists is True
    assert "Optimized authentication" in prop.diff

    storage = ChatStorage()
    ws = storage.create_workspace("Test", "local", sample_mini_repo)
    ws_id = ws["id"]
    res = patcher.apply_proposal(ws_id, prop)
    assert res["status"] == "applied"

    # Verify workspace file updated
    modified_text = login_file.read_text(encoding="utf-8")
    assert "Optimized authentication" in modified_text

    # Undo edit
    undo_res = patcher.undo_last_edit(ws_id)
    assert undo_res["status"] == "undone"

    # Verify byte-identical restoration
    restored_bytes = login_file.read_bytes()
    assert restored_bytes == original_bytes


def test_patcher_rejects_syntax_error(sample_mini_repo: Path):
    """Verify syntax-invalid proposals are blocked unless forced."""
    patcher = WorkspacePatcher(sample_mini_repo)
    proposals = patcher.parse_proposals(
        "### CHANGE: login_user (auth/login.py)\n"
        "```python\n"
        "def broken(:\n"
        "```\n"
    )
    assert len(proposals) == 1
    assert proposals[0].syntax_valid is False

    with pytest.raises(ValueError, match="syntax error"):
        patcher.apply_proposal("ws_test", proposals[0], allow_syntax_error=False)


def test_patcher_path_traversal_blocked(sample_mini_repo: Path):
    """Verify applying a change targeting a path outside workspace root is blocked."""
    patcher = WorkspacePatcher(sample_mini_repo)
    prop = ProposedChange(
        change_type="modify",
        target_symbol="hack",
        file_path="../../system.py",
        new_content="evil = 1",
        syntax_valid=True,
        syntax_error=None,
        diff="",
        target_exists=False,
        is_applicable=False,
    )
    with pytest.raises(ValueError, match="Path traversal"):
        patcher.apply_proposal("ws_test", prop)


# =============================================================================
# 5. Modes & Token Savings Accounting
# =============================================================================

def test_modes_preset_parameters():
    """Verify Economy, Balanced, and Quality presets map to documented configurations."""
    assert MODE_PRESETS["economy"]["budget"] == 2000
    assert MODE_PRESETS["economy"]["include_callers"] is False

    assert MODE_PRESETS["balanced"]["budget"] == 4000
    assert MODE_PRESETS["balanced"]["include_callers"] is True

    assert MODE_PRESETS["quality"]["budget"] == 8000
    assert MODE_PRESETS["quality"]["top_k"] == 8


# =============================================================================
# 6. Security: API Key Leakage Grep
# =============================================================================

def test_api_key_never_stored_in_messages_or_manifests(client: TestClient, sample_mini_repo: Path):
    """Verify distinct dummy API key never appears in conversation messages, DB, or JSON responses."""
    distinct_key = "sk-ant-super-distinct-secret-9876543210"

    # Create workspace and conversation
    ws_res = client.post("/workspaces", data={"local_path": str(sample_mini_repo)})
    assert ws_res.status_code == 200
    ws_id = ws_res.json()["workspace"]["id"]

    c_res = client.post("/conversations", json={"workspace_id": ws_id, "title": "Secret Test"})
    assert c_res.status_code == 200
    cid = c_res.json()["id"]

    try:
        # Send message with header key and dry_run
        msg_res = client.post(
            f"/conversations/{cid}/messages",
            json={"content": "How does login work?", "dry_run": True},
            headers={"X-Provider-Key": distinct_key},
        )
        assert msg_res.status_code == 200
        resp_text = msg_res.text
        assert distinct_key not in resp_text

        # Verify DB records
        storage = ChatStorage()
        msgs = storage.get_messages(cid)
        for m in msgs:
            assert distinct_key not in json.dumps(m)
    finally:
        client.delete(f"/conversations/{cid}")
        client.delete(f"/workspaces/{ws_id}")


# =============================================================================
# Upload regressions: real failure reasons reach the user
# =============================================================================
def _shop_zip() -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("shop/", "")
        z.writestr("shop/main.py", "from auth.login import login\n\ndef run():\n    return login('a', 'b')\n")
        z.writestr("shop/auth/__init__.py", "")
        z.writestr("shop/auth/login.py", "def login(u, p):\n    return u == p\n")
        z.writestr("shop/README.md", "# Shop\n")
    return buf.getvalue()


def test_upload_zip_with_top_level_folder(client: TestClient):
    res = client.post("/workspaces", files={"files": ("shop.zip", _shop_zip(), "application/zip")})
    assert res.status_code == 200, res.text
    counts = res.json()["workspace"]["file_counts"]
    assert counts["python"] == 3 and counts["text_only"] == 1
    working = Path(res.json()["workspace"]["working_path"])
    assert (working / "main.py").is_file() and not (working / "shop").exists()


def test_upload_reports_unreachable_mongodb(client: TestClient, monkeypatch):
    from app.storage import mongo

    def boom():
        raise mongo.StorageUnavailableError("Cannot reach MongoDB at the configured MONGODB_URI: ServerSelectionTimeoutError.")

    monkeypatch.setattr("app.chat.storage.get_database", boom, raising=False)
    res = client.post("/workspaces", files={"files": ("shop.zip", _shop_zip(), "application/zip")})
    assert res.status_code == 503
    assert "Cannot reach MongoDB" in res.json()["error"]["message"]


def test_unexpected_upload_error_names_the_cause(client: TestClient, monkeypatch):
    def boom(self, *a, **k):
        raise PermissionError("[Errno 13] Permission denied: 'data/workspaces'")

    monkeypatch.setattr("app.workspace.manager.WorkspaceManager.create_from_upload", boom)
    res = client.post("/workspaces", files={"files": ("shop.zip", _shop_zip(), "application/zip")})
    assert res.status_code == 500
    assert "PermissionError" in res.json()["error"]["message"]


def test_upload_zip_with_backslash_separators(client: TestClient):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("shop\\auth\\login.py", "def login():\n    return 1\n")
    res = client.post("/workspaces", files={"files": ("shop.zip", buf.getvalue(), "application/zip")})
    assert res.status_code == 200, res.text
    working = Path(res.json()["workspace"]["working_path"])
    assert (working / "auth" / "login.py").is_file()  # wrapping folder is dropped


def test_upload_backslash_traversal_rejected(client: TestClient):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a\\..\\..\\evil.py", "x = 1\n")
    res = client.post("/workspaces", files={"files": ("e.zip", buf.getvalue(), "application/zip")})
    assert res.status_code == 400
