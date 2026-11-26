"""Unit and integration tests for the SQLite metadata store and index orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.indexer import build_index
from app.parser.python_parser import parse_file
from app.scanner.repository_scanner import scan_repository
from app.storage.mongo import get_database, repo_key
from app.storage.metadata_store import (
    MetadataStore,
    SchemaVersionMismatchError,
)


def test_round_trip_preserves_all_entities(tmp_path: Path) -> None:
    """Verify parse -> save -> load reproduces identical ParsedFile structure."""
    code = (
        '"""Authentication and user management module."""\n'
        "\n"
        "import os\n"
        "import sys as system\n"
        "from database.connection import get_db, close_db as shutdown\n"
        "from .local_util import helper\n"
        "\n"
        "@decorator_one\n"
        "@decorator_two(param='value')\n"
        "class UserManager(BaseManager, AuthMixin):\n"
        '    """Manages system users."""\n'
        "\n"
        "    def __init__(self, db_conn: any = None) -> None:\n"
        '        """Initialize manager."""\n'
        "        self.db = db_conn\n"
        "        self.setup()\n"
        "\n"
        "    @audit_action\n"
        "    async def create_user(self, username: str, role: str = 'user', *flags: str, **meta: any) -> bool:\n"
        "        self.validate(username)\n"
        "        return True\n"
        "\n"
        "@standalone\n"
        "def verify_token(token: str, secret: str = 'key') -> bool:\n"
        '    """Verify user token."""\n'
        "    helper(token)\n"
        "    return True\n"
    )
    src_file = tmp_path / "auth_mgr.py"
    src_file.write_text(code, encoding="utf-8")

    ns = "test"
    original = parse_file(src_file, "auth_mgr.py")

    with MetadataStore.open(tmp_path, namespace=ns) as store:
        store.save_parsed_file(original)
        loaded = store.load_parsed_file("auth_mgr.py")

    assert loaded == original
    assert loaded.module_docstring == "Authentication and user management module."
    assert len(loaded.imports) == 5
    assert len(loaded.classes) == 1
    assert len(loaded.functions) == 1

    cls = loaded.classes[0]
    assert cls.name == "UserManager"
    assert cls.bases == ["BaseManager", "AuthMixin"]
    assert cls.decorators == ["decorator_one", "decorator_two(param='value')"]
    assert len(cls.methods) == 2

    m0 = cls.methods[0]
    assert m0.name == "__init__"
    assert [c.name for c in m0.calls] == ["self.setup"]

    m1 = cls.methods[1]
    assert m1.name == "create_user"
    assert m1.is_async is True
    assert [c.name for c in m1.calls] == ["self.validate"]

    fn = loaded.functions[0]
    assert fn.name == "verify_token"
    assert [c.name for c in fn.calls] == ["helper"]


def test_reopening_db_returns_same_data_without_parsing(tmp_path: Path) -> None:
    """Verify that reopening a DB in a fresh instance returns identical data."""
    src_file = tmp_path / "calc.py"
    src_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")

    ns = "persist"
    original = parse_file(src_file, "calc.py")

    with MetadataStore.open(tmp_path, namespace=ns) as store1:
        store1.save_parsed_file(original)

    # Re-open in a new store instance
    with MetadataStore.open(tmp_path, namespace=ns) as store2:
        loaded = store2.load_parsed_file("calc.py")

    assert loaded == original


def test_upsert_does_not_duplicate_rows(tmp_path: Path) -> None:
    """Verify that saving the same file multiple times does not duplicate records."""
    code = (
        "import os\n"
        "class A:\n"
        "    def m(self): pass\n"
        "def f(): pass\n"
    )
    src_file = tmp_path / "dup.py"
    src_file.write_text(code, encoding="utf-8")

    ns = "dup"
    parsed = parse_file(src_file, "dup.py")

    with MetadataStore.open(tmp_path, namespace=ns) as store:
        store.save_parsed_file(parsed)
        stats1 = store.get_stats()

        # Save again
        store.save_parsed_file(parsed)
        stats2 = store.get_stats()

    assert stats1.total_files == 1
    assert stats2.total_files == 1
    assert stats2.total_functions == stats1.total_functions
    assert stats2.total_classes == stats1.total_classes
    assert stats2.total_methods == stats1.total_methods
    assert stats2.total_imports == stats1.total_imports


def test_delete_file_cascades_to_all_children(tmp_path: Path) -> None:
    """Verify delete_file cascades to imports, classes, functions, and calls."""
    code = (
        "import sys\n"
        "class Service:\n"
        "    def execute(self):\n"
        "        sys.exit(0)\n"
    )
    src_file = tmp_path / "casc.py"
    src_file.write_text(code, encoding="utf-8")

    ns = "casc"
    parsed = parse_file(src_file, "casc.py")

    with MetadataStore.open(tmp_path, namespace=ns) as store:
        store.save_parsed_file(parsed)
        stats_before = store.get_stats()
        assert stats_before.total_files == 1
        assert stats_before.total_calls >= 1

        store.delete_file("casc.py")
        stats_after = store.get_stats()

    assert stats_after.total_files == 0
    assert stats_after.total_classes == 0
    assert stats_after.total_methods == 0
    assert stats_after.total_imports == 0
    assert stats_after.total_calls == 0


def test_diff_against_scan(tmp_path: Path) -> None:
    """Verify diff_against_scan accurately categorizes added, modified, deleted, unchanged."""
    f_a = tmp_path / "a.py"
    f_a.write_text("x = 1\n", encoding="utf-8")
    f_b = tmp_path / "b.py"
    f_b.write_text("y = 2\n", encoding="utf-8")

    ns = "diff"

    # Initial indexing of a.py and b.py
    build_index(tmp_path, namespace=ns)

    # 1. Add c.py
    f_c = tmp_path / "c.py"
    f_c.write_text("z = 3\n", encoding="utf-8")

    # 2. Modify a.py
    f_a.write_text("x = 9999\n", encoding="utf-8")

    # 3. Delete b.py
    f_b.unlink()

    scan_res = scan_repository(tmp_path)
    with MetadataStore.open(tmp_path, namespace=ns) as store:
        changes = store.diff_against_scan(scan_res)

    assert changes.added == ["c.py"]
    assert changes.modified == ["a.py"]
    assert changes.deleted == ["b.py"]
    assert changes.unchanged == []


def test_build_index_twice_no_changes(tmp_path: Path) -> None:
    """Verify running build_index twice with no changes parses 0 files on second run."""
    (tmp_path / "mod1.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "mod2.py").write_text("b = 2\n", encoding="utf-8")
    ns = "incremental"

    rep1 = build_index(tmp_path, namespace=ns)
    assert rep1.parsed == 2
    assert rep1.unchanged == 0

    rep2 = build_index(tmp_path, namespace=ns)
    assert rep2.parsed == 0
    assert rep2.unchanged == 2
    assert rep2.added == 0
    assert rep2.modified == 0
    assert rep2.deleted == 0


def test_build_index_after_editing_one_file(tmp_path: Path) -> None:
    """Verify editing one file causes exactly that 1 file to be parsed."""
    f1 = tmp_path / "file1.py"
    f1.write_text("def f1(): pass\n", encoding="utf-8")
    f2 = tmp_path / "file2.py"
    f2.write_text("def f2(): pass\n", encoding="utf-8")
    ns = "edit"

    build_index(tmp_path, namespace=ns)

    # Edit only file1
    f1.write_text("def f1():\n    # new comment\n    pass\n", encoding="utf-8")

    rep = build_index(tmp_path, namespace=ns)
    assert rep.parsed == 1
    assert rep.modified == 1
    assert rep.unchanged == 1


def test_build_index_after_deleting_file(tmp_path: Path) -> None:
    """Verify deleting a file removes its records and parses 0 files."""
    f1 = tmp_path / "f1.py"
    f1.write_text("x = 1\n", encoding="utf-8")
    f2 = tmp_path / "f2.py"
    f2.write_text("x = 2\n", encoding="utf-8")
    ns = "del"

    build_index(tmp_path, namespace=ns)

    f2.unlink()
    rep = build_index(tmp_path, namespace=ns)

    assert rep.deleted == 1
    assert rep.parsed == 0
    assert rep.unchanged == 1

    with MetadataStore.open(tmp_path, namespace=ns) as store:
        with pytest.raises(FileNotFoundError):
            store.load_parsed_file("f2.py")
        assert store.load_parsed_file("f1.py") is not None


def test_find_functions_by_name(tmp_path: Path) -> None:
    """Verify exact and partial name matching for functions."""
    code = (
        "def compute_total(items): pass\n"
        "def compute_tax(subtotal): pass\n"
        "def print_receipt(): pass\n"
    )
    src_file = tmp_path / "billing.py"
    src_file.write_text(code, encoding="utf-8")
    ns = "find"

    parsed = parse_file(src_file, "billing.py")
    with MetadataStore.open(tmp_path, namespace=ns) as store:
        store.save_parsed_file(parsed)

        exact_res = store.find_functions_by_name("compute_total", exact=True)
        assert len(exact_res) == 1
        assert exact_res[0].name == "compute_total"

        no_exact = store.find_functions_by_name("compute", exact=True)
        assert len(no_exact) == 0

        partial_res = store.find_functions_by_name("compute", exact=False)
        assert len(partial_res) == 2
        names = {f.name for f in partial_res}
        assert names == {"compute_total", "compute_tax"}


def test_find_call_sites(tmp_path: Path) -> None:
    """Verify find_call_sites locates all call occurrences and line numbers."""
    code = (
        "def service_a():\n"
        "    audit_log('a')\n"
        "\n"
        "class Worker:\n"
        "    def run(self):\n"
        "        audit_log('worker')\n"
    )
    src_file = tmp_path / "callers.py"
    src_file.write_text(code, encoding="utf-8")
    ns = "calls"

    parsed = parse_file(src_file, "callers.py")
    with MetadataStore.open(tmp_path, namespace=ns) as store:
        store.save_parsed_file(parsed)
        call_sites = store.find_call_sites("audit_log")

    assert len(call_sites) == 2
    qnames = [site[0] for site in call_sites]
    assert "callers.service_a" in qnames
    assert "callers.Worker.run" in qnames


def test_schema_version_mismatch_raises_clear_error(tmp_path: Path) -> None:
    """Verify opening a database with mismatched schema version raises SchemaVersionMismatchError."""
    ns = "mismatch"
    get_database().index_meta.insert_one({"_id": repo_key(tmp_path, ns), "schema_version": "999"})

    with pytest.raises(SchemaVersionMismatchError) as exc_info:
        MetadataStore(tmp_path, namespace=ns)

    assert "Schema version mismatch" in str(exc_info.value)
    assert "Please re-index" in str(exc_info.value)


def test_files_with_parse_error_stored_and_counted(tmp_path: Path) -> None:
    """Verify files with parse_error are recorded and represented in stats without breaking loading."""
    broken_file = tmp_path / "broken.py"
    broken_file.write_text("def broken_syntax :::: (\n", encoding="utf-8")
    ns = "broken"

    parsed = parse_file(broken_file, "broken.py")
    assert parsed.parse_error is not None

    with MetadataStore.open(tmp_path, namespace=ns) as store:
        store.save_parsed_file(parsed)
        stats = store.get_stats()
        assert stats.files_with_errors == 1
        assert stats.total_files == 1

        loaded = store.load_parsed_file("broken.py")
        assert loaded.parse_error is not None
        assert len(loaded.functions) == 0


def test_special_characters_unicode_and_large_sources(tmp_path: Path) -> None:
    """Verify unicode, special characters, and long strings survive database round trip."""
    long_comment = "# " + ("x" * 20000) + "\n"
    code = (
        '"""Unicode module: 🚀 中文 Čeština Café 🐍"""\n'
        + long_comment
        + 'def rocket_launch(emoji: str = "🚀") -> str:\n'
        '    """Returns 🌟"""\n'
        '    return f"blastoff {emoji}"\n'
    )
    src_file = tmp_path / "unicode_test.py"
    src_file.write_text(code, encoding="utf-8")
    ns = "unicode"

    parsed = parse_file(src_file, "unicode_test.py")
    with MetadataStore.open(tmp_path, namespace=ns) as store:
        store.save_parsed_file(parsed)
        loaded = store.load_parsed_file("unicode_test.py")

    assert loaded == parsed
    assert "🚀" in loaded.module_docstring
    assert "blastoff" in loaded.functions[0].source
