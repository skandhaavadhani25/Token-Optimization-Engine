"""Unit and integration tests for the Python AST parser."""

from __future__ import annotations

from pathlib import Path


from app.main import format_file_tree
from app.models.schemas import FileInfo, ScanResult
from app.parser.python_parser import (
    compute_module_name,
    parse_file,
    parse_repository,
)


def test_login_user_example_calls_extracted(tmp_path: Path) -> None:
    """Verify that login_user extracts get_user, verify_password, and create_session."""
    code = (
        "def login_user(username, password):\n"
        "    user = get_user(username)\n"
        "    if verify_password(password, user.password):\n"
        "        return create_session(user)\n"
    )
    test_file = tmp_path / "auth.py"
    test_file.write_text(code, encoding="utf-8")

    parsed = parse_file(test_file, "auth.py")

    assert parsed.parse_error is None
    assert len(parsed.functions) == 1
    fn = parsed.functions[0]
    assert fn.name == "login_user"
    assert fn.qualified_name == "auth.login_user"

    call_names = [c.name for c in fn.calls]
    assert call_names == ["get_user", "verify_password", "create_session"]
    assert [c.lineno for c in fn.calls] == [2, 3, 4]


def test_all_import_styles(tmp_path: Path) -> None:
    """Verify all import styles including relative imports, aliases, and wildcards."""
    code = (
        "import x\n"
        "import x as y\n"
        "from x import y\n"
        "from x import y as z\n"
        "from . import y\n"
        "from ..pkg import y\n"
        "from x import *\n"
    )
    test_file = tmp_path / "imports_test.py"
    test_file.write_text(code, encoding="utf-8")

    parsed = parse_file(test_file, "imports_test.py")

    assert parsed.parse_error is None
    assert len(parsed.imports) == 7

    imp0 = parsed.imports[0]
    assert imp0.module == "x" and imp0.name is None and imp0.alias is None and not imp0.is_from_import and imp0.level == 0

    imp1 = parsed.imports[1]
    assert imp1.module == "x" and imp1.name is None and imp1.alias == "y" and not imp1.is_from_import and imp1.level == 0

    imp2 = parsed.imports[2]
    assert imp2.module == "x" and imp2.name == "y" and imp2.alias is None and imp2.is_from_import and imp2.level == 0

    imp3 = parsed.imports[3]
    assert imp3.module == "x" and imp3.name == "y" and imp3.alias == "z" and imp3.is_from_import and imp3.level == 0

    imp4 = parsed.imports[4]
    assert imp4.module == "" and imp4.name == "y" and imp4.alias is None and imp4.is_from_import and imp4.level == 1

    imp5 = parsed.imports[5]
    assert imp5.module == "pkg" and imp5.name == "y" and imp5.alias is None and imp5.is_from_import and imp5.level == 2

    imp6 = parsed.imports[6]
    assert imp6.module == "x" and imp6.name == "*" and imp6.alias is None and imp6.is_from_import and imp6.level == 0


def test_class_with_bases_decorators_docstring_methods(tmp_path: Path) -> None:
    """Verify class with base classes, decorators, docstrings, and member methods."""
    code = (
        "@singleton\n"
        "@trace\n"
        "class UserService(BaseService, AuthMixin):\n"
        '    """User account operations."""\n'
        "\n"
        "    def __init__(self, db):\n"
        '        """Initialize service."""\n'
        "        self.db = db\n"
        "        self.init_db()\n"
        "\n"
        "    @audit\n"
        "    def login(self, username, password):\n"
        "        self.verify(username)\n"
        "        return True\n"
    )
    test_file = tmp_path / "service.py"
    test_file.write_text(code, encoding="utf-8")

    parsed = parse_file(test_file, "service.py")

    assert parsed.parse_error is None
    assert len(parsed.classes) == 1
    assert len(parsed.functions) == 0

    cls = parsed.classes[0]
    assert cls.name == "UserService"
    assert cls.qualified_name == "service.UserService"
    assert cls.bases == ["BaseService", "AuthMixin"]
    assert cls.decorators == ["singleton", "trace"]
    assert cls.docstring == "User account operations."
    assert len(cls.methods) == 2

    init_method = cls.methods[0]
    assert init_method.name == "__init__"
    assert init_method.qualified_name == "service.UserService.__init__"
    assert init_method.is_method is True
    assert init_method.parent_class == "UserService"
    assert [c.name for c in init_method.calls] == ["self.init_db"]

    login_method = cls.methods[1]
    assert login_method.name == "login"
    assert login_method.qualified_name == "service.UserService.login"
    assert login_method.is_method is True
    assert login_method.parent_class == "UserService"
    assert login_method.decorators == ["audit"]
    assert [c.name for c in login_method.calls] == ["self.verify"]


def test_async_decorated_functions_and_argument_varieties(tmp_path: Path) -> None:
    """Verify async functions, annotations, defaults, *args, and **kwargs."""
    code = (
        "@rate_limit(requests=100)\n"
        "async def fetch_records(\n"
        "    endpoint: str,\n"
        "    timeout: float = 30.0,\n"
        "    *flags: str,\n"
        "    retry_limit: int = 3,\n"
        "    **headers: any,\n"
        ") -> list[dict]:\n"
        '    """Fetch data asynchronously."""\n'
        "    res = await http.get(endpoint)\n"
        "    return res.data\n"
    )
    test_file = tmp_path / "fetcher.py"
    test_file.write_text(code, encoding="utf-8")

    parsed = parse_file(test_file, "fetcher.py")

    assert len(parsed.functions) == 1
    fn = parsed.functions[0]
    assert fn.is_async is True
    assert fn.decorators == ["rate_limit(requests=100)"]
    assert fn.return_annotation == "list[dict]"
    assert fn.docstring == "Fetch data asynchronously."

    arg_dict = {a.name: a for a in fn.arguments}
    assert arg_dict["endpoint"].kind == "positional"
    assert arg_dict["endpoint"].annotation == "str"
    assert arg_dict["endpoint"].default is None

    assert arg_dict["timeout"].kind == "positional"
    assert arg_dict["timeout"].annotation == "float"
    assert arg_dict["timeout"].default == "30.0"

    assert arg_dict["flags"].kind == "vararg"
    assert arg_dict["flags"].annotation == "str"

    assert arg_dict["retry_limit"].kind == "keyword_only"
    assert arg_dict["retry_limit"].annotation == "int"
    assert arg_dict["retry_limit"].default == "3"

    assert arg_dict["headers"].kind == "kwarg"
    assert arg_dict["headers"].annotation == "any"

    assert [c.name for c in fn.calls] == ["http.get"]


def test_nested_functions_calls_roll_up_to_parent(tmp_path: Path) -> None:
    """Verify nested function calls roll up to parent and nested functions are not listed top-level."""
    code = (
        "def outer_action(data):\n"
        "    step1_prepare(data)\n"
        "\n"
        "    def inner_subtask(sub):\n"
        "        step2_sub(sub)\n"
        "\n"
        "    inner_subtask(data)\n"
        "    step3_finalize()\n"
    )
    test_file = tmp_path / "nested.py"
    test_file.write_text(code, encoding="utf-8")

    parsed = parse_file(test_file, "nested.py")

    assert len(parsed.functions) == 1
    fn = parsed.functions[0]
    assert fn.name == "outer_action"

    call_names = [c.name for c in fn.calls]
    assert "step1_prepare" in call_names
    assert "step2_sub" in call_names
    assert "inner_subtask" in call_names
    assert "step3_finalize" in call_names


def test_method_calls_dotted_names_and_complex_expressions(tmp_path: Path) -> None:
    """Verify dotted names self.helper(), module.func(), and safe handling of complex expressions."""
    code = (
        "def execute_pipeline():\n"
        "    self.helper()\n"
        "    module.sub.func()\n"
        "    data[0].execute()\n"
        "    get_builder().chain().run()\n"
        "    (lambda x: x)()\n"
    )
    test_file = tmp_path / "pipeline.py"
    test_file.write_text(code, encoding="utf-8")

    parsed = parse_file(test_file, "pipeline.py")
    assert len(parsed.functions) == 1
    call_names = [c.name for c in parsed.functions[0].calls]

    assert "self.helper" in call_names
    assert "module.sub.func" in call_names
    assert "execute" in call_names
    assert "run" in call_names
    assert "chain" in call_names
    assert "get_builder" in call_names


def test_exact_source_text_and_line_ranges(tmp_path: Path) -> None:
    """Verify start_line/end_line slice reproduces exact source including decorators."""
    code = (
        "# Header line\n"
        "@dec1\n"
        "@dec2\n"
        "def compute(a, b):\n"
        "    # Inside function\n"
        "    return a + b\n"
        "\n"
        "class Model:\n"
        "    def run(self):\n"
        "        pass\n"
    )
    test_file = tmp_path / "source_slice.py"
    test_file.write_text(code, encoding="utf-8")

    lines = test_file.read_text(encoding="utf-8").splitlines(keepends=True)
    parsed = parse_file(test_file, "source_slice.py")

    fn = parsed.functions[0]
    expected_fn_slice = "".join(lines[fn.start_line - 1 : fn.end_line])
    assert fn.source == expected_fn_slice
    assert fn.source.startswith("@dec1")
    assert "return a + b" in fn.source

    cls = parsed.classes[0]
    expected_cls_slice = "".join(lines[cls.start_line - 1 : cls.end_line])
    assert cls.source == expected_cls_slice
    assert cls.source.startswith("class Model:")


def test_module_name_computation() -> None:
    """Verify dotted module name generation from relative file paths."""
    assert compute_module_name("auth/login.py") == "auth.login"
    assert compute_module_name("auth/__init__.py") == "auth"
    assert compute_module_name("pkg/subpkg/worker.py") == "pkg.subpkg.worker"
    assert compute_module_name("pkg/subpkg/__init__.py") == "pkg.subpkg"
    assert compute_module_name("main.py") == "main"
    assert compute_module_name("__init__.py") == "__init__"


def test_syntax_error_handled_gracefully(tmp_path: Path) -> None:
    """Verify syntax errors result in parse_error and do not stop parse_repository on other files."""
    valid_file = tmp_path / "valid.py"
    valid_file.write_text("def ok():\n    pass\n", encoding="utf-8")

    invalid_file = tmp_path / "broken.py"
    invalid_file.write_text("def broken_syntax :::: (\n", encoding="utf-8")

    scan_result = ScanResult(
        repo_root=str(tmp_path),
        files=[
            FileInfo(
                path="valid.py",
                absolute_path=str(valid_file),
                size_bytes=valid_file.stat().st_size,
                line_count=2,
                extension=".py",
                last_modified=valid_file.stat().st_mtime,
            ),
            FileInfo(
                path="broken.py",
                absolute_path=str(invalid_file),
                size_bytes=invalid_file.stat().st_size,
                line_count=1,
                extension=".py",
                last_modified=invalid_file.stat().st_mtime,
            ),
        ],
        total_files=2,
        total_size_bytes=valid_file.stat().st_size + invalid_file.stat().st_size,
        total_lines=3,
        skipped_dirs=0,
        skipped_files=0,
        scan_time_seconds=0.01,
    )

    parsed_files, summary = parse_repository(scan_result)

    assert summary.total_files == 2
    assert summary.parsed_ok == 1
    assert summary.failed == 1
    assert len(summary.errors) == 1
    assert summary.errors[0]["file"] == "broken.py"
    assert "SyntaxError" in summary.errors[0]["error"]

    broken_parsed = [f for f in parsed_files if f.file_path == "broken.py"][0]
    assert broken_parsed.parse_error is not None
    assert len(broken_parsed.functions) == 0

    valid_parsed = [f for f in parsed_files if f.file_path == "valid.py"][0]
    assert valid_parsed.parse_error is None
    assert len(valid_parsed.functions) == 1


def test_empty_and_comment_only_files(tmp_path: Path) -> None:
    """Verify empty files and comment-only files parse cleanly with empty element lists."""
    empty = tmp_path / "empty.py"
    empty.write_text("", encoding="utf-8")

    comments = tmp_path / "comments.py"
    comments.write_text("# only comments\n# line 2\n", encoding="utf-8")

    p_empty = parse_file(empty, "empty.py")
    assert p_empty.parse_error is None
    assert p_empty.line_count == 0
    assert len(p_empty.functions) == 0
    assert len(p_empty.classes) == 0
    assert len(p_empty.imports) == 0

    p_comments = parse_file(comments, "comments.py")
    assert p_comments.parse_error is None
    assert p_comments.line_count == 2
    assert len(p_comments.functions) == 0
    assert len(p_comments.classes) == 0
    assert len(p_comments.imports) == 0


def test_content_hash_stability(tmp_path: Path) -> None:
    """Verify content_hash is stable for identical content and updates on modification."""
    test_file = tmp_path / "hashed.py"
    test_file.write_text("x = 1\n", encoding="utf-8")

    p1 = parse_file(test_file, "hashed.py")
    p2 = parse_file(test_file, "hashed.py")
    assert p1.content_hash == p2.content_hash
    assert len(p1.content_hash) == 64

    test_file.write_text("x = 2\n", encoding="utf-8")
    p3 = parse_file(test_file, "hashed.py")
    assert p3.content_hash != p1.content_hash


def test_cli_tree_formatting(tmp_path: Path) -> None:
    """Verify format_file_tree outputs matching structure."""
    code = (
        "import bcrypt\n"
        "from database.user import get_user\n"
        "\n"
        "class UserService(Base):\n"
        "    def login(self, username, password):\n"
        "        user = get_user(username)\n"
        "        verify_password(password)\n"
        "\n"
        "def login_user(username, password):\n"
        "    get_user(username)\n"
    )
    test_file = tmp_path / "auth" / "login.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(code, encoding="utf-8")

    parsed = parse_file(test_file, "auth/login.py")
    tree = format_file_tree(parsed)

    assert "File: auth/login.py" in tree
    assert "imports: bcrypt, database.user.get_user" in tree
    assert "class UserService (bases: Base)" in tree
    assert "method login(self, username, password)" in tree
    assert "function login_user(username, password)" in tree
