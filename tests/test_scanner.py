"""Unit and integration tests for repository scanner and file filtering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.main import run_scan
from app.models.schemas import ScanResult
from app.scanner.file_filter import (
    is_binary_file,
    should_include_file,
    should_skip_dir,
)
from app.scanner.repository_scanner import (
    InvalidRepositoryError,
    scan_repository,
)


def test_py_files_found_and_counted(tmp_path: Path) -> None:
    """Verify that Python files are located, counted, and have accurate metadata."""
    root_file = tmp_path / "root.py"
    root_file.write_text("print('hello')\nprint('world')\n", encoding="utf-8")

    sub_dir = tmp_path / "pkg"
    sub_dir.mkdir()
    sub_file = sub_dir / "module.py"
    sub_file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    nested_dir = sub_dir / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "deep.py"
    nested_file.write_text("# single line\n", encoding="utf-8")

    result = scan_repository(tmp_path)

    assert isinstance(result, ScanResult)
    assert result.total_files == 3
    assert result.total_lines == 2 + 2 + 1
    assert result.total_size_bytes == root_file.stat().st_size + sub_file.stat().st_size + nested_file.stat().st_size
    assert len(result.files) == 3

    paths = [f.path for f in result.files]
    assert paths == ["pkg/module.py", "pkg/nested/deep.py", "root.py"]


def test_ignored_directories_excluded(tmp_path: Path) -> None:
    """Verify ignored directories such as venv, .git, node_modules are pruned and not walked."""
    ignored_names = [
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "venv",
        ".venv",
        "env",
        "node_modules",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        "build",
        "dist",
        "site-packages",
        ".eggs",
        "sample.egg-info",
    ]

    for name in ignored_names:
        dir_path = tmp_path / name
        dir_path.mkdir(parents=True, exist_ok=True)
        py_in_ignored = dir_path / "ignored.py"
        py_in_ignored.write_text("x = 1\n", encoding="utf-8")

    # Add one valid file outside ignored folders
    valid_file = tmp_path / "valid.py"
    valid_file.write_text("y = 2\n", encoding="utf-8")

    result = scan_repository(tmp_path)

    assert result.total_files == 1
    assert result.files[0].path == "valid.py"
    assert result.skipped_dirs >= len(ignored_names)
    for f in result.files:
        for ignored in ignored_names:
            assert not f.path.startswith(f"{ignored}/")


def test_non_py_files_excluded(tmp_path: Path) -> None:
    """Verify non-Python files are excluded and counted as skipped."""
    (tmp_path / "script.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# README\n", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"key": "value"}\n', encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("notes\n", encoding="utf-8")

    result = scan_repository(tmp_path)

    assert result.total_files == 1
    assert result.files[0].path == "script.py"
    assert result.skipped_files == 4


def test_binary_files_skipped(tmp_path: Path) -> None:
    """Verify binary files with null bytes are detected and skipped."""
    valid_file = tmp_path / "good.py"
    valid_file.write_text("print('clean')\n", encoding="utf-8")

    binary_file = tmp_path / "bad.py"
    binary_file.write_bytes(b"import os\x00\x01\x02\x03\x04")

    assert is_binary_file(binary_file) is True
    assert is_binary_file(valid_file) is False

    result = scan_repository(tmp_path)

    assert result.total_files == 1
    assert result.files[0].path == "good.py"
    assert result.skipped_files == 1


def test_oversized_files_skipped(tmp_path: Path) -> None:
    """Verify files larger than max_file_size are skipped."""
    small = tmp_path / "small.py"
    small.write_text("a = 1\n", encoding="utf-8")

    large = tmp_path / "large.py"
    large.write_text("a = 1\n" * 1000, encoding="utf-8")

    small_size = small.stat().st_size
    large_size = large.stat().st_size
    assert large_size > small_size

    result = scan_repository(tmp_path, max_file_size=small_size + 10)

    assert result.total_files == 1
    assert result.files[0].path == "small.py"
    assert result.skipped_files == 1


def test_paths_are_relative_and_posix_style(tmp_path: Path) -> None:
    """Verify file paths are formatted as POSIX paths relative to repo root."""
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    f = nested / "nested_module.py"
    f.write_text("pass\n", encoding="utf-8")

    result = scan_repository(tmp_path)

    assert len(result.files) == 1
    info = result.files[0]
    assert info.path == "a/b/c/nested_module.py"
    assert "\\" not in info.path
    assert info.extension == ".py"
    assert Path(info.absolute_path).is_absolute()
    assert info.line_count == 1


def test_invalid_repo_path_raises_clear_error(tmp_path: Path) -> None:
    """Verify non-existent paths and non-directory paths raise clear errors."""
    non_existent = tmp_path / "missing_dir"
    with pytest.raises((ValueError, FileNotFoundError, InvalidRepositoryError)):
        scan_repository(non_existent)

    regular_file = tmp_path / "some_file.txt"
    regular_file.write_text("not a dir", encoding="utf-8")
    with pytest.raises((ValueError, NotADirectoryError, InvalidRepositoryError)):
        scan_repository(regular_file)


def test_gitignore_patterns_respected(tmp_path: Path) -> None:
    """Verify that .gitignore patterns at the repo root are properly respected."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(
        "# Comments and blank lines\n\n"
        "ignored_folder/\n"
        "secret_*.py\n"
        "temp.py\n"
        "*.generated.py\n",
        encoding="utf-8",
    )

    (tmp_path / "kept.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "secret_key.py").write_text("SECRET = 123\n", encoding="utf-8")
    (tmp_path / "temp.py").write_text("tmp = True\n", encoding="utf-8")
    (tmp_path / "schema.generated.py").write_text("code = 1\n", encoding="utf-8")

    ignored_dir = tmp_path / "ignored_folder"
    ignored_dir.mkdir()
    (ignored_dir / "mod.py").write_text("x = 1\n", encoding="utf-8")

    allowed_dir = tmp_path / "pkg"
    allowed_dir.mkdir()
    (allowed_dir / "service.py").write_text("service = True\n", encoding="utf-8")

    result = scan_repository(tmp_path)

    paths = [f.path for f in result.files]
    assert paths == ["kept.py", "pkg/service.py"]
    assert "secret_key.py" not in paths
    assert "temp.py" not in paths
    assert "schema.generated.py" not in paths
    assert "ignored_folder/mod.py" not in paths


def test_deterministic_sorting(tmp_path: Path) -> None:
    """Verify that returned file lists are always sorted deterministically by path."""
    (tmp_path / "z_module.py").write_text("z = 1\n", encoding="utf-8")
    (tmp_path / "a_module.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "m_module.py").write_text("m = 1\n", encoding="utf-8")

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b_file.py").write_text("b = 1\n", encoding="utf-8")
    (sub / "a_file.py").write_text("a = 1\n", encoding="utf-8")

    result = scan_repository(tmp_path)
    paths = [f.path for f in result.files]
    assert paths == sorted(paths)


def test_configurable_extensions(tmp_path: Path) -> None:
    """Verify that allowed file extensions can be configured beyond .py."""
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "query.sql").write_text("SELECT 1;\n", encoding="utf-8")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")

    result = scan_repository(tmp_path, extensions=(".py", ".sql"))
    paths = [f.path for f in result.files]
    assert paths == ["app.py", "query.sql"]
    assert result.total_files == 2
    assert result.skipped_files == 1


def test_encoding_error_handled_gracefully(tmp_path: Path) -> None:
    """Verify files with non-UTF8 encoding errors are skipped gracefully without crashing."""
    valid_file = tmp_path / "valid.py"
    valid_file.write_text("x = 1\n", encoding="utf-8")

    # Invalid UTF-8 sequence that does not contain null byte
    bad_encoding_file = tmp_path / "bad_encoding.py"
    bad_encoding_file.write_bytes(b"\x80\x81\x82\xff\xfe\xfd")

    result = scan_repository(tmp_path)
    assert result.total_files == 1
    assert result.files[0].path == "valid.py"
    assert result.skipped_files == 1


def test_filter_api_standalone_functions(tmp_path: Path) -> None:
    """Verify should_skip_dir and should_include_file can be called directly."""
    assert should_skip_dir(".git") is True
    assert should_skip_dir("venv") is True
    assert should_skip_dir("src") is False

    valid_py = tmp_path / "standalone.py"
    valid_py.write_text("print('test')\n", encoding="utf-8")
    txt_file = tmp_path / "standalone.txt"
    txt_file.write_text("test\n", encoding="utf-8")

    assert should_include_file(valid_py) is True
    assert should_include_file(txt_file) is False


def test_cli_human_readable_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI produces formatted human-readable summary output."""
    (tmp_path / "main.py").write_text("print('cli')\n", encoding="utf-8")

    code = run_scan(str(tmp_path), as_json=False)
    assert code == 0

    captured = capsys.readouterr()
    assert "Code Context Engine - Repository Scan" in captured.out
    assert "Total Python Files:   1" in captured.out
    assert "Top 10 Largest Files:" in captured.out
    assert "main.py" in captured.out


def test_cli_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI produces valid JSON output with --json flag."""
    (tmp_path / "sample.py").write_text("print('json')\n", encoding="utf-8")

    code = run_scan(str(tmp_path), as_json=True)
    assert code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "repo_root" in data
    assert data["total_files"] == 1
    assert len(data["files"]) == 1
    assert data["files"][0]["path"] == "sample.py"
