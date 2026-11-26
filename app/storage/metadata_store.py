"""MongoDB metadata storage engine for parsed repository data.

Collections (all documents carry a `repo` key so many repositories share them):
  index_meta     one document per repo: schema version, timestamps
  index_files    one document per source file: hashes, sizes, imports
  index_symbols  one document per class, function, or method (with source)
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from app.models.schemas import (
    ChangeSet,
    ClassInfo,
    FunctionInfo,
    ImportInfo,
    ParsedFile,
    ScanResult,
    StoreStats,
)
from app.storage.mongo import get_database, repo_key

SCHEMA_VERSION = "1"


class StorageError(Exception):
    """Base exception for storage errors."""


class SchemaVersionMismatchError(StorageError, ValueError):
    """Raised when the stored schema version does not match the expected version."""


def _function_doc(repo: str, fn: FunctionInfo, class_qn: str | None) -> dict[str, Any]:
    return {
        "repo": repo,
        "kind": "method" if class_qn else "function",
        "qualified_name": fn.qualified_name,
        "name": fn.name,
        "file_path": fn.file_path,
        "start_line": fn.start_line,
        "class_qn": class_qn,
        "call_names": sorted({c.name for c in fn.calls}),
        "data": fn.model_dump(),
    }


def _class_doc(repo: str, cls: ClassInfo) -> dict[str, Any]:
    return {
        "repo": repo,
        "kind": "class",
        "qualified_name": cls.qualified_name,
        "name": cls.name,
        "file_path": cls.file_path,
        "start_line": cls.start_line,
        "class_qn": None,
        "call_names": [],
        "data": cls.model_copy(update={"methods": []}).model_dump(),
    }


class MetadataStore:
    """Manages MongoDB persistence and retrieval of repository AST metadata."""

    def __init__(
        self,
        repo_path: Path | str,
        namespace: str | None = None,
    ) -> None:
        self.repo_root: Path = Path(repo_path).resolve()
        self.repo: str = repo_key(self.repo_root, namespace)
        self.db = get_database()
        self._ensure_meta()

    @classmethod
    def open(
        cls,
        repo_path: Path | str,
        namespace: str | None = None,
    ) -> MetadataStore:
        """Context manager factory to open and manage a MetadataStore instance."""
        return cls(repo_path=repo_path, namespace=namespace)

    @classmethod
    def exists(cls, repo_path: Path | str, namespace: str | None = None) -> bool:
        """True when an index has been created for this repository."""
        return get_database().index_meta.count_documents({"_id": repo_key(repo_path, namespace)}, limit=1) > 0

    @classmethod
    def drop(cls, repo_path: Path | str, namespace: str | None = None) -> None:
        """Delete every stored record for this repository."""
        db = get_database()
        repo = repo_key(repo_path, namespace)
        db.index_symbols.delete_many({"repo": repo})
        db.index_files.delete_many({"repo": repo})
        db.index_meta.delete_one({"_id": repo})

    def __enter__(self) -> MetadataStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """No-op: the Mongo client is shared and long-lived."""

    def _ensure_meta(self) -> None:
        """Create the repo's meta document, or verify its schema version."""
        meta = self.db.index_meta.find_one({"_id": self.repo})
        if meta is None:
            now = time.time()
            self.db.index_meta.update_one(
                {"_id": self.repo},
                {
                    "$setOnInsert": {
                        "schema_version": SCHEMA_VERSION,
                        "repo_root": str(self.repo_root),
                        "created_at": now,
                        "last_indexed_at": now,
                    }
                },
                upsert=True,
            )
        elif meta.get("schema_version") != SCHEMA_VERSION:
            raise SchemaVersionMismatchError(
                f"Schema version mismatch: index has version '{meta.get('schema_version', 'unknown')}', "
                f"expected '{SCHEMA_VERSION}'. Please re-index the repository with --force."
            )

    # -------------------------------------------------------------------------
    # Writing Operations
    # -------------------------------------------------------------------------

    def save_parsed_file(
        self,
        parsed: ParsedFile,
        size_bytes: int = 0,
        last_modified: float = 0.0,
    ) -> None:
        """Replace a file's record and all of its symbols."""
        now = time.time()
        file_disk_path = self.repo_root / parsed.file_path
        if size_bytes == 0 and file_disk_path.is_file():
            try:
                st = file_disk_path.stat()
                size_bytes = st.st_size
                last_modified = st.st_mtime
            except OSError:
                pass

        self._remove_file_records(parsed.file_path)

        self.db.index_files.insert_one(
            {
                "repo": self.repo,
                "path": parsed.file_path,
                "module_name": parsed.module_name,
                "content_hash": parsed.content_hash,
                "size_bytes": size_bytes,
                "line_count": parsed.line_count,
                "last_modified": last_modified,
                "module_docstring": parsed.module_docstring,
                "parse_error": parsed.parse_error,
                "indexed_at": now,
                "imports": [imp.model_dump() for imp in parsed.imports],
            }
        )

        # A later definition with the same qualified name replaces an earlier one.
        symbols: dict[tuple[str, str], dict[str, Any]] = {}
        for cls in parsed.classes:
            symbols[("class", cls.qualified_name)] = _class_doc(self.repo, cls)
            for method in cls.methods:
                symbols[("fn", method.qualified_name)] = _function_doc(self.repo, method, cls.qualified_name)
        for fn in parsed.functions:
            symbols[("fn", fn.qualified_name)] = _function_doc(self.repo, fn, None)
        if symbols:
            self.db.index_symbols.insert_many(list(symbols.values()))

        self.db.index_meta.update_one({"_id": self.repo}, {"$set": {"last_indexed_at": now}})

    def _remove_file_records(self, path: str) -> None:
        self.db.index_files.delete_many({"repo": self.repo, "path": path})
        self.db.index_symbols.delete_many({"repo": self.repo, "file_path": path})

    def delete_file(self, path: str) -> None:
        """Delete a file record and its symbols from the index."""
        self._remove_file_records(path)

    def clear(self) -> None:
        """Remove all files and symbols but keep the repo's meta document."""
        self.db.index_symbols.delete_many({"repo": self.repo})
        self.db.index_files.delete_many({"repo": self.repo})

    def diff_against_scan(self, scan_result: ScanResult) -> ChangeSet:
        """Compare current scan result against stored index to detect changes."""
        cursor = self.db.index_files.find(
            {"repo": self.repo}, {"path": 1, "content_hash": 1, "size_bytes": 1, "last_modified": 1}
        )
        db_records: dict[str, tuple[str, int, float]] = {
            d["path"]: (d["content_hash"], d["size_bytes"], d["last_modified"]) for d in cursor
        }

        db_paths = set(db_records.keys())
        scan_map = {f.path: f for f in scan_result.files}
        disk_paths = set(scan_map.keys())

        added = sorted(disk_paths - db_paths)
        deleted = sorted(db_paths - disk_paths)
        common = sorted(disk_paths & db_paths)

        modified: list[str] = []
        unchanged: list[str] = []

        for p in common:
            db_hash, db_size, db_mtime = db_records[p]
            f_info = scan_map[p]

            # Fast pre-check: if size and mtime are identical, content is unchanged
            if f_info.size_bytes == db_size and abs(f_info.last_modified - db_mtime) < 1e-4:
                unchanged.append(p)
            else:
                # Content changed or timestamp updated; compare content hash of bytes
                try:
                    current_hash = hashlib.sha256(Path(f_info.absolute_path).read_bytes()).hexdigest()
                    (unchanged if current_hash == db_hash else modified).append(p)
                except OSError:
                    modified.append(p)

        return ChangeSet(added=added, modified=modified, deleted=deleted, unchanged=unchanged)

    # -------------------------------------------------------------------------
    # Reading Operations
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_function(doc: dict[str, Any]) -> FunctionInfo:
        data = dict(doc["data"])
        data["calls"] = sorted(data.get("calls", []), key=lambda c: c["lineno"])
        return FunctionInfo(**data)

    def _methods_of(self, class_qn: str) -> list[FunctionInfo]:
        cursor = self.db.index_symbols.find(
            {"repo": self.repo, "kind": "method", "class_qn": class_qn}
        ).sort("start_line", 1)
        return [self._to_function(d) for d in cursor]

    def _to_class(self, doc: dict[str, Any]) -> ClassInfo:
        data = dict(doc["data"])
        data["methods"] = self._methods_of(doc["qualified_name"])
        return ClassInfo(**data)

    def load_parsed_file(self, path: str) -> ParsedFile:
        """Reconstruct and return the complete ParsedFile model for a given path."""
        file_doc = self.db.index_files.find_one({"repo": self.repo, "path": path})
        if not file_doc:
            raise FileNotFoundError(f"File not found in index: {path}")

        symbols = list(self.db.index_symbols.find({"repo": self.repo, "file_path": path}).sort("start_line", 1))
        methods_by_class: dict[str, list[FunctionInfo]] = {}
        for d in symbols:
            if d["kind"] == "method":
                methods_by_class.setdefault(d["class_qn"], []).append(self._to_function(d))

        classes = [
            ClassInfo(**{**d["data"], "methods": methods_by_class.get(d["qualified_name"], [])})
            for d in symbols
            if d["kind"] == "class"
        ]
        functions = [self._to_function(d) for d in symbols if d["kind"] == "function"]
        imports = sorted((ImportInfo(**i) for i in file_doc.get("imports", [])), key=lambda i: i.lineno)

        return ParsedFile(
            file_path=file_doc["path"],
            module_name=file_doc["module_name"],
            imports=imports,
            classes=classes,
            functions=functions,
            module_docstring=file_doc.get("module_docstring"),
            line_count=file_doc["line_count"],
            content_hash=file_doc["content_hash"],
            parse_error=file_doc.get("parse_error"),
        )

    def load_all(self) -> list[ParsedFile]:
        """Load and reconstruct all ParsedFile instances stored for this repo."""
        return [self.load_parsed_file(p) for p in self.get_all_file_paths()]

    def get_all_files(self) -> list[ParsedFile]:
        """Alias for load_all."""
        return self.load_all()

    def get_all_file_paths(self) -> list[str]:
        """Return all indexed file paths in deterministic sorted order."""
        return [d["path"] for d in self.db.index_files.find({"repo": self.repo}, {"path": 1}).sort("path", 1)]

    def get_function(self, qualified_name: str) -> FunctionInfo | None:
        """Lookup a function or method by exact qualified name."""
        doc = self.db.index_symbols.find_one(
            {"repo": self.repo, "kind": {"$in": ["function", "method"]}, "qualified_name": qualified_name}
        )
        return self._to_function(doc) if doc else None

    def find_functions_by_name(self, name: str, exact: bool = True) -> list[FunctionInfo]:
        """Search functions and methods by short name (substring match, case-insensitive, when not exact)."""
        name_query: Any = name if exact else {"$regex": re.escape(name), "$options": "i"}
        cursor = self.db.index_symbols.find(
            {"repo": self.repo, "kind": {"$in": ["function", "method"]}, "name": name_query}
        ).sort("qualified_name", 1)
        return [self._to_function(d) for d in cursor]

    def get_class(self, qualified_name: str) -> ClassInfo | None:
        """Lookup a class by exact qualified name."""
        doc = self.db.index_symbols.find_one(
            {"repo": self.repo, "kind": "class", "qualified_name": qualified_name}
        )
        return self._to_class(doc) if doc else None

    def find_classes_by_name(self, name: str, exact: bool = True) -> list[ClassInfo]:
        """Search classes by short name (case-insensitive substring match when not exact)."""
        name_query: Any = name if exact else {"$regex": re.escape(name), "$options": "i"}
        cursor = self.db.index_symbols.find({"repo": self.repo, "kind": "class", "name": name_query}).sort(
            "qualified_name", 1
        )
        return [self._to_class(d) for d in cursor]

    def find_symbol_span(self, path: str, symbol: str) -> tuple[int, int] | None:
        """Line span of the tightest function in a file matching a qualified name, short name, or name suffix."""
        spans = []
        cursor = self.db.index_symbols.find(
            {"repo": self.repo, "file_path": path, "kind": {"$in": ["function", "method"]}}
        )
        for d in cursor:
            qn = d["qualified_name"]
            if symbol in (qn, d["name"]) or qn.endswith(symbol):
                spans.append((d["data"]["end_line"] - d["data"]["start_line"], d["data"]["start_line"], d["data"]["end_line"]))
        if not spans:
            return None
        _, start, end = min(spans)
        return start, end

    def find_imports_by_file(self, path: str) -> list[ImportInfo]:
        """Find all imports defined within a specific file."""
        doc = self.db.index_files.find_one({"repo": self.repo, "path": path}, {"imports": 1})
        if not doc:
            return []
        return sorted((ImportInfo(**i) for i in doc.get("imports", [])), key=lambda i: i.lineno)

    def find_call_sites(self, call_name: str) -> list[tuple[str, int]]:
        """Find callers and invocation line numbers for a given call name."""
        sites: list[tuple[str, int]] = []
        cursor = self.db.index_symbols.find(
            {"repo": self.repo, "kind": {"$in": ["function", "method"]}, "call_names": call_name}
        )
        for d in cursor:
            for call in d["data"]["calls"]:
                if call["name"] == call_name:
                    sites.append((d["qualified_name"], call["lineno"]))
        return sorted(sites)

    def get_stats(self) -> StoreStats:
        """Calculate and return repository metadata store aggregate metrics."""
        sym = self.db.index_symbols
        base = {"repo": self.repo}
        total_calls = 0
        total_imports = 0
        for d in sym.find({**base, "kind": {"$in": ["function", "method"]}}, {"data.calls": 1}):
            total_calls += len(d["data"]["calls"])
        for d in self.db.index_files.find(base, {"imports": 1}):
            total_imports += len(d.get("imports", []))

        meta = self.db.index_meta.find_one({"_id": self.repo}) or {}
        return StoreStats(
            total_files=self.db.index_files.count_documents(base),
            total_functions=sym.count_documents({**base, "kind": "function"}),
            total_classes=sym.count_documents({**base, "kind": "class"}),
            total_methods=sym.count_documents({**base, "kind": "method"}),
            total_imports=total_imports,
            total_calls=total_calls,
            files_with_errors=self.db.index_files.count_documents({**base, "parse_error": {"$ne": None}}),
            db_size_bytes=0,
            last_indexed_at=meta.get("last_indexed_at"),
        )
