"""Repository index orchestrator for building and maintaining the metadata store."""

from __future__ import annotations

import time
from pathlib import Path

from app.models.schemas import IndexReport
from app.parser.python_parser import parse_file
from app.scanner.repository_scanner import scan_repository
from app.storage.metadata_store import MetadataStore


def build_index(
    repo_path: Path | str,
    force: bool = False,
    namespace: str | None = None,
) -> IndexReport:
    """Scan and index repository into the MongoDB metadata store with incremental diffing.

    Args:
        repo_path: Path to root repository directory.
        force: If True, clear existing index and rebuild from scratch.
        namespace: Optional explicit index namespace (default: derived from the repo path).

    Returns:
        IndexReport containing breakdown of files indexed, modified, deleted, and timings.
    """
    total_start = time.perf_counter()
    repo = Path(repo_path).resolve()

    # 1. Scan repository files
    scan_start = time.perf_counter()
    scan_result = scan_repository(repo)
    scan_time = time.perf_counter() - scan_start

    scan_map = {f.path: f for f in scan_result.files}
    files_scanned = len(scan_result.files)

    parse_time = 0.0
    store_time = 0.0
    parse_errors = 0

    with MetadataStore.open(repo, namespace=namespace) as store:
        if force:
            # Purge existing files on force rebuild
            store.clear()

            added = files_scanned
            modified = 0
            deleted = 0
            unchanged = 0
            to_parse = [f.path for f in scan_result.files]
        else:
            changes = store.diff_against_scan(scan_result)
            added = len(changes.added)
            modified = len(changes.modified)
            deleted = len(changes.deleted)
            unchanged = len(changes.unchanged)

            # Purge removed files
            for del_path in changes.deleted:
                store.delete_file(del_path)

            to_parse = changes.added + changes.modified

        # 2. Parse added and modified files
        parsed_entries = []
        if to_parse:
            parse_start = time.perf_counter()
            for rel_p in to_parse:
                f_info = scan_map[rel_p]
                parsed = parse_file(abs_path=f_info.absolute_path, rel_path=rel_p)
                if parsed.parse_error:
                    parse_errors += 1
                parsed_entries.append((parsed, f_info))
            parse_time = time.perf_counter() - parse_start

            # 3. Store parsed records
            store_start = time.perf_counter()
            for parsed, f_info in parsed_entries:
                store.save_parsed_file(
                    parsed=parsed,
                    size_bytes=f_info.size_bytes,
                    last_modified=f_info.last_modified,
                )
            store_time = time.perf_counter() - store_start

    total_time = time.perf_counter() - total_start

    return IndexReport(
        files_scanned=files_scanned,
        added=added,
        modified=modified,
        deleted=deleted,
        unchanged=unchanged,
        parsed=len(to_parse),
        parse_errors=parse_errors,
        scan_time=round(scan_time, 4),
        parse_time=round(parse_time, 4),
        store_time=round(store_time, 4),
        total_time=round(total_time, 4),
    )
