"""Watch mode service for automatic incremental indexing on file system changes."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from app.scanner.repository_scanner import scan_repository
from app.services.incremental import IncrementalReport, update_index
from app.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class RepositoryWatcher:
    """Watches a repository for file modifications and triggers debounced incremental indexing."""

    def __init__(
        self,
        repo_path: Path | str,
        interval: float = 2.0,
        debounce_window: float = 0.5,
        verify: bool = False,
        generator: Any = None,
        on_update: Callable[[IncrementalReport], None] | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.interval = interval
        self.debounce_window = debounce_window
        self.verify = verify
        self.generator = generator
        self.on_update = on_update
        self.running = False

    def poll_once(self) -> IncrementalReport | None:
        """Perform a single check for changes. If changes are detected, run debounced update.

        Returns:
            IncrementalReport if an update was executed, None otherwise.
        """
        scan_res = scan_repository(self.repo_path)
        with MetadataStore.open(self.repo_path) as store:
            changes = store.diff_against_scan(scan_res)

        if not changes.added and not changes.modified and not changes.deleted:
            return None

        # Coalesce / debounce rapid changes (e.g. multi-file save or IDE burst)
        time.sleep(self.debounce_window)

        # Run incremental update
        report = update_index(
            repo_path=self.repo_path,
            verify=self.verify,
            generator=self.generator,
        )

        if self.on_update:
            self.on_update(report)

        return report

    def watch(self) -> None:
        """Start polling loop until interrupted."""
        self.running = True
        logger.info(f"Started watcher on {self.repo_path} (interval={self.interval}s, debounce={self.debounce_window}s)")
        try:
            while self.running:
                try:
                    report = self.poll_once()
                    if report and (report.added or report.modified or report.deleted):
                        logger.info(
                            f"[Watcher] Incremental update complete in {report.total_seconds:.2f}s: "
                            f"+{len(report.added)} ~{len(report.modified)} -{len(report.deleted)} files, "
                            f"{report.graph_nodes_changed} nodes changed, {report.items_reembedded} items re-embedded"
                        )
                except Exception as exc:
                    logger.error(f"[Watcher] Error during incremental update: {exc}")

                time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("[Watcher] Stopped by user (KeyboardInterrupt).")
        finally:
            self.running = False

    def stop(self) -> None:
        """Signal watcher to stop polling."""
        self.running = False
