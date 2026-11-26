"""Orchestrates building and incremental updating of semantic code embeddings."""

from __future__ import annotations

import time
from pathlib import Path

from app.embeddings.embedding_generator import BaseEmbeddingGenerator, EmbeddingGenerator
from app.embeddings.text_builder import build_items_for_file
from app.embeddings.vector_store import VectorStore
from app.models.schemas import EmbeddedItem, EmbeddingReport, ParsedFile
from app.storage.metadata_store import MetadataStore


def build_embeddings(
    repo_path: Path | str,
    force: bool = False,
    model_name: str | None = None,
    generator: BaseEmbeddingGenerator | None = None,
    show_progress: bool = True,
    max_code_tokens: int = 256,
    min_body_lines: int = 2,
    namespace: str | None = None,
) -> EmbeddingReport:
    """Build or incrementally update semantic vector index for a repository.

    Args:
        repo_path: Root path of repository.
        force: If True, re-embed all entities from scratch.
        model_name: Optional override for embedding model name.
        generator: Optional pre-configured BaseEmbeddingGenerator instance.
        show_progress: Whether to display encoding progress bar.
        max_code_tokens: Maximum token length for function_code items.
        min_body_lines: Minimum lines of body for non-trivial function code.
        namespace: Optional explicit namespace for the stored vector index (defaults to the repo key).

    Returns:
        EmbeddingReport summarizing operations and index metrics.

    Raises:
        RuntimeError: If repository has not been indexed yet.
    """
    start_time = time.time()
    repo = Path(repo_path).resolve()

    if not MetadataStore.exists(repo):
        raise RuntimeError(
            f"Repository has not been indexed yet. "
            f"Please run 'python -m app.main index {repo_path}' first."
        )

    with MetadataStore.open(repo) as store:
        stats = store.get_stats()
        if stats.total_files == 0:
            raise RuntimeError(
                f"Repository metadata store contains 0 files. "
                f"Please run 'python -m app.main index {repo_path}' first."
            )

        if generator is None:
            generator = EmbeddingGenerator(model_name=model_name)

        if force:
            vector_store = VectorStore(
                dimension=generator.dimension,
                model_name=generator.model_name,
                repo_path=repo,
                namespace=namespace,
            )
        else:
            vector_store = VectorStore.load_or_create(
                dimension=generator.dimension,
                model_name=generator.model_name,
                repo_path=repo,
                namespace=namespace,
            )

        all_parsed: list[ParsedFile] = store.load_all()
        current_file_paths = {pf.file_path for pf in all_parsed}

        items_new = 0
        items_changed = 0
        items_removed = 0
        items_unchanged = 0
        items_skipped = 0

        existing_items_by_id = {item.item_id: item for item in vector_store.items}
        existing_files = {item.file for item in vector_store.items}

        is_first_run = (len(vector_store.items) == 0)
        items_to_embed: list[EmbeddedItem] = []

        if is_first_run or force:
            for pf in all_parsed:
                items, skipped = build_items_for_file(
                    pf,
                    max_code_tokens=max_code_tokens,
                    tokenizer=generator.tokenizer,
                    min_body_lines=min_body_lines,
                )
                items_skipped += skipped
                items_to_embed.extend(items)
            items_new = len(items_to_embed)
        else:
            # Incremental update:
            # 1. Purge items for deleted files
            deleted_files = existing_files - current_file_paths
            for df in deleted_files:
                items_removed += vector_store.remove_by_file(df)

            # 2. Check each file for new or modified items
            for pf in all_parsed:
                file_items, skipped = build_items_for_file(
                    pf,
                    max_code_tokens=max_code_tokens,
                    tokenizer=generator.tokenizer,
                    min_body_lines=min_body_lines,
                )
                items_skipped += skipped

                existing_for_file = [
                    it for it in vector_store.items if it.file == pf.file_path
                ]
                file_needs_update = False

                if len(existing_for_file) != len(file_items):
                    file_needs_update = True
                else:
                    for item in file_items:
                        if item.item_id not in existing_items_by_id:
                            file_needs_update = True
                            break
                        if existing_items_by_id[item.item_id].text_hash != item.text_hash:
                            file_needs_update = True
                            break

                if file_needs_update:
                    if pf.file_path in existing_files:
                        vector_store.remove_by_file(pf.file_path)
                        items_changed += len(file_items)
                    else:
                        items_new += len(file_items)
                    items_to_embed.extend(file_items)
                else:
                    items_unchanged += len(file_items)

        # Batch encode new and changed items
        emb_start = time.time()
        if items_to_embed:
            texts = [it.text for it in items_to_embed]
            vectors = generator.encode(texts, batch_size=64, show_progress=show_progress)
            vector_store.add(items_to_embed, vectors)
        emb_time = time.time() - emb_start

        vector_store.last_indexed_at = stats.last_indexed_at
        vector_store.save()

        vs_stats = vector_store.stats()
        total_time = time.time() - start_time

        return EmbeddingReport(
            items_total=vs_stats["total_items"],
            items_new=items_new,
            items_changed=items_changed,
            items_removed=items_removed,
            items_unchanged=items_unchanged,
            items_skipped=items_skipped,
            by_kind=vs_stats["by_kind"],
            model_name=generator.model_name,
            embedding_time_seconds=emb_time,
            total_time_seconds=total_time,
            index_size_bytes=vs_stats["index_size_bytes"],
        )
