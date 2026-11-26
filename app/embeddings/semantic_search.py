"""Semantic search interface querying vector store and deduplicating by entity."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from app.embeddings.embedding_generator import BaseEmbeddingGenerator, EmbeddingGenerator
from app.embeddings.vector_store import VectorStore
from app.models.schemas import ScoredItem, SemanticHit
from app.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

DEFAULT_KIND_WEIGHTS: dict[str, float] = {
    "function_desc": 1.0,
    "function_code": 0.9,
    "class_desc": 0.9,
    "file_summary": 0.7,
}


class SemanticSearch:
    """Performs semantic vector search over code entities with entity deduplication."""

    def __init__(
        self,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        generator: BaseEmbeddingGenerator | None = None,
        kind_weights: dict[str, float] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.generator = generator or EmbeddingGenerator(model_name=vector_store.model_name)
        self.kind_weights = dict(DEFAULT_KIND_WEIGHTS)
        if kind_weights:
            self.kind_weights.update(kind_weights)

    def search(
        self,
        query: str,
        top_k: int = 10,
        kinds: Sequence[str] | None = None,
        files: Sequence[str] | None = None,
    ) -> list[SemanticHit]:
        """Perform semantic search, weighting by kind and deduplicating by entity.

        Args:
            query: Natural language or code search query string.
            top_k: Number of deduplicated entities to return.
            kinds: Optional filter on vector kinds.
            files: Optional filter on file paths.

        Returns:
            List of SemanticHit instances sorted descending by score.
        """
        if not query.strip():
            return []

        # Encode query
        query_vec = self.generator.encode([query], show_progress=False)

        # Over-fetch raw hits to allow thorough deduplication
        raw_k = max(top_k * 5, 50)
        scored_items: list[ScoredItem] = self.vector_store.search(
            query_vector=query_vec,
            top_k=raw_k,
            kinds=kinds,
            files=files,
        )

        # Group by entity reference: ref -> dict of info
        entity_groups: dict[str, dict[str, Any]] = {}

        for item in scored_items:
            ref = item.ref
            raw_score = item.score
            weight = self.kind_weights.get(item.kind, 1.0)
            weighted_score = raw_score * weight

            if ref not in entity_groups:
                entity_groups[ref] = {
                    "ref": ref,
                    "file": item.file,
                    "best_score": weighted_score,
                    "matched_kinds": [item.kind],
                    "kind_scores": {item.kind: round(raw_score, 4)},
                }
            else:
                group = entity_groups[ref]
                group["matched_kinds"].append(item.kind)
                group["kind_scores"][item.kind] = round(raw_score, 4)
                if weighted_score > group["best_score"]:
                    group["best_score"] = weighted_score

        # Sort entities by best weighted score descending
        sorted_groups = sorted(
            entity_groups.values(),
            key=lambda g: (-g["best_score"], g["ref"]),
        )[:top_k]

        hits: list[SemanticHit] = []
        for g in sorted_groups:
            ref = g["ref"]
            file_path = g["file"]

            # Resolve entity metadata and line numbers from metadata store
            entity_type = "file"
            start_line: int | None = None
            end_line: int | None = None

            fn = self.metadata_store.get_function(ref)
            if fn:
                entity_type = "method" if fn.is_method else "function"
                start_line = fn.start_line
                end_line = fn.end_line
                file_path = fn.file_path
            else:
                cls = self.metadata_store.get_class(ref)
                if cls:
                    entity_type = "class"
                    start_line = cls.start_line
                    end_line = cls.end_line
                    file_path = cls.file_path
                else:
                    entity_type = "file"
                    start_line = 1

            hits.append(
                SemanticHit(
                    ref=ref,
                    entity_type=entity_type,
                    file=file_path,
                    score=round(float(g["best_score"]), 4),
                    matched_kinds=sorted(list(set(g["matched_kinds"]))),
                    kind_scores=g["kind_scores"],
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        return hits

def load_semantic_search(repo_path: Path | str, metadata_store: MetadataStore) -> SemanticSearch | None:
    """Load the stored vector index for a repository, or None when absent or unreadable."""
    manifest = VectorStore.read_manifest(repo_path)
    if manifest is None:
        return None
    try:
        vstore = VectorStore(
            dimension=manifest.get("dimension", 384),
            model_name=manifest.get("model_name", EmbeddingGenerator.DEFAULT_MODEL),
            repo_path=repo_path,
        )
        vstore.load()
        return SemanticSearch(vector_store=vstore, metadata_store=metadata_store)
    except Exception as exc:
        logger.debug(f"Could not load semantic search for {repo_path}: {exc}")
        return None
