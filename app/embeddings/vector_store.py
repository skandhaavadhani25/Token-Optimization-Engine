"""FAISS vector store with parallel item metadata, persisted in MongoDB."""

from __future__ import annotations

import io
import json
import time
import zlib
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from app.models.schemas import EmbeddedItem, ScoredItem
from app.storage.mongo import delete_blobs, get_blob, get_database, put_blob, repo_key

SCHEMA_VERSION = "1"
_BLOB_NS = "embeddings"


class VectorStore:
    """FAISS-based vector index with exact cosine inner product search.

    Note on index type:
    IndexFlatIP evaluates exact inner product across all vectors. For repositories
    up to 100,000 items, Flat search is sub-millisecond, highly accurate, and
    requires no training clusters. If scaling beyond 500,000 vectors, migrating
    to IndexIVFFlat or IndexHNSWFlat will balance speed and recall.

    Persistence: the serialized FAISS index, raw vectors, and item metadata are
    stored as chunked blobs in MongoDB; a small manifest document describes them.
    """

    def __init__(
        self,
        dimension: int,
        model_name: str,
        repo_path: Path | str,
        namespace: str | None = None,
    ) -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.repo_path = Path(repo_path).resolve()
        self.key = repo_key(self.repo_path, namespace)
        self.namespace = namespace

        # IndexFlatIP over L2-normalized vectors calculates cosine similarity
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.items: list[EmbeddedItem] = []
        self.vectors: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)

        self.created_at: float = time.time()
        self.last_updated_at: float = self.created_at
        self.last_indexed_at: float | None = None
        self.index_size_bytes: int = 0

    @staticmethod
    def exists(repo_path: Path | str, namespace: str | None = None) -> bool:
        """True when embeddings have been saved for this repository."""
        return VectorStore.read_manifest(repo_path, namespace) is not None

    @staticmethod
    def read_manifest(repo_path: Path | str, namespace: str | None = None) -> dict[str, Any] | None:
        """Return the stored manifest (model, dimension, item count) or None."""
        doc = get_database().embedding_manifests.find_one({"_id": repo_key(repo_path, namespace)})
        if doc is None:
            return None
        doc = dict(doc)
        doc.pop("_id", None)
        return doc

    @staticmethod
    def drop(repo_path: Path | str, namespace: str | None = None) -> None:
        """Delete all stored embedding data for a repository."""
        key = repo_key(repo_path, namespace)
        delete_blobs(f"{_BLOB_NS}:{key}")
        get_database().embedding_manifests.delete_one({"_id": key})

    @classmethod
    def load_or_create(
        cls,
        dimension: int,
        model_name: str,
        repo_path: Path | str,
        namespace: str | None = None,
    ) -> VectorStore:
        """Load an existing VectorStore or instantiate a fresh one if absent."""
        store = cls(
            dimension=dimension,
            model_name=model_name,
            repo_path=repo_path,
            namespace=namespace,
        )
        if cls.exists(repo_path, namespace):
            store.load()
        return store

    def add(self, items: Sequence[EmbeddedItem], vectors: np.ndarray) -> None:
        """Add new items and their embedding vectors into index.

        Args:
            items: EmbeddedItem metadata objects.
            vectors: Float32 array of shape (len(items), dimension).
        """
        if len(items) == 0:
            return

        if vectors.shape != (len(items), self.dimension):
            raise ValueError(
                f"Vectors shape {vectors.shape} does not match ({len(items)}, {self.dimension})"
            )

        float_vectors = vectors.astype(np.float32)
        self.index.add(float_vectors)
        self.items.extend(items)

        if self.vectors.shape[0] == 0:
            self.vectors = float_vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, float_vectors])

        self.last_updated_at = time.time()

    def remove_by_file(self, file_path: str) -> int:
        """Remove all embedded items originating from the given file path.

        Rebuilds FAISS Flat index from remaining stored raw vectors.

        Args:
            file_path: Relative POSIX file path.

        Returns:
            Number of removed items.
        """
        keep_indices = [
            i for i, item in enumerate(self.items)
            if item.file != file_path
        ]
        removed_count = len(self.items) - len(keep_indices)

        if removed_count == 0:
            return 0

        new_items = [self.items[i] for i in keep_indices]
        new_vectors = self.vectors[keep_indices] if len(keep_indices) > 0 else np.empty((0, self.dimension), dtype=np.float32)

        # Rebuild index
        new_index = faiss.IndexFlatIP(self.dimension)
        if new_vectors.shape[0] > 0:
            new_index.add(new_vectors)

        self.index = new_index
        self.items = new_items
        self.vectors = new_vectors
        self.last_updated_at = time.time()

        return removed_count

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        kinds: Sequence[str] | None = None,
        files: Sequence[str] | None = None,
    ) -> list[ScoredItem]:
        """Search the index for items nearest to query_vector.

        Args:
            query_vector: 1D or 2D array of query embedding.
            top_k: Maximum number of ranked results to return.
            kinds: Optional filter on vector kinds.
            files: Optional filter on file paths.

        Returns:
            List of ScoredItem objects ordered descending by similarity score.
        """
        if self.index.ntotal == 0 or top_k <= 0:
            return []

        q_vec = query_vector.reshape(1, -1).astype(np.float32)

        # Over-fetch if filters are active
        fetch_k = self.index.ntotal if (kinds or files) else min(self.index.ntotal, top_k)
        if kinds or files:
            fetch_k = min(self.index.ntotal, max(top_k * 5, 50))

        distances, indices = self.index.search(q_vec, fetch_k)

        allowed_kinds = set(kinds) if kinds else None
        allowed_files = set(files) if files else None

        results: list[ScoredItem] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.items):
                continue

            item = self.items[idx]

            if allowed_kinds is not None and item.kind not in allowed_kinds:
                continue
            if allowed_files is not None and item.file not in allowed_files:
                continue

            results.append(
                ScoredItem(
                    item_id=item.item_id,
                    kind=item.kind,
                    ref=item.ref,
                    file=item.file,
                    text_hash=item.text_hash,
                    score=float(dist),
                )
            )

            if len(results) >= top_k:
                break

        return results

    def save(self) -> None:
        """Persist index, vectors, items metadata, and manifest to MongoDB."""
        ns = f"{_BLOB_NS}:{self.key}"

        index_bytes = zlib.compress(faiss.serialize_index(self.index).tobytes())
        buf = io.BytesIO()
        np.save(buf, self.vectors)
        vectors_bytes = zlib.compress(buf.getvalue())
        items_bytes = zlib.compress(
            json.dumps(
                [
                    {
                        "item_id": it.item_id,
                        "kind": it.kind,
                        "ref": it.ref,
                        "file": it.file,
                        "text_hash": it.text_hash,
                    }
                    for it in self.items
                ]
            ).encode("utf-8")
        )
        put_blob(ns, "index.faiss", index_bytes)
        put_blob(ns, "vectors.npy", vectors_bytes)
        put_blob(ns, "items.json", items_bytes)
        self.index_size_bytes = len(index_bytes) + len(vectors_bytes) + len(items_bytes)

        get_database().embedding_manifests.replace_one(
            {"_id": self.key},
            {
                "_id": self.key,
                "model_name": self.model_name,
                "dimension": self.dimension,
                "item_count": len(self.items),
                "created_at": self.created_at,
                "last_updated_at": self.last_updated_at,
                "schema_version": SCHEMA_VERSION,
                "last_indexed_at": self.last_indexed_at,
                "index_size_bytes": self.index_size_bytes,
            },
            upsert=True,
        )

    def load(self) -> None:
        """Load index, vectors, and items metadata from MongoDB.

        Raises:
            FileNotFoundError: If the manifest or index data are missing.
            ValueError: If saved model_name or dimension differs from current config.
        """
        manifest = self.read_manifest(self.repo_path, self.namespace)
        if manifest is None:
            raise FileNotFoundError(
                f"No embeddings found for {self.repo_path}. "
                f"Run 'python -m app.main embed {self.repo_path}' first."
            )

        if manifest.get("model_name") != self.model_name:
            raise ValueError(
                f"Model mismatch: index was built with model '{manifest.get('model_name')}', "
                f"but current configuration is '{self.model_name}'. "
                f"Please rebuild embeddings with 'python -m app.main embed {self.repo_path} --force'."
            )

        if manifest.get("dimension") != self.dimension:
            raise ValueError(
                f"Dimension mismatch: index has dimension {manifest.get('dimension')}, "
                f"but current configuration requires {self.dimension}. "
                f"Please rebuild embeddings with 'python -m app.main embed {self.repo_path} --force'."
            )

        self.created_at = manifest.get("created_at", time.time())
        self.last_updated_at = manifest.get("last_updated_at", time.time())
        self.last_indexed_at = manifest.get("last_indexed_at")
        self.index_size_bytes = manifest.get("index_size_bytes", 0)

        ns = f"{_BLOB_NS}:{self.key}"
        index_blob = get_blob(ns, "index.faiss")
        if index_blob is None:
            raise FileNotFoundError(f"Missing FAISS index data for {self.repo_path}")
        self.index = faiss.deserialize_index(np.frombuffer(zlib.decompress(index_blob), dtype=np.uint8))

        vec_blob = get_blob(ns, "vectors.npy")
        if vec_blob is not None:
            self.vectors = np.load(io.BytesIO(zlib.decompress(vec_blob))).astype(np.float32)
        else:
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)

        items_blob = get_blob(ns, "items.json")
        if items_blob is not None:
            self.items = [EmbeddedItem(**d) for d in json.loads(zlib.decompress(items_blob))]
        else:
            self.items = []

    def stats(self) -> dict[str, Any]:
        """Compute index metrics and stored size."""
        by_kind: dict[str, int] = {}
        for item in self.items:
            by_kind[item.kind] = by_kind.get(item.kind, 0) + 1

        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_items": len(self.items),
            "by_kind": by_kind,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "last_indexed_at": self.last_indexed_at,
            "index_size_bytes": self.index_size_bytes,
        }
