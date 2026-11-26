"""Embedding generator wrapping SentenceTransformers with lazy loading and batching."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

HASHING_MODEL_NAME = "hashing-bow-384"


class BaseEmbeddingGenerator(Protocol):
    """Protocol defining the interface for embedding generators."""

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    @property
    def tokenizer(self) -> Any | None:
        ...

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        ...


class EmbeddingGenerator:
    """Production embedding generator wrapping SentenceTransformer models.

    Supports lazy model loading, automatic device detection, L2 normalization,
    and configurable model selection via parameter or EMBEDDING_MODEL env var.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = (
            model_name
            or os.environ.get("EMBEDDING_MODEL")
            or self.DEFAULT_MODEL
        )
        self._device = device or self._detect_device()
        self._model: Any | None = None
        self._dimension: int | None = None
        self._fallback: HashingEmbeddingGenerator | None = None
        if importlib.util.find_spec("sentence_transformers") is None:
            # Keeps indexing and retrieval working on machines without the heavy
            # neural stack; quality is lexical, not semantic.
            logger.warning(
                "sentence-transformers is not installed; using the lexical hashing embedder. "
                "Install it for true semantic search."
            )
            self._fallback = HashingEmbeddingGenerator()
            self._model_name = self._fallback.model_name

    @staticmethod
    def _detect_device() -> str:
        """Detect best available hardware accelerator."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @property
    def model_name(self) -> str:
        """Name of the underlying sentence embedding model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Dimensionality of embedding vectors produced by this model."""
        if self._fallback is not None:
            return self._fallback.dimension
        if self._dimension is None:
            self._ensure_loaded()
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._dimension

    @property
    def tokenizer(self) -> Any | None:
        """Underlying HuggingFace tokenizer, if available."""
        if self._fallback is not None:
            return None
        self._ensure_loaded()
        return getattr(self._model, "tokenizer", None)

    def _ensure_loaded(self) -> None:
        """Lazy load the sentence transformer model on first access."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
            if self._dimension is None:
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()

            if hasattr(self._model, "max_seq_length"):
                max_len = getattr(self._model, "max_seq_length", None)
                if not max_len or max_len > 512:
                    self._model.max_seq_length = 512

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate L2-normalized float32 embedding vectors for input texts.

        Args:
            texts: List of strings to encode.
            batch_size: Inference batch size.
            show_progress: Whether to render progress bar during encoding.

        Returns:
            2D numpy array of shape (len(texts), dimension) with float32 values.
        """
        if self._fallback is not None:
            return self._fallback.encode(texts, batch_size=batch_size, show_progress=False)
        if not texts:
            dim = self.dimension
            return np.empty((0, dim), dtype=np.float32)

        self._ensure_loaded()
        # Truncate any extreme text inputs to prevent tokenizer overflow
        safe_texts = [t[:10000] if len(t) > 10000 else t for t in texts]
        embeddings = self._model.encode(
            safe_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


class FakeEmbeddingGenerator:
    """Fast, deterministic, offline embedding generator for unit tests.

    Produces normalized float32 vectors based on word-level hash projections
    without downloading or evaluating neural network weights.
    """

    def __init__(self, model_name: str = "fake-model", dimension: int = 64) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def tokenizer(self) -> Any | None:
        return None

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            if not words:
                vectors[i, 0] = 1.0
                continue
            for word in words:
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dimension
                vectors[i, idx] += 1.0

            norm = float(np.linalg.norm(vectors[i]))
            if norm > 0.0:
                vectors[i] /= norm
            else:
                vectors[i, 0] = 1.0

        return vectors.astype(np.float32)


_WORD_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


class HashingEmbeddingGenerator(FakeEmbeddingGenerator):
    """Offline lexical embedder: hashed bag of identifier sub-words.

    Splits snake_case and camelCase so `login_user` and "log in user" overlap.
    Used when sentence-transformers is unavailable.
    """

    def __init__(self, model_name: str = HASHING_MODEL_NAME, dimension: int = 384) -> None:
        super().__init__(model_name=model_name, dimension=dimension)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in _WORD_SPLIT.findall(text[:10000]):
                h = int(hashlib.md5(word.lower().encode("utf-8")).hexdigest(), 16)
                vectors[i, h % self._dimension] += 1.0
            norm = float(np.linalg.norm(vectors[i]))
            if norm > 0.0:
                vectors[i] /= norm
            else:
                vectors[i, 0] = 1.0
        return vectors
