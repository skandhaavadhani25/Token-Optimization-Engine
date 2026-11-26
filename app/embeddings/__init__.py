"""Semantic embeddings and vector search package for code-context-engine."""

from app.embeddings.embedding_generator import (
    BaseEmbeddingGenerator,
    EmbeddingGenerator,
    FakeEmbeddingGenerator,
)
from app.embeddings.indexer import build_embeddings
from app.embeddings.semantic_search import SemanticSearch
from app.embeddings.text_builder import (
    build_class_desc,
    build_file_summary,
    build_function_code,
    build_function_desc,
    build_items_for_file,
    split_identifier,
)
from app.embeddings.vector_store import VectorStore

__all__ = [
    "BaseEmbeddingGenerator",
    "EmbeddingGenerator",
    "FakeEmbeddingGenerator",
    "SemanticSearch",
    "VectorStore",
    "build_class_desc",
    "build_embeddings",
    "build_file_summary",
    "build_function_code",
    "build_function_desc",
    "build_items_for_file",
    "split_identifier",
]
