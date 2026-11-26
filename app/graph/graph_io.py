"""Serialization and deserialization routines for the code context graph (stored in MongoDB)."""

from __future__ import annotations

import json
import time
import warnings
import zlib
from pathlib import Path

import networkx as nx

from app.storage.mongo import delete_blobs, get_blob, get_database, has_blob, put_blob, repo_key

_BLOB_NS = "graph"
_BLOB_KEY = "node_link"


def save_graph(
    graph: nx.MultiDiGraph,
    repo_path: Path | str,
    namespace: str | None = None,
) -> str:
    """Persist the NetworkX code graph (node-link JSON, compressed) in MongoDB.

    Returns:
        The repo key the graph was stored under.
    """
    key = repo_key(repo_path, namespace)
    payload = zlib.compress(json.dumps(nx.node_link_data(graph)).encode("utf-8"))
    put_blob(f"{_BLOB_NS}:{key}", _BLOB_KEY, payload)
    get_database().graph_meta.replace_one(
        {"_id": key},
        {"_id": key, "last_indexed_at": graph.graph.get("last_indexed_at"), "saved_at": time.time()},
        upsert=True,
    )
    return key


def graph_exists(repo_path: Path | str, namespace: str | None = None) -> bool:
    """True when a graph has been saved for this repository."""
    return has_blob(f"{_BLOB_NS}:{repo_key(repo_path, namespace)}", _BLOB_KEY)


def delete_graph(repo_path: Path | str, namespace: str | None = None) -> None:
    """Remove the stored graph for a repository."""
    key = repo_key(repo_path, namespace)
    delete_blobs(f"{_BLOB_NS}:{key}")
    get_database().graph_meta.delete_one({"_id": key})


def is_graph_stale(repo_path: Path | str, namespace: str | None = None) -> bool:
    """True when the index changed after the stored graph was built."""
    key = repo_key(repo_path, namespace)
    db = get_database()
    graph_meta = db.graph_meta.find_one({"_id": key})
    index_meta = db.index_meta.find_one({"_id": key})
    if not graph_meta or not index_meta:
        return False
    built_at = graph_meta.get("last_indexed_at")
    indexed_at = index_meta.get("last_indexed_at")
    return built_at is not None and indexed_at is not None and float(indexed_at) > float(built_at) + 0.1


def load_graph(
    repo_path: Path | str,
    namespace: str | None = None,
) -> nx.MultiDiGraph:
    """Load the NetworkX code graph from MongoDB.

    Warns if the metadata index was updated after the graph was saved.

    Raises:
        FileNotFoundError: If no graph has been stored for the repository.
    """
    key = repo_key(repo_path, namespace)
    raw = get_blob(f"{_BLOB_NS}:{key}", _BLOB_KEY)
    if raw is None:
        raise FileNotFoundError(
            f"Code graph not found for {repo_path}. "
            f"Please build the graph first with 'python -m app.main graph-build {repo_path}'."
        )

    graph = nx.node_link_graph(json.loads(zlib.decompress(raw)), directed=True, multigraph=True)

    meta = get_database().index_meta.find_one({"_id": key})
    if meta and meta.get("last_indexed_at"):
        graph_last_indexed = graph.graph.get("last_indexed_at")
        # Allow slight timing tolerance (0.1s)
        if graph_last_indexed is not None and float(meta["last_indexed_at"]) > float(graph_last_indexed) + 0.1:
            warnings.warn(
                "Code graph is stale: the index was updated after the graph was built. "
                "Run 'python -m app.main graph-build' to rebuild.",
                UserWarning,
                stacklevel=2,
            )

    return graph
