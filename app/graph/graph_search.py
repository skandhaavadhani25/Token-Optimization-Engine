"""Graph traversal and query helpers for the code context graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import networkx as nx

from app.models.schemas import CallEdgeResult

CONFIDENCE_LEVELS: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "exact": 4,
}


class CodeGraph:
    """Wrapper around NetworkX MultiDiGraph providing code context search helpers."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def find_nodes(
        self,
        name: str,
        node_type: str | None = None,
        exact: bool = True,
    ) -> list[str]:
        """Search for nodes by ID, short name, or qualified name.

        Args:
            name: Search query string.
            node_type: Optional node type filter (e.g. 'function', 'class', 'file').
            exact: True for exact name match, False for substring search.

        Returns:
            Sorted list of matching node IDs.
        """
        matches: set[str] = set()
        query = name.strip()
        query_lower = query.lower()

        # Check if query matches a node_id directly
        if query in self.graph:
            data = self.graph.nodes[query]
            if node_type is None or data.get("type") == node_type:
                matches.add(query)

        for nid, data in self.graph.nodes(data=True):
            if node_type is not None and data.get("type") != node_type:
                continue

            n_name = str(data.get("name") or "")
            q_name = str(data.get("qualified_name") or "")

            if exact:
                if query in (nid, n_name, q_name):
                    matches.add(nid)
            else:
                if (
                    query_lower in nid.lower()
                    or query_lower in n_name.lower()
                    or query_lower in q_name.lower()
                ):
                    matches.add(nid)

        return sorted(list(matches))

    resolve_node = find_nodes

    def callees(
        self,
        node_id: str,
        depth: int = 1,
        min_confidence: str = "low",
        max_results: int = 100,
    ) -> list[CallEdgeResult]:
        """Find functions/methods called by the given node, up to depth hops.

        Args:
            node_id: Source function or method node ID.
            depth: Maximum traversal distance.
            min_confidence: Minimum edge confidence: 'low', 'medium', 'high', 'exact'.
            max_results: Maximum number of results to return.

        Returns:
            List of CallEdgeResult objects.
        """
        if node_id not in self.graph:
            return []

        min_val = CONFIDENCE_LEVELS.get(min_confidence.lower(), 1)
        results: list[CallEdgeResult] = []
        visited_nodes: set[str] = {node_id}
        queue: list[tuple[str, int]] = [(node_id, 1)]

        while queue and len(results) < max_results:
            curr_id, curr_depth = queue.pop(0)
            if curr_depth > depth:
                continue

            for _, target_id, data in self.graph.out_edges(curr_id, data=True):
                if data.get("rel") != "CALLS":
                    continue

                conf = data.get("confidence", "low")
                if CONFIDENCE_LEVELS.get(conf.lower(), 0) < min_val:
                    continue

                t_data = self.graph.nodes.get(target_id, {})
                q_name = t_data.get("qualified_name") or t_data.get("name") or target_id
                t_file = t_data.get("file", "")

                results.append(
                    CallEdgeResult(
                        node_id=target_id,
                        qualified_name=q_name,
                        file=t_file,
                        depth=curr_depth,
                        confidence=conf,
                        resolution=data.get("resolution", "exact"),
                        lineno=data.get("lineno", 1),
                    )
                )

                if len(results) >= max_results:
                    break

                if target_id not in visited_nodes and curr_depth < depth:
                    visited_nodes.add(target_id)
                    queue.append((target_id, curr_depth + 1))

        return sorted(results, key=lambda r: (r.depth, r.lineno, r.node_id))

    def callers(
        self,
        node_id: str,
        depth: int = 1,
        min_confidence: str = "low",
        max_results: int = 100,
    ) -> list[CallEdgeResult]:
        """Find functions/methods that call the given node, up to depth hops.

        Args:
            node_id: Target function or method node ID.
            depth: Maximum traversal distance.
            min_confidence: Minimum edge confidence: 'low', 'medium', 'high', 'exact'.
            max_results: Maximum number of results to return.

        Returns:
            List of CallEdgeResult objects.
        """
        if node_id not in self.graph:
            return []

        min_val = CONFIDENCE_LEVELS.get(min_confidence.lower(), 1)
        results: list[CallEdgeResult] = []
        visited_nodes: set[str] = {node_id}
        queue: list[tuple[str, int]] = [(node_id, 1)]

        while queue and len(results) < max_results:
            curr_id, curr_depth = queue.pop(0)
            if curr_depth > depth:
                continue

            for src_id, _, data in self.graph.in_edges(curr_id, data=True):
                if data.get("rel") != "CALLS":
                    continue

                conf = data.get("confidence", "low")
                if CONFIDENCE_LEVELS.get(conf.lower(), 0) < min_val:
                    continue

                s_data = self.graph.nodes.get(src_id, {})
                q_name = s_data.get("qualified_name") or s_data.get("name") or src_id
                s_file = s_data.get("file", "")

                results.append(
                    CallEdgeResult(
                        node_id=src_id,
                        qualified_name=q_name,
                        file=s_file,
                        depth=curr_depth,
                        confidence=conf,
                        resolution=data.get("resolution", "exact"),
                        lineno=data.get("lineno", 1),
                    )
                )

                if len(results) >= max_results:
                    break

                if src_id not in visited_nodes and curr_depth < depth:
                    visited_nodes.add(src_id)
                    queue.append((src_id, curr_depth + 1))

        return sorted(results, key=lambda r: (r.depth, r.lineno, r.node_id))

    def neighborhood(
        self,
        node_id: str,
        hops: int = 2,
        rels: Sequence[str] = ("CALLS", "INHERITS", "DEFINES"),
    ) -> nx.MultiDiGraph:
        """Extract an N-hop direction-agnostic neighborhood subgraph filtered by relationship types.

        Args:
            node_id: Focal node identifier.
            hops: Traversal radius.
            rels: Relationship types to follow.

        Returns:
            Induced subgraph MultiDiGraph with only matching edges.
        """
        if node_id not in self.graph:
            sub = nx.MultiDiGraph()
            return sub

        allowed_rels = set(rels)
        discovered: set[str] = {node_id}
        queue: list[tuple[str, int]] = [(node_id, 0)]

        while queue:
            curr, dist = queue.pop(0)
            if dist >= hops:
                continue

            # Direction-agnostic: explore out-edges
            for _, target, data in self.graph.out_edges(curr, data=True):
                if data.get("rel") in allowed_rels:
                    if target not in discovered:
                        discovered.add(target)
                        queue.append((target, dist + 1))

            # Explore in-edges
            for source, _, data in self.graph.in_edges(curr, data=True):
                if data.get("rel") in allowed_rels:
                    if source not in discovered:
                        discovered.add(source)
                        queue.append((source, dist + 1))

        sub = self.graph.subgraph(discovered).copy()

        # Remove edges with relationships not in allowed_rels
        edges_to_remove = [
            (u, v, k)
            for u, v, k, data in sub.edges(data=True, keys=True)
            if data.get("rel") not in allowed_rels
        ]
        for u, v, k in edges_to_remove:
            sub.remove_edge(u, v, key=k)

        return sub

    def dependency_path(
        self,
        source_id: str,
        target_id: str,
    ) -> list[str] | None:
        """Find the shortest path of CALLS edges from source to target.

        Args:
            source_id: Originating node ID.
            target_id: Destination node ID.

        Returns:
            List of node IDs forming path, or None if unreachable.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return None

        # Build simple directed graph containing only CALLS edges
        calls_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if data.get("rel") == "CALLS":
                calls_graph.add_edge(u, v)

        try:
            return list(nx.shortest_path(calls_graph, source=source_id, target=target_id))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def file_dependencies(self, file_path: str) -> dict[str, list[str]]:
        """Return incoming and outgoing dependencies for a given source file.

        Args:
            file_path: Relative POSIX path or file node ID.

        Returns:
            Dict containing imports, imported_by, depends_on, depended_on_by lists.
        """
        clean_path = file_path[5:] if file_path.startswith("file:") else file_path
        f_id = f"file:{clean_path}"

        if f_id not in self.graph:
            return {
                "imports": [],
                "imported_by": [],
                "depends_on": [],
                "depended_on_by": [],
            }

        imports_out: set[str] = set()
        imported_by: set[str] = set()
        depends_on: set[str] = set()
        depended_on_by: set[str] = set()

        for _, target, data in self.graph.out_edges(f_id, data=True):
            rel = data.get("rel")
            t_clean = target[5:] if target.startswith("file:") else target
            if rel == "IMPORTS":
                imports_out.add(t_clean)
            elif rel == "DEPENDS_ON":
                depends_on.add(t_clean)

        for source, _, data in self.graph.in_edges(f_id, data=True):
            rel = data.get("rel")
            s_clean = source[5:] if source.startswith("file:") else source
            if rel == "IMPORTS":
                imported_by.add(s_clean)
            elif rel == "DEPENDS_ON":
                depended_on_by.add(s_clean)

        return {
            "imports": sorted(list(imports_out)),
            "imported_by": sorted(list(imported_by)),
            "depends_on": sorted(list(depends_on)),
            "depended_on_by": sorted(list(depended_on_by)),
        }

    def related_files(
        self,
        node_id: str,
        hops: int = 2,
    ) -> list[tuple[str, float]]:
        """Compute ranking of files associated with a node's graph neighborhood.

        Files touched closer in the graph receive higher scores; multiple
        connecting edges accumulate score weight.

        Args:
            node_id: Starting node ID.
            hops: Maximum neighborhood radius.

        Returns:
            List of (file_path, score) tuples ordered descending by score.
        """
        if node_id not in self.graph:
            return []

        scores: dict[str, float] = defaultdict(float)
        allowed_rels = {"CALLS", "DEFINES", "IMPORTS", "INHERITS"}

        queue: list[tuple[str, int]] = [(node_id, 0)]
        visited: set[str] = {node_id}

        while queue:
            curr, dist = queue.pop(0)
            if dist >= hops:
                continue

            next_dist = dist + 1
            # Step weight: closer distance gives higher score contribution
            step_weight = float(hops - next_dist + 1) / next_dist

            # Out-edges
            for _, nbr, data in self.graph.out_edges(curr, data=True):
                if data.get("rel") in allowed_rels:
                    nbr_file = (
                        nbr[5:]
                        if nbr.startswith("file:")
                        else self.graph.nodes.get(nbr, {}).get("file")
                    )
                    if nbr_file:
                        scores[nbr_file] += step_weight

                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, next_dist))

            # In-edges
            for nbr, _, data in self.graph.in_edges(curr, data=True):
                if data.get("rel") in allowed_rels:
                    nbr_file = (
                        nbr[5:]
                        if nbr.startswith("file:")
                        else self.graph.nodes.get(nbr, {}).get("file")
                    )
                    if nbr_file:
                        scores[nbr_file] += step_weight

                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, next_dist))

        sorted_files = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(path, round(score, 3)) for path, score in sorted_files]

    def stats(self) -> dict[str, Any]:
        """Return aggregate summary counts of graph nodes and edges."""
        node_counts: dict[str, int] = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_counts[data.get("type", "unknown")] += 1

        edge_counts: dict[str, int] = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            edge_counts[data.get("rel", "unknown")] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_counts": dict(node_counts),
            "edge_counts": dict(edge_counts),
        }
