from typing import Any

import networkx as nx
import numpy as np

from genesis.analysis.config import ArteryLevel
from genesis.analysis.scores.utils import derive_missing_obstruction_attrs
from genesis.data.utils import networkx_find_root


@derive_missing_obstruction_attrs(graph_arg=0, attrs_args=["obstruction_attr"])
def qanadli(
    graph: nx.DiGraph,
    partial_obstruction_thresh: float = 0.25,
    total_obstruction_thresh: float = 0.75,
    obstruction_attr: str = "transversal_obstruction_max",
    debug: bool = False,
) -> float | tuple[float, dict[tuple[int, int], str]]:
    """Compute the Qanadli score for a directed graph.

    Args:
        graph: Directed graph representing the arterial tree.
        partial_obstruction_thresh: Transversal obstruction threshold to consider a segment partially obstructed.
        total_obstruction_thresh: Transversal obstruction threshold to consider a segment totally obstructed.
        obstruction_attr: The name of the edge attribute to use for obstruction values.
        debug: If True, return debug information for visualization.

    Returns:
        float or tuple: If debug is False, returns the Qanadli score (float between 0 and 1).
            If debug is True, returns a tuple (score, debug_info).
    """
    # Data structures to save data for each selected edge,
    # mapped by edge (u, v) to facilitate debugging if needed
    descendant_segments_counts: dict[tuple[int, int], int] = {}
    obstructions: dict[tuple[int, int], float] = {}
    debug_info: dict[tuple[int, int], str] = {}

    def _store_edge_data(
        edge: tuple[int, int], obstruction: float, descendant_segments_count: int, level: ArteryLevel
    ) -> None:
        """Store data for a given edge to the data structures used to compute/debug the score."""
        obstructions[edge] = obstruction
        descendant_segments_counts[edge] = descendant_segments_count

        if debug:
            degree = np.digitize(obstruction, [partial_obstruction_thresh, total_obstruction_thresh])
            artery_type = ArteryLevel(level).name
            debug_info[edge] = f"{artery_type[0]}: {obstruction:.2f} (n_seg:{descendant_segments_count}, deg:{degree})"

    def _depth_first_search(node: Any) -> None:
        for child in graph.successors(node):
            edge_attrs = graph.edges[node, child]
            artery_obstruction = edge_attrs[obstruction_attr]
            artery_level = edge_attrs["level"]

            if artery_level <= ArteryLevel.SEGMENTAL:  # Only consider arteries of segmental level or above
                if artery_obstruction > partial_obstruction_thresh:
                    # If artery is obstructed enough:
                    # - Count the whole subtree (number of segmental descendants) as obstructed to the same degree
                    # - Stop recursion
                    descendant_segments_count = _count_terminal_descendants(graph, (node, child), ArteryLevel.SEGMENTAL)
                    _store_edge_data((node, child), artery_obstruction, descendant_segments_count, artery_level)

                elif _is_terminal(graph, (node, child), ArteryLevel.SEGMENTAL):
                    # If artery has no descendants of at least segmental level:
                    # - Count it as one non-obstructed artery
                    # - Stop recursion
                    _store_edge_data((node, child), artery_obstruction, 1, artery_level)

                else:
                    # Recursively visit children
                    _depth_first_search(child)

    _depth_first_search(networkx_find_root(graph))

    # From the lists of obstruction degrees and number of segmental descendants, compute the Qanadli score
    obstructions_vals = list(obstructions.values())
    descendant_segments_counts_vals = list(descendant_segments_counts.values())
    # Discretize obstruction values into degrees: 0 (no obstruction), 1 (partial), 2 (total)
    degrees = np.digitize(obstructions_vals, [partial_obstruction_thresh, total_obstruction_thresh])
    # Compute the Qanadli score as the obstruction degrees weighted by the number of segmental descendants
    weighted_degrees = descendant_segments_counts_vals * degrees
    # Original paper summed the weighted degrees across the landmark arteries, but here we normalize by the maximum
    # possible score (2 * total number of segmental arteries) to get a score between 0 and 1
    score = sum(weighted_degrees) / (2 * sum(descendant_segments_counts_vals))

    if debug:
        return score, debug_info
    return score


def _is_terminal(graph: nx.DiGraph, edge: tuple[int, int], terminal_level: int) -> bool:
    """Check if an edge is terminal, i.e. of defined level at most with no successors of that level at most.

    Args:
        graph: Directed graph representing the arterial tree.
        edge: IDs of the source and destination nodes of the edge to check.
        terminal_level: Level at the end of which descendants are considered terminal.

    Returns:
        True if the edge is terminal, i.e. of defined level at most with no successors of that level at most.
    """
    parent, node = edge
    if graph.edges[parent, node]["level"] > terminal_level:
        return False  # Edge above the terminal level cannot be terminal
    return all(graph.edges[node, child]["level"] > terminal_level for child in graph.successors(node))


def _count_terminal_descendants(graph: nx.DiGraph, edge: tuple[int, int], terminal_level: int) -> int:
    """Count the number of terminal descendants of an edge.

    Args:
        graph: Directed graph representing the arterial tree.
        edge: IDs of the source and destination nodes of the edge from which to start counting.
        terminal_level: Artery level at the end of which descendants are considered terminal.

    Returns:
        Number of terminal descendants of an edge.
    """

    def _depth_first_search(_edge: tuple[int, int]) -> int:
        if _is_terminal(graph, _edge, terminal_level):
            return 1
        _parent, node = _edge
        return sum(_depth_first_search((node, child)) for child in graph.successors(node))

    return _depth_first_search(edge)
