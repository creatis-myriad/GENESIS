from typing import Any

import networkx as nx
import numpy as np

from genesis.analysis.scores.utils import derive_missing_obstruction_attrs
from genesis.analysis.scores.vascular_tree import ArteryLevel
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
    weights: dict[tuple[int, int], int] = {}
    obstructions: dict[tuple[int, int], float] = {}
    debug_info: dict[tuple[int, int], str] = {}

    def _save_edge_data(edge_key: tuple[int, int], level: int, weight: int, obstruction: float) -> None:
        """Save data for a given edge to the data structures used to compute/debug the score."""
        weights[edge_key] = weight
        obstructions[edge_key] = obstruction

        if debug:
            degree_value = np.digitize(obstruction, [partial_obstruction_thresh, total_obstruction_thresh])
            artery_type = ArteryLevel(level).name
            debug_info[edge_key] = f"{artery_type[0]}: {obstruction:.2f} (w:{weight}, d:{degree_value})"

    def _depth_first_search(node: Any) -> None:
        for child in graph.successors(node):
            edge_attrs = graph.edges[node, child]
            artery_obstruction = edge_attrs[obstruction_attr]

            match artery_level := edge_attrs["level"]:
                case ArteryLevel.ROOT:
                    # Recursively visit children, but do not add root arteries to the score
                    _depth_first_search(child)
                case ArteryLevel.MEDIASTINAL | ArteryLevel.LOBAR:
                    if artery_obstruction > partial_obstruction_thresh:
                        _save_edge_data((node, child), artery_level, edge_attrs["segments_below"], artery_obstruction)
                    else:
                        # Recursively visit children if artery is not obstructed enough
                        _depth_first_search(child)
                case ArteryLevel.SEGMENTAL:
                    _save_edge_data((node, child), artery_level, 1, artery_obstruction)
                case _:
                    artery_type = ArteryLevel(artery_level).name.lower()
                    raise ValueError(
                        f"Unexpected artery level '{artery_level} ({artery_type})' for edge ({node}, {child})"
                    )

    _depth_first_search(networkx_find_root(graph))
    score = _compute_qanadli_score(
        list(weights.values()), list(obstructions.values()), partial_obstruction_thresh, total_obstruction_thresh
    )
    if debug:
        return score, debug_info
    return score


def _compute_qanadli_score(
    weights: list[int],
    obstruction_vals: list[float],
    partial_obstruction_thresh: float,
    total_obstruction_thresh: float,
) -> float:
    """Compute the Qanadli score from a list of obstruction values.

    Args:
        weights: Weights of the segments, representing the number of segments below each artery.
        obstruction_vals: Obstruction value for each segment.
        partial_obstruction_thresh: Transversal obstruction threshold to consider a segment partially obstructed.
        total_obstruction_thresh: Transversal obstruction threshold to consider a segment totally obstructed.

    Returns:
        The Qanadli score, between 0 and 1.
    """
    # Discretize obstruction values into degrees
    degrees = np.digitize(obstruction_vals, [partial_obstruction_thresh, total_obstruction_thresh])
    weighted_degrees = weights * degrees
    return sum(weighted_degrees) / (2 * sum(weights))
