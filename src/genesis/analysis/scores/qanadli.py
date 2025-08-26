from typing import Any

import networkx as nx
import numpy as np

from genesis.analysis.scores.utils import aggregate_score_input
from genesis.data.utils import networkx_find_root


@aggregate_score_input
def qanadli(
    graph: nx.DiGraph,
    partial_obstruction_thresh: float = 0.25,
    total_obstruction_thresh: float = 0.75,
    obstruction_attr: str = "transversal_obstruction_max",
    debug: bool = False,
) -> float | tuple[float, list[tuple], list[str]]:
    """Compute the Qanadli score for a directed graph.

    Args:
        graph: Directed graph representing the arterial tree.
        partial_obstruction_thresh: Transversal obstruction threshold to consider a segment partially obstructed.
        total_obstruction_thresh: Transversal obstruction threshold to consider a segment totally obstructed.
        obstruction_attr: The name of the edge attribute to use for obstruction values.
        debug: If True, return debug information for visualization.

    Returns:
        float or tuple: If debug is False, returns the Qanadli score (float between 0 and 1).
            If debug is True, returns a tuple (score, debug_edges, debug_labels).
    """
    root = networkx_find_root(graph)
    weights: list[int] = []
    obstruction_vals: list[float] = []
    debug_edges: list[tuple] = []
    debug_labels: list[str] = []

    def _depth_first_search(node: Any) -> None:
        for child in graph.successors(node):
            edge_attrs = graph.edges[node, child]
            artery_obstruction = edge_attrs.get(obstruction_attr, 0.0)
            artery_type = _get_artery_type(edge_attrs)

            if artery_type == "mediastinal" or artery_type == "lobar":
                if artery_obstruction > partial_obstruction_thresh:
                    weight = edge_attrs["segments_below"]
                    weights.append(weight)
                    obstruction_vals.append(artery_obstruction)

                    if debug:
                        debug_edges.append((node, child))
                        degree_value = np.digitize(
                            artery_obstruction, [partial_obstruction_thresh, total_obstruction_thresh]
                        )
                        debug_labels.append(
                            f"{artery_type[0].upper()}: {artery_obstruction:.2f} (w:{weight}, d:{degree_value})"
                        )
                else:
                    _depth_first_search(child)
            elif artery_type == "segmental":
                weights.append(1)
                obstruction_vals.append(artery_obstruction)

                if debug:
                    debug_edges.append((node, child))
                    degree_value = np.digitize(
                        artery_obstruction, [partial_obstruction_thresh, total_obstruction_thresh]
                    )
                    debug_labels.append(f"S: {artery_obstruction:.2f} (w:1, d:{degree_value})")
            elif artery_type == "root":
                _depth_first_search(child)

    _depth_first_search(root)
    score = (
        _compute_qanadli_score(weights, obstruction_vals, partial_obstruction_thresh, total_obstruction_thresh)
        if obstruction_vals
        else 0.0
    )
    return score, debug_edges, debug_labels if debug_edges else score


def _get_artery_type(edge: dict[str, Any]) -> str:
    """Get the type of artery based on the edge attributes.

    Args:
        edge: Edge attributes containing 'level'.

    Returns:
        Type of artery ('root', 'mediastinal', 'lobar' or 'segmental').
    """
    level = edge.get("level", 0)
    match level:
        case 1:
            artery_type = "root"
        case 2:
            artery_type = "mediastinal"
        case 3:
            # Previous implementation is commented below for reference, although it does not seem to be able to return
            # anything other than "lobar"
            # if all(succ.get("level", 0) == 4 for succ in edge.get("successors", [])):
            #     return "lobar"  # return "segmental" in normal cases
            # return "lobar"
            artery_type = "lobar"
        case 4:
            artery_type = "segmental"
        case _:
            artery_type = ""

    return artery_type


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
