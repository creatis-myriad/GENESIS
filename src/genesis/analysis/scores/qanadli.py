from typing import Any

import networkx as nx
import numpy as np

from genesis.data.utils import networkx_find_root


def qanadli(
    graph: nx.DiGraph,
    min_obstruction_thresh: float = 0.25,
    max_obstruction_thresh: float = 0.75,
    obstruction_attr: str = "max_transversal_obstruction",
    debug: bool = False,
) -> float | tuple[float, list[tuple], list[str]]:
    """Compute the Qanadli score for a directed graph.

    Args:
        graph: Directed graph representing the arterial tree.
        min_obstruction_thresh: Minimum obstruction threshold for considering a segment.
        max_obstruction_thresh: Maximum obstruction threshold for considering a segment.
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
                if artery_obstruction > min_obstruction_thresh:
                    weight = _count_segmental_descendants(edge_attrs)
                    weights.append(weight)
                    obstruction_vals.append(artery_obstruction)

                    if debug:
                        debug_edges.append((node, child))
                        degree_value = (
                            0
                            if artery_obstruction < min_obstruction_thresh
                            else 1
                            if artery_obstruction < max_obstruction_thresh
                            else 2
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
                    degree_value = (
                        0
                        if artery_obstruction < min_obstruction_thresh
                        else 1
                        if artery_obstruction < max_obstruction_thresh
                        else 2
                    )
                    debug_labels.append(f"S: {artery_obstruction:.2f} (w:1, d:{degree_value})")
            elif artery_type == "root":
                _depth_first_search(child)

    _depth_first_search(root)
    score = (
        _compute_qanadli_score(weights, obstruction_vals, min_obstruction_thresh, max_obstruction_thresh)
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


def _count_segmental_descendants(edge_attrs: dict[str, Any]) -> int:
    """Count the number of descendants of an artery that are at the segmental level or lower.

    Args:
        edge_attrs: Edge attributes, which must contain the 'segments_below' key.

    Returns:
        Number of descendants of an artery that are at the segmental level or lower.
    """

    def _inner_count_subsegmental_descendants(edge: dict[str, Any]) -> int:
        """Recursive function to count subsegmental descendants."""
        if edge.get("level", 0) <= 4:
            return edge.get("segments_below", 0)
        count_by_child = [_inner_count_subsegmental_descendants(succ) for succ in edge.get("successors", [])]
        return sum(count_by_child)

    return _inner_count_subsegmental_descendants(edge_attrs)


def _compute_qanadli_score(
    weights: list[int], obstruction_vals: list[float], min_obstruction_thresh: float, max_obstruction_thresh: float
) -> float:
    """Compute the Qanadli score from a list of obstruction values.

    Args:
        weights: Weights of the segments, representing the number of segments below each artery.
        obstruction_vals: Obstruction value for each segment.
        min_obstruction_thresh: Minimum obstruction threshold for considering a segment.
        max_obstruction_thresh: Maximum obstruction threshold for considering a segment.

    Returns:
        The Qanadli score, between 0 and 1.
    """
    # Discretize obstruction values into degrees
    degrees = np.digitize(obstruction_vals, [min_obstruction_thresh, max_obstruction_thresh])
    weighted_degrees = weights * degrees
    return sum(weighted_degrees) / (2 * sum(weights))
