from typing import Any

import networkx as nx

from genesis.data.utils import networkx_find_root


def compute_qanadli(
    graph: nx.DiGraph,
    min_obstruction_thresh: float = 0.25,
    max_obstruction_thresh: float = 0.75,
    obstruction_attr: str = "max_transversal_obstruction",
    debug: bool = False,
) -> tuple[float, list[tuple], list[str]] | float:
    """Compute the Qanadli score for a directed graph.

    Args:
        graph (nx.DiGraph): Directed graph representing the arterial tree.
        min_obstruction_thresh (float, optional): Minimum obstruction threshold for considering a segment.
            Defaults to 0.25.
        max_obstruction_thresh (float, optional): Maximum obstruction threshold for considering a segment.
            Defaults to 0.75.
        obstruction_attr (str, optional): The name of the edge attribute to use for obstruction values.
            Defaults to "max_transversal_obstruction".
        debug (bool, optional): If True, return debug information for visualization.
            Defaults to False.

    Returns:
        float or tuple: If debug is False, returns the Qanadli score (float between 0 and 1).
            If debug is True, returns a tuple (score, debug_edges, debug_labels).
    """
    root = networkx_find_root(graph)
    weights: list[int] = []
    degrees: list[float] = []
    debug_edges: list[tuple] = []
    debug_labels: list[str] = []

    def _dfs(node: Any) -> None:
        for child in graph.successors(node):
            edge_attrs = graph.edges[node, child]
            mto = edge_attrs.get(obstruction_attr, 0.0)
            arterie_type = get_arterie_type(edge_attrs)

            if arterie_type == "mediastinal" or arterie_type == "lobar":
                if mto > min_obstruction_thresh:
                    weight = get_subsegments_below(edge_attrs)
                    weights.append(weight)
                    degrees.append(mto)

                    if debug:
                        debug_edges.append((node, child))
                        degree_value = 0 if mto < min_obstruction_thresh else 1 if mto < max_obstruction_thresh else 2
                        debug_labels.append(f"{arterie_type[0].upper()}: {mto:.2f} (w:{weight}, d:{degree_value})")
                else:
                    _dfs(child)
            elif arterie_type == "segmental":
                weights.append(1)
                degrees.append(mto)

                if debug:
                    debug_edges.append((node, child))
                    degree_value = 0 if mto < min_obstruction_thresh else 1 if mto < max_obstruction_thresh else 2
                    debug_labels.append(f"S: {mto:.2f} (w:1, d:{degree_value})")
            elif arterie_type == "root":
                _dfs(child)

    _dfs(root)
    score = compute_qanadli_score(weights, degrees, min_obstruction_thresh, max_obstruction_thresh) if degrees else 0.0
    return score, debug_edges, debug_labels if debug_edges else score


def get_arterie_type(edge: dict[str, Any]) -> str:
    """Get the type of artery based on the edge attributes.

    Args:
        edge (dict[str, Any]): Edge attributes containing 'level'.

    Returns:
        str: Type of artery ('root', 'mediastinal', 'lobar' or 'segmental').
    """
    level = edge.get("level", 0)
    if level == 1:
        return "root"
    if level == 2:
        return "mediastinal"
    if level == 4:
        return "segmental"
    if level == 3:
        if all(succ.get("level", 0) == 4 for succ in edge.get("successors", [])):
            return "lobar"  # return "segmental" in normal cases
        return "lobar"
    return ""


def get_subsegments_below(edge_attrs: dict[str, Any]) -> int:
    """Get the number of subsegments below an artery segment.

    Args:
        edge_attrs (dict[str, Any]): Edge attributes containing 'segments_below'.

    Returns:
        int: Number of subsegments below the artery segment.
    """
    subsegments_below = []

    def _dfs(edge: dict[str, Any]) -> int:
        """Recursive function to count segments below."""
        level = edge.get("level", 0)
        if level <= 4:
            subsegments_below.append(edge.get("segments_below", 0))
        else:
            for succ in edge.get("successors", []):
                _dfs(succ)

    _dfs(edge_attrs)
    return sum(subsegments_below) if subsegments_below else 0


def compute_qanadli_score(
    weights: list[int], degrees: list[float], min_obstruction_thresh: float, max_obstruction_thresh: float
) -> float:
    """Compute the Qanadli score for a list of degrees.

    Args:
        weights (list[int]): Weights of the segments, representing the number of segments below each artery.
        degrees (list[float]): Degrees of obstruction for each segment.
        min_obstruction_thresh (float): Minimum obstruction threshold for considering a segment.
        max_obstruction_thresh (float): Maximum obstruction threshold for considering a segment.

    Returns:
        float: The Qanadli score, a float between 0 and 1.
    """
    degrees = [
        0 if degree < min_obstruction_thresh else 1 if degree < max_obstruction_thresh else 2 for degree in degrees
    ]
    weighted_degrees = [weight * degree for weight, degree in zip(weights, degrees, strict=False)]
    return sum(weighted_degrees) / (2 * sum(weights))
