from typing import Any

import networkx as nx

from genesis.data.utils import networkx_find_root


def compute_mastora(
    graph: nx.DiGraph,
    use_percentage: bool = False,
    mode: str = "mls",
    obstruction_attr: str = "max_transversal_obstruction",
    debug: bool = False,
) -> tuple[float, list[tuple], list[str]] | float:
    """Compute the Mastora score for a directed graph.

    Args:
        graph (nx.DiGraph): Directed graph representing the arterial tree.
        use_percentage (bool, optional): If set, treat degrees as obstruction percentages (0 to 1).
            Otherwise, use degrees (0 to 5).
        mode (str, optional): Artery levels to include: 'm' (mediastinal), 'l' (lobar), 's' (segmental).
            Any combination (e.g., 'mls').
        obstruction_attr (str, optional): The name of the edge attribute to use for obstruction values.
            Defaults to "max_transversal_obstruction".
        debug (bool, optional): If True, return debug information for visualization.
            Defaults to False.

    Returns:
        float or tuple: If debug is False, returns the Mastora score (float between 0 and 1).
            If debug is True, returns a tuple (score, debug_edges, debug_labels).
    """
    level_map = {
        "m": [1, 2],  # mediastinal
        "l": [3],  # lobar
        "s": [4],  # segmental
    }
    levels = [lvl for key in mode for lvl in level_map.get(key, [])]

    debug_edges = []
    debug_labels = []

    def _dfs(node: Any) -> list:
        degs = []
        for succ in graph.successors(node):
            attrs = graph.edges[node, succ]
            if attrs.get("level", 0) in levels:
                obs_value = attrs.get(obstruction_attr, 0.0)
                degs.append(obs_value)

                if debug:
                    debug_edges.append((node, succ))
                    artery_level = attrs.get("level", 0)
                    level_type = (
                        "M" if artery_level in level_map["m"] else "L" if artery_level in level_map["l"] else "S"
                    )
                    debug_labels.append(f"{level_type}: {obs_value:.2f}")

            degs.extend(_dfs(succ))
        return degs

    root = networkx_find_root(graph)
    degrees = _dfs(root)
    score = compute_mastora_score(degrees, use_percentage) if degrees else 0.0

    if debug:
        return score, debug_edges, debug_labels
    return score


def compute_mastora_score(degrees: list[float], use_percentage: bool = False) -> float:
    """Compute the Mastora score for a list of degrees.

    Args:
        degrees (list[float]): Degrees of the mediastinal, lobar and segmental arteries.
            Converted to integer between 0 and 5 if `use_obstruction_percentage` is False, otherwise float between
            0 and 1.
        use_percentage (bool, optional): If set, treat degrees as obstruction percentages (0 to 1).
            Otherwise, use degrees (0 to 5).

    Returns:
        float: The Mastora score, a float between 0 and 1.
    """
    # click.echo(degrees)
    if not use_percentage:
        degrees = [int(float(degree) / 0.25) + 1 for degree in degrees]
    # click.echo(degrees)
    sum_degrees = sum(degrees)
    n = len(degrees)
    return sum_degrees / n if use_percentage else sum_degrees / (n * 5)
