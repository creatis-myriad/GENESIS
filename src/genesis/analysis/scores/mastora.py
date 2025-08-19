from typing import Any

import networkx as nx
import numpy as np

from genesis.data.utils import networkx_find_root


def mastora(
    graph: nx.DiGraph,
    use_percentage: bool = False,
    mode: str = "mls",
    obstruction_attr: str = "max_transversal_obstruction",
    debug: bool = False,
) -> float | tuple[float, list[tuple], list[str]]:
    """Compute the Mastora score for a directed graph.

    Args:
        graph: Directed graph representing the arterial tree.
        use_percentage: If set, use obstruction ratio directly. Otherwise, convert to discrete {1...5} point scale.
        mode: Artery levels to include: 'm' (mediastinal), 'l' (lobar), 's' (segmental).
            Combinations of multiple levels (e.g., 'mls') are also accepted.
        obstruction_attr: The name of the edge attribute to use for obstruction values.
        debug: If True, return debug information for visualization.

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

    def _depth_first_search(node: Any) -> list:
        obstruction_vals = []
        for succ in graph.successors(node):
            attrs = graph.edges[node, succ]
            if attrs.get("level", 0) in levels:
                artery_obstruction = attrs.get(obstruction_attr, 0.0)
                obstruction_vals.append(artery_obstruction)

                if debug:
                    debug_edges.append((node, succ))
                    artery_level = attrs.get("level", 0)
                    level_type = (
                        "M" if artery_level in level_map["m"] else "L" if artery_level in level_map["l"] else "S"
                    )
                    debug_labels.append(f"{level_type}: {artery_obstruction:.2f}")

            obstruction_vals.extend(_depth_first_search(succ))
        return obstruction_vals

    root = networkx_find_root(graph)
    obstruction_vals = _depth_first_search(root)
    score = _compute_mastora_score(obstruction_vals, use_percentage) if obstruction_vals else 0.0

    if debug:
        return score, debug_edges, debug_labels
    return score


def _compute_mastora_score(obstruction_vals: list[float], use_percentage: bool = False) -> float:
    """Compute the Mastora score from a list of obstruction values.

    Args:
        obstruction_vals: Obstruction values of the mediastinal, lobar and segmental arteries.
            If `use_percentage` is `True`, they are used directly as obstruction percentages.
            Otherwise, they are converted to a discrete point scale {1...5}.
        use_percentage: If set, use obstruction ratio directly. Otherwise, convert to discrete {1...5} point scale.

    Returns:
        The Mastora score, between 0 and 1.
    """
    if not use_percentage:
        # Discretize obstruction values between {0...4} + shift by 1 to {1...5}
        obstruction_vals = np.digitize(obstruction_vals, [0.25, 0.5, 0.75, 1]) + 1
    num_arteries = len(obstruction_vals)
    return sum(obstruction_vals) / (num_arteries if use_percentage else num_arteries * 5)
