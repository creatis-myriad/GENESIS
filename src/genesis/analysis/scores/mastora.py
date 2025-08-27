from typing import Any

import networkx as nx
import numpy as np

from genesis.analysis.scores.utils import derive_missing_obstruction_attrs
from genesis.analysis.scores.vascular_tree import ArteryLevel
from genesis.data.utils import networkx_find_root


@derive_missing_obstruction_attrs(graph_arg=0, attrs_args=["obstruction_attr"])
def mastora(
    graph: nx.DiGraph,
    use_percentage: bool = False,
    mode: str = "rmls",
    obstruction_attr: str = "transversal_obstruction_max",
    debug: bool = False,
) -> float | tuple[float, dict[tuple[int, int], str]]:
    """Compute the Mastora score for a directed graph.

    Args:
        graph: Directed graph representing the arterial tree.
        use_percentage: If set, use obstruction ratio directly. Otherwise, convert to discrete {1...5} point scale.
        mode: Artery levels to include: 'r' (root), 'm' (mediastinal), 'l' (lobar), 's' (segmental).
            Combinations of multiple levels (e.g., 'rmls') are also accepted.
        obstruction_attr: The name of the edge attribute to use for obstruction values.
        debug: If True, return debug information for visualization.

    Returns:
        float or tuple: If debug is False, returns the Mastora score (float between 0 and 1).
            If debug is True, returns a tuple (score, debug_info).
    """
    map_lvl_shorthands_to_enum = {level.name.lower()[0]: level for level in ArteryLevel}
    if not set(mode) <= set(map_lvl_shorthands_to_enum.keys()):
        raise ValueError(
            f"Invalid mode '{mode}'. Allowed levels are {list(map_lvl_shorthands_to_enum.keys())}, "
            f"or any combination of them (e.g. '{''.join(map_lvl_shorthands_to_enum.keys())}')."
        )
    # Determine numerical levels for which to look for obstructions, based on requested artery levels
    levels_to_search = {map_lvl_shorthands_to_enum[level_shorthand] for level_shorthand in mode}

    obstructions: dict[tuple[int, int], float] = {}
    debug_info: dict[tuple[int, int], str] = {}

    def _depth_first_search(node: Any) -> None:
        for child in graph.successors(node):
            edge_attrs = graph.edges[node, child]
            artery_obstruction = edge_attrs[obstruction_attr]

            if (artery_level := edge_attrs["level"]) in levels_to_search:
                obstructions[(node, child)] = artery_obstruction

                if debug:
                    artery_type = ArteryLevel(artery_level).name
                    debug_info[(node, child)] = f"{artery_type[0]}: {artery_obstruction:.2f}"

                # Recursively visit children, only if artery is in a level to be inspected
                _depth_first_search(child)

    _depth_first_search(networkx_find_root(graph))
    score = _compute_mastora_score(list(obstructions.values()), use_percentage=use_percentage)

    if debug:
        return score, debug_info
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
