from typing import Any, Literal

import networkx as nx
import numpy as np

from genesis.analysis.scores.config import ArteryLevel
from genesis.analysis.scores.utils import derive_missing_obstruction_attrs
from genesis.data.utils import networkx_find_root

# Thresholds to discretize obstruction values into 5 levels (1-5) + 0 (no obstruction)
# Left edges are included in intervals, but 0 should be excluded from bin 1, thus the use of 1e-6 for left-most edge
DEGREE_THRESHOLDS = [1e-6, 0.25, 0.5, 0.75, 1]


@derive_missing_obstruction_attrs(graph_arg=0, attrs_args=["obstruction_attr"])
def mastora(
    graph: nx.DiGraph,
    mode: Literal["central", "peripheral", "global"],
    obstruction_attr: str = "transversal_obstruction_max",
    debug: bool = False,
) -> float | tuple[float, dict[tuple[int, int], str]]:
    """Compute the Mastora score for a directed graph.

    Args:
        graph: Directed graph representing the arterial tree.
        mode: Variant of the Mastora score to compute.
            - 'central': Considers obstructions in the mediastinal and lobar arteries
            - 'peripheral': Considers obstructions in the segmental arteries
            - 'global': Considers obstructions in all arteries (i.e. both central and peripheral)
        obstruction_attr: The name of the edge attribute to use for obstruction values.
        debug: If True, return debug information for visualization.

    Returns:
        float or tuple: If debug is False, returns the Mastora score (float between 0 and 1).
            If debug is True, returns a tuple (score, debug_info).
    """
    # Determine numerical levels for which to look for obstructions, based on mode
    match mode:
        case "global":
            levels_to_search = {ArteryLevel.MEDIASTINAL, ArteryLevel.LOBAR, ArteryLevel.SEGMENTAL}
        case "central":
            levels_to_search = {ArteryLevel.MEDIASTINAL, ArteryLevel.LOBAR}
        case "peripheral":
            levels_to_search = {ArteryLevel.SEGMENTAL}
        case _:
            raise ValueError(f"Invalid mode '{mode}'. Allowed values are 'central', 'peripheral' or 'global'.")

    # Data structures to save data for each selected edge,
    # mapped by edge (u, v) to facilitate debugging if needed
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

            # Recursively visit children
            _depth_first_search(child)

    _depth_first_search(networkx_find_root(graph))

    # From the lists of obstruction degrees and number of descendant segments, compute the Qanadli score
    obstructions_vals = list(obstructions.values())
    # Discretize obstruction values between {0...5}, so that normalization will be in [0, 1]
    degrees = np.digitize(obstructions_vals, DEGREE_THRESHOLDS)
    # Original paper summed the degrees across the landmark arteries, but here we normalize and average degrees
    # to get a score between 0 and 1, invariant to the exact number of arteries of the patient
    score = np.mean(degrees / len(DEGREE_THRESHOLDS)).item()

    if debug:
        return score, debug_info
    return score
