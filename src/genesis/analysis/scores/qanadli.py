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
    # Data structures to save data for each selected edge,
    # mapped by edge (u, v) to facilitate debugging if needed
    num_descendant_segments: dict[tuple[int, int], int] = {}
    obstructions: dict[tuple[int, int], float] = {}
    debug_info: dict[tuple[int, int], str] = {}

    def _save_edge_data(edge_key: tuple[int, int], level: int, segments_below: int, obstruction: float) -> None:
        """Save data for a given edge to the data structures used to compute/debug the score."""
        num_descendant_segments[edge_key] = segments_below
        obstructions[edge_key] = obstruction

        if debug:
            degree = np.digitize(obstruction, [partial_obstruction_thresh, total_obstruction_thresh])
            artery_type = ArteryLevel(level).name
            debug_info[edge_key] = f"{artery_type[0]}: {obstruction:.2f} (num_seg:{segments_below}, deg:{degree})"

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

    # From the lists of obstruction degrees and number of descendant segments, compute the Qanadli score
    obstructions_vals = list(obstructions.values())
    num_descendant_segments_vals = list(num_descendant_segments.values())
    # Discretize obstruction values into degrees: 0 (no obstruction), 1 (partial), 2 (total)
    degrees = np.digitize(obstructions_vals, [partial_obstruction_thresh, total_obstruction_thresh])
    # Compute the Qanadli score as the obstruction degrees weighted by the number of descendant segments
    weighted_degrees = num_descendant_segments_vals * degrees
    # Original paper summed the weighted degrees across the landmark arteries, but here we normalize by the maximum
    # possible score (2 * total number of segmental arteries) to get a score between 0 and 1
    score = sum(weighted_degrees) / (2 * sum(num_descendant_segments_vals))

    if debug:
        return score, debug_info
    return score
