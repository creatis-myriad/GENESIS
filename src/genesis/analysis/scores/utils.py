from collections.abc import Callable
from functools import wraps
from typing import Any

import networkx as nx

from genesis.data.utils import networkx_aggregate_attrs, networkx_has_edge_attributes


def aggregate_score_input(score_fn: Callable) -> Callable:
    """Decorator to aggregate list-valued edge attributes used as input by global obstruction scores.

    The decorator ensures that the requested edge-wise aggregation of the base obstruction score is available
    for the global obstruction score computation.

    Args:
        score_fn: Base scoring function that takes aggregated inputs and produces a global score.

    Returns:
        A function that computes the global score after aggregating the required edge attributes.
    """

    @wraps(score_fn)
    def _aggregate_and_score(graph: nx.DiGraph, *args, obstruction_attr: str, **kwargs) -> Any:
        # Only aggregate if we don't detect the aggregated attribute already exists
        if not networkx_has_edge_attributes(graph, attrs=[obstruction_attr]):
            base_obstr_attr, agg = obstruction_attr.rsplit("_", 1)
            # If debugging, modify graph in place to retain aggregated attributes for visualization
            in_place = kwargs.get("debug", False)
            graph = networkx_aggregate_attrs(graph, {base_obstr_attr: agg}, element="edges", in_place=in_place)
        return score_fn(graph, *args, obstruction_attr=obstruction_attr, **kwargs)

    return _aggregate_and_score
