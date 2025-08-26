from collections.abc import Callable
from typing import Any

import networkx as nx

from genesis.data.utils import networkx_find_root


def ancestors_obstruction_max(
    graph: nx.DiGraph,
    input_attr: str = "transversal_obstruction_max",
    output_attr: str = "ancestors_obstruction_max",
    **kwargs,
) -> nx.DiGraph:
    """For each edge in the directed tree, add the maximum obstruction from its ancestors as a new attribute.

    This function propagates new obstruction values in the vascular tree that are assumed to be more representative of
    the impact of upstream obstructions than local vessel obstruction values.

    Args:
        graph: Directed acyclic graph representing a tree.
        input_attr: Name of the edge attribute with the local obstruction values.
        output_attr: Name of a new edge attribute where to store the value derived from ancestor obstructions.
        **kwargs: Additional parameters to pass along to `update_edge_attr_top_down`.

    Returns:
        A copy of `graph` where each edge has as new attribute the maximum obstruction from its ancestors.
    """
    return update_edge_attr_top_down(graph, update_fn=max, input_attr=input_attr, output_attr=output_attr, **kwargs)


def ancestors_obstruction_cumulated(
    graph: nx.DiGraph,
    input_attr: str = "transversal_obstruction_max",
    output_attr: str = "ancestors_obstruction_cumulated",
    **kwargs,
) -> nx.DiGraph:
    """For each edge in the directed tree, add a weighted function of its ancestors' obstructions as a new attribute.

    This function propagates new obstruction values in the vascular tree that are assumed to be more representative of
    the impact of upstream obstructions on vessel hemodynamic than local vessel obstruction values.

    Args:
        graph: Directed acyclic graph representing a tree.
        input_attr: Name of the edge attribute with the local obstruction values.
        output_attr: Name of a new edge attribute where to store the value derived from ancestor obstructions.
        **kwargs: Additional parameters to pass along to `update_edge_attr_top_down`.

    Returns:
        A copy of `graph` where each edge has as new attribute the weighted function of its ancestors' obstructions.
    """

    def cumulate_fn(parent_obstruction: float, own_obstruction: float) -> float:
        """Return new cumulated obstruction for the edge based on parent's and own values."""
        return 1 - (1 - own_obstruction) * (1 - parent_obstruction)

    return update_edge_attr_top_down(
        graph, update_fn=cumulate_fn, input_attr=input_attr, output_attr=output_attr, **kwargs
    )


def update_edge_attr_top_down(
    graph: nx.DiGraph,
    update_fn: Callable[[float, float], float],
    input_attr: str,
    output_attr: str | None = None,
    root: Any = None,
) -> nx.DiGraph:
    """In a directed tree, update edge attributes based on their own value and their parent's, starting from a root.

    Args:
        graph: Directed acyclic graph representing a tree.
        update_fn: Function to compute the new value based on the edge's parent's and own values.
        input_attr: Name of the edge attribute with the raw values to update.
        output_attr: Name of a new edge attribute where to store the updated values. If not provided, "input_attr" will
            be updated.
        root: The root node (in-degree == 0). If None, it is auto-detected. Defaults to None.

    Returns:
        A copy of `graph` where each edge has a new/updated attribute.

    Raises:
        ValueError: If `graph` is not a valid arborescence.
    """
    if output_attr is None:
        output_attr = input_attr

    graph = graph.copy()
    if root is None:
        root = networkx_find_root(graph)

    def _depth_first_traversal(node: Any, parent_attr_val: float) -> None:
        for child in graph.successors(node):
            own = graph.edges[node, child].get(input_attr, 0.0)
            updated_val = update_fn(parent_attr_val, own)
            graph.edges[node, child][output_attr] = updated_val
            _depth_first_traversal(child, updated_val)

    _depth_first_traversal(root, 0.0)
    return graph
