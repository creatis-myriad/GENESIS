from typing import Any, Literal

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def networkx_to_pyg(graph: nx.Graph, target_attr: str, target_dtype: torch.dtype, **from_networkx_kwargs) -> Data:
    """Convert NetworkX `Graph` to PyG `Data`.

    Args:
        graph: NetworkX graph.
        target_attr: Key of the graph attribute to use as target.
        target_dtype: Data type of the target attribute.
        **from_networkx_kwargs: Node and edge filter lists to pass to `from_networkx`.

    Returns:
        PyG `Data` representation of the NetworkX `Graph`.
    """
    if not _has_node_attributes(graph):
        from_networkx_kwargs["group_node_attrs"] = None
    elif from_networkx_kwargs.get("group_node_attrs") is None:
        from_networkx_kwargs["group_node_attrs"] = "all"

    if not _has_edge_attributes(graph):
        from_networkx_kwargs["group_edge_attrs"] = None
    elif from_networkx_kwargs.get("group_edge_attrs") is None:
        from_networkx_kwargs["group_edge_attrs"] = "all"

    # Don't manage string attributes
    data = from_networkx(graph, **from_networkx_kwargs)
    data = _clean_data_attributes(data)
    target_label = graph.graph[target_attr]
    data.y = torch.tensor(target_label, dtype=target_dtype)
    return data


def networkx_line_graph(graph: nx.Graph) -> nx.Graph:
    """Convert NetworkX `Graph` to its line graph.

    Only transpose original edge features to dual graph nodes, as edge features are not used in most GNNs.

    Args:
        graph: Original graph.

    Returns:
        Line graph.
    """
    dual_graph = nx.line_graph(graph)
    dual_graph.graph.update(graph.graph)
    for source, target, feats in graph.edges(data=True):
        dual_graph.nodes[(source, target)].update(feats)
    return dual_graph


def networkx_add_attrs(
    graph: nx.Graph, element: Literal["graph", "nodes", "edges", "links"], attrs: dict[str, Any]
) -> nx.Graph:
    """Add attributes to the graph, nodes, or edges.

    Args:
        graph: NetworkX graph.
        element: Element to add attributes to; should be 'graph', 'nodes', or 'edges'/'links'.
        attrs: Attributes to add.

    Returns:
        Graph with added attributes.
    """
    match element:
        case "graph":
            graph.graph.update(attrs)
        case "nodes":
            nx.set_node_attributes(graph, attrs)
        case "edges" | "links":
            nx.set_edge_attributes(graph, attrs)
        case _:
            raise ValueError("Element must be 'graph', 'nodes', or 'edges'/'links'.")
    return graph


def networkx_remove_attrs(
    graph: nx.Graph, element: Literal["graph", "nodes", "edges", "links"], attrs: list[str]
) -> nx.Graph:
    """Remove attributes from the graph, nodes, or edges.

    Args:
        graph: NetworkX graph.
        element: Element to remove attributes from; should be 'graph', 'nodes', or 'edges'/'links'.
        attrs: Attributes to remove.

    Returns:
        Graph with removed attributes.
    """
    match element:
        case "graph":
            for attr in attrs:
                graph.graph.pop(attr, None)
        case "nodes":
            for _, node_attrs in graph.nodes(data=True):
                for attr in attrs:
                    node_attrs.pop(attr, None)
        case "edges" | "links":
            for _, _, edge_attrs in graph.edges(data=True):
                for attr in attrs:
                    edge_attrs.pop(attr, None)
        case _:
            raise ValueError("Element must be 'graph', 'nodes', or 'edges'/'links'.")
    return graph


def networkx_setdefault_attrs(graph: nx.Graph, element: Literal["nodes", "edges", "links"], default: Any) -> nx.Graph:
    """Set default attribute values if not present in all nodes or edges.

    Args:
        graph: NetworkX graph.
        element: Element to set default attribute values; should be 'nodes' or 'edges'/'links'.
        default: Default value to set for missing attributes.

    Returns:
        Graph with zero-covered attributes.
    """
    match element:
        case "nodes":
            all_node_attrs = {node_attr for _, node_attrs in graph.nodes(data=True) for node_attr in node_attrs}
            for _, node_attrs in graph.nodes(data=True):
                for key in all_node_attrs:
                    node_attrs.setdefault(key, default)
        case "edges" | "links":
            all_edge_attrs = {edge_attr for _, _, edge_attrs in graph.edges(data=True) for edge_attr in edge_attrs}
            for _, _, edge_attrs in graph.edges(data=True):
                for key in all_edge_attrs:
                    edge_attrs.setdefault(key, default)
        case _:
            raise ValueError("Element must be 'nodes' or 'edges'/'links'.")
    return graph


def networkx_aggregate_list_attrs(
    graph: nx.Graph,
    agg_func: dict[str, list[Literal["sum", "max", "min", "mean"]]],
    element: Literal["nodes", "edges", "links"],
    remove_original: bool = False,
) -> nx.Graph:
    """Aggregate list-valued attributes on nodes or edges.

    Writes each result under a new key `<op>_<attr>`, and (optionally) deletes the original list-valued attribute.

    Args:
        graph: NetworkX graph whose nodes or edges hold list-valued attrs.
        agg_func: Mapping from attribute name to aggregation operators to apply.
        element: Elements on which to aggregate attribute values: should be 'nodes' or 'edges'/'links'.
        remove_original: If True, drop the original list-valued attribute after aggregation.

    Returns:
        The same graph, mutated in-place with new scalar attributes.

    Raises:
        ValueError: if `element` is not one of "nodes" or "edges"/"links".
        NotImplementedError: if any op in `agg_func` is not in {"sum", "max", "min", "mean"}.
    """
    if element not in ("nodes", "edges", "links"):
        raise ValueError("`element` must be either 'nodes' or 'edges'/'links'.")

    # Supported operations
    for attr, ops in agg_func.items():
        # Re-create iterator for each attribute
        items = graph.nodes(data=True) if element == "nodes" else graph.edges(data=True)
        for *_, data in items:
            attr_vals = data.get(attr, [])

            for op in ops:
                match op:
                    case "sum":
                        v = np.nansum(attr_vals)
                    case "max":
                        v = np.nanmax(attr_vals)
                    case "min":
                        v = np.nanmin(attr_vals)
                    case "mean":
                        v = np.nanmean(attr_vals)
                    case _:
                        raise NotImplementedError(f"Unsupported aggregation operation on '{attr}': {op}")

                data[f"{op}_{attr}"] = v

            if remove_original:
                data.pop(attr, None)

    return graph


def networkx_find_root(graph: nx.DiGraph) -> Any:
    """Find the unique root node (in-degree == 0) in a directed tree.

    Args:
        graph: A directed acyclic graph representing an arborescence where each node has in-degree ≤ 1 and the
            underlying undirected graph is connected.

    Returns:
        Any: The root node of the tree (the only node with in-degree 0).

    Raises:
        ValueError: If no node with in-degree 0 is found.
        ValueError: If more than one node with in-degree 0 is found.
    """
    roots = [node for node, deg in graph.in_degree() if deg == 0]
    if not roots:
        raise ValueError("No root found: graph has no node with in-degree 0.")
    if len(roots) > 1:
        raise ValueError(f"Multiple roots found: {roots}")
    return roots[0]


def _has_node_attributes(graph: nx.Graph) -> bool:
    """Check if a NetworkX graph has node attributes."""
    return any(feats for _, feats in graph.nodes(data=True))


def _has_edge_attributes(graph: nx.Graph) -> bool:
    """Check if the NetworkX Graph has edge attributes."""
    return any(feats for _, _, feats in graph.edges(data=True))


def _clean_data_attributes(data: Data) -> Data:
    """Remove all attributes from PyG `Data` object except 'x', 'y', 'edge_index', and 'edge_attr'.

    Args:
        data: PyG graph.

    Returns:
        PyG `Data` object cleaned from non-essential attributes
    """
    for key in data.keys():  # noqa: SIM118
        if key not in ["x", "y", "edge_index", "edge_attr"]:
            delattr(data, key)
    return data
