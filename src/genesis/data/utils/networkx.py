import warnings
from typing import Any, Literal

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from genesis.data.data import GraphAttrData
from genesis.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def networkx_to_pyg(
    graph: nx.Graph, target_attr: str | None = None, target_dtype: torch.dtype | None = None, **from_networkx_kwargs
) -> Data:
    """Convert NetworkX `Graph` to PyG `Data`.

    Args:
        graph: NetworkX graph.
        target_attr: Key of the graph attribute to use as target. If `None`, `data.y` is not set.
        target_dtype: Data type of the target attribute. If `None`, `data.y` uses the default tensor dtype.
        **from_networkx_kwargs: Node and edge filter lists to pass to `from_networkx`.

    Returns:
        PyG `Data` representation of the NetworkX `Graph`.
    """
    if networkx_has_attributes(graph, "nodes") and from_networkx_kwargs.get("group_node_attrs") is None:
        from_networkx_kwargs["group_node_attrs"] = "all"
    if networkx_has_attributes(graph, "edges") and from_networkx_kwargs.get("group_edge_attrs") is None:
        from_networkx_kwargs["group_edge_attrs"] = "all"

    # Catch `group_graph_attrs`, as it is not supported by `from_networkx` and we mimic the expected behavior here
    group_graph_attrs = from_networkx_kwargs.pop("group_graph_attrs", None)

    # Don't manage string attributes
    data = from_networkx(graph, **from_networkx_kwargs)
    if target_attr:
        target_label = graph.graph[target_attr]
        data.y = torch.tensor(target_label, dtype=target_dtype)

    # Manually group graph attributes automatically added by `from_networkx` into a single `graph_attr` attr
    if group_graph_attrs is None:
        # By default, group all graph attributes except the target attribute
        group_graph_attrs = graph.graph.keys() - {target_attr}
    if group_graph_attrs:
        # If graph attributes are present, convert/cast PyG `Data` to `GraphAttrData` to handle batching properly
        data = GraphAttrData.from_dict(data.to_dict())
        data.graph_attr = torch.stack([getattr(data, graph_attr) for graph_attr in group_graph_attrs])

    # After having grouped requested graph attributes, remove leftover graph attributes from `data`
    return _clean_data_attributes(data)


def networkx_line_graph(graph: nx.Graph) -> nx.Graph:
    """Convert NetworkX `Graph` to its line graph.

    Only transpose original edge features to line graph nodes, as edge features are not used in most GNNs.

    Args:
        graph: Original graph.

    Returns:
        Line graph.
    """
    line_graph = nx.line_graph(graph)
    line_graph.graph.update(graph.graph)
    for source, target, feats in graph.edges(data=True):
        line_graph.nodes[(source, target)].update(feats)
    return line_graph


def networkx_add_attrs[G: nx.Graph](
    graph: G, element: Literal["graph", "nodes", "edges", "links"], attrs: dict[str, Any], in_place: bool = False
) -> G:
    """Add attributes to the graph, nodes, or edges.

    Args:
        graph: NetworkX graph.
        element: Element to add attributes to; should be 'graph', 'nodes', or 'edges'/'links'.
        attrs: Attributes to add.
        in_place: If True, modify the input graph in place. Otherwise, return a modified copy of the graph.

    Returns:
        Graph with added attributes.
    """
    if not in_place:
        graph = graph.copy()

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


def networkx_remove_attrs[G: nx.Graph](
    graph: G, element: Literal["graph", "nodes", "edges", "links"], attrs: list[str], in_place: bool = False
) -> G:
    """Remove attributes from the graph, nodes, or edges.

    Args:
        graph: NetworkX graph.
        element: Element to remove attributes from; should be 'graph', 'nodes', or 'edges'/'links'.
        attrs: Attributes to remove.
        in_place: If True, modify the input graph in place. Otherwise, return a modified copy of the graph.

    Returns:
        Graph with removed attributes.
    """
    if not in_place:
        graph = graph.copy()

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


def networkx_setdefault_attrs[G: nx.Graph](
    graph: G, element: Literal["nodes", "edges", "links"], default: Any, in_place: bool = False
) -> G:
    """Set default attribute values if not present in all nodes or edges.

    Args:
        graph: NetworkX graph.
        element: Element to set default attribute values; should be 'nodes' or 'edges'/'links'.
        default: Default value to set for missing attributes.
        in_place: If True, modify the input graph in place. Otherwise, return a modified copy of the graph.

    Returns:
        Graph with missing attributes set to the default value.
    """
    if not in_place:
        graph = graph.copy()

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


def networkx_aggregate_attrs[G: nx.Graph](
    graph: G,
    agg_func: dict[str, Literal["sum", "max", "min", "mean"] | list[Literal["sum", "max", "min", "mean"]]],
    element: Literal["nodes", "edges", "links"],
    nan_default: Any = 0,
    remove_original: bool = False,
    in_place: bool = False,
) -> G:
    """Aggregate multivalued attributes on nodes or edges.

    Writes each result under a new key `<op>_<attr>`, and (optionally) deletes the original multivalued attribute.

    Args:
        graph: NetworkX graph whose nodes or edges hold multivalued attrs.
        agg_func: Mapping from attribute name to aggregation operator(s) to apply.
        element: Elements on which to aggregate attribute values: should be 'nodes' or 'edges'/'links'.
        nan_default: Value to use if all values are NaN during aggregation.
        remove_original: If True, drop the original multivalued attribute after aggregation.
        in_place: If True, modify the input graph in place. Otherwise, return a modified copy of the graph.

    Returns:
        Graph with new scalar attributes, aggregated from multivalued ones.

    Raises:
        ValueError: if `element` is not one of "nodes" or "edges"/"links".
        NotImplementedError: if any op in `agg_func` is not in {"sum", "max", "min", "mean"}.
    """
    if element not in ("nodes", "edges", "links"):
        raise ValueError("`element` must be either 'nodes' or 'edges'/'links'.")

    if not in_place:
        graph = graph.copy()

    # Supported operations
    for attr, ops in agg_func.items():
        if isinstance(ops, str):  # If a single op is provided for the attribute, wrap it in a list
            ops = [ops]  # noqa: PLW2901

        # Re-create iterator for each attribute
        items = graph.nodes(data=True) if element == "nodes" else graph.edges(data=True)
        for *elem_key, data in items:
            attr_data = data[attr]
            if isinstance(attr_data, dict):
                attr_data = attr_data.values()

            for op in ops:
                # Handle warning from np.nan* functions when all values are NaN
                # to silence default RuntimeWarning and give a custom warning message instead
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", r"All-NaN (slice|axis) encountered")
                    try:
                        match op:
                            case "sum":
                                v = np.nansum(attr_data)
                            case "max":
                                v = np.nanmax(attr_data)
                            case "min":
                                v = np.nanmin(attr_data)
                            case "mean":
                                v = np.nanmean(attr_data)
                            case _:
                                raise NotImplementedError(f"Unsupported aggregation operation on '{attr}': {op}")
                    except Warning:
                        match element:
                            case "nodes":
                                elem_desc = f"node {elem_key}"
                            case "edges" | "links":
                                elem_desc = f"edge between nodes {' and '.join(map(str, elem_key))}"

                        log.warning(
                            f"Attribute '{attr}' on {elem_desc} was all-NaN; "
                            f"default to setting '{attr}_{op}' to {nan_default}."
                        )
                        v = nan_default

                data[f"{attr}_{op}"] = v

            if remove_original:
                data.pop(attr)

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


def networkx_has_attributes(
    graph: nx.Graph, element: Literal["nodes", "edges", "links"], attrs: list[str] | None = None
) -> bool:
    """Check if the NetworkX Graph has any edge attributes, or the requested ones if specified.

    Args:
        graph: NetworkX graph whose elements to check for attributes.
        element: Element to check for attributes; should be 'nodes' or 'edges'/'links'.
        attrs: List of attribute names to check for. If None, check if attributes are present on any element.

    Returns:
        True if the graph has the requested attributes (or any) on the specified elements, False otherwise.

    Raises:
        ValueError: if `element` is not one of "nodes" or "edges"/"links".
    """
    if element not in ("nodes", "edges", "links"):
        raise ValueError("`element` must be either 'nodes' or 'edges'/'links'.")

    elems = graph.nodes(data=True) if element == "nodes" else graph.edges(data=True)
    if attrs:
        # If attrs are requested, return True if all attrs are present on all elements
        return all(set(attrs) <= set(elem_attrs) for *_, elem_attrs in elems)
    # If no attrs are requested, return True if any attribute is present on any element
    return any(bool(elem_attrs) for *_, elem_attrs in elems)


def _clean_data_attributes(data: Data) -> Data:
    """Remove all attributes from PyG `Data` object except 'x', 'y', 'edge_index', and 'edge_attr'.

    Args:
        data: PyG graph.

    Returns:
        PyG `Data` object cleaned from non-essential attributes
    """
    for key in data.keys():  # noqa: SIM118
        if key not in ["x", "y", "edge_index", "edge_attr", "graph_attr", "pos", "time", "num_nodes"]:
            delattr(data, key)
    return data
