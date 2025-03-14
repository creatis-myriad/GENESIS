import json
from pathlib import Path
from typing import Any

import networkx as nx
from torch_geometric.data import Data

from .networkx import (
    networkx_line_graph,
    networkx_to_torch_geometric,
)


def json_to_pyg(
    json_path: Path,
    target_key: str,
    target_num_classes: int,
    line_graph: bool = False,
    json_to_nx_kwargs: dict[str, Any] = None,
    nx_to_pyg_kwargs: dict[str, Any] = None,
) -> Data:
    """Parses JSON file into a PyG Data object.

    :param json_path: File path to read as PyG Data object
    :param target_key: Key of the graph attribute to use as target
    :param target_num_classes: Number of classes of the target final tensor
    :param line_graph: Whether to convert the parsed graph to its line graph
    :param json_to_nx_kwargs: Keys for serialized attribute names to pass to `nx.node_link_graph`
    :param nx_to_pyg_kwargs: Node and edge features filters to pass to `from_networkx`
    :return: PyG Data object
    """
    if json_to_nx_kwargs is None:
        json_to_nx_kwargs = {}
    if nx_to_pyg_kwargs is None:
        nx_to_pyg_kwargs = {}
    nx_graph = json_graph_to_networkx(json_path, line_graph, **json_to_nx_kwargs)
    return networkx_to_torch_geometric(nx_graph, target_key, target_num_classes, **nx_to_pyg_kwargs)


def json_graph_to_networkx(json_path: Path, line_graph: bool = False, **node_link_graph_kwargs) -> nx.Graph:
    """Parses JSON file into a nx graph.

    :param json_path: File path to read as nx graph
    :param line_graph: Whether to convert the parsed graph to its line graph
    :param node_link_graph_kwargs: Keys for serialized attribute names to pass to `nx.node_link_graph`
    :return: nx graph
    """
    with open(json_path) as file:
        json_graph = json.load(file)

    # If no keys for serialized attribute names are provided,
    # use default keys + set edges key to avoid warning
    if not node_link_graph_kwargs:
        node_link_graph_kwargs = {"edges": "edges"}
    graph = nx.node_link_graph(json_graph, **node_link_graph_kwargs)
    if line_graph:
        graph = networkx_line_graph(graph)
    return graph
