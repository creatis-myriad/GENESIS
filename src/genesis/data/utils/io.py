import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from genesis.data.utils import networkx_line_graph, networkx_to_torch_geometric


def json_to_pyg(
    json_path: Path,
    target_attr: str,
    target_dtype: torch.dtype,
    line_graph: bool = False,
    json_to_nx_kwargs: dict[str, Any] | None = None,
    nx_to_pyg_kwargs: dict[str, Any] | None = None,
) -> Data:
    """Parse a JSON file into a PyG `Data`, representing a graph.

    Args:
        json_path: File path to read as PyG graph.
        target_attr: Key of the graph attribute to use as target.
        target_dtype: Data type of the target attribute.
        line_graph: Whether to convert the parsed graph to its line graph.
        json_to_nx_kwargs: Keys for serialized attribute names to pass to `nx.node_link_graph`.
        nx_to_pyg_kwargs: Node and edge features filters to pass to `pyg.utils.from_networkx`.

    Returns:
        PyG `Data` object loaded from the JSON file.
    """
    if json_to_nx_kwargs is None:
        json_to_nx_kwargs = {}
    if nx_to_pyg_kwargs is None:
        nx_to_pyg_kwargs = {}
    nx_graph = json_graph_to_networkx(json_path, line_graph, **json_to_nx_kwargs)
    return networkx_to_torch_geometric(nx_graph, target_attr, target_dtype, **nx_to_pyg_kwargs)


def json_graph_to_networkx(json_path: Path, line_graph: bool = False, **node_link_graph_kwargs) -> nx.Graph:
    """Parses JSON file into a NetworkX `Graph`.

    Args:
        json_path: File path to read as NetworkX graph.
        line_graph: Whether to convert the parsed graph to its line graph.
        **node_link_graph_kwargs: Keys for serialized attribute names to pass to `nx.node_link_graph`.

    Returns:
        NetworkX `Graph` loaded from the JSON file.
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


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy data types."""

    def default(self, obj: object) -> object:
        """Convert NumPy data types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(self).default(obj)
