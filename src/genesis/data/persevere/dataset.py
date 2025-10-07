from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import InMemoryDataset

from genesis.data.utils import json_to_pyg


class PersevereDataset(InMemoryDataset):
    """PyG dataset for the PERSEVERE dataset, which contains multiple graphs of pulmonary vessel trees.

    The dataset is stored as JSON files (1 per graph) implementing the node-link format of NetworkX. See
    https://networkx.org/documentation/stable/reference/readwrite/json_graph.html
    for more information about the JSON data format.

    This dataset reconstructs the graphs from the JSON files, converts them to PyG Data objects, and groups them in a
    single dataset.

    We bypass saving a processed version of the dataset to disk, instead processing the raw JSON on-the-fly when
    initializing the dataset object. This is done to allow loading the dataset in parallel with different parameters
    (e.g. targets, graph attributes), which would cause conflicting processed files if done using a standard
    `InMemoryDataset`. This is feasible because loading the scale of JSON files used by PERSEVERE is inexpensive.
    """

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        line_graph: bool = True,
        target_attr: str = "risk_ESC-2014",
        target_dtype: str | torch.dtype = torch.long,
        node_attrs_filter: list[str] | None = None,
        edge_attrs_filter: list[str] | None = None,
        graph_attrs_filter: list[str] | None = None,
        json_to_nx_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes a `PersevereDataset`.

        Args:
            root: Root directory where the dataset is saved.
            transform: PyG data transform, applies on-access transformation without altering stored data.
            line_graph: Whether to convert graphs to their line graphs.
            target_attr: Key of the graph attribute to use as target.
            target_dtype: Data type of the target attribute.
            node_attrs_filter: List of node features to keep in the `Data` objects, from all available node features.
                If `None`, defaults to keeping all node features.
            edge_attrs_filter: List of edge features to keep in the `Data` objects, from all available edge features.
                If `None`, defaults to keeping all edge features.
            graph_attrs_filter: List of graph features to keep in the `Data` objects, from all available graph features.
                If `None`, defaults to keeping all graph features (except the target attribute).
            json_to_nx_kwargs: Keys for serialized attribute names to pass to `nx.node_link_graph`.
        """
        super().__init__(root, transform)

        self._line_graph = line_graph
        self._target_attr = target_attr
        self._target_dtype = target_dtype
        self._json_to_nx_kwargs = json_to_nx_kwargs

        self._nx_to_pyg_kwargs = {
            "group_node_attrs": edge_attrs_filter if line_graph else node_attrs_filter,
            "group_edge_attrs": node_attrs_filter if line_graph else edge_attrs_filter,
            "group_graph_attrs": graph_attrs_filter,
        }

        data_list = [
            json_to_pyg(
                json_path,
                self._target_attr,
                self._target_dtype,
                line_graph=self._line_graph,
                json_to_nx_kwargs=self._json_to_nx_kwargs,
                nx_to_pyg_kwargs=self._nx_to_pyg_kwargs,
            )
            # NOTE: Sort the JSON files to ensure a deterministic order of the graphs in the dataset
            #  necessary to guarantee reproducibility of training/validation/test splits.
            for json_path in sorted(Path(self.raw_dir).glob("*.json"))
        ]
        self.data, self.slices = self.collate(data_list)
