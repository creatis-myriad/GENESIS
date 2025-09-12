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
    """

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
        line_graph: bool = True,
        target_attr: str = "risk_ESC-2014",
        target_dtype: str | torch.dtype = torch.long,
        node_attrs_filter: list[str] | None = None,
        edge_attrs_filter: list[str] | None = None,
        json_to_nx_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes a `PersevereDataset`.

        Args:
            root: Root directory where the dataset is saved.
            transform: PyG data transform, applies on-access transformation without altering stored data.
            pre_transform: PyG data pre-transform, applies transformation before storing.
            pre_filter: PyG data pre-filter, filters data before storing.
            force_reload: PyG force reload, forces reprocessing to update pre_transform/filter changes.
            line_graph: Whether to convert graphs to their line graphs.
            target_attr: Key of the graph attribute to use as target.
            target_dtype: Data type of the target attribute.
            node_attrs_filter: List of node features to keep in the `Data` objects, from all available node features.
                If `None`, defaults to keeping all node features.
            edge_attrs_filter: List of edge features to keep in the `Data` objects, from all available edge features.
                If `None`, defaults to keeping all edge features.
            json_to_nx_kwargs: Keys for serialized attribute names to pass to `nx.node_link_graph`.
        """
        self._line_graph = line_graph
        self._target_attr = target_attr
        self._target_dtype = target_dtype
        self._json_to_nx_kwargs = json_to_nx_kwargs

        self._nx_to_pyg_kwargs = {
            "group_node_attrs": edge_attrs_filter if line_graph else node_attrs_filter,
            "group_edge_attrs": node_attrs_filter if line_graph else edge_attrs_filter,
        }

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        """By default, processed data is saved in 'data.pt'."""
        return "data.pt"

    def process(self) -> None:
        """Process the raw data and save it to data.pt."""
        data_list = [
            json_to_pyg(
                json_path,
                self._target_attr,
                self._target_dtype,
                line_graph=self._line_graph,
                json_to_nx_kwargs=self._json_to_nx_kwargs,
                nx_to_pyg_kwargs=self._nx_to_pyg_kwargs,
            )
            for json_path in Path(self.raw_dir).glob("*.json")
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
