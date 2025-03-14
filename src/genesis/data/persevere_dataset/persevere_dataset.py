from collections.abc import Callable
from pathlib import Path

from torch_geometric.data import InMemoryDataset

from .utils.io import json_to_pyg


class PersevereDataset(InMemoryDataset):
    """PyG dataset for the PERSEVERE dataset."""

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
        line_graph: bool = True,
        target_key: str = "risk",
        target_num_classes: int = 3,
        node_attrs_filter: list | None = None,
        edge_attrs_filter: list | None = None,
        json_to_nx_kwargs: dict | None = None,
    ) -> None:
        """Initializes a `PersevereDataset`.

        :param root: Root directory
        :param transform: PyG data transform, applies on-access transformation without altering stored data
        :param pre_transform: PyG data pre-transform, applies transformation before storing
        :param pre_filter: PyG data pre-filter, filters data before storing
        :param force_reload: PyG force reload, forces reprocessing to update pre_transform/filter changes
        :param line_graph: Whether to convert graphs to their line graphs
        :param target_key: Key of the graph attribute to use as target
        :param target_num_classes: Number of classes of the target final tensor
        :param node_attrs_filter: List of node features to keep in the `Data` objects, from all available node features.
        If `None`, defaults to keeping all node features.
        :param edge_attrs_filter: List of edge features to keep in the `Data` objects, from all available edge features.
        If `None`, defaults to keeping all edge features.
        :param json_to_nx_kwargs: Keys for serialized attribute names to pass to `nx.node_link_graph`
        """
        self.__line_graph = line_graph
        self.__target_key = target_key
        self.__target_num_classes = target_num_classes
        self.__json_to_nx_kwargs = json_to_nx_kwargs

        self.__nx_to_pyg_kwargs = {
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
                self.__target_key,
                self.__target_num_classes,
                line_graph=self.__line_graph,
                json_to_nx_kwargs=self.__json_to_nx_kwargs,
                nx_to_pyg_kwargs=self.__nx_to_pyg_kwargs,
            )
            for json_path in Path(self.raw_dir).glob("*.json")
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
