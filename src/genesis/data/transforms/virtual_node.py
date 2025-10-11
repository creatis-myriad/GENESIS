import copy
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class VirtualNodes(BaseTransform):
    """Appends virtual node(s) (i.e. connected to all nodes) with optional initialization, otherwise zero-filled.

    Inspired by the `VirtualNode` transform from the PyTorch Geometric library (see link below), which hard-codes a
    zero-filled virtual node to the graph. This implementation was made to make the virtual node more flexible, by
    allowing multiple virtual nodes and non-zero initial features.

    References:
        - PyTorch Geometric `VirtualNode` transform:
          https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.VirtualNode.html
    """

    def __init__(self, num: int | None = None) -> None:
        """Initializes a `VirtualNodes` instance.

        Args:
            num: Number of virtual nodes to add to each graph. If `None` and no initial features are provided when
                calling `forward`, one virtual node will be added. Must be at least 1 if specified.
        """
        super().__init__()
        self.num = num

    def forward(self, data: Data, init_values: Tensor = None) -> Any:
        """Appends virtual node(s) (i.e. connected to all nodes) with optional initialization, otherwise zero-filled.

        Args:
            data: The input graph to which to add virtual node(s).
            init_values: ([`num`,] `features`), Optional initial features of the virtual node(s) to add.
                If `None`, virtual node will be zero-filled.
        """
        if init_values is None:
            num_virtual_nodes = self.num or 1  # Default to adding one virtual node if no initial features are provided
        else:
            if init_values.ndim == 1:
                init_values = init_values.unsqueeze(0)  # Prepend batch dim if one unbatched virtual node is provided
            if self.num is not None and len(init_values) != self.num:
                raise ValueError(
                    f"Number of provided virtual node initial features ({len(init_values)}) does not match the number "
                    f"of virtual nodes to add ({self.num})."
                )
            num_virtual_nodes = len(init_values)

        # Extract node and edge information (number, nodes connected by edges, types of edges, etc.)
        assert data.edge_index is not None
        edge_index = data.edge_index
        row, col = edge_index
        edge_type = data.get("edge_type", torch.zeros_like(row))
        num_nodes = data.num_nodes
        assert num_nodes is not None

        # Add edges between the virtual node and all other nodes in sparse COO format `edge_index`
        arange = torch.arange(num_nodes, device=row.device)
        for virtual_node_idx in range(num_nodes, num_nodes + num_virtual_nodes):
            # Connect all nodes to the current virtual node and vice versa
            full = edge_index.new_full((num_nodes,), virtual_node_idx)
            row = torch.cat([arange, full], dim=0)
            col = torch.cat([full, arange], dim=0)
            new_edges = torch.stack([row, col], dim=0)
            # Add the edges for the current virtual node to the overall edge_index
            edge_index = torch.cat([edge_index, new_edges], dim=1)
            # TOIMPROVE: Sort edges by source node to keep full compatibility with PyG operations requiring sorted edges
            # e.g. `sort_edge_index(edge_index)`, but should take into account `edge_attr` and `edge_type` if present

        # Add new virtual edge type (for edges connected to virtual nodes)
        num_edge_types = int(edge_type.max()) if edge_type.numel() > 0 else 0
        # Create a pair of new edge types in/out of virtual nodes (e.g. [1, 2])
        new_types = edge_type.new_tensor([num_edge_types + 1, num_edge_types + 2])
        # Broadcast the pair of new edge types to all edges connected to virtual nodes
        # (e.g. [1, ..., 1, 2, ..., 2] of length 2 * num_nodes)
        new_types = torch.repeat_interleave(new_types, num_nodes)
        # Repeat the above pattern for each virtual node
        # (e.g. [1, ..., 1, 2, ..., 2, 1, ..., 2] for two virtual nodes, of length 2 * num_nodes * num_virtual_nodes)
        new_types = new_types.repeat(num_virtual_nodes)
        edge_type = torch.cat([edge_type, new_types], dim=0)

        old_data = copy.copy(data)
        for key, value in old_data.items():
            # updating these attributes is handled above
            if key == "edge_index" or key == "edge_type":
                continue

            if isinstance(value, Tensor):
                dim = old_data.__cat_dim__(key, value)  # Dimension along which to concatenate the data attribute
                size = list(value.size())

                new_value = None
                if key == "edge_weight":
                    # Each virtual node connects to all other nodes, so 2 * num_nodes new edge weights are added for
                    # each virtual node, with default value of 1
                    size[dim] = 2 * num_nodes * num_virtual_nodes
                    new_value = value.new_ones(size)
                elif old_data.is_edge_attr(key):
                    # Each virtual node connects to all other nodes, so 2 * num_nodes new edge attributes are added for
                    # each virtual node, with default value of 0
                    size[dim] = 2 * num_nodes * num_virtual_nodes
                    new_value = value.new_zeros(size)
                elif key == "batch":
                    # Assign the same batch index as the first node in the graph to all virtual nodes
                    size[dim] = num_virtual_nodes
                    new_value = value.new_full(size, int(value[0]))
                elif old_data.is_node_attr(key):
                    # Add initial virtual node features if provided, num_virtual_nodes zero-filled nodes are added
                    if init_values is not None:
                        if list(init_values.size())[1:] != size[1:]:
                            raise ValueError(
                                f"Dimensionality of virtual nodes initial features ({list(init_values.size())[1:]}) "
                                f"does not match the node feature dimensionality ({size[1:]})."
                            )
                        new_value = init_values
                    else:
                        size[dim] = num_virtual_nodes
                        new_value = value.new_zeros(size)

                if new_value is not None:
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        if "num_nodes" in data:
            data.num_nodes = num_nodes + num_virtual_nodes

        return data
