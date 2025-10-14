from abc import ABC

from torch import nn
from torch.nn import Linear
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform

from genesis.data.transforms.virtual_node import VirtualNodes


class LearnableTransform(BaseTransform, nn.Module, ABC):
    """A base class for transformations with learnable parameters."""


class GraphAttributesVirtualNodes(LearnableTransform):
    """Appends a virtual node (i.e. connected to all nodes) initialized with an embedding learned from graph attributes.

    This implementation was made to support multimodal data (i.e. graph attributes) as input features of virtual nodes.
    """

    def __init__(self, graph_dim: int, embed_dim: int) -> None:
        """Initializes a `GraphAttributesVirtualNodes` instance.

        Args:
            graph_dim: Number of input graph features.
            embed_dim: Dimension to embed the graph features to. Should match the node dimensionality (after transforms
                like positional encoding have been applied).
        """
        super().__init__()
        self.virtual_nodes = VirtualNodes()
        self.graph_lin = Linear(graph_dim, embed_dim)

    def forward(self, data: Data) -> Data:  # noqa: D102
        data_list = data.to_data_list() if (is_batch := isinstance(data, Batch)) else [data]

        new_data_list = []
        for graph in data_list:
            # Project graph features to embedding dimension
            graph_attr_embedding = self.graph_lin(graph.graph_attr)

            # Add virtual node initialized with graph attribute embedding
            graph_with_vn = self.virtual_nodes.forward(graph, init_values=graph_attr_embedding)
            new_data_list.append(graph_with_vn)

        return Batch.from_data_list(new_data_list) if is_batch else new_data_list[0]
