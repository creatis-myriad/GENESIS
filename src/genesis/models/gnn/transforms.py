from abc import ABC

from torch import nn
from torch.nn import Linear
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform

from genesis.data.transforms.virtual_node import VirtualNodes as VirtualNodesPerGraph


class LearnableTransform(BaseTransform, nn.Module, ABC):
    """A base class for transformations with learnable parameters."""


class VirtualNodes(LearnableTransform):
    """Wrapper around our `VirtualNodes` to fit into the `LearnableTransform` API and apply to batched graphs.

    Inherits from `nn.Module` indirectly through `LearnableTransform`, even though it contains no learnable parameters,
    which enables chaining with other learnable transforms inside collections like `nn.Sequential` or `nn.ModuleDict`.
    """

    def __init__(self) -> None:
        """Initializes a `GraphAttributesVirtualNodes` instance."""
        super().__init__()
        self.virtual_nodes = VirtualNodesPerGraph()

    def forward(self, data: Data) -> Data:  # noqa: D102
        data_list = data.to_data_list() if (is_batch := isinstance(data, Batch)) else [data]
        new_data_list = [self._add_virtual_nodes_to_graph(graph) for graph in data_list]
        return Batch.from_data_list(new_data_list) if is_batch else new_data_list[0]

    def _add_virtual_nodes_to_graph(self, graph: Data) -> Data:
        """Adds virtual nodes to a single graph.

        Args:
            graph: The input graph to which to add virtual node(s).

        Returns:
            Input graph with virtual node(s) added.
        """
        return self.virtual_nodes.forward(graph)


class GraphAttributesVirtualNodes(VirtualNodes):
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
        self.graph_lin = Linear(graph_dim, embed_dim)

    def _add_virtual_nodes_to_graph(self, graph: Data) -> Data:
        # Project graph features to embedding dimension
        graph_attr_embedding = self.graph_lin(graph.graph_attr)

        # Add virtual node initialized with graph attribute embedding
        return self.virtual_nodes.forward(graph, init_values=graph_attr_embedding)
