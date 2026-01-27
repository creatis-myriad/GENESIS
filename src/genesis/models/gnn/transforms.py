from abc import ABC

import torch
from rtdl_revisiting_models import CategoricalEmbeddings
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


class AtomBondEmbedding(LearnableTransform):
    """Encode categorical atom and bond features describing molecular graphs using embedding layers."""

    def __init__(
        self,
        atom_cardinalities: list[int],
        atom_embed_dim: int,
        bond_cardinalities: list[int],
        bond_embed_dim: int,
    ) -> None:
        """Initializes an `AtomBondEmbedding` instance.

        Args:
            atom_cardinalities: Cardinalities of each categorical atom feature.
            atom_embed_dim: Embedding dimension for atom (i.e. node) features.
            bond_cardinalities: Cardinalities of each categorical bond feature.
            bond_embed_dim: Embedding dimension for bond (i.e. edge) features.
        """
        super().__init__()

        self.atom_num_features = len(atom_cardinalities)
        self.bond_num_features = len(bond_cardinalities)
        self.atom_encoder = CategoricalEmbeddings(atom_cardinalities, atom_embed_dim, bias=False)
        self.bond_encoder = CategoricalEmbeddings(bond_cardinalities, bond_embed_dim, bias=False)

    def forward(self, data: Data) -> Data:
        """Encodes categorical atom and bond features in the provided graph data object.

        Args:
            data: The input graph data object.

        Returns:
            Graph data object with categorical atom and bond features to continuous embeddings.
        """
        # Extract categorical features from atoms (i.e. nodes) and bonds (i.e. edges),
        # ensuring they are in long format to be used as indices for embedding lookup
        atom_features = data.x[..., : self.atom_num_features].long()
        bond_features = data.edge_attr[..., : self.bond_num_features].long()

        # Encode categorical features, summing over the features dimensions to get a single embedding per atom/bond
        # e.g. (*, len(atom_cardinalities), embed_dim) -> (*, embed_dim)
        atom_encodings = self.atom_encoder(atom_features).sum(dim=-2)
        bond_encodings = self.bond_encoder(bond_features).sum(dim=-2)

        # Concatenate encodings to leftover non-encoded features (e.g. positional encoding)
        leftover_node_features = data.x[..., self.atom_num_features :]
        data.x = torch.cat([atom_encodings, leftover_node_features], dim=-1)
        leftover_edge_features = data.edge_attr[..., self.bond_num_features :]
        data.edge_attr = torch.cat([bond_encodings, leftover_edge_features], dim=-1)

        return data
