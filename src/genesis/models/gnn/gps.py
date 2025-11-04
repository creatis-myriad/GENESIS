import copy
import itertools
from typing import Any, Final, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import BatchNorm1d, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GINConv, GINEConv, GPSConv, MessagePassing
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

from genesis.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class GPS(torch.nn.Module):
    """Implementation of the General, Powerful, Scalable (GPS) graph transformer model.

    Inspired by an example from the PyTorch Geometric library (see link below), adapted to be simpler and more
    configurable under the default configuration of `GINEConv` message-passing layer and multihead attention.

    Notes:
        - If graph-level features are provided, this implementation uses our proposed GAGPSConv extension of the GPSConv
          layer, which adds support for Graph-level Attributes.
        - Also adapted to be a more generic encoder (as the reference impl. focused on graph-level tasks). To use for
          graph-level tasks, use as the encoder of a `genesis.models.GraphLevelLitModule`.

    References:
        - Model introduced by the "Recipe for a General, Powerful, Scalable Graph Transformer" paper:
          https://arxiv.org/abs/2205.12454
        - Original PyTorch Geometric example:
          https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
    """

    # Add indicators of supported params in forward pass, for compatibility with PyG's `BasicGNN`
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool]
    supports_graph_attr: Final[bool]
    supports_norm_batch: Final[bool] = False
    supports_pe: Final[bool] = True
    supports_batch: Final[bool] = True

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        pe_dim: int,
        pe_embed_dim: int,
        num_layers: int,
        edge_dim: int | None = None,
        graph_dim: int | None = None,
        out_channels: int | None = None,
        gps_kwargs: dict[str, Any] | None = None,
        mpnn_type: Literal["gine", "gatedgcn", "pna"] = "gine",
        mpnn_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes a `GPS` model.

        Args:
            in_channels: Number of input node features.
            hidden_channels: Number of features to use in the hidden layers. For the first layer, features will be a
                concatenation of embeddings of the positional encoding (`pe_embed_dim` channels) and input node features
                (rest of the channels).
            pe_dim: Number of input positional encoding features.
            pe_embed_dim: Number of features to embed the positional encodings to.
            num_layers: Number of `GPSConv` layers to use.
            edge_dim: Number of input edge features. If `None`, edge features are not used.
            graph_dim: Number of input graph features. If `None`, graph features are not used.
            out_channels: Number of output features. If `None`, the number of output features is equal to
                `hidden_channels`.
            gps_kwargs: Additional keyword arguments to pass to `GPSConv` layers.
            mpnn_type: Type of message-passing layer to use inside each `GPSConv`.
            mpnn_kwargs: Additional keyword arguments to pass to the message-passing layers inside each `GPSConv`.
        """
        super().__init__()

        if attn_type := gps_kwargs.get("attn_type"):
            # TOFIX: GPSConv supports only default "multihead" attention for now
            if attn_type == "performer":
                raise NotImplementedError("Performer attention is not implemented for GPS yet.")
            if attn_type != "multihead":
                raise ValueError(f"Unsupported attention type: {attn_type}.")

        if pe_embed_dim > hidden_channels:
            raise ValueError(
                f"The size of the positional encoding embedding ({pe_embed_dim=}) must be less than or equal to "
                f"the number of hidden channels ({hidden_channels=})."
            )

        self.supports_edge_attr = bool(edge_dim)
        self.supports_graph_attr = bool(graph_dim)

        self.pe_norm = BatchNorm1d(pe_dim)
        self.pe_lin = Linear(pe_dim, pe_embed_dim)
        self.node_lin = Linear(in_channels, hidden_channels - pe_embed_dim)
        self.edge_lin = Linear(edge_dim, hidden_channels) if edge_dim else None
        self.graph_lin = Linear(graph_dim, hidden_channels) if graph_dim else None

        self._mpnn_type = mpnn_type
        self._mpnn_kwargs = mpnn_kwargs or {}

        self.convs = ModuleList()
        layers_channels = ([hidden_channels] * num_layers) + [out_channels or hidden_channels]
        for layer_in, layer_out in itertools.pairwise(layers_channels):
            conv = self.init_conv(layer_in, layer_out, **(gps_kwargs or {}))
            self.convs.append(conv)

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> torch.nn.Module:  # noqa: D102
        conv_cls = GPSConv if not self.supports_graph_attr else GAGPSConv
        return conv_cls(in_channels, self.init_mpnn(in_channels, out_channels, **self._mpnn_kwargs), **kwargs)

    def init_mpnn(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:  # noqa: D102
        match self._mpnn_type:
            case "gine":
                nn = Sequential(
                    Linear(in_channels, out_channels),
                    ReLU(),
                    Linear(out_channels, out_channels),
                )
                if self.edge_lin is not None:
                    mpnn = GINEConv(nn, edge_dim=self.edge_lin.out_features, **kwargs)
                else:
                    log.info(
                        f"No edge features provided to {self.__class__.__name__}, but GINECONV ('gine') chosen as MPNN "
                        f"layer. Using GINConv instead, as it is the equivalent without edge features."
                    )
                    mpnn = GINConv(nn, **kwargs)

            case "gatedgcn":
                raise NotImplementedError("GatedGCN is not implemented for GPS yet.")

            case "pna":
                raise NotImplementedError("PNA is not implemented for GPS yet.")

            case _:
                raise ValueError(
                    f"Unsupported type for the MPNN layer: {self._mpnn_type}. "
                    f"Supported types are 'gine' and 'gatedgcn'."
                )

        return mpnn

    def reset_parameters(self) -> None:
        """Resets all learnable parameters of the module."""
        self.node_lin.reset_parameters()
        self.pe_lin.reset_parameters()
        self.pe_norm.reset_parameters()
        self.edge_lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.supports_edge_attr:
            self.edge_lin.reset_parameters()
        if self.supports_graph_attr:
            self.graph_lin.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        pe: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        graph_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x: Node features of shape `[num_nodes, in_channels]`.
            edge_index: Edge indices of shape `[2, num_edges]`.
            pe: Positional encodings of shape `[num_nodes, pe_in_channels]`.
            batch: Batch vector assigning each element to a specific graph of shape `[num_nodes]`.
            edge_attr: Edge features of shape `[num_edges, edge_in_channels]`, if any.
            graph_attr: Graph-level features of shape `[graph_dim]`, if any.
            **kwargs: Additional keyword arguments to pass to the `GPSConv` layers.

        Returns:
            Updated node features of shape `[num_nodes, out_channels or hidden_channels]`.
        """
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_lin(x), self.pe_lin(x_pe)), 1)
        if edge_attr is not None:
            assert self.supports_edge_attr
            # Pass `edge_attr` to hybrid MPNN/GT layer only if supported,
            # otherwise layer might not support an `edge_attr` kwarg
            kwargs["edge_attr"] = self.edge_lin(edge_attr)

        if graph_attr is not None:
            assert self.supports_graph_attr
            graph_attr = self.graph_lin(graph_attr)

        for conv in self.convs:
            if graph_attr is None:
                x = conv(x, edge_index, batch, **kwargs)
            else:
                # Pass `graph_attr` to hybrid MPNN/GT layer only if supported,
                # otherwise layer might not support a `graph_attr` kwarg
                # And if passed, update `graph_attr` at each layer
                x, graph_attr = conv(x, edge_index, batch, graph_attr, **kwargs)
        return x


class GAGPSConv(GPSConv):
    """Extension of the GPSConv layer to handle Graph-level Attributes.

    References:
        - Implementation copies parts of PyG's `GPSConv`:
          https://github.com/pyg-team/pytorch_geometric/blob/76ff9c2ce18c8cebf52122b57e2aeadce9793d10/torch_geometric/nn/conv/gps_conv.py#L20-L185
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        super().__init__(*args, **kwargs)

        if isinstance(self.attn, PerformerAttention):
            raise ValueError(
                "'PerformerAttention' is currently not supported for extension of GPSConv that supports graph-level "
                "attributes."
            )

        # Init attention layers for both cross-attention directions (graph to node / node to graph),
        # with a structure identical to GPSConv's self-attention (because the embedding size is the same)
        self.graph2node_attn = copy.deepcopy(self.attn)
        self.node2graph_attn = copy.deepcopy(self.attn)

        self.graph_attr_mlp = copy.deepcopy(self.mlp)
        self.norm4 = copy.deepcopy(self.norm1)
        self.norm5 = copy.deepcopy(self.norm1)

    def reset_parameters(self) -> None:
        """Resets all learnable parameters of the module."""
        super().reset_parameters()
        self.graph2node_attn._reset_parameters()
        self.node2graph_attn._reset_parameters()
        reset(self.graph_attr_mlp)
        if self.norm4 is not None:
            self.norm4.reset_parameters()
        if self.norm5 is not None:
            self.norm5.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        batch: torch.Tensor,
        graph_attr: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the forward pass of the module.

        The code was copied from `torch_geometric.nn.GPSConv`'s `forward` method, and only modified where indicated to
        include graph-level features in global self-attention.

        Notes:
            - `GPSConv` leaves the `batch` arg optional, but in practice it is required to properly group graphs as
              sequences of tokens for attention layers. Therefore, `batch` is made required here, even if it makes the
              signature not match that of the parent, to avoid wrongly applying attention.
        """
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:  # noqa: SIM108
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)

        ###############################################################################
        #                          Start custom code block                            #
        ###############################################################################

        # Add sequence dimension to graph-level token
        # (num_graphs, channels) -> (num_graphs, 1, channels)
        graph_attr = graph_attr.unsqueeze(1)

        # 1) Update graph-level token using cross-attention with node tokens
        h_graph_attr, _ = self.graph2node_attn(graph_attr, h, h, key_padding_mask=~mask, need_weights=False)

        # 2) Update node tokens using both self-attention and cross-attention with graph-level token
        h_self, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        # No mask is needed in the cross-attention below since padded tokens (in `h`) appear only in the queries.
        # Queries don't interact with each other, so padded queries don't affect the computations of non-padded queries.
        # See this PyTorch issue's comment: https://github.com/pytorch/pytorch/issues/34453#issuecomment-1955116737
        h_cross, _ = self.node2graph_attn(h, graph_attr, graph_attr, need_weights=False)
        h = h_self + h_cross  # Combine self-attention between nodes and cross-attention with graph token

        # Remove sequence dimension from graph-level tokens
        # i.e. (num_graphs, 1, channels) -> (num_graphs, channels)
        graph_attr = graph_attr.squeeze(1)
        h_graph_attr = h_graph_attr.squeeze(1)

        ###############################################################################
        #                           End custom code block                             #
        ###############################################################################

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:  # noqa: SIM108
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:  # noqa: SIM108
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        ###############################################################################
        #                          Start custom code block                            #
        ###############################################################################

        # 1) Update graph-level features token after attention layer

        # Since operations below manipulate graph-level features, i.e. one vector representation per graph,
        # use a batch vector for norm layers where each vector is assigned to a different graph
        graph_batch = torch.arange(len(graph_attr), device=graph_attr.device)

        h_graph_attr = F.dropout(h_graph_attr, p=self.dropout, training=self.training)
        h_graph_attr = h_graph_attr + graph_attr  # Residual connection.
        if self.norm4 is not None:
            if self.norm_with_batch:
                h_graph_attr = self.norm4(h_graph_attr, batch=graph_batch)
            else:
                h_graph_attr = self.norm4(h_graph_attr)

        h_graph_attr = h_graph_attr + self.graph_attr_mlp(h_graph_attr)
        if self.norm5 is not None:
            if self.norm_with_batch:
                h_graph_attr = self.norm5(h_graph_attr, batch=graph_batch)
            else:
                h_graph_attr = self.norm5(h_graph_attr)

        # 2) Additionally return updated graph-level features
        return out, h_graph_attr
