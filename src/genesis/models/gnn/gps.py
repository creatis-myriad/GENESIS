import itertools
from typing import Any, Final, Literal

import torch
from torch.nn import BatchNorm1d, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GINConv, GINEConv, GPSConv, MessagePassing
from torch_geometric.typing import Adj

from genesis.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class GPS(torch.nn.Module):
    """Implementation of the General, Powerful, Scalable (GPS) graph transformer model.

    Inspired by an example from the PyTorch Geometric library (see link below), adapted to be simpler and more
    configurable under the default configuration of `GINEConv` message-passing layer and multihead attention.

    Also adapted to be a more generic encoder (as the reference impl. focused on graph-level tasks). To add readout and
    a prediction head for graph-level tasks, use as the encoder of a `genesis.models.GraphLevelLitModule`.

    References:
        - Model introduced by the "Recipe for a General, Powerful, Scalable Graph Transformer" paper:
          https://arxiv.org/abs/2205.12454
        - Original PyTorch Geometric example:
          https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
    """

    # Add indicators of supported params in forward pass, for compatibility with PyG's `BasicGNN`
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool]
    supports_norm_batch: Final[bool]
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

        self.pe_norm = BatchNorm1d(pe_dim)
        self.pe_lin = Linear(pe_dim, pe_embed_dim)
        self.node_lin = Linear(in_channels, hidden_channels - pe_embed_dim)
        self.edge_lin = Linear(edge_dim, hidden_channels) if edge_dim else None

        self._mpnn_type = mpnn_type
        self._mpnn_kwargs = mpnn_kwargs or {}

        self.convs = ModuleList()
        layers_channels = ([hidden_channels] * num_layers) + [out_channels or hidden_channels]
        for layer_in, layer_out in itertools.pairwise(layers_channels):
            conv = self.init_conv(layer_in, layer_out, **(gps_kwargs or {}))
            self.convs.append(conv)

        self.supports_edge_attr = bool(edge_dim)
        self.supports_norm_batch = self.convs[0].norm_with_batch

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> torch.nn.Module:  # noqa: D102
        return GPSConv(in_channels, self.init_mpnn(in_channels, out_channels, **self._mpnn_kwargs), **kwargs)

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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        pe: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x: Node features of shape `[num_nodes, in_channels]`.
            edge_index: Edge indices.
            pe: Positional encodings of shape `[num_nodes, pe_in_channels]`.
            batch: Batch vector assigning each element to a specific graph of shape `[num_nodes]`.
            edge_attr: Edge features of shape `[num_edges, edge_in_channels]`, if any.
            **kwargs: Additional keyword arguments to pass to the `GPSConv` layers.
        """
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_lin(x), self.pe_lin(x_pe)), 1)
        if edge_attr is not None:
            assert self.supports_edge_attr
            # Pass `edge_attr` to MPNN layer only if supported, since otherwise layer won't expect an `edge_attr` kwarg
            kwargs["edge_attr"] = self.edge_lin(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, **kwargs)
        return x
