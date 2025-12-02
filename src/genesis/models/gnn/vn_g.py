from collections.abc import Callable
from typing import Any, Final, Literal

import torch
import torch_geometric.nn as pyg_nn
from torch import Tensor, masked, nn
from torch_geometric.nn import MeanAggregation, SumAggregation
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import to_dense_batch


class VN_G_Mixin(nn.Module):  # noqa: N801
    """Mixin meant to work with children of PyG's `BasicGNN` to implement VN_G formulation of virtual nodes.

    References:
        - Implementation copies parts of PyG's `BasicGNN`:
          https://github.com/pyg-team/pytorch_geometric/blob/76ff9c2ce18c8cebf52122b57e2aeadce9793d10/torch_geometric/nn/models/basic_gnn.py#L32-L386
    """

    supports_batch: Final[bool] = True  # VN_G formulation requires to know the batch vector to compute global update
    supports_graph_attr: Final[bool]

    def __init__(
        self,
        *args,
        v2: bool = False,
        graph_dim: int | None = None,
        vn_g_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initializes the VN_G mixin.

        Args:
            *args: Additional positional arguments to pass to the class inheriting form `BasicGNN`.
            v2: If true, will make use of `VN_Gv2_MessagePassing` rather than `VN_G_MessagePassing`.
            graph_dim: Global features dimensionality (in case there are any). Only used in case `v2` is `True`, since
                only v2 supports global features.
            vn_g_kwargs: Additional keyword arguments to pass to the VN_G message passing layers' constructor.
            **kwargs: Additional keyword arguments to pass to the class inheriting form `BasicGNN`.
        """
        super().__init__(*args, **kwargs)

        self.v2 = v2
        self.supports_graph_attr = v2

        # Initialize message passing layers through the virtual node for each "regular" message passing layer
        # These layers never deal with `in_channels`, since even the 1st one comes after the 1st GNN layer,
        # which has already projected node features to `hidden_channels`
        vn_g_message_passing_cls = VN_G_MessagePassing if not v2 else VN_Gv2_MessagePassing
        self.vn_g_message_passing_layers = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.vn_g_message_passing_layers.append(
                vn_g_message_passing_cls(self.hidden_channels, self.hidden_channels, **vn_g_kwargs)
            )
        self.vn_g_message_passing_layers.append(
            vn_g_message_passing_cls(self.hidden_channels, self.out_channels, **vn_g_kwargs)
        )

        # Add an initial layer to project global features dimensionality to hidden dimensionality,
        # if global features are provided and supported by the model
        if graph_dim and self.supports_graph_attr:
            self.graph_attr_lin = pyg_nn.Linear(graph_dim, self.hidden_channels)

    def reset_parameters(self) -> None:
        """Resets all learnable parameters of the mixin and the main module."""
        super().reset_parameters()
        for vn_g_layer in self.vn_g_message_passing_layers:
            vn_g_layer.reset_parameters()
        if hasattr(self, "graph_attr_lin"):
            self.graph_attr_lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        graph_attr: OptTensor = None,
        batch_size: int | None = None,
        num_sampled_nodes_per_hop: list[int] | None = None,
        num_sampled_edges_per_hop: list[int] | None = None,
    ) -> Tensor:
        """Forward pass.

        The code was copied from `torch_geometric.nn.BasicGNN`'s `forward` method, and only modified for global message
        passing where indicated.

        References:
            - See documentation of the concrete implementation of PyG's `BasicGNN` for details regarding the arguments.
              e.g. GCN: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html#torch_geometric.nn.models.GCN.forward
        """
        if num_sampled_nodes_per_hop is not None and isinstance(edge_weight, Tensor) and isinstance(edge_attr, Tensor):
            raise NotImplementedError(
                "'trim_to_layer' functionality does not yet support trimming of both 'edge_weight' and 'edge_attr'"
            )

        #######################################################################################
        # Added steps to support global features in VN_Gv2:                                   #
        # IF global features are provided:                                                    #
        #   Initialize global state by projecting global features them to node dimensionality #
        # ELSE                                                                                #
        #   Initialize global state with zeros                                                #
        #######################################################################################
        if hasattr(self, "graph_attr_lin"):
            if graph_attr is None:
                raise ValueError(
                    f"{self.__class__.__name__} has been configured to expect global features, but no "
                    f"`graph_attr` has been provided to the forward pass."
                )
            x_global = self.graph_attr_lin(graph_attr)  # (num_graphs, num_graph_features -> node_channels)
        elif self.v2:
            num_graphs = torch.unique_consecutive(batch).numel()
            x_global = x.new_zeros((num_graphs, self.hidden_channels))  # (num_graphs, node_channels)
        #######################################################################################
        #                                  End custom code block                              #
        #######################################################################################

        xs: list[Tensor] = []
        assert len(self.convs) == len(self.norms)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=False)):
            if not torch.jit.is_scripting() and num_sampled_nodes_per_hop is not None:
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                x = norm(x, batch, batch_size) if self.supports_norm_batch else norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)

                ###############################################################################
                #                          Start custom code block                            #
                ###############################################################################
                if self.v2:
                    x, x_global = self.vn_g_message_passing_layers[i](x, x_global, batch)
                else:
                    x = self.vn_g_message_passing_layers[i](x, batch)
                ###############################################################################
                #                           End custom code block                             #
                ###############################################################################

                if hasattr(self, "jk"):
                    xs.append(x)

        ###############################################################################
        #                          Start custom code block                            #
        ###############################################################################
        # Normally, global message passing is done inside the layers loop, after non-linearities,
        # but before storing layer output for jumping knowledge pooling.
        # However, when jumping knowledge is not used, non-linearities are skipped for the last layer.
        # Therefore, when jumping knowledge is not used, global message passing has to be done outside the layers
        # loop for the last layer
        if self.jk_mode is None:
            if self.v2:
                x, _ = self.vn_g_message_passing_layers[self.num_layers - 1](x, x_global, batch)
            else:
                x = self.vn_g_message_passing_layers[self.num_layers - 1](x, batch)
        ###############################################################################
        #                           End custom code block                             #
        ###############################################################################

        x = self.jk(xs) if hasattr(self, "jk") else x
        x = self.lin(x) if hasattr(self, "lin") else x

        return x  # noqa: RET504


class VN_G_MessagePassing(nn.Module):  # noqa: N801
    """Implementation of the VN_G formulation of virtual nodes, as described in the original paper.

    References:
        - Inspired by the paper and published code from:
          "Understanding Virtual Nodes: Oversquashing and Node Heterogeneity": https://arxiv.org/abs/2405.13526
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        agg: Literal["mean", "sum"] = "mean",
        norm_weighting: bool = True,
        global_local_agg: Literal["mean", "sum"] = "sum",
    ) -> None:
        """Initializes the global message passing layer.

        Args:
            in_channels: Number of input channels for local node representations.
            out_channels: Number of output channels for node representations.
            agg: Aggregation to use across real nodes representations to compute global update, emulating message
                passing through a virtual node.
            norm_weighting: Whether to normalize global node representations based on graph degrees before combining
                them with local node representations.
            global_local_agg: Aggregation to use to combine global and local node representations.
        """
        super().__init__()

        # Initialize learnable weights to update nodes' local representations
        self.local_linear = pyg_nn.Linear(in_channels, out_channels)

        match agg:
            case "mean":
                self.agg = MeanAggregation()
            case "sum":
                self.agg = SumAggregation()
            case _:
                raise NotImplementedError(f"Invalid `agg` value: {agg}. Available options: 'mean', 'sum'.")

        match global_local_agg:
            case "mean":
                self.global_local_agg = torch.mean
            case "sum":
                self.global_local_agg = torch.sum
            case _:
                raise NotImplementedError(
                    f"Invalid `global_local_agg` value '{global_local_agg}'. Available options: 'mean', 'sum'."
                )

        self.norm_weighting = norm_weighting

    def reset_parameters(self) -> None:
        """Resets all learnable parameters of the module."""
        self.local_linear.reset_parameters()
        self.agg.reset_parameters()

    def forward(self, x_local: Tensor, batch: Tensor) -> Tensor:
        """Perform global message passing through the virtual node to compute updated node representations.

        Args:
            x_local: (num_nodes, node_channels), Local node representations, after the local message passing update.
            batch: (num_nodes,), The batch vector which assigns each element to a specific graph.

        Returns:
            (num_nodes, node_channels), Updated node representations.
        """
        ################################ STEP 1 ################################
        # GLOBAL MESSAGE PASSING AND AGGREGATION OF LOCAL NODE REPRESENTATIONS #
        ########################################################################
        # Linear transformation of updated local representations
        x_local = self.local_linear(x_local)  # (num_nodes, node_channels)
        # Global aggregation to emulate message passing through a virtual node connected to all other nodes
        x_global = self.agg(x_local, index=batch)  # (num_nodes -> num_graphs, node_channels)
        # Normalization of global representations based on graph degree
        if self.norm_weighting:
            _, graph_degrees = torch.unique_consecutive(batch, return_counts=True)  # (num_graphs,)
            deg_inv_sqrt = torch.pow(graph_degrees, -0.5)  # (num_graphs,)
            x_global = deg_inv_sqrt.unsqueeze(-1) * x_global  # (num_graphs, node_channels)

        ############################## STEP 2 ##############################
        # AGGREGATE LOCAL AND GLOBAL UPDATES FOR FINAL NODE REPRESENTATION #
        ####################################################################
        # Index by batch vector to broadcast global update per graph to all the nodes in their respective graphs
        x_global = x_global[batch]  # (num_graphs -> num_nodes, node_channels)
        x = torch.stack([x_local, x_global])  # (2, num_nodes, node_channels)
        return self.global_local_agg(x, dim=0)  # (2 -> 0, num_nodes, node_channels)


class VN_Gv2_MessagePassing(VN_G_MessagePassing):  # noqa: N801
    """Update to the VN_G formulation of virtual nodes, with explicit VN and support for global features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *args,
        act: str | Callable | None = "relu",
        act_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """Initializes the global message passing layer.

        Args:
            in_channels: Number of input channels for local node representations.
            out_channels: Number of output channels for node representations.
            *args: Additional arguments passed to `VN_G_MessagePassing`.
            act: Activation function to apply to global node representations before combining them with local node
                representations.
            act_kwargs: Keyword arguments to pass to the activation function defined by `act`.
            dropout: Dropout to apply to global node representations before combining them with local node
                representations.
            **kwargs: Additional keyword arguments passed to `VN_G_MessagePassing`.
        """
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.dropout = nn.Dropout(dropout)

        # Override the aggregation function defined in the parent, to use one that works on data in masked dense format
        # We have to first explicitly delete the aggregation module set by the parent, to avoid an error of the type:
        # 'TypeError("cannot assign <FUNCTION> as child module 'agg' (torch.nn.Module or None expected)")'
        del self.agg
        match agg := kwargs.get("agg", "mean"):
            case "mean":
                self.agg = masked.mean
            case "sum":
                self.agg = masked.sum
            case _:
                raise NotImplementedError(f"Invalid `agg` value: {agg}. Available options: 'mean', 'sum'.")

        # Initialize learnable weights to update nodes' global representations
        self.global_linear = pyg_nn.Linear(in_channels, out_channels)

    def reset_parameters(self) -> None:
        """Resets all learnable parameters of the module."""
        # NOTE: Do not call `super.reset_parameters()` because it resets `self.agg`'s parameters, which this child
        # overrides to be masked function instead of an `Aggregation` instance.
        # Therefore, make sure that we also reset here the parameters still relevant from the parent.
        self.local_linear.reset_parameters()
        self.global_linear.reset_parameters()

    def forward(self, x_local: Tensor, x_global: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Perform global message passing through the virtual node to compute updated node/global representations.

        Args:
            x_local: (num_nodes, node_channels), Local node representations, after the local message passing update.
            x_global: (num_graphs, node_channels), Global node representations from the previous layer.
            batch: (num_nodes,), The batch vector which assigns each element to a specific graph.

        Returns:
            (num_nodes, node_channels), Updated node representations.
            (num_graphs, node_channels), Updated virtual node representation for each graph.
        """
        ################################ STEP 1 ################################
        # GLOBAL MESSAGE PASSING AND AGGREGATION OF LOCAL NODE REPRESENTATIONS #
        ########################################################################

        # Linear transformation of updated local representations and previous global representations
        x_local = self.local_linear(x_local)  # (num_nodes, node_channels)
        x_global = self.global_linear(x_global)  # (num_graphs, node_channels)

        # Use dense batch format to concatenate global representations to their respective graphs
        x_dense, nodes_mask = to_dense_batch(x_local, batch=batch)  # (num_graphs, num_nodes_max, node_channels)
        x_dense = torch.cat([x_global.unsqueeze(1), x_dense], dim=1)  # (num_graphs, num_nodes_max+1, node_channels)
        # Account for concatenated global representations in mask
        x_global_mask = nodes_mask.new_ones((len(x_global), 1))  # (num_graphs, 1)
        nodes_mask = torch.cat([x_global_mask, nodes_mask], dim=1)  # (num_graphs, num_nodes_max+1)
        # Expand mask to include all data dims (required by torch's masked ops)
        # (num_graphs, num_nodes_max+1, 0 -> node_channels)
        nodes_mask = nodes_mask.unsqueeze(-1).expand(-1, -1, x_dense.shape[-1])

        # Aggregate updated local representations and previous global representations to compute global update,
        # equivalent to message passing through the virtual node
        x_global = self.agg(x_dense, dim=1, mask=nodes_mask)  # (num_graphs, num_nodes_max+1 -> 0, node_channels)

        # Apply non-linearities to the updated global representations
        x_global = self.act(x_global)
        x_global = self.dropout(x_global)

        ############################## STEP 2 ##############################
        # AGGREGATE LOCAL AND GLOBAL UPDATES FOR FINAL NODE REPRESENTATION #
        ####################################################################

        # Index by batch vector to broadcast global update per graph to all the nodes in their respective graphs
        x_global_broadcast = x_global[batch]  # (num_graphs -> num_nodes, node_channels)
        x = torch.stack([x_local, x_global_broadcast])  # (2, num_nodes, node_channels)
        x = self.global_local_agg(x, dim=0)  # (2 -> 0, num_nodes, node_channels)

        return x, x_global


class GCN_VN_G(VN_G_Mixin, pyg_nn.GCN):  # noqa: N801
    """Extension of PyG's `GCN` model that implements the VN_G formulation of virtual nodes."""


class GAT_VN_G(VN_G_Mixin, pyg_nn.GAT):  # noqa: N801
    """Extension of PyG's `GAT` model that implements the VN_G formulation of virtual nodes."""


class GIN_VN_G(VN_G_Mixin, pyg_nn.GIN):  # noqa: N801
    """Extension of PyG's `GIN` model that implements the VN_G formulation of virtual nodes."""
