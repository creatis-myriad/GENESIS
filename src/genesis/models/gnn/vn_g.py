import copy
from typing import Any, Final, Literal

import torch
import torch_geometric.nn as pyg_nn
from torch import Tensor, nn
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import Adj, OptTensor


class VN_G_Mixin(nn.Module):  # noqa: N801
    """Mixin meant to work with children of PyG's `BasicGNN` to implement VN_G formulation of virtual nodes.

    References:
        - Inspired by the VN_G variant proposed in the paper
          "Understanding Virtual Nodes: Oversquashing and Node Heterogeneity": https://arxiv.org/abs/2405.13526
        - Implementation copies parts of PyG's `BasicGNN`:
          https://github.com/pyg-team/pytorch_geometric/blob/76ff9c2ce18c8cebf52122b57e2aeadce9793d10/torch_geometric/nn/models/basic_gnn.py#L32-L386
    """

    supports_batch: Final[bool] = True  # VN_G formulation requires to know the batch vector to compute global update

    def __init__(
        self,
        *args,
        global_mp_agg: str | Any = "MeanAggregation",
        global_mp_agg_kwargs: dict[str, Any] | None = None,
        global_norm_weighting: bool = True,
        global_dropout: float = 0.0,
        global_local_agg: Literal["mean", "sum"] = "sum",
        **kwargs,
    ) -> None:
        """Initializes the VN_G mixin.

        Args:
            *args: Additional positional arguments to pass to the class inheriting form `BasicGNN`.
            global_mp_agg: Aggregation to use across all real nodes to emulate message passing through a virtual node.
            global_mp_agg_kwargs: Arguments passed to the respective aggregation function defined by `global_agg`.
            global_norm_weighting: Whether to normalize global node representations based on graph degrees before
                combining them with local node representations.
            global_dropout: Dropout to apply to global node representations before combining them with local node
                representations.
            global_local_agg: Aggregation to use to combine global and local node representations.
            **kwargs: Additional keyword arguments to pass to the class inheriting form `BasicGNN`.
        """
        super().__init__(*args, **kwargs)

        # Initialize learnable linear layer for local representations at each message passing layer
        # These layers never deal with input dimensions, since even the 1st one comes after the 1st GNN layer,
        # which has already projected node features to `hidden_channels`
        self.local_linears = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.local_linears.append(pyg_nn.Linear(self.hidden_channels, self.hidden_channels))
        self.local_linears.append(pyg_nn.Linear(self.hidden_channels, self.out_channels))

        # Initialize identical copies of aggregation layer at each message passing layer
        # (in case aggregation contains learnable parameters)
        self.global_mp_aggs = nn.ModuleList()
        global_mp_agg = aggregation_resolver(global_mp_agg, **(global_mp_agg_kwargs or {}))
        for _ in range(self.num_layers):
            self.global_mp_aggs.append(copy.deepcopy(global_mp_agg))

        match global_local_agg:
            case "mean":
                self.global_local_agg = torch.mean
            case "sum":
                self.global_local_agg = torch.sum
            case _:
                raise NotImplementedError(
                    f"Invalid `global_local_agg` value '{global_local_agg}'. Available options: 'mean', 'sum'."
                )

        self.global_norm_weighting = global_norm_weighting
        self.global_dropout = nn.Dropout(global_dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
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
                # Added steps defined by VN_G:                                                #
                # 1) Update virtual node through global message passing and aggregation       #
                # 2) Update nodes representation by aggregating local and global node updates #
                ###############################################################################
                x_global = self.global_message_and_aggregate(x, batch, i)
                x = self.agg_global_local_updates(x, x_global, batch)
                ###############################################################################
                #                           End custom code block                             #
                ###############################################################################

                if hasattr(self, "jk"):
                    xs.append(x)

        ###############################################################################
        # Added steps defined by VN_G:                                                #
        # 1) Update virtual node through global message passing and aggregation       #
        # 2) Update nodes representation by aggregating local and global node updates #
        ###############################################################################
        # Normally, we perform the global message passing + update inside the layers loop, after act/norm/dropout,
        # but before storing layer output for jumping knowledge pooling.
        # However, when jumping knowledge is not used, act/norm/dropout are skipped for the last layer.
        # Therefore, when jumping knowledge is not used, we perform global message passing + update outside the layers
        # loop to compute it for the last layer
        if self.jk_mode is None:
            x_global = self.global_message_and_aggregate(x, batch, self.num_layers - 1)
            x = self.agg_global_local_updates(x, x_global, batch)
        ###############################################################################
        #                           End custom code block                             #
        ###############################################################################

        x = self.jk(xs) if hasattr(self, "jk") else x
        x = self.lin(x) if hasattr(self, "lin") else x

        return x  # noqa: RET504

    def global_message_and_aggregate(self, x_local: Tensor, batch: Tensor, layer_idx: int) -> Tensor:
        """Message passing step between updated local node representations and virtual node.

        Args:
            x_local: (num_nodes, node_channels), Local node representations, after the local message passing update.
            batch: (num_nodes,), The batch vector which assigns each element to a specific graph.
            layer_idx: The index of the current message passing layer.

        Returns:
            (num_graphs, node_channels), Updated virtual node representation for each graph.
        """
        # Extract layers to use from the module lists
        local_linear = self.local_linears[layer_idx]
        agg = self.global_mp_aggs[layer_idx]

        # Linear transformation of updated local representations
        x_local = local_linear(x_local)  # (num_nodes, node_channels)
        # Global aggregation to emulate message passing through a virtual node connected to all other nodes
        x_global = agg(x_local, index=batch)  # (num_nodes -> num_graphs, node_channels)
        # Normalization of global representations based on graph degree
        if self.global_norm_weighting:
            _, graph_degrees = torch.unique_consecutive(batch, return_counts=True)  # (num_graphs,)
            deg_inv_sqrt = torch.pow(graph_degrees, -0.5)  # (num_graphs,)
            x_global = deg_inv_sqrt.unsqueeze(-1) * x_global  # (num_graphs, node_channels)

        return x_global

    def agg_global_local_updates(self, x_local: Tensor, x_global: Tensor, batch: Tensor) -> Tensor:
        """Combine local and global updates, computed in separate steps, into one representation per node.

        Args:
            x_local: (num_nodes, node_channels), Local node representations, after the local message passing update.
            x_global: (num_graphs, node_channels), Global node representations, after the global message passing update.
            batch: (num_nodes,), Batch vector which assigns each element to a specific graph.

        Returns:
            (num_nodes, node_channels), Final updated node representations.
        """
        # Index by batch vector to broadcast aggregated values per graph to all the nodes in their respective graphs
        x_global = x_global[batch]  # (num_graphs -> num_nodes, node_channels)

        # Apply dropout on the broadcasted global update, to obtain different dropout per node in the same graph
        x_global = self.global_dropout(x_global)

        x = torch.stack([x_local, x_global])
        return self.global_local_agg(x, dim=0)


class GCN_VN_G(VN_G_Mixin, pyg_nn.GCN):  # noqa: N801
    """Extension of PyG's `GCN` model that implements the VN_G formulation of virtual nodes."""


class GAT_VN_G(VN_G_Mixin, pyg_nn.GAT):  # noqa: N801
    """Extension of PyG's `GAT` model that implements the VN_G formulation of virtual nodes."""


class GIN_VN_G(VN_G_Mixin, pyg_nn.GIN):  # noqa: N801
    """Extension of PyG's `GIN` model that implements the VN_G formulation of virtual nodes."""
