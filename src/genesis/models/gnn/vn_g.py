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

    # VN_G formulation requires to know the batch vector to compute global update
    supports_batch: Final[bool] = True

    def __init__(
        self,
        *args,
        global_agg: str | Any = "MeanAggregation",
        global_agg_kwargs: dict[str, Any] | None = None,
        norm_weighting: bool = True,
        global_dropout: float = 0.0,
        global_local_agg: Literal["mean", "sum"] = "sum",
        **kwargs,
    ) -> None:
        """Initializes the VN_G mixin.

        Args:
            *args: Additional positional arguments to pass to the class inheriting form `BasicGNN`.
            global_agg: Aggregation to use across all real nodes to emulate message passing through a virtual node.
            global_agg_kwargs: Arguments passed to the respective aggregation function defined by `global_agg`.
            norm_weighting: Whether to normalize global node representations based on graph degrees before combining
                them with local node representations.
            global_dropout: Dropout to apply to global node representations before combining them with local node
                representations.
            global_local_agg: Aggregation to use to combine global and local node representations.
            **kwargs: Additional keyword arguments to pass to the class inheriting form `BasicGNN`.
        """
        super().__init__(*args, **kwargs)

        # Initialize learnable linear layer for each message passing layer
        # These layers never deal with input dimensions, since even the 1st one comes after the 1st GNN layer,
        # which has already projected node features to `hidden_channels`
        self.global_linears = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.global_linears.append(pyg_nn.Linear(self.hidden_channels, self.hidden_channels))
        self.global_linears.append(pyg_nn.Linear(self.hidden_channels, self.out_channels))

        # Initialize identical aggregation layer for each message passing layer
        # (in case aggregation contains learnable parameters)
        self.global_aggs = nn.ModuleList()
        global_agg = aggregation_resolver(global_agg, **(global_agg_kwargs or {}))
        for _ in range(self.num_layers):
            self.global_aggs.append(copy.deepcopy(global_agg))

        self.norm_weighting = norm_weighting
        self.global_dropout = nn.Dropout(global_dropout)
        self.global_local_agg = global_local_agg

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

        The code was copied from `torch_geometric.nn.BasicGNN`'s `forward` method, and only modified to call the global
        update function where relevant.

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

                ##################################################
                # ADDED CALL TO GLOBAL UPDATE AS DEFINED BY VN_G #
                ##################################################
                x_global = self._global_update(x, batch, i)
                x = self._agg_global_local_updates(x, x_global)

                if hasattr(self, "jk"):
                    xs.append(x)

        ##################################################
        # ADDED CALL TO GLOBAL UPDATE AS DEFINED BY VN_G #
        ##################################################
        # Normally, we call global update inside the layers loop, after activation/normalization/dropout but
        # before storing layer output for jumping knowledge pooling.
        # However, when jumping knowledge is not used, activation/normalization/dropout are skipped for the last layer.
        # Therefore, when jumping knowledge is not used, we have to call the global update outside the layers loop to
        # compute it on the last layer
        if self.jk_mode is None:
            x_global = self._global_update(x, batch, self.num_layers - 1)
            x = self._agg_global_local_updates(x, x_global)

        x = self.jk(xs) if hasattr(self, "jk") else x
        x = self.lin(x) if hasattr(self, "lin") else x

        return x  # noqa: RET504

    def _global_update(self, x: Tensor, batch: Tensor, layer_idx: int) -> Tensor:
        """Global update of the VN_G formulation of virtual nodes.

        Args:
            x: (num_nodes, num_node_features), The node features from the local update step (i.e. message passing).
            batch: (num_nodes,), The batch vector which assigns each element to a specific graph. Necessary to know how
                nodes to aggregate into a global message per graph.
            layer_idx: The index of the current message passing layer.

        Returns:
            (num_nodes, num_node_features), The global update to combine to local node representations.
        """
        # Extract layers to use from the module lists
        global_linear = self.global_linears[layer_idx]
        global_agg = self.global_aggs[layer_idx]

        # Linear transformation of updated local representations
        x_global = global_linear(x)  # (num_nodes, num_node_features)
        # Global aggregation to emulate message passing through a virtual node connected to all other nodes
        x_global = global_agg(x_global, index=batch)  # (num_nodes -> num_graphs, num_node_features)
        # Normalization of global representations based on graph degree
        if self.norm_weighting:
            _, graph_degrees = torch.unique_consecutive(batch, return_counts=True)  # (num_graphs,)
            deg_inv_sqrt = torch.pow(graph_degrees, -0.5)  # (num_graphs,)
            x_global = deg_inv_sqrt.unsqueeze(-1) * x_global  # (num_graphs, num_node_features)

        # Index by batch vector to broadcast aggregated values per graph to all the nodes in their respective graphs
        x_global = x_global[batch]  # (num_graphs -> num_nodes, num_node_features)

        return self.global_dropout(x_global)

    def _agg_global_local_updates(self, x_local: Tensor, x_global: Tensor) -> Tensor:
        """Combine local and global updates, computed in separate steps, into one representation per node.

        Args:
            x_local: (num_nodes, num_node_features), The local node representations.
            x_global: (num_nodes, num_node_features), The global node representations.

        Returns:
            (num_nodes, num_node_features), The updated node representations.
        """
        x = torch.stack([x_local, x_global])
        match self.global_local_agg:
            case "mean":
                x = x.mean(dim=0)
            case "sum":
                x = x.sum(dim=0)
            case _:
                raise ValueError(
                    f"Invalid `global_local_agg` value '{self.global_local_agg}'. Allowed values are 'mean' or 'sum'."
                )

        return x


class GCN_VN_G(VN_G_Mixin, pyg_nn.GCN):  # noqa: N801
    """Extension of PyG's `GCN` model that implements the VN_G formulation of virtual nodes."""


class GAT_VN_G(VN_G_Mixin, pyg_nn.GAT):  # noqa: N801
    """Extension of PyG's `GAT` model that implements the VN_G formulation of virtual nodes."""


class GIN_VN_G(VN_G_Mixin, pyg_nn.GIN):  # noqa: N801
    """Extension of PyG's `GIN` model that implements the VN_G formulation of virtual nodes."""
