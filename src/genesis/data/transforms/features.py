from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ConcatFeatures(BaseTransform):
    """Concatenate graph features together."""

    def __init__(
        self,
        attrs: dict[str, Literal["graph", "node"]],
        level: Literal["graph", "node"],
        concat_attr: str,
        node_agg: Literal["mean", "sum", "max", "min"] = "mean",
    ) -> None:
        """Initializes a `ConcatFeatures` instance.

        Args:
            attrs: Mapping between attribute fields to concatenate and at what level they appear in the graph, to
                determine how to reshape them for the concatenation.
            level: At which level to project/aggregate the features before concatenating them.
            concat_attr: Attribute name where to save the concatenated features.
            node_agg: When concatenating node features to graph-level features, which aggregation operation to use to
                reduce across the node dimensionality. Ignored for other features.
        """
        if level not in ["graph", "node"]:
            raise ValueError(
                f"Invalid `level` value, i.e. target for concatenation, '{level}'. "
                f"Allowed values are 'graph' and 'node'."
            )
        for attr, attr_type in attrs.items():
            if attr_type not in ["graph", "node"]:
                raise ValueError(
                    f"Invalid type of features to concatenate '{attr_type}' for feature '{attr}'. "
                    f"Allowed types are 'graph' and 'node'."
                )
        if node_agg not in [None, "mean", "sum", "max", "min"]:
            raise ValueError(
                f"Invalid `node_agg` value '{node_agg}'. Allowed values are: ['mean', 'sum', 'max', 'min']."
            )

        self.attrs = attrs
        self.level = level
        self.concat_attr = concat_attr
        self.node_aggr = getattr(torch, node_agg)

    def forward(self, data: Data) -> Data:
        """Concatenates features together on the provided graph.

        Args:
            data: The input graph for which to concatenate features.

        Returns:
            Graph with features concatenated as configured.
        """
        # Collect all the features to concatenate to the target
        feat2concat = []
        for attr, attr_type in self.attrs.items():
            attr_val = getattr(data, attr)

            if attr_type == "graph" and self.level == "node":
                # Broadcast global features to node dimension
                attr_val = attr_val.expand(data.num_nodes, -1)

            elif attr_type == "node" and self.level == "graph":
                # Aggregate node features across the graph
                attr_val = self.node_aggr(attr_val, dim=0)

            feat2concat.append(attr_val)

        # Concatenate the features and save to the target
        data[self.concat_attr] = torch.hstack(feat2concat)

        return data
