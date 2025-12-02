from typing import Any

from torch_geometric.data import Data


class GraphAttrData(Data):
    """`Data` class that batches `graph_attr` along a new batch dimension, to properly handle global features.

    This is a workaround for the fact that natively PyG's `Data` class concatenates global features along the feature
    dimension when batching, i.e. [num_examples * num_features] rather than [num_examples, num_features].

    References:
        - This is the recommended way to implement this behavior, as per PyG docs:
          https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html#batching-along-new-dimensions

    Examples:
        >>> import torch
        >>> from torch_geometric.data import Batch
        >>> num_graphs = 4
        >>> num_graph_features = 10
        >>> data_list = [GraphAttrData(graph_attr=torch.randn(num_graph_features)) for _ in range(num_graphs)]
        >>> data_list[0].graph_attr.shape  # (num_graph_features,)
        torch.Size([10])
        >>> batch = Batch.from_data_list(data_list)
        >>> batch.graph_attr.shape  # (num_graphs, num_graph_features)
        torch.Size([4, 10])
    """

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:  # noqa: D105
        if key == "graph_attr":
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
