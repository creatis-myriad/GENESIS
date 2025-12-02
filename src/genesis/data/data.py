from typing import Any

from torch_geometric.data import Data


class GraphAttrData(Data):
    """`Data` class that batches `graph_attr` along a new batch dimension, to properly handle global features.

    This is a workaround for the fact that natively PyG's `Data` class concatenates global features along the feature
    dimension when batching, i.e. [num_examples * num_features] rather than [num_examples, num_features].

    References:
        - This is the recommended way to implement this behavior, as per PyG docs:
          https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html#batching-along-new-dimensions
    """

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:  # noqa: D105
        if key == "graph_attr":
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
