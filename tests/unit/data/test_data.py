import pytest
import torch
from torch_geometric.data import Batch

from genesis.data.data import GraphAttrData


@pytest.mark.parametrize("graph_attr_size", [(3,), (1, 3)])
def test_graph_attr_data(graph_attr_size: tuple[int, ...]) -> None:
    """Test that `GraphAttrData` batches `graph_attr` attribute along a new batch dimension.

    Args:
        graph_attr_size: Size of the randomly generated `graph_attr` attribute.
    """
    data = GraphAttrData(graph_attr=torch.randn(graph_attr_size))

    data_list = [data, data]
    batch = Batch.from_data_list(data_list)

    # Check that `graph_attr` is batched along a new dimension
    assert batch.graph_attr.shape == (2, *graph_attr_size)

    # Check that unbatched `graph_attr` have the batch dimension removed
    unbatched_data = batch.to_data_list()
    assert all(unbatched.graph_attr.shape == graph_attr_size for unbatched in unbatched_data)
