import copy
import itertools
import re
from contextlib import nullcontext

import pytest
import torch
from _pytest.fixtures import FixtureRequest
from torch_geometric.data import Data

from genesis.data.transforms.virtual_node import VirtualNodes


@pytest.fixture(scope="module")
def full_graph() -> Data:
    """Pytest fixture that returns a simple connected graph with most PyG built-in features."""
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[2, 0, 2], [3, 1, 0]])
    edge_weight = torch.rand(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr, num_nodes=x.size(0))


@pytest.fixture(scope="module")
def minimal_disconnected_graph() -> Data:
    """Pytest fixture that returns a disconnected graph with only node features."""
    x = torch.randn(4, 16)
    return Data(x=x, edge_index=torch.empty(2, 0, dtype=torch.long))


@pytest.fixture(scope="module", params=["full_graph", "minimal_disconnected_graph"])
def data(request: FixtureRequest) -> Data:
    """Pytest fixture for graphs to use in tests."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("num_virtual_nodes", [None, 1, 2])
def test_virtual_nodes_zero_fill(data: Data, num_virtual_nodes: int | None) -> None:
    """Test adding a single virtual node to a connected graph, by specifying it explicitly or by default."""
    data_with_virtual_nodes = VirtualNodes(num=num_virtual_nodes)(data)
    zero_filled_virtual_nodes = data.x.new_zeros((num_virtual_nodes or 1, *data.x.shape[1:]))
    _validate_data_with_virtual_nodes(data, data_with_virtual_nodes, zero_filled_virtual_nodes)


@pytest.fixture(params=[(16,), (1, 16), (2, 16)])
def init_values(request: FixtureRequest) -> torch.Tensor:
    """Pytest fixture that returns initial features for virtual nodes."""
    return torch.randn(*request.param)


@pytest.mark.parametrize("num_virtual_nodes", [None, 1, 2])
def test_virtual_nodes_init_values(data: Data, num_virtual_nodes: int | None, init_values: torch.Tensor) -> None:
    """Test adding a single virtual node with initialized features to a connected graph."""
    num_init_values = 1 if init_values.ndim == 1 else len(init_values)
    if num_virtual_nodes is not None and num_virtual_nodes != num_init_values:
        expectation = pytest.raises(
            ValueError,
            match=re.escape(
                f"Number of provided virtual node initial features ({num_init_values}) does not match the number "
                f"of virtual nodes to add ({num_virtual_nodes})."
            ),
        )
    else:
        expectation = nullcontext()

    with expectation:
        data_with_virtual_nodes = VirtualNodes(num=num_virtual_nodes).forward(copy.copy(data), init_values=init_values)
    if not isinstance(expectation, nullcontext):
        return  # Stop test here if exception was raised as expected

    _validate_data_with_virtual_nodes(
        data, data_with_virtual_nodes, init_values if init_values.ndim == 2 else init_values.unsqueeze(0)
    )


def _validate_data_with_virtual_nodes(original_data: Data, data: Data, init_values: torch.Tensor) -> None:
    """Helper function to validate that the data with virtual nodes has been correctly modified."""
    x, edge_index, num_nodes = original_data.x, original_data.edge_index, original_data.num_nodes
    num_nodes = original_data.num_nodes
    num_edges = edge_index.size(1)
    node_attr_dim = x.shape[1:]

    edge_attr, edge_weight = original_data.get("edge_attr"), original_data.get("edge_weight")

    num_virtual_nodes = len(init_values)

    # Check that no other attribute than `edge_type` gets added (if not already present)
    assert len(data) == len(original_data) + ("edge_type" not in original_data)
    assert data.num_nodes == num_nodes + num_virtual_nodes

    assert data.x.size() == (num_nodes + num_virtual_nodes, *node_attr_dim)
    assert torch.allclose(data.x[:num_nodes], x)
    assert torch.allclose(data.x[num_nodes:], init_values)

    new_row_by_virtual_node = [list(range(num_nodes)) + [num_nodes + i] * num_nodes for i in range(num_virtual_nodes)]
    new_col_by_virtual_node = [[num_nodes + i] * num_nodes + list(range(num_nodes)) for i in range(num_virtual_nodes)]
    assert data.edge_index.tolist() == [
        list(itertools.chain(edge_index[0].tolist(), *new_row_by_virtual_node)),
        list(itertools.chain(edge_index[1].tolist(), *new_col_by_virtual_node)),
    ]

    assert data.edge_type.tolist() == [0] * num_edges + ([1] * num_nodes + [2] * num_nodes) * num_virtual_nodes

    num_new_edges = 2 * num_nodes * num_virtual_nodes  # Each virtual node connects to all other nodes
    if edge_attr is not None:
        edge_attr_dim = edge_attr.shape[1:]
        assert data.edge_attr.size() == (num_edges + num_new_edges, *edge_attr_dim)
        assert torch.allclose(data.edge_attr[:num_edges], edge_attr)
        assert data.edge_attr[num_edges:].abs().sum() == 0

    if edge_weight is not None:
        assert data.edge_weight.size() == (num_edges + num_new_edges,)
        assert torch.allclose(data.edge_weight[:num_edges], edge_weight)
        assert data.edge_weight[num_edges:].abs().sum() == num_new_edges
