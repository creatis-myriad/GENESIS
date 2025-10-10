from typing import Literal

import pytest
import torch
from torch import Tensor

from genesis.models.gnn.agg import SelectAggregation


@pytest.fixture(scope="module")
def data_to_agg() -> tuple[Tensor, Tensor, Tensor]:
    """A Pytest fixture of hard-coded data to aggregate."""
    x = torch.arange(10)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    ptr = torch.tensor([0, 2, 4, 8, 10])
    return x, index, ptr


@pytest.mark.parametrize(
    ("mode", "expectation"), [("first", torch.tensor([0, 2, 4, 8])), ("last", torch.tensor([1, 3, 7, 9]))]
)
def test_select_aggregation_ptr(
    data_to_agg: tuple[Tensor, Tensor, Tensor], mode: Literal["first", "last"], expectation: Tensor
) -> None:
    """Test `SelectAggregation` when groups are defined by `ptr`."""
    x, _, ptr = data_to_agg
    agg = SelectAggregation(mode)
    out = agg(x, ptr=ptr, dim=0)
    torch.testing.assert_close(out, expectation)


@pytest.mark.parametrize("mode", ["first", "last"])
def test_select_aggregation_index(data_to_agg: tuple[Tensor, Tensor, Tensor], mode: Literal["first", "last"]) -> None:
    """Test that `SelectAggregation` raises an error when groups are defined by `index`."""
    x, index, _ = data_to_agg
    agg = SelectAggregation(mode)
    with pytest.raises(
        NotImplementedError,
        match="`SelectAggregation` is only defined for reductions where indices are ordered and grouped together",
    ):
        agg(x, index=index, dim=0)
