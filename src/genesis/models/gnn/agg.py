from typing import Literal

from torch import Tensor
from torch_geometric.nn import aggr


class SelectAggregation(aggr.Aggregation):
    """An aggregation operator that selects an item at a specific position in a sequence of elements.

    Only defined with `ptr`-based aggregation, where elements are ordered and grouped together. The concept of
    "first" and "last" elements does not make sense in the context of `index`-based aggregation, where groups are
    potentially disjoint and unordered. If `index` is ordered and grouped, then `ptr` is typically also available.
    """

    def __init__(self, mode: Literal["first", "last"]) -> None:
        """Initializes a `SelectAggregation`.

        Args:
            mode: Indicates which element to select from the sequence.
                - `"first"`: Select the first element in each sequence, e.g. CLS token prepended to a set of tokens.
                - `"last"`: Select the last element in each sequence, e.g. virtual node appended at the end of a graph.
        """
        super().__init__()
        self.mode = mode

    def forward(  # noqa: D102
        self,
        x: Tensor,
        index: Tensor | None = None,  # ignored, but kept for compatibility with base class
        ptr: Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,  # ignored, but kept for compatibility with base class
        max_num_elements: int | None = None,  # ignored, but kept for compatibility with base class
    ) -> Tensor:
        if ptr is None:
            raise NotImplementedError(
                "`SelectAggregation` is only defined for reductions where indices are ordered and grouped together, "
                "in which case `ptr` is typically available. If you tried to provide `index` from a batch of data "
                "(e.g. `data.batch`) that is already ordered and grouped, you can easily switch to using `ptr` "
                "provided by the batch (e.g. `data.ptr`)."
            )

        match self.mode:
            case "first":
                # only keep pointers to the first element of each group
                # as the last pointer points to one past the last element
                select = ptr[:-1]
            case "last":
                # shift pointers back by one to get the last element of each group
                select = ptr[1:] - 1

        return x[select]
