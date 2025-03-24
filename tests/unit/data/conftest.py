import functools
from collections.abc import Callable
from pathlib import Path

import pytest
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset


@pytest.fixture
def mutag_dataset_fn(tmp_path: Path) -> Callable[[], Dataset]:
    """A Pytest fixture for a function that returns a `TUDataset` instance for the MUTAG dataset.

    Args:
        tmp_path: The temporary data path.

    Returns:
        A function that returns a dataset.
    """
    return functools.partial(TUDataset, root=str(tmp_path), name="MUTAG")
