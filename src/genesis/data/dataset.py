import copy
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset


class CSVDataset(Dataset):
    """Dataset for loading tabular data from a CSV file."""

    def __init__(self, src: str | Path, target_attr: str | None = None, **read_csv_kwargs) -> None:
        """Initializes a `CSVDataset`.

        Args:
            src: Path to the CSV file.
            target_attr: Name of the target attribute (column) in the CSV file. If None, the dataset will not return
                targets.
            **read_csv_kwargs: Additional keyword arguments to pass to `pandas.read_csv`.
        """
        self.root = Path(src).parent
        self.data = pd.read_csv(src, **read_csv_kwargs)
        self._target_attr = target_attr

    @property
    def x(self) -> pd.DataFrame:
        """Get the input features of the dataset.

        Mirrors the PyG convention of using `x` for input features, allowing this dataset to work transparently with
        dataset utils (e.g. `SplitLightningDataset`) expecting the PyG convention.
        """
        features = self.data.columns.difference([self._target_attr])
        return self.data[features]

    @property
    def y(self) -> pd.Series | None:
        """Get the target feature of the dataset.

        Mirrors the PyG convention of using `y` for target features, allowing this dataset to work transparently with
        dataset utils (e.g. `SplitLightningDataset`) expecting the PyG convention.
        """
        return self.data[self._target_attr] if self._target_attr is not None else None

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int | slice | Sequence[int]) -> np.ndarray | tuple[np.ndarray, np.ndarray] | Subset:
        """In case `index` is of type integer, will return the data object at index `index`.

        Otherwise, `index` is interpreted as a slicing object, e.g. `slice(2, 5)`, will return a subset of the dataset
        at the specified indices.

        Args:
            index: Numerical index (i.e. row number) of the item to retrieve.

        Returns:
            The item at the specified index. If `target_attr` is not None, returns a tuple of (features, target),
            otherwise returns only the features.
        """
        # Check if index is an int, and return the data item at that index
        if isinstance(index, int):
            item = self.x.iloc[index].to_numpy()
            if self.y is not None:
                return item, self.y.iloc[index]
            return item

        # Otherwise, interpret index as a slice or sequence and return a subset of the dataset
        return self.index_select(index)

    def index_select(self, index: slice | Sequence[int]) -> "CSVDataset":
        """Select a subset of the dataset based on the provided indices.

        Notes:
            - This approach to copying the dataset and modifying its data, within an `index_select` method, is inspired
              by PyG's way of slicing datasets, see:
              https://github.com/pyg-team/pytorch_geometric/blob/2.6.1/torch_geometric/data/dataset.py#L276-L346
            - The specific checks on `index` type and behavior in each case mirrors MONAI's dataset, see:
              https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/data/dataset.py#L196-L108

        Args:
            index: Index to select from the dataset.

        Returns:
            A new `CSVDataset` containing only the selected indices.
        """
        if isinstance(index, slice):  # e.g. dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
        elif isinstance(index, Sequence):  # e.g. dataset[[1, 3, 4]]
            indices = index
        else:
            raise IndexError(f"Only slices (':'), list, and tuples are valid indices (got '{type(index).__name__}')")

        dataset = copy.copy(self)
        dataset.data = self.data.iloc[indices]
        return dataset

    def loc(self, key: int | str) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Get item by label.

        Args:
            key: Label of the item to retrieve.

        Returns:
            The item at the specified label. If `target_attr` is not None, returns a tuple of (features, target),
            otherwise returns only the features.
        """
        return self[self.data.index.get_loc(key)]
