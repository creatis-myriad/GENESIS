import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.impute._base import _BaseImputer
from torch.utils.data import Dataset

from genesis.data.utils import impute


class CSVDataset(Dataset):
    """Dataset for loading tabular data from a CSV file."""

    def __init__(
        self,
        src: str | Path,
        split: str | None = None,
        target_attr: str | None = None,
        imputer: _BaseImputer | None = None,
        impute_cols: list[str] | None = None,
        drop_na: bool | list[str] = False,
        **read_csv_kwargs,
    ) -> None:
        """Initializes a `CSVDataset`.

        Args:
            src: If `split` is None, path to the CSV file. If `split` is provided, path to the directory containing the
                CSV files for each split (e.g. 'train.csv', 'val.csv', 'test.csv').
            split: If provided, specifies which split (e.g. 'train', 'val', 'test') to load from the `src` directory.
            target_attr: Name of the target attribute (column) in the CSV file. If None, the dataset will not return
                targets.
            imputer: Imputer to complete missing values. If None, no imputation will be performed. In any case, any
                remaining missing values will be dropped.
            impute_cols: Columns for which to complete missing values. If None, default to all columns except the target
                attribute. Can be made to impute the target by explicitly including it in the list.
            drop_na: If imputer is not provided, determines whether to drop rows with any missing value.
                If a list, subset of columns in which to consider missing values.
                Otherwise, if True, missing values in any column will lead to the row being dropped.
                If False, no rows are dropped.
            **read_csv_kwargs: Additional keyword arguments to pass to `pandas.read_csv`.
        """
        csv_file = Path(src)
        if split:
            csv_file /= f"{split}.csv"
        self.root = csv_file.parent  # For compatibility with `SplitLightningDataset` to generate splits at runtime
        self.data = pd.read_csv(csv_file, **read_csv_kwargs)
        self._target_attr = target_attr

        # Complete missing values if imputer is provided
        if imputer is not None:
            self.data = impute(
                self.data,
                imputer,
                # By default, impute all columns except the target attribute
                impute_cols=self.data.columns.difference([target_attr]) if impute_cols is None else impute_cols,
            )

        # Otherwise, drop missing values if requested
        elif drop_na:
            self.data.dropna(inplace=True, subset=drop_na if isinstance(drop_na, list) else None)

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

    def __getitem__(
        self, index: int | slice | Sequence[int]
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray], "CSVDataset"]:
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
