from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


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
        self.data = pd.read_csv(src, **read_csv_kwargs)
        self.target_attr = target_attr

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> np.ndarray | tuple[np.ndarray, Number]:
        """Get item by index.

        Args:
            index: Numerical index (i.e. row number) of the item to retrieve.

        Returns:
            The item at the specified index. If `target_attr` is not None, returns a tuple of (features, target),
            otherwise returns only the features.
        """
        item = self.data.iloc[index]
        if self.target_attr is not None:
            target = item.pop(self.target_attr)
            return item.to_numpy(), target
        return item.to_numpy()

    def indices(self) -> list[int | str]:
        """Get the index labels of the dataset."""
        return self.data.index.tolist()

    def loc(self, key: int | str) -> np.ndarray | tuple[np.ndarray, Number]:
        """Get item by label.

        Args:
            key: Label of the item to retrieve.

        Returns:
            The item at the specified label. If `target_attr` is not None, returns a tuple of (features, target),
            otherwise returns only the features.
        """
        return self[self.data.index.get_loc(key)]
