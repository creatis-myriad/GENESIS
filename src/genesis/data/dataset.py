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
        self.root = Path(src).parent
        self.data = pd.read_csv(src, **read_csv_kwargs)
        # Assign input and target features to `x` and `y`, respectively, to follow the PyG convention and work
        # transparently with dataset utils (e.g. `SplitLightningDataset`) expecting the PyG convention
        self.x = self.data.copy()
        self.y = self.x.pop(target_attr) if target_attr is not None else None

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Get item by index.

        Args:
            index: Numerical index (i.e. row number) of the item to retrieve.

        Returns:
            The item at the specified index. If `target_attr` is not None, returns a tuple of (features, target),
            otherwise returns only the features.
        """
        item = self.x.iloc[index].to_numpy()
        if self.y is not None:
            return item, self.y.iloc[index]
        return item

    # TODO: Check if `__getitem__` provides support for indexing the dataset, e.g. to extract train/val/test splits

    def indices(self) -> list[int | str]:
        """Get the index labels of the dataset."""
        return self.data.index.tolist()

    def loc(self, key: int | str) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Get item by label.

        Args:
            key: Label of the item to retrieve.

        Returns:
            The item at the specified label. If `target_attr` is not None, returns a tuple of (features, target),
            otherwise returns only the features.
        """
        return self[self.data.index.get_loc(key)]
