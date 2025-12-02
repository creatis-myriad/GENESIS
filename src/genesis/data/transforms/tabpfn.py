from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

try:
    from tabpfn import TabPFNClassifier

    _tabpfn_is_available = True
except ImportError:
    _tabpfn_is_available = False


class TabPFNEmbedding(BaseTransform):
    """Replaces global features by their embedding obtained from a pre-fitted TabPFN model."""

    def __init__(self, tabpfn_ckpt: Path, reduce_estimators: Literal["mean"] | None = "mean") -> None:
        """Initializes a `TabPFNVirtualNodes` instance.

        Args:
            tabpfn_ckpt: Path to the TabPFN model pre-fitted to the dataset's graph attributes.
            reduce_estimators: How to reduce across the features dimension, if TabPFN uses multiple estimators.
        """
        super().__init__()

        if not _tabpfn_is_available:
            raise ModuleNotFoundError(
                "No module named 'tabpfn' found in your Python environment. Install it through 'baselines' extra when "
                "installing the project, e.g. pip install genesis[baselines], or manually via 'pip install tabpfn'."
            )

        self._tabpfn = TabPFNClassifier.load_from_fit_state(
            tabpfn_ckpt, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reduce_estimators = reduce_estimators

    def forward(self, data: Data) -> Data:
        """Replaces global features by their embedding obtained from a pre-fitted TabPFN model.

        Args:
            data: The input graph for which to embed global features.

        Returns:
            Input graph with TabPFN-embedded global features.
        """
        # Convert input to numpy array and add samples dimension
        tabpfn_x = data.graph_attr.unsqueeze(0).cpu().numpy()
        # If model was fitted on DataFrame, i.e. with named feature, convert input to DataFrame to avoid warning:
        # "X does not have valid feature names, but TabPFNClassifier was fitted with feature names"
        if self._tabpfn.feature_names_in_ is not None:
            tabpfn_x = pd.DataFrame(tabpfn_x, columns=self._tabpfn.feature_names_in_)

        # Configure autocast to use float16 dtype, since TabPFN might try to use AMP and `get_embeddings` does not
        # properly convert from bfloat16 (not supported by numpy) before casting to numpy array
        # This is the best workaround I found because I did not manage to configure TabPFN not to use AMP:
        # - Using `enabled=False` in autocast context manager didn't force disable AMP (probably overridden internally)
        # - Configuring `tabpfn.inference_precision=torch.float` seems to be ignored by `get_embedding`
        with torch.autocast(self._tabpfn.device, dtype=torch.float16):
            tabpfn_embedding = self._tabpfn.get_embeddings(tabpfn_x)

        # Convert back to a tensor, making sure the dtype and device are the same as the original tensor
        tabpfn_embedding = data.graph_attr.new_tensor(tabpfn_embedding)

        # If TabPFN used multiple estimators, either reduce or concat features from the ensemble of estimators
        if self._tabpfn.n_estimators > 1:
            match self.reduce_estimators:
                case "mean":
                    tabpfn_embedding = tabpfn_embedding.mean(dim=0)
                case None:
                    tabpfn_embedding = tabpfn_embedding.view(-1)

        # Update the global features with the TabPFN embedding
        data.graph_attr = tabpfn_embedding
        return data
