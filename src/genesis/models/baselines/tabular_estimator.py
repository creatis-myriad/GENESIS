import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning_utilities import apply_to_collection
from torchmetrics import MetricCollection

from genesis.data import split


@runtime_checkable
class BaseClassifier(Protocol):
    """Protocol that a backbone classifier, i.e. a sklearn model, must implement."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "BaseClassifier": ...  # noqa: D102,N803

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...  # noqa: D102,N803


@runtime_checkable
class BaseRegressor(Protocol):
    """Protocol that a backbone regressor, i.e. a sklearn model, must implement."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "BaseRegressor": ...  # noqa: D102,N803

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...  # noqa: D102,N803


class TabularEstimator:
    """ML estimator that uses tabular data, providing an API similar to sklearn estimators."""

    def __init__(
        self,
        model: BaseClassifier | BaseRegressor,
        metrics: MetricCollection | None = None,
    ) -> None:
        """Initializes a `SklearnClassifierLitModule`.

        Args:
            model: Backbone estimator model.
            metrics: A collection of metrics to use for evaluation.
        """
        if not isinstance(model, BaseClassifier | BaseRegressor):
            raise ValueError("Model must be an instance of either `BaseClassifier` or `BaseRegressor`")

        self.model = model
        self.metrics = metrics

    @staticmethod
    def _setup_data(datamodule: LightningDataModule, subset: str) -> tuple[pd.DataFrame, np.ndarray]:
        """Extract and process samples from the data module, handling missing values and categorical attributes.

        Args:
            datamodule: Data module.
            subset: Subset of the data to extract (e.g. "train", "test").

        Returns:
            Tuple of data extracted from the subset:
                - DataFrame of tabular data to use as input features, w/ missing values marked as np.nan.
                - Numpy array of target labels.
        """
        # Make sure the subset has been set up before extracting the data
        match subset:
            case split.TRAIN_SET | split.VAL_SET:
                datamodule.setup(stage=TrainerFn.FITTING)
            case split.TEST_SET:
                datamodule.setup(stage=TrainerFn.TESTING)
            case "predict":
                datamodule.setup(stage=TrainerFn.PREDICTING)
            case _:
                raise ValueError(f"Invalid subset: {subset}")

        # Select the appropriate subset of the data
        dataset = getattr(datamodule, f"{subset}_dataset")

        # Extract the input and target features from the dataset
        return dataset.x, dataset.y.to_numpy()

    def fit(self, datamodule: LightningDataModule, subset: str) -> "TabularEstimator":
        """Fit the model to a subset of the data.

        Args:
            datamodule: Data module.
            subset: Subset of the data to fit the model on (e.g. "train").

        Returns:
            The fitted model.
        """
        # Extract the input features and target labels from the specified subset
        X, y = self._setup_data(datamodule, subset)  # noqa: N806

        # Fit the model based on sklearn's `BaseEstimator` API
        self.model = self.model.fit(X, y)

        return self

    def _predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Make predictions on input samples.

        Args:
            X: (n_samples, n_features) Input features for all samples.

        Returns:
            Array of model predictions.
            (n_samples, n_classes) Probabilities for each class with a `BaseClassifier` backbone.
            (n_samples,) Predicted value with a `BaseRegressor` backbone.
        """
        # Perform inference, keeping intermediate predictions (i.e. class probabilities) if the model supports it
        match self.model:
            case BaseClassifier():
                y_pred = self.model.predict_proba(X)
            case BaseRegressor():
                y_pred = self.model.predict(X)
            case _:
                raise AssertionError("Model should be an instance of either `BaseClassifier` or `BaseRegressor")

        return y_pred

    def predict(self, datamodule: LightningDataModule, subset: str) -> np.ndarray:
        """Make predictions on a subset of the data.

        Args:
            datamodule: Data module.
            subset: Subset of the data to predict on (e.g. "predict").

        Returns:
            Array of model predictions.
            (n_samples, n_classes) Probabilities for each class with a `BaseClassifier` backbone.
            (n_samples,) Predicted value with a `BaseRegressor` backbone.
        """
        # Extract the input features from the specified subset
        X, _ = self._setup_data(datamodule, subset)  # noqa: N806

        return self._predict(X)

    def score(self, datamodule: LightningDataModule, subset: str) -> dict[str, float]:
        """Measure the model's performance on a subset of the data.

        Args:
            datamodule: Data module.
            subset: Subset of the data to evaluate the model on (e.g. "train", "test").

        Returns:
            Dictionary of model's metrics (e.g. accuracy, AUROC, etc.).
        """
        if self.metrics is None:
            raise ValueError(
                "This model instance has no defined evaluation metrics, i.e. no `metrics` arg was provided when "
                "instantiating the model. Calling `score` is only valid for models with metrics."
            )

        # Extract the input features and target labels from the specified subset
        X, y_true = self._setup_data(datamodule, subset)  # noqa: N806

        y_pred = self._predict(X)

        # Compute the model's performance metrics
        # Because we use the `MetricCollection` API from torchmetrics, we have to convert the predictions/targets from
        # numpy arrays to torch tensors, and conversely for the computed metrics
        scores = self.metrics(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        return apply_to_collection(scores, torch.Tensor, lambda x: x.cpu().numpy())

    def save(self, ckpt: Path | str) -> None:
        """Save the model to disk.

        Args:
            ckpt: Path to save the model to.
        """
        with Path(ckpt).open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, ckpt: Path | str) -> "TabularEstimator":
        """Load a model from disk.

        Args:
            ckpt: Path to the model checkpoint.

        Returns:
            The loaded model.
        """
        with Path(ckpt).open("rb") as f:
            return pickle.load(f)  # noqa: S301
