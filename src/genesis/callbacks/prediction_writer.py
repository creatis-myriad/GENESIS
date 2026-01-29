from collections.abc import Sequence
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter


class GraphLevelPredictionWriter(BasePredictionWriter):
    """Callback to log graph-level predictions to the configured logger/experiment tracker."""

    def __init__(
        self,
        output_dir: str,
        save_fit_predictions: bool,
        save_test_predictions: bool,
        filename_format: str = "{subset}_predictions.csv",
    ) -> None:
        """Initializes a `GraphLevelPredictionWriter` instance.

        Args:
            output_dir: Directory where prediction files will be saved.
            save_fit_predictions: Whether the dataloaders passed to the prediction loop will include training and
                validation dataloaders.
            save_test_predictions: Whether the dataloaders passed to the prediction loop will include the test
                dataloader.
            filename_format: A format string for naming the output files. It should include a `{subset}` placeholder
                that will be replaced with the name of the dataloader subset (e.g., "train", "val", "test").
        """
        self.output_dir = output_dir
        self.filename_format = filename_format
        self.predictions_dataloaders = []
        if save_fit_predictions:
            self.predictions_dataloaders.extend(["train", "val"])
        if save_test_predictions:
            self.predictions_dataloaders.append("test")
        super().__init__(write_interval="epoch")

    def write_on_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]
    ) -> None:
        """Logs predictions at the end of an epoch, saving each dataloader's predictions to a separate file."""
        raise NotImplementedError
