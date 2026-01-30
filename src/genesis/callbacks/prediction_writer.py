from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.loggers import WandbLogger


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

    def _preprocess_predictions(
        self, dataloader_preds: Sequence[Any], dataloader_batch_indices: Sequence[Any]
    ) -> tuple[np.ndarray, list[int]]:
        """Preprocess predictions and batch indices from a single dataloader.
        
        Args:
            dataloader_preds: Predictions from a single dataloader (list of batch predictions).
            dataloader_batch_indices: Batch indices corresponding to the predictions.
            
        Returns:
            A tuple containing:
                - Predictions as a numpy array
                - Flattened list of batch indices
                
        Raises:
            TypeError: If predictions are not in the expected format (torch.Tensor).
            ValueError: If the number of batch indices is less than the number of predictions.
        """
        # Concatenate all batch predictions into a single tensor
        if isinstance(dataloader_preds[0], torch.Tensor):
            all_predictions = torch.cat([pred for pred in dataloader_preds])
        else:
            # Handle case where predictions might already be concatenated
            if isinstance(dataloader_preds, torch.Tensor):
                all_predictions = dataloader_preds
            else:
                raise TypeError(
                    f"Expected predictions to be a list of torch.Tensor or a single torch.Tensor, "
                    f"but got {type(dataloader_preds)}"
                )
        
        # Convert predictions to numpy for DataFrame creation
        if isinstance(all_predictions, torch.Tensor):
            all_predictions = all_predictions.cpu().numpy()
        
        # Flatten batch indices if they are nested
        all_batch_indices = []
        for batch_idx_list in dataloader_batch_indices:
            if isinstance(batch_idx_list, (list, tuple)):
                all_batch_indices.extend(batch_idx_list)
            else:
                all_batch_indices.append(batch_idx_list)
        
        # Validate that we have enough batch indices for all predictions
        if len(all_batch_indices) < len(all_predictions):
            raise ValueError(
                f"Number of batch indices ({len(all_batch_indices)}) is less than "
                f"number of predictions ({len(all_predictions)})"
            )
        
        # Truncate batch indices to match predictions if necessary
        all_batch_indices = all_batch_indices[:len(all_predictions)]
        
        return all_predictions, all_batch_indices

    def write_on_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]
    ) -> None:
        """Logs predictions at the end of an epoch, saving each dataloader's predictions to a separate file.
        
        This method collects predictions from different dataloaders (train/val/test), saves them to separate
        CSV files, and logs them to the configured experiment tracker if available.
        
        Note:
            For WandbLogger, predictions are logged as interactive Tables.
        
        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.
            predictions: A sequence of predictions from each dataloader. Each element is a list of batch predictions.
            batch_indices: A sequence of batch indices corresponding to the predictions.
        """
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate through each dataloader's predictions
        for subset, dataloader_preds, dataloader_batch_indices in zip(
            self.predictions_dataloaders, predictions, batch_indices, strict=True
        ):
            # Skip if no predictions for this dataloader
            if not dataloader_preds:
                continue
            
            # Preprocess predictions and batch indices
            all_predictions, all_batch_indices = self._preprocess_predictions(
                dataloader_preds, dataloader_batch_indices
            )
            
            # Create DataFrame based on prediction dimensionality
            if all_predictions.ndim == 1:
                # Binary classification or regression: single value per sample
                df = pd.DataFrame({
                    "prediction": all_predictions,
                    "batch_idx": all_batch_indices,
                })
            else:
                # Multi-class or multi-label: multiple values per sample
                prediction_cols = {f"prediction_{i}": all_predictions[:, i] for i in range(all_predictions.shape[1])}
                df = pd.DataFrame({
                    **prediction_cols,
                    "batch_idx": all_batch_indices,
                })
            
            # Save to CSV file
            filename = self.filename_format.format(subset=subset)
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            
            # Log to experiment tracker if available
            if trainer.logger is not None:
                # Handle both single logger and list of loggers
                loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
                
                for logger in loggers:
                    # Log as a WandB Table for WandbLogger
                    if isinstance(logger, WandbLogger):
                        import wandb
                        
                        # Create WandB Table from DataFrame
                        table = wandb.Table(dataframe=df)
                        wandb_run = logger.experiment
                        wandb_run.log({f"{subset}_predictions": table})
