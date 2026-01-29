from collections.abc import Sequence
from pathlib import Path
from typing import Any

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

    def write_on_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]
    ) -> None:
        """Logs predictions at the end of an epoch, saving each dataloader's predictions to a separate file.
        
        This method collects predictions from different dataloaders (train/val/test), saves them to separate
        CSV files, and logs them to the configured experiment tracker if available.
        
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
        for dataloader_idx, (dataloader_preds, dataloader_batch_indices) in enumerate(
            zip(predictions, batch_indices, strict=False)
        ):
            # Get the subset name for this dataloader
            if dataloader_idx >= len(self.predictions_dataloaders):
                # Skip if we have more predictions than expected dataloaders
                continue
            
            # Skip if no predictions for this dataloader
            if not dataloader_preds or len(dataloader_preds) == 0:
                continue
            
            subset = self.predictions_dataloaders[dataloader_idx]
            
            # Concatenate all batch predictions into a single tensor
            if isinstance(dataloader_preds[0], torch.Tensor):
                all_predictions = torch.cat([pred for pred in dataloader_preds])
            else:
                # Handle case where predictions might already be concatenated or in a different format
                all_predictions = dataloader_preds
            
            # Convert predictions to numpy for DataFrame creation
            if isinstance(all_predictions, torch.Tensor):
                all_predictions = all_predictions.cpu().numpy()
            
            # Create a DataFrame with predictions and batch indices
            # Flatten batch indices if they are nested
            all_batch_indices = []
            for batch_idx_list in dataloader_batch_indices:
                if isinstance(batch_idx_list, (list, tuple)):
                    all_batch_indices.extend(batch_idx_list)
                else:
                    all_batch_indices.append(batch_idx_list)
            
            # Create DataFrame based on prediction dimensionality
            if all_predictions.ndim == 1:
                # Binary classification or regression: single value per sample
                df = pd.DataFrame({
                    "prediction": all_predictions,
                    "batch_idx": all_batch_indices[:len(all_predictions)],
                })
            else:
                # Multi-class or multi-label: multiple values per sample
                prediction_cols = {f"prediction_{i}": all_predictions[:, i] for i in range(all_predictions.shape[1])}
                df = pd.DataFrame({
                    **prediction_cols,
                    "batch_idx": all_batch_indices[:len(all_predictions)],
                })
            
            # Save to CSV file
            filename = self.filename_format.format(subset=subset, split=subset)
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            
            # Log to experiment tracker if available
            if trainer.logger is not None:
                # Handle both single logger and list of loggers
                loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
                
                for logger in loggers:
                    # Log the file as an artifact for WandbLogger
                    if isinstance(logger, WandbLogger):
                        wandb_run = logger.experiment
                        wandb_run.save(str(filepath), base_path=str(output_dir.parent))
                    
                    # Log summary statistics
                    summary_stats = {
                        f"{subset}/predictions_mean": float(all_predictions.mean()),
                        f"{subset}/predictions_std": float(all_predictions.std()),
                        f"{subset}/predictions_min": float(all_predictions.min()),
                        f"{subset}/predictions_max": float(all_predictions.max()),
                        f"{subset}/num_predictions": len(all_predictions),
                    }
                    logger.log_metrics(summary_stats)
