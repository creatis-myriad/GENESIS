"""Tests for the GraphLevelPredictionWriter callback."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from genesis.callbacks.prediction_writer import GraphLevelPredictionWriter


class DummyModule(LightningModule):
    """Dummy LightningModule for testing."""

    def __init__(self) -> None:
        """Initialize DummyModule."""
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)


def test_prediction_writer_initialization() -> None:
    """Test that GraphLevelPredictionWriter initializes correctly."""
    writer = GraphLevelPredictionWriter(
        output_dir="/tmp/predictions",
        save_fit_predictions=True,
        save_test_predictions=True,
        filename_format="{subset}_predictions.csv",
    )
    
    assert writer.output_dir == "/tmp/predictions"
    assert writer.filename_format == "{subset}_predictions.csv"
    assert writer.predictions_dataloaders == ["train", "val", "test"]
    assert isinstance(writer, BasePredictionWriter)


def test_prediction_writer_fit_only() -> None:
    """Test that GraphLevelPredictionWriter initializes with only fit predictions."""
    writer = GraphLevelPredictionWriter(
        output_dir="/tmp/predictions",
        save_fit_predictions=True,
        save_test_predictions=False,
    )
    
    assert writer.predictions_dataloaders == ["train", "val"]


def test_prediction_writer_test_only() -> None:
    """Test that GraphLevelPredictionWriter initializes with only test predictions."""
    writer = GraphLevelPredictionWriter(
        output_dir="/tmp/predictions",
        save_fit_predictions=False,
        save_test_predictions=True,
    )
    
    assert writer.predictions_dataloaders == ["test"]


def test_write_on_epoch_end_binary_classification(tmp_path: Path) -> None:
    """Test write_on_epoch_end with binary classification predictions."""
    writer = GraphLevelPredictionWriter(
        output_dir=str(tmp_path),
        save_fit_predictions=True,
        save_test_predictions=False,
    )
    
    # Create dummy predictions (binary classification: 1D outputs)
    # Two dataloaders (train, val), each with 2 batches
    predictions = [
        [torch.tensor([0.8, 0.6, 0.9]), torch.tensor([0.7, 0.5])],  # train: 5 samples total
        [torch.tensor([0.4, 0.3]), torch.tensor([0.6])],  # val: 3 samples total
    ]
    
    batch_indices = [
        [[0, 1, 2], [3, 4]],  # train batch indices
        [[0, 1], [2]],  # val batch indices
    ]
    
    # Create dummy trainer and module
    trainer = Trainer(logger=None)
    module = DummyModule()
    
    # Call write_on_epoch_end
    writer.write_on_epoch_end(trainer, module, predictions, batch_indices)
    
    # Check that CSV files were created
    train_file = tmp_path / "train_predictions.csv"
    val_file = tmp_path / "val_predictions.csv"
    
    assert train_file.exists()
    assert val_file.exists()
    
    # Check train file contents
    train_df = pd.read_csv(train_file)
    assert len(train_df) == 5
    assert "prediction" in train_df.columns
    assert "batch_idx" in train_df.columns
    assert list(train_df["prediction"]) == pytest.approx([0.8, 0.6, 0.9, 0.7, 0.5])
    # Verify batch indices are correct
    assert list(train_df["batch_idx"]) == [0, 1, 2, 3, 4]
    
    # Check val file contents
    val_df = pd.read_csv(val_file)
    assert len(val_df) == 3
    assert "prediction" in val_df.columns
    assert "batch_idx" in val_df.columns
    assert list(val_df["prediction"]) == pytest.approx([0.4, 0.3, 0.6])
    # Verify batch indices are correct
    assert list(val_df["batch_idx"]) == [0, 1, 2]


def test_write_on_epoch_end_multiclass_classification(tmp_path: Path) -> None:
    """Test write_on_epoch_end with multiclass classification predictions."""
    writer = GraphLevelPredictionWriter(
        output_dir=str(tmp_path),
        save_fit_predictions=False,
        save_test_predictions=True,
    )
    
    # Create dummy predictions (multiclass: 2D outputs with 3 classes)
    # One dataloader (test), with 2 batches
    predictions = [
        [
            torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]),  # batch 1: 2 samples
            torch.tensor([[0.1, 0.2, 0.7]]),  # batch 2: 1 sample
        ],
    ]
    
    batch_indices = [
        [[0, 1], [2]],  # test batch indices
    ]
    
    # Create dummy trainer and module
    trainer = Trainer(logger=None)
    module = DummyModule()
    
    # Call write_on_epoch_end
    writer.write_on_epoch_end(trainer, module, predictions, batch_indices)
    
    # Check that CSV file was created
    test_file = tmp_path / "test_predictions.csv"
    assert test_file.exists()
    
    # Check test file contents
    test_df = pd.read_csv(test_file)
    assert len(test_df) == 3
    assert "prediction_0" in test_df.columns
    assert "prediction_1" in test_df.columns
    assert "prediction_2" in test_df.columns
    assert "batch_idx" in test_df.columns
    
    # Check first row
    assert list(test_df.iloc[0][["prediction_0", "prediction_1", "prediction_2"]]) == pytest.approx([0.8, 0.1, 0.1])


def test_write_on_epoch_end_empty_predictions(tmp_path: Path) -> None:
    """Test write_on_epoch_end with empty predictions."""
    writer = GraphLevelPredictionWriter(
        output_dir=str(tmp_path),
        save_fit_predictions=True,
        save_test_predictions=False,
    )
    
    # Create empty predictions
    predictions = [[], []]
    batch_indices = [[], []]
    
    # Create dummy trainer and module
    trainer = Trainer(logger=None)
    module = DummyModule()
    
    # Call write_on_epoch_end - should not crash
    writer.write_on_epoch_end(trainer, module, predictions, batch_indices)
    
    # Check that no CSV files were created (since predictions were empty)
    train_file = tmp_path / "train_predictions.csv"
    val_file = tmp_path / "val_predictions.csv"
    
    assert not train_file.exists()
    assert not val_file.exists()


def test_write_on_epoch_end_with_logging(tmp_path: Path) -> None:
    """Test write_on_epoch_end with logger configured."""
    from unittest.mock import Mock
    
    writer = GraphLevelPredictionWriter(
        output_dir=str(tmp_path),
        save_fit_predictions=False,
        save_test_predictions=True,
    )
    
    # Create dummy predictions
    predictions = [
        [torch.tensor([0.5, 0.6, 0.7])],
    ]
    
    batch_indices = [
        [[0, 1, 2]],
    ]
    
    # Create mock logger (non-WandbLogger)
    mock_logger = Mock()
    trainer = Trainer(logger=mock_logger)
    module = DummyModule()
    
    # Call write_on_epoch_end
    writer.write_on_epoch_end(trainer, module, predictions, batch_indices)
    
    # Verify that the CSV file was created
    test_file = tmp_path / "test_predictions.csv"
    assert test_file.exists()
    
    # Verify that log_metrics was not called (no summary statistics)
    mock_logger.log_metrics.assert_not_called()


def test_write_on_epoch_end_fewer_batch_indices_than_predictions(tmp_path: Path) -> None:
    """Test write_on_epoch_end when there are fewer batch indices than predictions."""
    writer = GraphLevelPredictionWriter(
        output_dir=str(tmp_path),
        save_fit_predictions=False,
        save_test_predictions=True,
    )
    
    # Create predictions with more samples than batch indices
    predictions = [
        [torch.tensor([0.5, 0.6, 0.7])],
    ]
    
    batch_indices = [
        [[0, 1]],  # Only 2 batch indices for 3 predictions
    ]
    
    trainer = Trainer(logger=None)
    module = DummyModule()
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="Number of batch indices .* is less than number of predictions"):
        writer.write_on_epoch_end(trainer, module, predictions, batch_indices)


def test_write_on_epoch_end_invalid_prediction_format(tmp_path: Path) -> None:
    """Test write_on_epoch_end with invalid prediction format."""
    writer = GraphLevelPredictionWriter(
        output_dir=str(tmp_path),
        save_fit_predictions=False,
        save_test_predictions=True,
    )
    
    # Create predictions with invalid format (not tensors)
    predictions = [
        [123, 456, 789],  # Invalid: list of integers instead of tensors
    ]
    
    batch_indices = [
        [[0, 1, 2]],
    ]
    
    trainer = Trainer(logger=None)
    module = DummyModule()
    
    # Should raise TypeError
    with pytest.raises(TypeError, match="Expected predictions to be"):
        writer.write_on_epoch_end(trainer, module, predictions, batch_indices)
