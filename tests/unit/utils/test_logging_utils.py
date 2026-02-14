from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genesis.utils.logging_utils import create_predictions_dataframe, save_predictions_to_csv


class TestCreatePredictionsDataframe:
    """Tests for create_predictions_dataframe function."""

    def test_regression_predictions_without_batch_indices(self) -> None:
        """Test creating DataFrame for regression predictions without batch indices."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = create_predictions_dataframe(predictions)

        assert list(df.columns) == ["prediction"]
        assert len(df) == 5
        assert np.allclose(df["prediction"].values, predictions)

    def test_regression_predictions_with_batch_indices(self) -> None:
        """Test creating DataFrame for regression predictions with batch indices."""
        predictions = np.array([1.0, 2.0, 3.0])
        batch_indices = [0, 1, 2]
        df = create_predictions_dataframe(predictions, batch_indices=batch_indices)

        assert list(df.columns) == ["prediction", "batch_idx"]
        assert len(df) == 3
        assert np.allclose(df["prediction"].values, predictions)
        assert df["batch_idx"].tolist() == batch_indices

    def test_classification_predictions_without_labels(self) -> None:
        """Test creating DataFrame for classification predictions without output labels."""
        predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        df = create_predictions_dataframe(predictions)

        assert list(df.columns) == ["0", "1"]
        assert len(df) == 3
        assert np.allclose(df["0"].values, predictions[:, 0])
        assert np.allclose(df["1"].values, predictions[:, 1])

    def test_classification_predictions_with_labels(self) -> None:
        """Test creating DataFrame for classification predictions with output labels."""
        predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        output_labels = ["class_a", "class_b"]
        df = create_predictions_dataframe(predictions, output_labels=output_labels)

        assert list(df.columns) == ["class_a", "class_b"]
        assert len(df) == 3
        assert np.allclose(df["class_a"].values, predictions[:, 0])
        assert np.allclose(df["class_b"].values, predictions[:, 1])

    def test_classification_predictions_with_batch_indices(self) -> None:
        """Test creating DataFrame for classification predictions with batch indices."""
        predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        batch_indices = [0, 1]
        df = create_predictions_dataframe(predictions, batch_indices=batch_indices)

        assert list(df.columns) == ["0", "1", "batch_idx"]
        assert len(df) == 2
        assert df["batch_idx"].tolist() == batch_indices

    def test_softmax_operation(self) -> None:
        """Test applying softmax operation to predictions."""
        predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
        df = create_predictions_dataframe(predictions, samplewise_op="softmax")

        # Check that values are valid probabilities
        row_sums = df[["0", "1"]].sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_argmax_operation(self) -> None:
        """Test applying argmax operation to predictions."""
        predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        df = create_predictions_dataframe(predictions, samplewise_op="argmax")

        assert list(df.columns) == ["prediction"]
        assert len(df) == 3
        assert df["prediction"].tolist() == [1, 0, 1]

    def test_invalid_samplewise_op(self) -> None:
        """Test that invalid samplewise operation raises ValueError."""
        predictions = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unsupported samplewise operation"):
            create_predictions_dataframe(predictions, samplewise_op="invalid_op")


class TestSavePredictionsToCsv:
    """Tests for save_predictions_to_csv function."""

    def test_save_to_csv(self, tmp_path: Path) -> None:
        """Test saving predictions DataFrame to CSV file."""
        df = pd.DataFrame({"prediction": [1.0, 2.0, 3.0], "batch_idx": [0, 1, 2]})
        output_dir = tmp_path / "predictions"
        filename = "test_predictions.csv"

        filepath = save_predictions_to_csv(df, output_dir, filename)

        assert filepath.exists()
        assert filepath.name == filename
        assert filepath.parent == output_dir

        # Verify file contents
        loaded_df = pd.read_csv(filepath)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        df = pd.DataFrame({"prediction": [1.0, 2.0, 3.0]})
        output_dir = tmp_path / "new" / "directory" / "predictions"
        filename = "test.csv"

        filepath = save_predictions_to_csv(df, output_dir, filename)

        assert output_dir.exists()
        assert filepath.exists()
