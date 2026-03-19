from pathlib import Path

import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from genesis.tabular_baseline import fit_and_score

from ..helpers.run import RunIf  # noqa: TID252


@pytest.fixture(scope="module")
def cfg_global(cfg_path: Path, application_overrides: list[str]) -> DictConfig:
    """A pytest fixture for setting up a Hydra DictConfig for a tabular baseline run.

    Args:
        cfg_path: The directory containing the Hydra configuration files.
        application_overrides: The overrides to use to specify the application (i.e. data, model, etc.) for the tests.

    Returns:
        A DictConfig object containing a Hydra configuration for a tabular baseline run.
    """
    with initialize(version_base=None, config_path=str(cfg_path)):
        cfg = compose(config_name="tabular_baseline.yaml", return_hydra_config=True, overrides=application_overrides)

        # set defaults for all tests
        with open_dict(cfg):
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False

    return cfg


@pytest.fixture
def cfg(cfg_global: DictConfig, shared_datadir: Path, tmp_path: Path) -> DictConfig:
    """Modifies the `cfg_global()` fixture to use temporary dummy data and logging paths.

    This is called by each test which uses the `cfg` arg. Each test generates its own temporary logging path.

    Args:
        cfg_global: The input DictConfig object to be modified.
        shared_datadir: The directory containing the dummy data.
        tmp_path: The temporary logging path.

    Returns:
        A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_global.copy()

    with open_dict(cfg):
        # Use a shared directory of dummy data, structured like the real PERSEVERE data
        cfg.paths.data_dir = str(shared_datadir)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.output_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="module")
def application_overrides(data_overrides: list[str], model_overrides: list[str]) -> list[str]:
    """A pytest fixture for the overrides to use to specify the application (i.e. data, model, etc.) for the tests.

    Returns:
        A list of configuration overrides.
    """
    return [*data_overrides, *model_overrides]


@pytest.fixture(
    scope="module",
    params=[
        ("risk_ESC-2014", "multi_classification"),
        ("risk_ESC-2014_elevated", "binary_classification"),
        ("troponin", "regression"),
    ],
)
def data_overrides(request: FixtureRequest) -> list[str]:
    """A pytest fixture for the overrides to use to specify the data.

    Returns:
        A list of configuration overrides.
    """
    target, metrics = request.param
    return [
        "data=split_lightning_dataset",
        "data/dataset=persevere_global_features",
        f"data/dataset/target={target}",
        # Specify the metrics here, since they depend on the data task
        f"model/metrics={metrics}",
    ]


@pytest.fixture(scope="module", params=["tabicl", "tabpfn", "xgboost"])
def model_overrides(request: FixtureRequest) -> list[str]:
    """A pytest fixture for the overrides to use to specify the model for the tests.

    Returns:
        A list of configuration overrides.
    """
    model = request.param
    return [
        "model=tabular_estimator",
        f"model/model={model}",
    ]


@RunIf(baselines=True)
@pytest.mark.slow
def test_train(cfg: DictConfig) -> None:
    """Fit model on training data and score on training/validation data.

    Args:
        cfg: A DictConfig containing a valid configuration.
    """
    HydraConfig().set_config(cfg)
    metric_dict, _ = fit_and_score(cfg)

    # Check that both training and validation metrics (and only those two) are present
    assert any(metric.startswith("train/") for metric in metric_dict)
    assert any(metric.startswith("val/") for metric in metric_dict)
    assert all(metric.startswith(("train/", "val/")) for metric in metric_dict)


@RunIf(baselines=True)
@pytest.mark.slow
def test_train_eval(cfg: DictConfig) -> None:
    """Fit model on training data and score on training/validation/test data.

    Args:
        cfg: A DictConfig containing a valid configuration.
    """
    with open_dict(cfg):
        cfg.test = True

    HydraConfig().set_config(cfg)
    metric_dict, _ = fit_and_score(cfg)

    # Check that training, validation, and test metrics (and only those three) are present
    assert any(metric.startswith("train/") for metric in metric_dict)
    assert any(metric.startswith("val/") for metric in metric_dict)
    assert any(metric.startswith("test/") for metric in metric_dict)
    assert all(metric.startswith(("train/", "val/", "test/")) for metric in metric_dict)


@RunIf(baselines=True)
@pytest.mark.slow
def test_train_resume_eval(tmp_path: Path, cfg: DictConfig) -> None:
    """Fit model on training data, save model, then load model and score on test data.

    Args:
        tmp_path: The temporary logging path.
        cfg: A DictConfig containing a valid configuration.
    """
    with open_dict(cfg):
        cfg.test = True

    HydraConfig().set_config(cfg)
    train_metric_dict, _ = fit_and_score(cfg)

    with open_dict(cfg):
        cfg.train = False
        cfg.ckpt_path = str(tmp_path / "model.pickle")

    HydraConfig().set_config(cfg)
    test_metric_dict, _ = fit_and_score(cfg)

    # Check that only test metrics are present
    assert all(metric.startswith("test/") for metric in test_metric_dict)

    # Check that the training and test metrics reported on the test set are close,
    # i.e. that the model was saved and loaded correctly
    metric = next(iter(test_metric_dict))
    assert abs(train_metric_dict[metric].item() - test_metric_dict[metric].item()) < 0.001


@RunIf(baselines=True)
@pytest.mark.slow
def test_train_predict(tmp_path: Path, cfg: DictConfig) -> None:
    """Fit model on training data and generate predictions on train/val/test sets.

    Args:
        tmp_path: The temporary logging path.
        cfg: A DictConfig containing a valid configuration.
    """
    with open_dict(cfg):
        cfg.test = True
        cfg.predict = True
    predictions_dir = tmp_path / "predictions"

    HydraConfig().set_config(cfg)
    metric_dict, _ = fit_and_score(cfg)

    # Check that metrics were computed
    assert any(metric.startswith("train/") for metric in metric_dict)
    assert any(metric.startswith("val/") for metric in metric_dict)
    assert any(metric.startswith("test/") for metric in metric_dict)

    # Check that prediction CSV files were created
    assert (predictions_dir / "train_predictions.csv").exists()
    assert (predictions_dir / "val_predictions.csv").exists()
    assert (predictions_dir / "test_predictions.csv").exists()

    # Verify that the CSV files contain predictions with proper structure
    train_df = pd.read_csv(predictions_dir / "train_predictions.csv")
    val_df = pd.read_csv(predictions_dir / "val_predictions.csv")
    test_df = pd.read_csv(predictions_dir / "test_predictions.csv")

    # Check that all dataframes have predictions and proper structure
    for df, subset in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        assert len(df) > 0, f"{subset} DataFrame is empty"
        # Check that DataFrame has either 'prediction' column (regression) or numbered columns (classification)
        assert "prediction" in df.columns or "0" in df.columns, f"{subset} DataFrame missing prediction columns"
