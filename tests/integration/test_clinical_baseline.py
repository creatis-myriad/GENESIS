from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from genesis.clinical_baseline import fit_and_score

from ..helpers.run import RunIf  # noqa: TID252


@pytest.fixture(scope="module")
def cfg_global(cfg_path: Path, application_overrides: list[str]) -> DictConfig:
    """A pytest fixture for setting up a Hydra DictConfig for a clinical baseline run.

    Args:
        cfg_path: The directory containing the Hydra configuration files.
        application_overrides: The overrides to use to specify the application (i.e. data, model, etc.) for the tests.

    Returns:
        A DictConfig object containing a Hydra configuration for a clinical baseline run.
    """
    with initialize(version_base=None, config_path=str(cfg_path)):
        cfg = compose(config_name="clinical_baseline.yaml", return_hydra_config=True, overrides=application_overrides)

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


@pytest.fixture(scope="module", params=[("risk", "multi_classification"), ("bnp", "regression")])
def data_overrides(request: FixtureRequest) -> list[str]:
    """A pytest fixture for the overrides to use to specify the data.

    Returns:
        A list of configuration overrides.
    """
    target, metrics = request.param
    return [
        "data=split_lightning_dataset",
        "data/dataset=persevere_clinical",
        f"data/dataset/target={target}",
        # Specify the metrics here, since they depend on the data task
        f"model/metrics={metrics}",
    ]


@pytest.fixture(scope="module", params=["tabpfn", "xgboost"])
def model_overrides(request: FixtureRequest) -> list[str]:
    """A pytest fixture for the overrides to use to specify the model for the tests.

    Returns:
        A list of configuration overrides.
    """
    model = request.param
    return [
        "model=tabular_estimator",
        f"model/components@model.model={model}",
    ]


@RunIf(xgboost=True)
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


@RunIf(xgboost=True)
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


@RunIf(xgboost=True)
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
