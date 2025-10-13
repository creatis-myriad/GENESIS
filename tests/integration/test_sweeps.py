import os
from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest

from ..helpers.run import RunIf, run_sh_command  # noqa: TID252


@pytest.fixture(scope="module", params=["src/genesis/train.py"])
def script(request: FixtureRequest) -> Path:
    """A pytest fixture for the training script.

    Returns:
        The path to the training script.
    """
    # This has to be a fixture rather than a module-level variable because it relies on the PROJECT_ROOT env var
    # having been set by another session-scoped fixture
    return Path(os.environ["PROJECT_ROOT"], request.param)


@pytest.fixture
def testing_overrides(tmp_path: Path) -> list[str]:
    """A pytest fixture for the overrides to use for tests (e.g. logging, accelerator, compilation, etc.).

    Returns:
        A list of configuration overrides.
    """
    return [
        "hydra.run.dir=" + str(tmp_path),
        "hydra.sweep.dir=" + str(tmp_path),
        "logger=[]",
        # Uncomment the following line if compilation is enabled by default to disable it for tests
        # "compile=false",  # Disable compilation to speed up tests
    ]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(script: Path, testing_overrides: list[str]) -> None:
    """Test running all available experiment configs (except for tabular baselines) with `fast_dev_run=True`.

    Args:
        script: The path of the script to invoke.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(script),
        "-m",
        "experiment=glob(*,exclude=tabular_baseline/*)",
        "++trainer.fast_dev_run=true",
        *testing_overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.parametrize("script", ["src/genesis/tabular_baseline.py"], indirect=True)
def test_tabular_baseline_experiments(script: Path, shared_datadir: Path, testing_overrides: list[str]) -> None:
    """Test running all available tabular baseline experiment configs.

    Args:
        script: The path of the script to invoke.
        shared_datadir: The directory containing the dummy data.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(script),
        "-m",
        "experiment=glob(tabular_baseline/*)",
        f"paths.data_dir={shared_datadir}",  # Override the data path with the test dummy data path
        *testing_overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(script: Path, testing_overrides: list[str]) -> None:
    """Test default hydra sweep.

    Args:
        script: The path of the script to invoke.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(script),
        "-m",
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
        *testing_overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(script: Path, testing_overrides: list[str]) -> None:
    """Test Optuna hyperparam sweeping.

    Args:
        script: The path of the script to invoke.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(script),
        "-m",
        "hparams_search=graph_classification_optuna",
        "~serial_sweeper",  # Disable the serial sweeper here to test it separately
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
        *testing_overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_serial_sweep(script: Path, testing_overrides: list[str]) -> None:
    """Test single-process serial sweeping.

    Args:
        script: The path of the script to invoke.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(script),
        "serial_sweeper=splits",
        "++trainer.fast_dev_run=true",
        *testing_overrides,
    ]
    run_sh_command(command)
