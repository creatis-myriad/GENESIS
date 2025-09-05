from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from genesis.eval import evaluate
from genesis.train import train

XFAIL_HYDRA_CHOICES = {
    (
        ("model/encoder", "gps"),
        ("data/dataset", "enzymes"),
    ): "GPS takes longer to train than other models, and usually fails to achieve >0% test accuracy with only "
    "1 training epoch on the ENZYMES dataset.",
}


@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with `eval.py`.

    Args:
        tmp_path: The temporary logging path.
        cfg_train: A DictConfig containing a valid training configuration.
        cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    hydra_choices = HydraConfig().get().runtime.choices
    # Test if the current Hydra config matches any of the configs expected to fail
    for conditions, reason in XFAIL_HYDRA_CHOICES.items():
        if all(hydra_choices.get(param) == value for param, value in conditions):
            pytest.xfail(reason)

    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in {child.name for child in (tmp_path / "checkpoints").iterdir()}

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/acc"] > 0.0
    assert abs(train_metric_dict["test/acc"].item() - test_metric_dict["test/acc"].item()) < 0.001
