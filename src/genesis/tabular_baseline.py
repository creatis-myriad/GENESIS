from pathlib import Path
from typing import Any

import hydra
import lightning as L  # noqa: N812
from lightning import LightningDataModule
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig
from rich.console import Console

from genesis.models import TabularEstimator
from genesis.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    hydra_serial_sweeper,
    instantiate_loggers,
    log_hyperparameters,
    metrics_table,
    pre_hydra_routine,
    task_wrapper,
)
from genesis.utils.logging_utils import (
    create_predictions_dataframe,
    log_dataframe,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def fit_and_score(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the baseline model to tabular data."""
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: TabularEstimator = hydra.utils.instantiate(cfg.model, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict, logger=logger)

    train_metrics = {}
    if ckpt_path := cfg.get("ckpt_path"):
        log.info(f"Loading model from {ckpt_path}!")
        model = model.load(ckpt_path)
        if cfg.get("train"):
            log.warning(
                "ckpt is provided, but training is enabled. Unless the model supports resuming training from a "
                "checkpoint, the loaded weights will be discarded by the newly trained model. \n"
                "To disable this warning, set 'train=False' (to use the ckpt) or 'ckpt_path=null' (to train a new "
                "model)."
            )

    if cfg.get("train"):
        log.info("Starting training!")
        model = model.fit(datamodule=datamodule, subset="train")
        train_metrics = model.score(datamodule=datamodule, subset="train")
        val_metrics = model.score(datamodule=datamodule, subset="val")
        train_metrics.update(val_metrics)

        if cfg.get("ckpt_save_filename"):
            ckpt_save_path = Path(cfg.ckpt_save_dirpath or cfg.paths.output_dir) / cfg.ckpt_save_filename
            log.info(f"Saving trained model to {ckpt_save_path}!")
            model.save(ckpt_save_path)
            _log_model(logger, ckpt_save_path, aliases=["model"])

        if cfg.get("ckpt_backbone_save_filename"):
            ckpt_backbone_save_path = (
                Path(cfg.ckpt_backbone_save_dirpath or cfg.paths.output_dir) / cfg.ckpt_backbone_save_filename
            )
            log.info(f"Saving trained model backbone to {ckpt_backbone_save_path}!")
            model.save_backbone(ckpt_backbone_save_path)
            print(ckpt_backbone_save_path)
            _log_model(logger, ckpt_backbone_save_path, aliases=["backbone"])

    test_metrics = {}
    if cfg.get("test"):
        log.info("Starting testing!")
        if not (cfg.get("ckpt_path") or cfg.get("train")):
            log.warning("ckpt not found! Using untrained model for testing... This is likely a mistake in your config.")
        test_metrics = model.score(datamodule=datamodule, subset="test")

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    Console().print(metrics_table(metric_dict, cols_from_prefixes=["train", "val", "test"]))

    if cfg.get("predict"):
        log.info("Starting predicting!")
        if not (cfg.get("ckpt_path") or cfg.get("train")):
            log.warning(
                "ckpt not found! Using untrained model for predicting... This is likely a mistake in your config."
            )

        # Determine which subsets to predict on
        predict_subsets = []
        if cfg.get("train"):
            predict_subsets.extend(["train", "val"])
        if cfg.get("test"):
            predict_subsets.append("test")

        # Get prediction configuration
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_labels = cfg.get("predictions_output_labels")
        samplewise_ops = cfg.get("predictions_samplewise_op")
        if samplewise_ops is None or isinstance(samplewise_ops, str):
            # If a single samplewise operation is provided, convert it to a list for consistency
            samplewise_ops = [samplewise_ops]
        if None not in samplewise_ops:
            # If any samplewise operation is specified, also save unmodified predictions by adding None as an operation
            samplewise_ops.append(None)

        for subset in predict_subsets:
            log.info(f"Predicting on {subset} set...")
            predictions = model.predict(datamodule=datamodule, subset=subset)

            for samplewise_op in samplewise_ops:
                # Create DataFrame from predictions
                df = create_predictions_dataframe(
                    predictions=predictions,
                    batch_indices=None,
                    output_labels=output_labels,
                    samplewise_op=samplewise_op,
                )

                # Save to CSV file
                op_suffix = f"_{samplewise_op}" if samplewise_op is not None else ""
                filepath = output_dir / f"{subset}{op_suffix}_predictions.csv"
                log.info(f"Saved predictions to {filepath}")
                df.to_csv(filepath, index=False)

                # Log to experiment tracker if available
                if logger:
                    log_dataframe(df, logger, filepath.stem)

    return metric_dict, object_dict


def _log_model(logger: list[Logger], ckpt_path: Path, aliases: list[str] | None = None) -> None:
    """Logs the model checkpoint to a WandB logger, if one is present and configured to log models."""
    for logger_instance in logger:
        if isinstance(logger_instance, WandbLogger) and logger_instance._log_model:
            wandb_run = logger_instance.experiment
            wandb_run.log_model(path=ckpt_path, aliases=aliases)


@hydra.main(version_base=None, config_path="configs", config_name="tabular_baseline.yaml")
@hydra_serial_sweeper
def hydra_main(cfg: DictConfig) -> float | None:
    """Hydra entry point for training.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        (Optional) optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = fit_and_score(cfg)

    # safely retrieve optimized metric value for hydra-based hyperparameter optimization
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


def main() -> float | None:
    """Main entry point for training, before Hydra is called.

    This is a workaround for issues with Python packaging tools requiring a function to target for script entrypoints.
    It provides a target for entrypoints that comes before Hydra is called, allowing for pre-Hydra routines to be run
    (e.g. setting up environment variables, registering custom OmegaConf resolvers etc.)
    """
    pre_hydra_routine()
    return hydra_main()


if __name__ == "__main__":
    main()
