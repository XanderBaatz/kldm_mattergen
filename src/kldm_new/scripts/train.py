"""KLDM-New training entry-point with Hydra + PyTorch Lightning.

Usage
-----
# Local (defaults)
python -m kldm_new.scripts.train

# Override from CLI
python -m kldm_new.scripts.train trainer.devices=4 datamodule.train_batch_size=512

# Debug (tiny model, CPU, no logging)
python -m kldm_new.scripts.train +experiment=debug

# Resume from checkpoint
python -m kldm_new.scripts.train ckpt_path=/path/to/checkpoint.ckpt

# HPC / SLURM (multi-node)
python -m kldm_new.scripts.train trainer.devices=4 trainer.num_nodes=2 trainer.strategy=ddp
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import Logger

from kldm_new.utils import (
    close_loggers,
    get_metric_value,
    get_pylogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = get_pylogger(__name__)

# Resolve config path relative to *this* file (kldm_new/scripts/train.py)
_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------


def train(cfg: DictConfig) -> tuple[dict, dict[str, Any]]:
    """Instantiate all components from *cfg* and run training.

    Returns
    -------
    metric_dict : dict
        Merged train + test callback metrics.
    object_dict : dict
        References to all instantiated objects (for inspection / HPO).

    """
    # Seed
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # DataModule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # LightningModule
    log.info(f"Instantiating lit_module <{cfg.lit_module._target_}>")
    lit_module: pl.LightningModule = hydra.utils.instantiate(cfg.lit_module)

    # Callbacks
    log.info("Instantiating callbacks …")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Loggers
    log.info("Instantiating loggers …")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "lit_module": lit_module,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters …")
        log_hyperparameters(object_dict)

    # --- Train ---
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=lit_module,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    # --- Test ---
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = cfg.get("ckpt_path")
        if not ckpt_path:
            ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
            if not ckpt_path:
                log.warning("No best checkpoint found – using current weights.")
                ckpt_path = None
        trainer.validate(model=lit_module, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    close_loggers()

    return metric_dict, object_dict


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=_CONFIGS_DIR,
    config_name="train_mp_20",
    version_base="1.3",
)
def main(cfg: DictConfig) -> float | None:
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict, cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
