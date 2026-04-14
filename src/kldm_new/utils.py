"""KLDM-New utilities — logging, Hydra helpers, callback instantiation."""

from __future__ import annotations

import logging
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

SRC_ROOT: Path = Path(__file__).resolve().parents[1]

log = get_pylogger(__name__) if False else logging.getLogger(__name__)  # avoid circular


# ---------------------------------------------------------------------------
# Multi-GPU-safe logger
# ---------------------------------------------------------------------------


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Return a logger where all levels are rank-zero-only in multi-GPU runs."""
    logger = logging.getLogger(name)
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger


# Re-create module-level logger properly
log = get_pylogger(__name__)


# ---------------------------------------------------------------------------
# Hydra instantiation helpers
# ---------------------------------------------------------------------------


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Callback]:
    """Instantiate all callbacks from a Hydra DictConfig."""
    callbacks: list[Callback] = []
    if not callbacks_cfg:
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    """Instantiate all loggers from a Hydra DictConfig."""
    loggers: list[Logger] = []
    if not logger_cfg:
        return loggers
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")
    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


# ---------------------------------------------------------------------------
# Hyperparameter logging
# ---------------------------------------------------------------------------


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Log selected hyperparameters to all trainer loggers (e.g. W&B)."""
    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    lit_module = object_dict["lit_module"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        return

    hparams: dict[str, Any] = {}
    hparams["lit_module"] = cfg.get("lit_module")
    hparams["datamodule"] = cfg.get("datamodule")
    hparams["trainer"] = cfg.get("trainer")
    hparams["callbacks"] = cfg.get("callbacks")
    hparams["tags"] = cfg.get("tags")
    hparams["seed"] = cfg.get("seed")
    hparams["model/params/total"] = sum(p.numel() for p in lit_module.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in lit_module.parameters() if p.requires_grad)

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


# ---------------------------------------------------------------------------
# Metric retrieval (for Hydra HPO)
# ---------------------------------------------------------------------------


def get_metric_value(metric_dict: dict, metric_name: str | None) -> float | None:
    """Retrieve a metric value safely from the trainer callback dict."""
    if not metric_name:
        return None
    if metric_name not in metric_dict:
        raise KeyError(
            f"Metric '{metric_name}' not found in {list(metric_dict.keys())}. Check that LitKLDM logs it and the name matches `optimized_metric`."
        )
    return metric_dict[metric_name].item()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def close_loggers() -> None:
    """Ensure W&B (and similar) finish cleanly — critical for multirun."""
    if find_spec("wandb"):
        import wandb

        if wandb.run:
            wandb.finish()
