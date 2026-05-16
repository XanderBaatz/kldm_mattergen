"""Training entry point for kldm_plus.

Usage
-----
# Default config (de-novo, 3x3 lattice):
    uv run python -m kldm_plus.train

# Original KLDM config (6D lattice, kldm_frnct-equivalent):
    uv run python -m kldm_plus.train --config-name=kldm

# Crystal structure prediction:
    uv run python -m kldm_plus.train --config-name=csp

# Override individual parameters:
    uv run python -m kldm_plus.train --config-name=kldm trainer.devices=2

Environment variables
---------------------
PROJECT_ROOT  Root of the repository (default: cwd).  Used by data_module configs
              to locate datasets under $PROJECT_ROOT/../datasets/cache/.
OUTPUT_DIR    Directory for checkpoints / logs (default: outputs/<timestamp>).
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT  # noqa: F401 — registers eval resolver
from mattergen.diffusion.config import Config
from mattergen.diffusion.run import main
from omegaconf import DictConfig, OmegaConf

from kldm_plus.settings import Settings
from tools.logger import Logger, LogType

_settings = Settings()
logger = Logger(
    __name__,
    log_type=LogType.LOCAL,
)

_CONFIGS = Path(__file__).parent / "configs"


@hydra.main(
    config_path=str(_CONFIGS),
    config_name="default",
    version_base="1.1",
)
def train(cfg: DictConfig) -> None:
    """Train script."""
    # Use module-level _settings — Hydra changes cwd before calling this
    # function, so instantiating Settings() here would fail to find .env.local.
    settings = _settings

    # Inject paths and secrets from .env.local into os.environ so that
    # Hydra's oc.env resolver and WandB can read them.
    os.environ.setdefault("DATA_PATH", str(settings.data_path))
    if settings.WANDB_API_KEY:
        os.environ.setdefault("WANDB_API_KEY", settings.WANDB_API_KEY)
    os.environ.setdefault("WANDB_PROJECT", settings.WANDB_PROJECT)
    os.environ.setdefault("WANDB_MODE", settings.WANDB_MODE)

    logger.info("data_path=%s  config=%s", settings.data_path, settings.KLDM_CONFIG)
    torch.set_float32_matmul_precision("high")

    # Merge with mattergen's Config schema so checkpoint_path and other
    # structured fields are present (mirrors mattergen/scripts/run.py).
    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, cfg)
    OmegaConf.set_readonly(
        config,
        True,  # noqa: FBT003
    )

    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=False))  # noqa: G003

    main(config)


if __name__ == "__main__":
    train()
