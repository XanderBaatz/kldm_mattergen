from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """kldm_plus runtime settings loaded from environment variables.

    All fields can be set via environment variables (case-insensitive) or
    in a ``.env.local`` file at the repo root.  Sensible defaults are
    provided so the training script works out-of-the-box locally.

    Examples
    --------
    In ``.env.local``::

        DATA_PATH=/workspace/data
        KLDM_CONFIG=kldm_debug
        WANDB_API_KEY=wandb_v1_...

    """

    model_config = SettingsConfigDict(
        env_file=(
            ".env",
            Path.home() / ".env.local",  # user-level overrides (~/.env.local)
            ".env.local",  # repo-level overrides (takes precedence)
        ),
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrecognised env vars (e.g. system variables)
    )

    # ---------- Environment ----------
    IS_LOCAL: bool = False
    """True when running locally (enables coloured console logging)."""

    # ---------- Paths ----------
    PROJECT_ROOT: Path = Path()
    """Root of the repository.  Used by Hydra configs to locate datasets."""

    DATA_PATH: Path | None = None
    """Directory containing ``mp_20/processed/`` etc.
    Defaults to ``$HOME/kldm_mattergen/data`` when not set."""

    LOG_PATH: Path = Path.home() / "kldm_logs"
    """Directory for LSF/HPC log files."""

    # ---------- Training ----------
    KLDM_CONFIG: str = "kldm_csp"
    """Hydra config name (e.g. ``kldm``, ``kldm_csp``, ``kldm_debug``)."""

    # ---------- WandB ----------
    WANDB_API_KEY: str | None = None
    """WandB API key.  Set in ``.env.local`` — never commit this."""

    WANDB_PROJECT: str = "kldm_plus"
    WANDB_RUN_GROUP: str = "local_runs"
    WANDB_MODE: str = "online"

    @property
    def data_path(self) -> Path:
        """Resolved data root (falls back to ~/kldm_mattergen/data)."""
        return self.DATA_PATH if self.DATA_PATH is not None else Path.home() / "kldm_mattergen" / "data"
