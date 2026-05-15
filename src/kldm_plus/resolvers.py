"""Custom OmegaConf resolvers for kldm_plus configs.

Usage in YAML::

    scale_pos: ${eval:2*pi}
    batch_size: ${eval:'512 // 4'}
"""

import math

from omegaconf import OmegaConf

_EVAL_GLOBALS = {k: v for k, v in vars(math).items() if not k.startswith("_")}


def _eval_resolver(expr: str) -> float:
    """Safely evaluate a numeric expression using the ``math`` module namespace."""
    return float(eval(expr, {"__builtins__": {}}, _EVAL_GLOBALS))  # noqa: S307


OmegaConf.register_new_resolver("eval", _eval_resolver, replace=True)
