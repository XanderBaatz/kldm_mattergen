"""Kinetic Langevin predictor-corrector samplers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
from mattergen.diffusion.sampling.predictors_correctors import LangevinCorrector

from kldm_plus.diffusion.sampling.corrector import KineticLangevinCorrector
from kldm_plus.diffusion.sampling.predictor import KinLangevinEMPredictor, KinLangevinPosPredictor

if TYPE_CHECKING:
    from kldm_plus.diffusion.diffusion_module import KineticDiffusionModule

__all__ = [
    "KinLangevinEMPredictor",
    "KinLangevinPosPredictor",
    "KineticLangevinCorrector",
    "make_sampler",
]


def make_sampler(
    diffusion_module: KineticDiffusionModule,
    device: torch.device | str,
    N: int = 1000,  # noqa: N803
    n_corrector_steps: int = 1,
    eps_t: float = 1e-3,
    vel_snr: float = 0.2,
    cell_snr: float = 0.16,
) -> PredictorCorrector:
    """Assemble a kinetic-Langevin predictor-corrector sampler.

    Fields handled:
      * ``"vel"``  — EI predictor + Langevin corrector (both kinetic-Langevin aware)
      * ``"pos"``  — deterministic predictor driven by current velocity (no corrector)
      * ``"cell"`` — ancestral-sampling predictor + Langevin corrector (VPSDE)

    Args:
        diffusion_module: trained ``KineticDiffusionModule``.
        device: device to run sampling on.
        N: number of denoising steps.
        n_corrector_steps: Langevin corrector steps per noise level.
        eps_t: diffusion time to stop denoising at.
        vel_snr: SNR for the velocity Langevin corrector.
        cell_snr: SNR for the cell Langevin corrector.

    Returns:
        A configured ``PredictorCorrector`` ready for ``.sample(conditioning_data)``.

    """
    predictor_partials = {
        "vel": KinLangevinEMPredictor,
        "pos": KinLangevinPosPredictor,
        "cell": AncestralSamplingPredictor,
    }
    corrector_partials = {
        "vel": partial(KineticLangevinCorrector, snr=vel_snr),
        "cell": partial(LangevinCorrector, snr=cell_snr),
    }
    return PredictorCorrector(
        diffusion_module=diffusion_module,
        predictor_partials=predictor_partials,
        corrector_partials=corrector_partials,
        device=torch.device(device) if isinstance(device, str) else device,
        n_steps_corrector=n_corrector_steps,
        N=N,
        eps_t=eps_t,
    )
