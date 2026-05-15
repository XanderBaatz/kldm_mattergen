"""Kinetic Langevin Diffusion Model — Lightning wrapper.

Thin subclass of mattergen's DiffusionLightningModule; the only purpose is to
give the module a KLDM-specific name and to restrict the diffusion_module type
to KineticDiffusionModule so that type-checkers can surface misconfigurations.
All training logic (training_step, validation_step, configure_optimizers,
load_from_checkpoint, …) is inherited unchanged from the base class.
"""

from __future__ import annotations

from mattergen.diffusion.lightning_module import DiffusionLightningModule

from kldm_plus.diffusion.diffusion_module import KineticDiffusionModule


class KLDMLightningModule(DiffusionLightningModule[KineticDiffusionModule]):
    """LightningModule for the Kinetic Langevin Diffusion Model.

    Wraps a :class:`KineticDiffusionModule` and inherits all training logic
    (optimizer, scheduler, loss logging) from
    :class:`mattergen.diffusion.lightning_module.DiffusionLightningModule`.
    """
