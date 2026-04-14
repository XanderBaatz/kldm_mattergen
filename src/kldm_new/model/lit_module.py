"""PyTorch Lightning module for KLDM training and sampling."""

from __future__ import annotations

import warnings

import torch
from pymatgen.core import Lattice, Structure
from pytorch_lightning import LightningModule
from torch import Tensor

from kldm_new.data import add_velocity
from kldm_new.diffusion.corruption import KLDMMultiCorruption  # noqa: TC001
from kldm_new.diffusion.loss import KLDMLoss
from kldm_new.diffusion.sampling import KLDMSampler
from kldm_new.model import KLDMScoreModel  # noqa: TC001


class LitKLDM(LightningModule):
    """Lightning module for training and sampling with KLDM.

    Parameters
    ----------
    score_model : KLDMScoreModel
        The score model.
    multi_corruption : KLDMMultiCorruption
        The multi-corruption (TDM + cell SDE).
    loss_fn : KLDMLoss
        The combined loss function.
    lr : float
        Learning rate.
    with_ema : bool
        Enable exponential moving average of model weights.
    ema_decay : float
        EMA decay rate.
    ema_start : int
        Start EMA after this many epochs.
    sampling_N : int
        Number of reverse-time steps for sampling.
    sampling_corrector_steps : int
        Number of Langevin corrector steps per predictor step.

    """

    def __init__(  # noqa: D107, PLR0913
        self,
        score_model: KLDMScoreModel,
        multi_corruption: KLDMMultiCorruption,
        loss_fn: KLDMLoss | None = None,
        lr: float = 1e-4,
        with_ema: bool = True,
        ema_decay: float = 0.999,
        ema_start: int = 100,
        sampling_N: int = 1000,
        sampling_corrector_steps: int = 1,
    ) -> None:
        super().__init__()
        self.score_model = score_model
        self.multi_corruption = multi_corruption
        self.loss_fn = loss_fn or KLDMLoss()

        if with_ema:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                score_model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
            )
        else:
            self.ema_model = None

        self.save_hyperparameters(ignore=["score_model", "multi_corruption", "loss_fn"])

    # ---- Training ---------------------------------------------------------

    def _basic_step(self, batch):
        """Single training step: corrupt → predict → loss."""
        batch = add_velocity(batch)

        batch_size = batch.get_batch_size()
        t = self._sample_t(batch_size)

        # Corrupt
        noisy_batch = self.multi_corruption.sample_marginal(batch, t)

        # Predict
        score_out = self.score_model(noisy_batch, t)

        # Loss
        total_loss, loss_dict = self.loss_fn(
            multi_corruption=self.multi_corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_out,
            t=t,
        )
        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._basic_step(batch)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()})
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_model is not None and self.current_epoch > self.hparams.ema_start:
            self.ema_model.update_parameters(self.score_model)

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._basic_step(batch)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, on_epoch=True)
        return loss

    # ---- Sampling ---------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch,
        force_ema: bool = True,
        N: int | None = None,
        n_corrector_steps: int | None = None,
    ) -> list[Structure]:
        """Generate crystal structures via reverse-time sampling.

        Args:
            batch: A batch providing ``num_atoms`` and ``atomic_numbers``.
            force_ema: Use EMA model if available.
            N: Override number of sampling steps.
            n_corrector_steps: Override corrector steps.

        Returns:
            List of pymatgen :class:`Structure` objects.

        """
        model = self._get_model(ema=force_ema)
        batch = add_velocity(batch)

        sampler = KLDMSampler(
            multi_corruption=self.multi_corruption,
            score_fn=lambda b, t: model(b, t),
            N=N or self.hparams.sampling_N,
            n_corrector_steps=n_corrector_steps or self.hparams.sampling_corrector_steps,
        )

        result_batch = sampler.sample(conditioning_data=batch)

        try:
            structures = self._structures_from_batch(result_batch)
        except Exception as e:
            warnings.warn(f"Structure conversion failed: {e}")
            return []
        return structures

    # ---- Helpers ----------------------------------------------------------

    def _sample_t(self, batch_size: int) -> Tensor:
        """Sample uniform t ∈ (ε, T]."""
        T = self.multi_corruption.T
        eps = 1e-3
        return (eps - T) * torch.rand(batch_size, 1, device=self.device) + T

    def _get_model(self, ema: bool = False) -> KLDMScoreModel:  # noqa: FBT001, FBT002
        if self.ema_model and (ema or self.current_epoch > self.hparams.ema_start):
            return self.ema_model.module
        return self.score_model

    def _structures_from_batch(self, batch: int) -> list[Structure]:
        """Convert a sampled batch to pymatgen Structures."""
        from ase.data import chemical_symbols

        pos = batch["pos"].cpu().numpy()
        cell = batch["cell"].cpu().numpy()
        h = batch["atomic_numbers"].cpu()
        if h.ndim > 1:
            h = torch.argmax(h, dim=1)
        h = h.numpy()

        ptr = batch.ptr.cpu().numpy()
        structures = []

        for i, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):  # noqa: B905, RUF007
            coords = pos[start:end] % 1.0  # fractional, [0,1)
            symbols = [chemical_symbols[idx] for idx in h[start:end]]
            lattice = Lattice(cell[i])
            struct = Structure(
                lattice=lattice,
                species=symbols,
                coords=coords,
                coords_are_cartesian=False,
            )
            struct = struct.get_sorted_structure()
            structures.append(struct)

        return structures

    def configure_optimizers(self):  # noqa: ANN201, D102
        return torch.optim.AdamW(
            self.score_model.parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
            weight_decay=1e-12,
        )
