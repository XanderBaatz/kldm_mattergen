"""Neural-network utility modules: embeddings and scatter helpers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinEmbedding(nn.Module):
    """Sinusoidal distance / feature embedding.

    Embeds a scalar input into ``2 * n_frequencies`` dimensions using
    ``[sin(2^0 π x), cos(2^0 π x), ..., sin(2^{K-1} π x), cos(2^{K-1} π x)]``.
    """

    def __init__(self, n_frequencies: int = 10):
        super().__init__()
        self.n_frequencies = n_frequencies
        self._dim = 2 * n_frequencies
        # Compute in float64 to avoid overflow, then cast.
        # π·2^k overflows float32 at k≥128 (π·2^127 ≈ 5.3e38 > float32_max).
        # Clamp to float32 max so the buffer is always finite.
        freqs = torch.pi * (2.0 ** torch.arange(n_frequencies, dtype=torch.float64))
        freqs = freqs.clamp(max=torch.finfo(torch.float32).max).float()
        self.register_buffer("freqs", freqs)

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``(..., d)`` input vector (e.g. 3-D displacement).
               The L2 norm is computed and embedded.

        Returns:
            ``(..., 2 * n_frequencies)`` embedded tensor.

        """
        # Compute norm for vector inputs, keep scalars as-is.
        # Clamp before sqrt to avoid NaN gradient of norm at zero
        # (which occurs when two atoms share the same fractional coordinate).
        if x.ndim >= 2 and x.shape[-1] > 1:
            x = x.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-16).sqrt()  # (..., 1)
        elif x.ndim == 1:
            x = x.unsqueeze(-1)  # (..., 1)
        # x: (..., 1) × freqs: (K,) → (..., K)
        xf = x * self.freqs  # (..., K)
        return torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1)  # (..., 2K)


class FourierEmbedding(nn.Module):
    """Fourier time embedding (random Fourier features).

    Maps a scalar ``t`` to ``out_features`` dimensions via learned random
    frequencies, matching the implementation in kldm_frnct.
    """

    def __init__(self, in_features: int = 1, out_features: int = 128):
        super().__init__()
        # Random frequencies (not learned)
        self.register_buffer("W", torch.randn(in_features, out_features // 2) * 2 * math.pi)
        self.linear = nn.Linear(out_features, out_features)

    def forward(self, t: Tensor) -> Tensor:
        """Args:
            t: ``(B, 1)`` or ``(B,)`` time values.

        Returns:
            ``(B, out_features)`` Fourier features.

        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        proj = t @ self.W  # (B, out_features // 2)
        emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, out_features)
        return self.linear(emb)
