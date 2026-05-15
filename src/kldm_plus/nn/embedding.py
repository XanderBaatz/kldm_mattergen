import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class SinEmbedding(nn.Module):
    """Sinusoidal distance / feature embedding.

    Embeds a scalar input into 2 * n_frequencies dimensions using:

        [
            sin(2^0 π x),
            cos(2^0 π x),
            sin(2^1 π x),
            cos(2^1 π x),
            ...,
            sin(2^{K-2} π x),
            cos(2^{K-2} π x)
            sin(2^{K-1} π x),
            cos(2^{K-1} π x)
        ]

    """

    def __init__(
        self,
        n_frequencies: int = 10,  # K/2
        n_space: int = 3,
    ) -> None:
        """Initialize sinusoidal embedding settings.

        Args:
            n_frequencies (int): Number of sinusoidal frequencies.
            n_space (int): Number of spatial dimensions in the input.

        """
        super().__init__()

        self.n_frequencies = n_frequencies
        self.n_space = n_space
        # self._dim = 2 * n_frequencies # 2 * K
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): input vector, e.g. 3D displacement. The L2 norm is computed and embedded.

        Returns:
            Tensor: embedded tensor

        """
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class FourierEmbedding(nn.Module):
    """Random Fourier features (sine and cosine expansion)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std: float = 1.0,
        *,
        trainable: bool = False,
    ) -> None:
        """Initialize random Fourier feature projection.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension (must be even).
            std (float): Standard deviation for random weight initialization.
            trainable (bool): Whether projection weights are trainable.

        """
        super().__init__()
        if (out_features % 2) != 0:
            msg = "out_features must be even"
            raise ValueError(msg)
        weight = torch.normal(mean=torch.zeros(out_features // 2, in_features), std=std)

        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, x: Tensor) -> Tensor:
        """Project inputs with random Fourier features and return cos/sin concatenation.

        Args:
            x (Tensor): Input tensor of shape ``(..., in_features)``.

        Returns:
            Tensor: Fourier-embedded tensor of shape ``(..., out_features)``.

        """
        x = F.linear(x, self.weight)

        cos_features = torch.cos(2 * math.pi * x)
        sin_features = torch.sin(2 * math.pi * x)

        return torch.cat((cos_features, sin_features), dim=1)


class TimeEmbedding(nn.Module):
    """Sinusoidal time-step embedding module."""

    def __init__(
        self,
        out_features: int,
    ) -> None:
        """Initialize sinusoidal time embedding frequencies.

        Args:
            out_features (int): Output embedding dimension.

        """
        super().__init__()

        half = out_features // 2
        v = math.log(10_000) / (half - 1)
        f = torch.exp(torch.arange(half) * -v)
        self.register_buffer("f", f)

    def forward(
        self,
        t: Tensor,
    ) -> Tensor:
        """Compute sinusoidal embedding for time steps.

        Args:
            t (Tensor): Time-step tensor of shape ``(..., 1)`` or ``(..., half)``
                broadcastable with the internal frequency buffer.

        Returns:
            Tensor: Time embedding tensor of shape ``(..., out_features)``.

        """
        x = t * self.f[None, :]

        return torch.cat((torch.cos(x), torch.sin(x)), dim=1)
