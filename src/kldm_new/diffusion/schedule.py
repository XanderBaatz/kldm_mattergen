from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F  # noqa: N812


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""

    @abstractmethod
    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha(t) = exp(-0.5 * ∫_0^t β(s) ds). Shape mirrors t."""

    @abstractmethod
    def std(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the noise standard deviation at time t. Shape mirrors t."""


class VPNoiseSchedule(NoiseSchedule):
    """Variance Preserving (VP) noise schedule base class.

    Derived classes must implement :meth:`beta` and :meth:`_integral_beta`.
    Default :meth:`mean_coeff` and :meth:`std` are derived automatically:

    * ``mean_coeff(t) = exp(-0.5 * ∫_0^t β(s) ds)``
    * ``std(t) = sqrt(1 - mean_coeff(t)²)``

    For the Sub-VP SDE the variance used is ``(1 - mean_coeff²)²`` (not
    ``1 - mean_coeff²``); this class exposes the raw quantities and Sub-VP
    applies the additional squaring internally.
    """

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """alpha(t) = exp(-0.5 * ∫_0^t β(s) ds)."""
        return torch.exp(-0.5 * self._integral_beta(t))

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """VP noise standard deviation: sqrt(1 - alpha(t)²)."""
        return torch.sqrt(1.0 - self.mean_coeff(t) ** 2)

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise rate β(t). Shape mirrors t."""

    @abstractmethod
    def _integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Closed-form antiderivative ∫_0^t β(s) ds. Shape mirrors t."""


class LinearSchedule(VPNoiseSchedule):
    r"""Linear beta schedule (Song et al., 2021, ICLR; Appendix C).

    .. math::
        \beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})\,t

    Integral:

    .. math::
        \int_0^t \beta(s)\,\mathrm{d}s
        = \beta_{\min}\,t + \tfrac{1}{2}(\beta_{\max} - \beta_{\min})\,t^2
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2


class CosineSchedule(VPNoiseSchedule):
    r"""Cosine noise schedule from Nichol & Dhariwal (2021), Improved DDPM.

    Defines the mean coefficient as:

    .. math::
        \alpha(t) = \frac{\cos\!\bigl(\varphi(t)\bigr)}{\cos\!\bigl(\varphi(0)\bigr)},
        \quad \varphi(t) = \frac{t/T + s}{1 + s}\,\frac{\pi}{2}

    where ``s`` is a small offset that prevents ``β(t)`` from approaching zero
    near ``t = 0``.  Equivalently:

    .. math::
        \beta(t) = \frac{\pi}{T(1+s)}\,\tan\!\bigl(\varphi(t)\bigr)
    """

    def __init__(self, T: float = 1.0, s: float = 0.008) -> None:
        self.T = T
        self.s = s
        self._cos_phi0 = math.cos(s / (1.0 + s) * math.pi / 2)

    def _phi(self, t: torch.Tensor) -> torch.Tensor:
        return (t / self.T + self.s) / (1.0 + self.s) * (math.pi / 2)

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """alpha(t) = cos(φ(t)) / cos(φ(0))."""
        return torch.cos(self._phi(t)) / self._cos_phi0

    def _integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        return -2.0 * torch.log(self.mean_coeff(t).clamp(min=1e-8))

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        phi_prime = math.pi / (2.0 * self.T * (1.0 + self.s))
        return 2.0 * torch.tan(self._phi(t)) * phi_prime


class SigmoidSchedule(VPNoiseSchedule):
    r"""Sigmoid noise schedule with analytic integral.

    Defines beta via a sigmoid interpolation:

    .. math::
        \beta(t) = \beta_{\min}
                 + (\beta_{\max} - \beta_{\min})\,\sigma\!\bigl(h(t)\bigr),
        \quad h(t) = \mathtt{start} + (\mathtt{end} - \mathtt{start})\,t

    The integral has a closed form via the softplus function
    ``softplus(x) = log(1 + exp(x))``:

    .. math::
        \int_0^t \beta(s)\,\mathrm{d}s
        = \beta_{\min}\,t
          + \frac{\beta_{\max} - \beta_{\min}}{\mathtt{end} - \mathtt{start}}
            \bigl[\mathrm{softplus}(h(t)) - \mathrm{softplus}(\mathtt{start})\bigr]
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        start: float = -3.0,
        end: float = 3.0,
    ) -> None:
        if end <= start:
            msg = "end must be greater than start"
            raise ValueError(msg)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.start = start
        self.end = end
        self._slope = end - start
        self._softplus_start = math.log1p(math.exp(start))

    def _h(self, t: torch.Tensor) -> torch.Tensor:
        return self.start + self._slope * t

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self._h(t))

    def _integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        softplus_ht = F.softplus(self._h(t))
        return self.beta_min * t + (self.beta_max - self.beta_min) / self._slope * (softplus_ht - self._softplus_start)
