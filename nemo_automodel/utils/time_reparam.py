"""
Time reparameterization for discrete diffusion (MDLM / Block Diffusion).

Maps uniform τ ∈ [0,1] to non-uniform t ∈ [0,1] via the CDF of the ELBO integrand,
concentrating training samples where the loss signal is highest.

Reference: Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (2024).
"""

import math
from functools import lru_cache

import torch


def _log_rate_linear(t: torch.Tensor) -> torch.Tensor:
    """Log rate for linear (CondOT) noise schedule: log(1/(1-t))."""
    return -torch.log1p(-t)


def _log_rate_cosine(t: torch.Tensor) -> torch.Tensor:
    """Log rate for cosine noise schedule: log(pi/2 * tan(pi*t/2))."""
    return torch.log(torch.tensor(math.pi / 2)) + torch.log(torch.tan(math.pi * t / 2))


def _rate_fn(t: torch.Tensor, reparam_type: str) -> torch.Tensor:
    """Compute noise rate R(t) for the given schedule type."""
    if reparam_type in ("linear", "condot"):
        return 1.0 / (1.0 - t).clamp(min=1e-8)
    elif reparam_type == "cosine":
        return (math.pi / 2) * torch.tan(math.pi * t / 2).clamp(min=1e-8)
    else:
        raise ValueError(f"Unsupported time_reparam type: {reparam_type}")


def _elbo_weight(t: torch.Tensor, reparam_type: str, vocab_size: int) -> torch.Tensor:
    """ELBO integrand weight: R(t) * (K-1) * sigma_t * log(K-1) / K
    where sigma_t = 1-t for linear, cos(pi*t/2) for cosine.
    Simplified to just R(t) * sigma_t for relative weighting (constants cancel in CDF).
    """
    rate = _rate_fn(t, reparam_type)
    if reparam_type in ("linear", "condot"):
        sigma_t = 1.0 - t
    elif reparam_type == "cosine":
        sigma_t = torch.cos(math.pi * t / 2)
    else:
        sigma_t = 1.0 - t
    return rate * sigma_t.clamp(min=1e-8)


@lru_cache(maxsize=16)
def _build_lut(reparam_type: str, vocab_size: int, lut_size: int, quad_points: int):
    """Build a lookup table mapping uniform τ → reparameterized t.

    Uses trapezoidal quadrature to compute the CDF of the ELBO weight,
    then inverts it via a uniform grid.
    """
    # Build fine grid for quadrature
    n_quad = max(lut_size * quad_points, 10000)
    eps = 1e-4
    t_grid = torch.linspace(eps, 1.0 - eps, n_quad, dtype=torch.float64)

    # Compute weight at each grid point
    w = _elbo_weight(t_grid, reparam_type, vocab_size).double()
    w = w.clamp(min=0.0)

    # Trapezoidal CDF
    dt = t_grid[1] - t_grid[0]
    cdf = torch.cumsum(w, dim=0) * dt
    cdf = cdf / cdf[-1]  # normalize to [0, 1]

    # Build inverse CDF lookup: for uniform τ values, find corresponding t
    tau_grid = torch.linspace(0.0, 1.0, lut_size + 1, dtype=torch.float64)
    # Use searchsorted to invert CDF
    t_values = torch.zeros(lut_size + 1, dtype=torch.float64)
    indices = torch.searchsorted(cdf, tau_grid.clamp(0, 1))
    indices = indices.clamp(0, len(t_grid) - 1)
    t_values = t_grid[indices]
    # Boundary fix
    t_values[0] = eps
    t_values[-1] = 1.0 - eps

    return t_values.float()


def tau_to_t(tau: torch.Tensor, config) -> torch.Tensor:
    """Map uniform τ ∈ [0,1] to reparameterized t ∈ [0,1].

    Args:
        tau: Uniform time samples, any shape.
        config: Object with attributes:
            - time_reparam: str ("linear", "condot", "cosine", or "none")
            - vocab_size: int
            - reparam_lut_size: int (default 1000)
            - reparam_quad_points: int (default 64)

    Returns:
        t: Reparameterized time values, same shape as tau.
    """
    reparam_type = getattr(config, "time_reparam", "linear")
    if reparam_type in (None, "none", "linear"):
        return tau

    vocab_size = getattr(config, "vocab_size", 151936)
    lut_size = getattr(config, "reparam_lut_size", 1000)
    quad_points = getattr(config, "reparam_quad_points", 64)

    lut = _build_lut(reparam_type, vocab_size, lut_size, quad_points)
    lut = lut.to(tau.device)

    # Linear interpolation in the LUT
    tau_clamped = tau.float().clamp(0.0, 1.0)
    # Scale τ to LUT index space
    idx_float = tau_clamped * lut_size
    idx_low = idx_float.long().clamp(0, lut_size - 1)
    idx_high = (idx_low + 1).clamp(0, lut_size)
    frac = idx_float - idx_low.float()

    t_low = lut[idx_low]
    t_high = lut[idx_high]
    t = t_low + frac * (t_high - t_low)

    return t.to(dtype=tau.dtype)
