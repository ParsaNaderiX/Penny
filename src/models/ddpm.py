"""Minimal DDPM noise scheduler — drop-in replacement for diffusers.DDPMScheduler.

Only the subset used by JointDiffusion training is implemented:
  - ``config.num_train_timesteps``
  - ``add_noise(x0, noise, timesteps)``

This avoids the diffusers dependency (which requires torch >= 2.4 for xpu support).
"""

from __future__ import annotations

import torch


class _Config:
    def __init__(self, num_train_timesteps: int) -> None:
        self.num_train_timesteps = num_train_timesteps


class DDPMScheduler:
    """Linear-beta DDPM schedule.

    Args:
        num_train_timesteps: Total diffusion steps T (default 1000).
        beta_start:          β at t=0 (default 1e-4).
        beta_end:            β at t=T (default 0.02).
        beta_schedule:       Only ``"linear"`` is supported.
        clip_sample:         Unused; kept for API compatibility.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
    ) -> None:
        if beta_schedule != "linear":
            raise ValueError(
                f"Only 'linear' beta_schedule is supported, got {beta_schedule!r}"
            )
        self.config = _Config(num_train_timesteps)
        betas = torch.linspace(
            beta_start, beta_end, num_train_timesteps, dtype=torch.float64
        )
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0).float()
        self._sqrt_alpha_bar = alpha_bar.sqrt()
        self._sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * ε."""
        device = original_samples.device
        sa = self._sqrt_alpha_bar.to(device)[timesteps]
        sb = self._sqrt_one_minus_alpha_bar.to(device)[timesteps]
        # reshape for broadcasting over all dims except batch
        extra = original_samples.dim() - 1
        sa = sa.view(-1, *([1] * extra))
        sb = sb.view(-1, *([1] * extra))
        return sa * original_samples + sb * noise
