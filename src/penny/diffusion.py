"""DDPM / DDIM diffusion utilities for Penny.

Implements a linear-beta DDPM noise schedule with a forward diffusion process
and deterministic DDIM sampling.  All schedule buffers are created on the
configured device so that every tensor stays on the GPU throughout training,
validation and sampling.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class Diffusion:
    """Gaussian diffusion process with a linear beta schedule.

    Parameters
    ----------
    config : dict
        Provides ``T_max``, ``beta_start``, ``beta_end`` and ``ddim_steps``.
    device : torch.device
        Device on which all schedule buffers live.
    """

    def __init__(self, config: dict, device: torch.device) -> None:
        self.device = device
        self.T_max = int(config["T_max"])
        self.ddim_steps = int(config["ddim_steps"])
        betas = torch.linspace(
            config["beta_start"], config["beta_end"], self.T_max, device=device
        )
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        logger.info(
            "diffusion schedule: T_max=%d beta=[%.5f, %.5f] ddim_steps=%d",
            self.T_max,
            config["beta_start"],
            config["beta_end"],
            self.ddim_steps,
        )

    def forward_diffusion(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Sample ``x_t`` from ``q(x_t | x_0)``.

        Parameters
        ----------
        x0 : (B, T, F) clean trajectory.
        t : (B,) integer timesteps.
        noise : (B, T, F) standard normal noise.
        """
        ab = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int],
        regime: torch.Tensor,
        context: torch.Tensor,
        ddim_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate trajectories with deterministic (eta=0) DDIM sampling.

        Parameters
        ----------
        model : the :class:`~penny.model.Penny` network.
        shape : ``(B, T, F)`` of the trajectory to generate.
        regime : (B, regime_dim) conditioning regime vectors.
        context : (B, T, F) past sequence conditioning.
        ddim_steps : number of DDIM steps (defaults to the configured value).
        """
        steps = ddim_steps or self.ddim_steps
        b = shape[0]
        x = torch.randn(shape, device=self.device)
        ts = torch.linspace(self.T_max - 1, 0, steps, device=self.device).long()

        for i in range(steps):
            t_cur = ts[i]
            t_batch = t_cur.repeat(b)
            ab_t = self.alpha_bars[t_batch]
            eps, _, x0 = model(context, x, t_batch, regime, ab_t)

            ab_prev = (
                self.alpha_bars[ts[i + 1]]
                if i + 1 < steps
                else torch.ones((), device=self.device)
            )
            x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps
        return x
