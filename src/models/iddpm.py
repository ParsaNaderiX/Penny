"""Improved-DDPM Gaussian diffusion — cosine schedule + learned variance + hybrid loss.

A minimal, self-contained port of the noise/denoise recipe from Nichol & Dhariwal,
*Improved Denoising Diffusion Probabilistic Models* (2021,
github.com/openai/improved-diffusion), specialised for the joint LOB trend model
:class:`~models.stablelob.StableLOB`.

What differs from the plain linear-β DDPM in :mod:`models.ddpm`:

* **Cosine β schedule** — ``ᾱ(t) = cos((t/T + s)/(1 + s) · π/2)²`` (``s = 0.008``),
  ``β_i = min(1 − ᾱ(t₂)/ᾱ(t₁), 0.999)``.  Adds less noise early / late than a linear
  schedule, which improves log-likelihood and training stability.
* **Learned variance (LEARNED_RANGE)** — the model predicts, per element, ``(ε, v)``.
  ``v`` interpolates the reverse-process log-variance between the two natural bounds:
  ``log Σ_θ = frac·log β_t + (1 − frac)·log β̃_t``,  ``frac = (v + 1)/2`` where
  ``β̃_t`` is the true-posterior variance (``posterior_log_variance_clipped``).
* **Hybrid objective** ``L = L_simple + λ_vlb · L_vlb``:
    - ``L_simple = MSE(ε̂, ε)`` trains the mean.
    - ``L_vlb`` is the (rescaled) variational bound; the mean is **stop-gradded**
      inside it so the VLB trains *only* the variance ``v`` (the IDDPM trick that
      keeps ``L_vlb`` from swamping ``L_simple``).

Adaptation for LOB: the data are continuous, causally z-scored features — **not**
8-bit images — so the ``t = 0`` decoder term uses a **continuous Gaussian NLL**
rather than IDDPM's discretized-Gaussian (256-bin) likelihood, and ``x₀`` is not
clamped to ``[-1, 1]``.  Everything else follows the reference.

This class only holds schedule tensors + math (no parameters); it is passed to the
trainer alongside the model, mirroring ``DDPMScheduler`` / ``ForwardProcess``.
"""

from __future__ import annotations

import math

import torch

_LOG2 = math.log(2.0)


def cosine_betas(num_timesteps: int, s: float = 0.008, max_beta: float = 0.999):
    """IDDPM cosine β schedule as a ``(T,)`` float64 tensor."""

    def alpha_bar(u: float) -> float:
        return math.cos((u + s) / (1.0 + s) * math.pi / 2.0) ** 2

    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


def normal_kl(
    mean1: torch.Tensor,
    logvar1: torch.Tensor,
    mean2: torch.Tensor,
    logvar2: torch.Tensor,
) -> torch.Tensor:
    """KL( N(mean1, e^{logvar1}) || N(mean2, e^{logvar2}) ), elementwise, in nats."""
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * torch.exp(-logvar2)
    )


def gaussian_nll(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor):
    """Elementwise continuous Gaussian negative log-likelihood, in nats."""
    return 0.5 * (
        math.log(2.0 * math.pi) + logvar + (x - mean) ** 2 * torch.exp(-logvar)
    )


def _mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dims → ``(B,)``."""
    return x.flatten(1).mean(dim=1)


class ImprovedDiffusion:
    """Cosine-schedule Gaussian diffusion with learned variance + hybrid loss."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        cosine_s: float = 0.008,
        max_beta: float = 0.999,
    ) -> None:
        self.num_timesteps = num_timesteps
        betas = cosine_betas(num_timesteps, cosine_s, max_beta)  # (T,) float64
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)  # ᾱ_t
        acp_prev = torch.cat([torch.ones(1, dtype=torch.float64), acp[:-1]])

        self.betas = betas
        self.sqrt_acp = acp.sqrt()
        self.sqrt_one_minus_acp = (1.0 - acp).sqrt()
        # x0 reconstruction from ε
        self.sqrt_recip_acp = (1.0 / acp).sqrt()
        self.sqrt_recipm1_acp = (1.0 / acp - 1.0).sqrt()
        # true posterior q(x_{t-1} | x_t, x_0)
        post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.posterior_variance = post_var
        # variance is 0 at t=0; clip the log by substituting post_var[1] for index 0
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([post_var[1:2], post_var[1:]])
        )
        self.posterior_mean_coef1 = betas * acp_prev.sqrt() / (1.0 - acp)
        self.posterior_mean_coef2 = (1.0 - acp_prev) * alphas.sqrt() / (1.0 - acp)
        # LEARNED_RANGE upper bound uses log β (min-clamped so log is finite at t=0)
        self.log_betas = torch.log(betas.clamp_min(1e-20))

    @staticmethod
    def _extract(arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Gather ``arr[t]`` and reshape to broadcast over an ``ndim``-D tensor."""
        out = arr.to(t.device)[t].float()
        return out.view(-1, *([1] * (ndim - 1)))

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion ``x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε``."""
        n = x0.dim()
        return (
            self._extract(self.sqrt_acp, t, n) * x0
            + self._extract(self.sqrt_one_minus_acp, t, n) * noise
        )

    def predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        n = x_t.dim()
        return (
            self._extract(self.sqrt_recip_acp, t, n) * x_t
            - self._extract(self.sqrt_recipm1_acp, t, n) * eps
        )

    def q_posterior_mean_variance(
        self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        """True posterior mean and clipped log-variance of ``q(x_{t-1}|x_t,x_0)``."""
        n = x_t.dim()
        mean = (
            self._extract(self.posterior_mean_coef1, t, n) * x0
            + self._extract(self.posterior_mean_coef2, t, n) * x_t
        )
        log_var = self._extract(self.posterior_log_variance_clipped, t, n)
        return mean, log_var

    def p_mean_variance(
        self, eps: torch.Tensor, v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        """Reverse-process mean + learned log-variance from predicted ``(ε, v)``.

        ``frac = (v + 1)/2`` interpolates the log-variance between the true-posterior
        variance ``β̃_t`` (min) and ``β_t`` (max), per LEARNED_RANGE.
        """
        n = x_t.dim()
        min_log = self._extract(self.posterior_log_variance_clipped, t, n)
        max_log = self._extract(self.log_betas, t, n)
        frac = (v + 1.0) / 2.0
        model_log_var = frac * max_log + (1.0 - frac) * min_log
        x0_pred = self.predict_xstart_from_eps(x_t, t, eps)
        model_mean, _ = self.q_posterior_mean_variance(x0_pred, x_t, t)
        return model_mean, model_log_var, x0_pred

    def _vb_term(
        self,
        eps: torch.Tensor,
        v: torch.Tensor,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample variational-bound term (bits): KL for t>0, decoder NLL at t=0."""
        true_mean, true_log_var = self.q_posterior_mean_variance(x0, x_t, t)
        model_mean, model_log_var, _ = self.p_mean_variance(eps, v, x_t, t)
        kl = _mean_flat(normal_kl(true_mean, true_log_var, model_mean, model_log_var))
        nll = _mean_flat(gaussian_nll(x0, model_mean, model_log_var))
        return torch.where(t == 0, nll, kl) / _LOG2  # (B,) in bits

    def hybrid_loss(
        self,
        eps_hat: torch.Tensor,
        v_hat: torch.Tensor,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        lambda_vlb: float = 1.0,
    ):
        """``L_simple + λ_vlb · L_vlb`` (RESCALED_MSE hybrid).

        ``L_simple`` trains the mean (ε); the VLB stop-grads the mean so it trains
        only the variance ``v``.  The VLB is rescaled by ``num_timesteps / 1000`` (the
        reference RESCALED_MSE factor); ``lambda_vlb`` multiplies on top (1.0 = the
        paper's hybrid; the paper's λ=0.001 already lives in the rescale).
        """
        l_simple = torch.mean((eps_hat - noise) ** 2)
        vb = self._vb_term(eps_hat.detach(), v_hat, x0, x_t, t).mean()
        vb = vb * (self.num_timesteps / 1000.0) * lambda_vlb
        loss = l_simple + vb
        return loss, {"simple": float(l_simple.detach()), "vlb": float(vb.detach())}
