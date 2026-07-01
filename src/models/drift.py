"""Drift loss + memory bank (Generative Modeling via Drifting, Lambert et al.).

PyTorch port of the JAX reference (github.com/lambertae/drifting, ``drift_loss.py``).

The generator is trained as a *one-step* sampler (no iterative denoising) by
dragging its output particles toward a data-distribution "goal" computed with a
multi-scale kernel force field: generated particles are **attracted** to positive
(real) anchors and **repelled** from negative (unconditional / other generated)
anchors, at several temperatures ``R``.  The goal is built entirely under a
stop-gradient, so the trainable loss is a plain MSE onto that goal::

    goal = old_gen + sum_R  force_R / ||force_R||          (no grad)
    loss = mean( (gen/scale - goal/scale)^2 )

A class-agnostic ring buffer (:class:`WindowMemoryBank`) supplies a large, stable
pool of anchors beyond the current mini-batch, mirroring the reference's
``ArrayMemoryBank``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batched Euclidean distance. x: (B,N,D), y: (B,M,D) -> (B,N,M)."""
    xy = torch.einsum("bnd,bmd->bnm", x, y)
    xn = torch.einsum("bnd,bnd->bn", x, x)
    yn = torch.einsum("bmd,bmd->bm", y, y)
    sq = xn[:, :, None] + yn[:, None, :] - 2 * xy
    return sq.clamp_min(eps).sqrt()


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    R_list: tuple[float, ...] = (0.02, 0.05, 0.2),
):
    """Multi-scale kernel drift loss.

    Args:
        gen:       generated particles, ``(B, C_g, S)``.
        fixed_pos: positive (real) anchors, ``(B, C_p, S)``.
        fixed_neg: negative anchors, ``(B, C_n, S)`` or ``None``.
        weight_*:  optional per-particle weights ``(B, C_*)`` (default 1).
        R_list:    kernel temperatures (each self-normalised, then summed).

    Returns:
        ``(loss (B,), info dict)`` where ``info[f"loss_{R}"]`` is the (scalar)
        squared force magnitude at temperature ``R``.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = fixed_pos.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = fixed_neg.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    # Target order: [old_gen | neg | pos]; the first C_g + C_n block is repulsive,
    # the trailing C_p block is attractive.
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    with torch.no_grad():
        dist = _cdist(old_gen, targets)  # (B, C_g, C_g+C_n+C_p)
        weighted = dist * targets_w[:, None, :]
        scale = weighted.mean() / targets_w.mean()

        scale_inputs = (scale / (S**0.5)).clamp_min(1e-3)  # coords ~ order 1
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / scale.clamp_min(1e-3)

        # Mask a particle's affinity to its own copy (leading C_g block, diagonal).
        eye = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block = F.pad(eye, (0, C_n + C_p))[None]  # (1, C_g, tot)
        dist_normed = dist_normed + block * 100.0

        force = torch.zeros_like(old_gen_scaled)
        info: dict[str, torch.Tensor] = {}
        split = C_g + C_n
        for R in R_list:
            logits = -dist_normed / R
            aff = torch.softmax(logits, dim=-1)
            aff_t = torch.softmax(logits, dim=-2)
            aff = (aff * aff_t).clamp_min(1e-6).sqrt()  # bidirectional, geom mean
            aff = aff * targets_w[:, None, :]

            aff_neg = aff[:, :, :split]  # old_gen + neg -> repulsion
            aff_pos = aff[:, :, split:]  # pos -> attraction
            sum_pos = aff_pos.sum(-1, keepdim=True)
            r_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(-1, keepdim=True)
            r_pos = aff_pos * sum_neg
            R_coeff = torch.cat([r_neg, r_pos], dim=2)

            f_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            total_c = R_coeff.sum(-1)  # ~0 (attraction and repulsion balance)
            f_R = f_R - total_c[..., None] * old_gen_scaled
            f_norm = (f_R**2).mean()
            info[f"loss_{R}"] = f_norm
            force = force + f_R / f_norm.clamp_min(1e-8).sqrt()

        goal_scaled = old_gen_scaled + force

    gen_scaled = gen / scale_inputs
    loss = ((gen_scaled - goal_scaled) ** 2).mean(dim=(-1, -2))
    return loss, info


class WindowMemoryBank:
    """Class-agnostic ring buffer of real LOB windows for drift anchors.

    Stores windows on CPU; :meth:`sample` returns a ``(n, *window_shape)`` draw.
    Lazily initialised on the first :meth:`add` so the window shape need not be
    known ahead of time.
    """

    def __init__(self, max_size: int, dtype: torch.dtype = torch.float32) -> None:
        self.max_size = int(max_size)
        self.dtype = dtype
        self.buf: torch.Tensor | None = None
        self.labels = torch.zeros(self.max_size, dtype=torch.long)
        self.ptr = 0
        self.count = 0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Insert windows ``x (N, *shape)`` with labels ``y (N,)`` (both CPU-ok)."""
        x = x.detach().to(self.dtype).cpu()
        y = y.detach().cpu()
        if self.buf is None:
            self.buf = torch.zeros(self.max_size, *x.shape[1:], dtype=self.dtype)
        n = x.shape[0]
        for i in range(n):
            self.buf[self.ptr] = x[i]
            self.labels[self.ptr] = y[i]
            self.ptr = (self.ptr + 1) % self.max_size
            self.count = min(self.count + 1, self.max_size)

    def ready(self, n: int) -> bool:
        return self.count >= n

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n`` windows (with replacement); returns ``(x, labels)``."""
        if self.buf is None or self.count == 0:
            raise RuntimeError("WindowMemoryBank is empty; call add() first.")
        idx = torch.randint(0, self.count, (n,))
        return self.buf[idx], self.labels[idx]
