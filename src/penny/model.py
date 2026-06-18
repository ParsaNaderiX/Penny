"""Penny model components (spec section 5; paper §4.3).

- ``build_unet``: a diffusers ``UNet2DModel`` (5-channel in, 2-channel out) — the
  inpainting backbone.  No custom UNet.
- ``TrendHead``: a 1->3 linear layer mapping the scalar trend ratio ``l`` to
  class logits (6 parameters, trained jointly).
- ``painted_future_mid``: reconstructs the absolute future mid-price series from a
  Tweedie-denoised image — reading the price channel (LOB mode) or integrating
  best-level OFI with the fitted ``gamma`` (OFI mode).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DModel


def build_unet(config: dict) -> UNet2DModel:
    """Construct the inpainting UNet; self-attention at the deepest block (spec 5.1)."""
    filters = tuple(config["unet_filters"])
    n_blocks = len(filters)
    attn_at = config["self_attn_at_block"]  # 1-indexed
    down = tuple(
        "AttnDownBlock2D" if (b + 1) == attn_at else "DownBlock2D"
        for b in range(n_blocks)
    )
    up = tuple(
        "AttnUpBlock2D" if (n_blocks - b) == attn_at else "UpBlock2D"
        for b in range(n_blocks)
    )
    return UNet2DModel(
        sample_size=config["padded_size"],
        in_channels=5,
        out_channels=2,
        layers_per_block=2,
        block_out_channels=filters,
        down_block_types=down,
        up_block_types=up,
        dropout=config["dropout"],
    )


class TrendHead(nn.Module):
    """Linear map from the scalar trend ratio ``l`` to 3 class logits (spec 5.3)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 3)

    def forward(self, trend_value: torch.Tensor) -> torch.Tensor:
        """``l`` shape ``(B,)`` -> logits ``(B, 3)``."""
        return self.fc(trend_value.view(-1, 1))


def painted_future_mid(
    x0_hat: torch.Tensor,
    config: dict,
    norm,
    level_starts: np.ndarray,
    mid_ref: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Reconstruct the future mid series ``(B, T_future)`` from a denoised image.

    LOB mode reads the best bid/ask price offsets directly; OFI mode integrates
    the best-level OFI scaled by ``gamma`` and anchored at the boundary mid
    (spec 7.1 / 8.5, with the user-chosen OFI reconstruction).  Differentiable
    w.r.t. ``x0_hat`` for the trend loss.
    """
    n = config["n_levels"]
    t_past, t_total = config["T_past"], config["T_total"]
    ls = level_starts
    fut = slice(t_past, t_total)

    bb = x0_hat[:, 0, ls[n - 1] : ls[n], fut].mean(dim=1)  # (B, T_future) normalized
    ba = x0_hat[:, 0, ls[n] : ls[n + 1], fut].mean(dim=1)
    bb = bb * float(norm.std[n - 1, 0]) + float(norm.mean[n - 1, 0])
    ba = ba * float(norm.std[n, 0]) + float(norm.mean[n, 0])

    mid_ref = mid_ref.view(-1, 1)
    if config["feature_mode"] == "lob":
        offset = (bb + ba) / 2.0  # mid - anchor
        return offset + mid_ref
    ofi_best = bb + ba  # buy-positive flow
    return mid_ref + torch.cumsum(ofi_best, dim=1) * gamma


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
