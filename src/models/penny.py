"""Penny — asset-conditioned diffusion + trend classification head.

Combines the two ideas in this repo into a single multi-asset model:

  * **diffclf idea** — the U-Net diffusion branch is conditioned on the *asset
    identity* (which coin) via a learnable embedding + classifier-free-guidance
    condition dropout, so it learns ``p(x | asset)`` and the marginal ``p(x)``,
    i.e. the distribution of each coin's LOB windows.
  * **JointDiffusion idea** — a classification head on the U-Net bottleneck
    predicts the price-trend label (0=down, 1=stationary, 2=up), trained with
    cross-entropy.

One Penny model is trained across **every symbol of one exchange**.  At
inference the asset id is *known* (you always know which coin you are looking
at): condition on it and read the trend head at ``t = 0``.

Shapes
------
forward(x_t (B,1,T,F), t (B,), asset (B,)) -> (eps_hat (B,1,T,F), logits (B,3))
predict(batch, device)                     -> logits (B,3)   (clean window, t=0)

Indices
-------
asset ids ``0 .. n_assets-1``; the null (∅) asset id is ``n_assets``.

Config keys
-----------
n_assets            : number of symbols pooled into this run   (required)
penny_base_channels : base U-Net channel count                 (default 32)
penny_depth         : number of down/up stages                 (default 2)
penny_time_emb      : time + asset embedding dim               (default 128)
penny_dropout       : dropout in the trend head                (default 0.1)
penny_p_uncond      : asset condition-dropout prob at train    (default 0.15)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.jointdiff import Down, TimeDoubleConv, Up, sinusoidal_embedding


class Penny(nn.Module):
    """Asset-conditioned U-Net that denoises and classifies trend jointly."""

    family = "penny"

    def __init__(self, config: dict) -> None:
        super().__init__()
        n_assets = config["n_assets"]
        self.n_assets = n_assets
        self.null_asset = n_assets
        base = config.get("penny_base_channels", 32)
        depth = config.get("penny_depth", 2)
        temb_dim = config.get("penny_time_emb", 128)
        self.temb_dim = temb_dim
        self.p_uncond = config.get("penny_p_uncond", 0.15)

        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        # +1 row for the null (∅) asset used by classifier-free condition dropout
        self.asset_emb = nn.Embedding(n_assets + 1, temb_dim)

        chans = [base * (2**i) for i in range(depth + 1)]
        self.stem = TimeDoubleConv(1, base, temb_dim)
        self.downs = nn.ModuleList(
            Down(chans[i], chans[i + 1], temb_dim) for i in range(depth)
        )
        self.ups = nn.ModuleList(
            Up(chans[i + 1], chans[i], chans[i], temb_dim)
            for i in reversed(range(depth))
        )
        self.out_conv = nn.Conv2d(base, 1, 1)
        bottleneck = chans[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck, bottleneck),
            nn.SiLU(),
            nn.Dropout(config.get("penny_dropout", 0.1)),
            nn.Linear(bottleneck, 3),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, asset: torch.Tensor):
        """Return ``(eps_hat, trend_logits)`` conditioned on ``asset``."""
        temb = self.time_mlp(sinusoidal_embedding(t, self.temb_dim))
        temb = temb + self.asset_emb(asset)
        x = self.stem(x_t, temb)
        skips = [x]
        for down in self.downs:
            x = down(x, temb)
            skips.append(x)
        logits = self.classifier(skips[-1])
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip, temb)
        return self.out_conv(x), logits

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Trend logits for the clean window conditioned on the known asset."""
        x = batch["x"].to(device).float()
        asset = batch["asset"].to(device)
        t = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        _, logits = self(x, t, asset)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
