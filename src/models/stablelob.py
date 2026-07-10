"""StableLOB: improved-DDPM joint diffusion-classifier for feature-only LOB trend.

Shares the *trunk idea* of :class:`~models.jumpgatelob.JumpGateLOB` — a local
recurrent encoder followed by a **single** temporal self-attention layer, feeding a
trend head (kept at inference) and a grid diffusion head (a training-time auxiliary)
— but swaps JumpGateLOB's Lévy ε-prediction for the **improved-DDPM** noise/denoise
recipe (Nichol & Dhariwal 2021; :mod:`models.iddpm`):

  * a **cosine** noise schedule,
  * **learned reverse variance**: the diffusion head predicts *two* channels per
    element, ``(ε̂, v̂)``, where ``v̂`` interpolates the reverse log-variance between
    ``β̃_t`` and ``β_t``,
  * a **hybrid** ``L_simple + λ·L_vlb`` objective (in the trainer).

The name refers to the *training stability* the improved-DDPM likelihood weighting
buys (learned variance + cosine schedule) — it is **not** latent Stable Diffusion
(no VAE / latent space; the model diffuses the raw feature window directly).

Trunk (run once per pass):

  1. optional ``BiN`` front-end (per-window bilinear normalisation),
  2. **local encoder** — a (bi)GRU (or temporal-conv) → per-timestep context
     ``H₀ (B, T, D)``.  The window ends at the prediction point and the label lives
     strictly outside it, so a bidirectional pass leaks nothing,
  3. **one** DiT-style **temporal self-attention** layer over ``T`` with sinusoidal
     positions, adaLN-Zero conditioned on the timestep embedding ``c``.

Two heads share the trunk context ``H``:

  * **trend head** — attention-pool over ``T`` → 3 logits.  Feature-only inference
    runs only the trunk + this head on the clean window (no reverse sampling).
  * **diffusion head** — a small flat, constant-``(T,F)`` grid net; each block mixes
    book levels over ``F`` and injects the trunk context, adaLN-Zero conditioned on
    the timestep.  It outputs ``(ε̂, v̂)`` for the learned-variance objective.

Inference contract matches every other crypto model: ``predict(batch, device) →
logits (B, 3)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (
    AttentionPool,
    BiN,
    LevelAttention,
    count_parameters as count_parameters,  # re-export
    sinusoidal_embedding,
)


def _groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    # x: (B, N, D); shift/scale: (B, D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TemporalAttnBlock(nn.Module):
    """One DiT-style temporal self-attention layer over ``T`` (adaLN-Zero)."""

    def __init__(self, dim: int, heads: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sa, ca, ga, sm, cm, gm = self.ada(c).chunk(6, dim=1)
        h = _modulate(self.norm1(x), sa, ca)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + ga.unsqueeze(1) * a
        h = _modulate(self.norm2(x), sm, cm)
        x = x + gm.unsqueeze(1) * self.mlp(h)
        return x


class DiffBlock(nn.Module):
    """Grid diffusion block: feature-axis mixing over ``F`` + trunk-context injection,
    adaLN-Zero conditioned on ``t``.  Operates on ``(B, C, T, F)``."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        ctx_dim: int,
        feat_mix: str,
        feat_heads: int,
        pad_mode: str,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels, affine=False)
        self.ada = nn.Linear(cond_dim, 3 * channels)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.ctx = nn.Linear(ctx_dim, channels)  # per-timestep trunk context
        if feat_mix == "attn":
            self.mix = LevelAttention(channels, feat_heads)
        elif feat_mix == "conv":
            self.mix = nn.Conv2d(
                channels, channels, (1, 3), padding=(0, 1), padding_mode=pad_mode
            )
        else:
            raise ValueError(f"feat_mix must be attn|conv, got {feat_mix!r}")

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        shift, scale, gate = self.ada(c).chunk(3, dim=1)  # each (B, C)
        v = (-1, x.shape[1], 1, 1)
        h = self.norm(x) * (1 + scale.view(v)) + shift.view(v)
        h = h + self.ctx(H).permute(0, 2, 1).unsqueeze(-1)  # (B, C, T, 1) over F
        h = F.silu(self.mix(h))
        return x + gate.view(v) * h


class DiffHead(nn.Module):
    """Flat grid net predicting ``(ε̂, v̂)`` for the learned-variance objective."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        ctx_dim: int,
        n_blocks: int,
        feat_mix: str,
        feat_heads: int,
        pad_mode: str,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(1, channels, 1)
        self.blocks = nn.ModuleList(
            DiffBlock(channels, cond_dim, ctx_dim, feat_mix, feat_heads, pad_mode)
            for _ in range(n_blocks)
        )
        # 2 output channels: ε (mean) and v (variance interpolation).
        self.out = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t, c, H):
        x = self.input_projection(x_t)  # (B, C, T, F)
        for blk in self.blocks:
            x = blk(x, c, H)
        o = self.out(x)  # (B, 2, T, F)
        eps, v = o[:, :1], o[:, 1:2]  # each (B, 1, T, F)
        return eps, v


class StableLOB(nn.Module):
    """(bi)GRU + one temporal-attention trunk; trend head + learned-variance diff head."""

    family = "joint_diffusion"

    def __init__(self, config: dict) -> None:
        super().__init__()
        F_dim = config["n_features"]
        temb_dim = config.get("stable_time_emb", 128)
        self.temb_dim = temb_dim
        self.F = F_dim

        # ---- BiN front-end (matches JumpGateLOB) ----------------------------
        self.bin = (
            BiN(config["T_past"], F_dim) if config.get("use_bin", False) else None
        )

        # ---- local encoder --------------------------------------------------
        self.local = config.get("stable_local", "gru")
        hidden = config.get("stable_gru_hidden", 64)
        bidir = bool(config.get("stable_bidirectional", True))
        if self.local == "gru":
            self.gru = nn.GRU(
                input_size=F_dim,
                hidden_size=hidden,
                num_layers=config.get("stable_gru_layers", 2),
                dropout=config.get("stable_gru_dropout", 0.0)
                if config.get("stable_gru_layers", 2) > 1
                else 0.0,
                batch_first=True,
                bidirectional=bidir,
            )
            D = hidden * (2 if bidir else 1)
        elif self.local == "conv":
            D = hidden
            self.embed = nn.Linear(F_dim, D)
            self.tconv = nn.Sequential(
                nn.Conv1d(D, D, 3, padding=1, padding_mode="replicate"),
                nn.SiLU(),
                nn.Conv1d(D, D, 3, padding=1, padding_mode="replicate"),
            )
        else:
            raise ValueError(f"stable_local must be gru|conv, got {self.local!r}")
        self.D = D

        # ---- timestep conditioning c = MLP(emb(t)) --------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )

        # ---- one temporal-attention layer -----------------------------------
        self.temporal = TemporalAttnBlock(
            D,
            heads=config.get("stable_attn_heads", 4),
            cond_dim=temb_dim,
            dropout=config.get("stable_attn_dropout", 0.1),
        )

        # ---- trend head ------------------------------------------------------
        self.pool = AttentionPool(D, heads=config.get("stable_pool_heads", 4))
        self.cls_dropout = nn.Dropout(config.get("cls_dropout", 0.0))
        self.classifier = nn.Linear(D, 3)

        # ---- diffusion head (ε, v) ------------------------------------------
        self.diff_head = DiffHead(
            channels=config.get("stable_diff_channels", 16),
            cond_dim=temb_dim,
            ctx_dim=D,
            n_blocks=config.get("stable_diff_blocks", 2),
            feat_mix=config.get("stable_feat_mix", "conv"),
            feat_heads=config.get("stable_feat_heads", 2),
            pad_mode=config.get("stable_pad_mode", "reflect"),
        )

    # ---- trunk --------------------------------------------------------------
    def _local(self, x: torch.Tensor) -> torch.Tensor:
        s = x.squeeze(1)  # (B, T, F)
        if self.bin is not None:
            s = self.bin(s)
        if self.local == "gru":
            H, _ = self.gru(s)
            return H
        h = self.embed(s).transpose(1, 2)  # (B, D, T)
        return self.tconv(h).transpose(1, 2)  # (B, T, D)

    def _cond(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(sinusoidal_embedding(t, self.temb_dim))

    def trunk(self, x: torch.Tensor, t: torch.Tensor):
        """Return ``(H (B,T,D), c (B,temb_dim))``."""
        c = self._cond(t)
        H0 = self._local(x)  # (B, T, D)
        T = H0.shape[1]
        pos = sinusoidal_embedding(torch.arange(T, device=x.device), self.D).unsqueeze(
            0
        )
        H = self.temporal(H0 + pos, c)
        return H, c

    def _trend_logits(self, H: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cls_dropout(self.pool(H)))

    # ---- task-specific passes (training uses these separately) --------------
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Trend logits from the *clean* window at ``t = 0`` (matches inference)."""
        t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        H, _ = self.trunk(x, t)
        return self._trend_logits(H)

    def diffuse(self, x_t: torch.Tensor, t: torch.Tensor):
        """Predict ``(ε̂, v̂)`` on the *noised* window at timestep ``t``."""
        H, c = self.trunk(x_t, t)
        return self.diff_head(x_t, c, H)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """Joint pass: ``(ε̂, v̂, logits)``."""
        H, c = self.trunk(x_t, t)
        logits = self._trend_logits(H)
        eps_hat, v_hat = self.diff_head(x_t, c, H)
        return eps_hat, v_hat, logits

    def trunk_parameters(self):
        """All params except the trend head (for a frozen-trunk phase-2 probe)."""
        head = set(map(id, self.pool.parameters())) | set(
            map(id, self.classifier.parameters())
        )
        return (p for p in self.parameters() if id(p) not in head)

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Feature-only inference: trunk + trend head on the clean window."""
        x = batch["x"].to(device).float()
        return self.classify(x)
