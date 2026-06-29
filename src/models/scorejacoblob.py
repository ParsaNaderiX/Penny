"""ScoreJacobLOB — score-Jacobian-attention diffusion classifier for LOB trends.

A four-stage, two-phase model for mid-price trend prediction (down / flat / up):

  Stage 1  BiN              — adaptive bilinear normalisation along the time and
                              feature axes (removes price-vs-volume magnitude
                              disparity and non-stationarity).
  Stage 2  DiffBackbone1D   — a causal (autoregressive) 1-D U-Net diffusion model
                              with bottleneck self-attention, trained with the
                              v-parameterisation + min-SNR loss weighting.
  Stage 3  Score-Jacobian   — with the backbone frozen, the input-space Jacobian
                              of the score s_θ(x̂, t*) is probed by vector-Jacobian
                              products and marginalised into a temporal saliency
                              w_t ∈ ℝ^T and a feature saliency w_f ∈ ℝ^F, at K
                              noise levels (multi-scale). Stop-gradient throughout.
  Stage 4  Gated head       — the frozen bottleneck h is gated element-wise by the
                              attention-derived saliencies, pooled, and classified.

Why this avoids gradient surgery
--------------------------------
The diffusion objective (Stage 2) and the classification objective (Stage 4) are
optimised in **separate phases**: the backbone is frozen before the head is
trained, and the Jacobian extraction carries a stop-gradient. The two objectives
therefore never share a backward pass, so there is no inter-task gradient
conflict to resolve — the attention that a model like JointDiffCFG would learn
under PCGrad is here *read off* the frozen score field instead of being trained.

Batch format: ``{"x": (B, 1, T, F), "label": int}`` — identical to every other
crypto model.  ``predict(batch, device) → (B, 3)`` logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.jointdiff import _groups, sinusoidal_embedding


# ── Stage 1: adaptive Bilinear Normalisation ─────────────────────────────────


class BiN(nn.Module):
    """Bilinear normalisation: normalise across time and across features, then mix.

    Temporal branch z-scores each feature column across time (learnable per-feature
    γ_t, β_t); feature branch z-scores each time row across features (learnable
    per-time γ_f, β_f).  A softmax-mixed convex combination of the two branches is
    returned, so the layer learns how much temporal vs. cross-sectional
    normalisation each dataset needs.
    """

    def __init__(self, T: int, F_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma_t = nn.Parameter(torch.ones(F_dim))  # per-feature
        self.beta_t = nn.Parameter(torch.zeros(F_dim))
        self.gamma_f = nn.Parameter(torch.ones(T))  # per-time
        self.beta_f = nn.Parameter(torch.zeros(T))
        self.mix = nn.Parameter(torch.zeros(2))  # softmax → convex weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        mt = x.mean(dim=1, keepdim=True)
        st = x.std(dim=1, keepdim=True) + self.eps
        xt = (x - mt) / st * self.gamma_t + self.beta_t  # (B,T,F)

        mf = x.mean(dim=2, keepdim=True)
        sf = x.std(dim=2, keepdim=True) + self.eps
        xf = (x - mf) / sf * self.gamma_f[None, :, None] + self.beta_f[None, :, None]

        w = torch.softmax(self.mix, dim=0)
        return w[0] * xt + w[1] * xf


# ── Stage 2: causal 1-D U-Net diffusion backbone ─────────────────────────────


class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding so output[t] depends on input[≤ t]."""

    def __init__(self, ci: int, co: int, k: int, stride: int = 1) -> None:
        super().__init__(ci, co, k, stride=stride, padding=0)
        self.left = k - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left, 0)))


class CausalBlock(nn.Module):
    """Two causal convs + GroupNorm + SiLU with an additive time embedding."""

    def __init__(self, ci: int, co: int, temb_dim: int) -> None:
        super().__init__()
        self.c1 = CausalConv1d(ci, co, 3)
        self.n1 = nn.GroupNorm(_groups(co), co)
        self.temb = nn.Linear(temb_dim, co)
        self.c2 = CausalConv1d(co, co, 3)
        self.n2 = nn.GroupNorm(_groups(co), co)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.act(self.n1(self.c1(x)))
        x = x + self.temb(temb).unsqueeze(-1)
        return self.act(self.n2(self.c2(x)))


class Down1D(nn.Module):
    def __init__(self, ci: int, co: int, temb_dim: int) -> None:
        super().__init__()
        self.down = CausalConv1d(ci, ci, 3, stride=2)  # halve time, causally
        self.block = CausalBlock(ci, co, temb_dim)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x), temb)


class Up1D(nn.Module):
    def __init__(self, ci: int, skip_ch: int, co: int, temb_dim: int) -> None:
        super().__init__()
        self.reduce = nn.Conv1d(ci, co, 1)
        self.block = CausalBlock(co + skip_ch, co, temb_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], dim=1), temb)


class DiffBackbone1D(nn.Module):
    """Causal 1-D U-Net over time with bottleneck self-attention (v-prediction)."""

    def __init__(
        self, F_dim: int, base: int, depth: int, temb_dim: int, heads: int
    ) -> None:
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        chans = [base * (2**i) for i in range(depth + 1)]
        self.bottleneck_ch = chans[-1]
        self.stem = CausalBlock(F_dim, base, temb_dim)
        self.downs = nn.ModuleList(
            Down1D(chans[i], chans[i + 1], temb_dim) for i in range(depth)
        )
        self.attn = nn.MultiheadAttention(chans[-1], heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(chans[-1])
        self.ups = nn.ModuleList(
            Up1D(chans[i + 1], chans[i], chans[i], temb_dim)
            for i in reversed(range(depth))
        )
        self.out = nn.Conv1d(base, F_dim, 1)

    def _temb(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(sinusoidal_embedding(t, self.temb_dim))

    def _causal_attn(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C, L) → attend across L causally
        z = h.transpose(1, 2)  # (B, L, C)
        L = z.shape[1]
        mask = torch.triu(torch.ones(L, L, device=z.device, dtype=torch.bool), 1)
        a, _ = self.attn(z, z, z, attn_mask=mask)
        z = self.attn_norm(z + a)
        return z.transpose(1, 2)  # (B, C, L)

    def _encode(self, x_t: torch.Tensor, temb: torch.Tensor):
        h = x_t.transpose(1, 2)  # (B, F, T)
        x = self.stem(h, temb)
        skips = [x]
        for down in self.downs:
            x = down(x, temb)
            skips.append(x)
        x = self._causal_attn(x)  # bottleneck
        return x, skips

    def bottleneck(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the post-attention bottleneck feature ``(B, C_b, L_b)``."""
        h, _ = self._encode(x_t, self._temb(t))
        return h

    def forward_v(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the v-parameterised target, shape ``(B, T, F)``."""
        temb = self._temb(t)
        x, skips = self._encode(x_t, temb)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip, temb)
        return self.out(x).transpose(1, 2)  # (B, T, F)


# ── ScoreJacobLOB ────────────────────────────────────────────────────────────


class ScoreJacobLOB(nn.Module):
    """Score-Jacobian-attention diffusion classifier (see module docstring)."""

    family = "score_jacob"

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F_dim = config["n_features"]
        self.T = T
        self.F = F_dim

        base = config.get("sjl_base_channels", 64)
        depth = config.get("sjl_depth", 2)
        temb_dim = config.get("sjl_time_emb", 128)
        heads = config.get("sjl_attn_heads", 4)
        hidden = config.get("sjl_head_hidden", 128)

        self.bin = BiN(T, F_dim)
        self.backbone = DiffBackbone1D(F_dim, base, depth, temb_dim, heads)

        # Stage-3 hyperparameters
        self.t_star = list(config.get("sjl_t_star", [20, 50, 100]))  # K noise levels
        self.repr_t = int(config.get("sjl_repr_t", 0))  # level for h
        self.probe_stride = int(config.get("sjl_probe_stride", 1))
        self.min_snr_gamma = float(config.get("sjl_min_snr_gamma", 5.0))

        # diffusion schedule buffers (self-contained, like diffclf)
        T_max = config.get("T_max", 1000)
        betas = torch.linspace(
            config.get("beta_start", 1e-4),
            config.get("beta_end", 0.02),
            T_max,
            dtype=torch.float64,
        )
        ab = torch.cumprod(1.0 - betas, dim=0).float()
        self.register_buffer("sqrt_ab", ab.sqrt())
        self.register_buffer("sqrt_1mab", (1.0 - ab).sqrt())

        # Stage-4 gated head (the only part trained in phase 2)
        C_b = self.backbone.bottleneck_ch
        self.gate_feat = nn.Linear(F_dim, C_b)  # feature saliency → channel gate
        self.head = nn.Sequential(
            nn.Linear(C_b, hidden),
            nn.GELU(),
            nn.Dropout(config.get("sjl_dropout", 0.1)),
            nn.Linear(hidden, 3),
        )

    # ---- diffusion math ----

    def _ab(self, t: torch.Tensor):
        a = self.sqrt_ab[t].view(-1, 1, 1)
        s = self.sqrt_1mab[t].view(-1, 1, 1)
        return a, s

    def diffusion_loss(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """v-prediction MSE with min-SNR-γ weighting (phase 1)."""
        a, s = self._ab(t)
        x_t = a * x0 + s * noise
        v_pred = self.backbone.forward_v(x_t, t)
        v_tgt = a * noise - s * x0
        snr = (a[:, 0, 0] ** 2) / (s[:, 0, 0] ** 2)  # (B,)
        w = torch.clamp(snr, max=self.min_snr_gamma) / (snr + 1.0)  # v-param weight
        mse = ((v_pred - v_tgt) ** 2).mean(dim=(1, 2))  # (B,)
        return (w * mse).mean()

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score s_θ(x, t) = ∇_x log p_t(x), from the v-prediction."""
        a, s = self._ab(t)
        v = self.backbone.forward_v(x, t)
        eps = s * x + a * v  # ε = σ x_t + α v
        return -eps / s

    # ---- Stage 3: score-Jacobian saliency ----

    def extract_saliency(self, x_hat: torch.Tensor):
        """Marginal temporal/feature saliencies from the input-space score Jacobian.

        Returns ``(w_t (B, T), w_f (B, F))``, detached (stop-gradient), normalised
        so each row has mean 1 (a multiplicative gate centred on 1).
        """
        B, T, Fd = x_hat.shape
        wt = x_hat.new_zeros(B, T)
        wf = x_hat.new_zeros(B, Fd)
        with torch.enable_grad():
            for ts in self.t_star:
                x = x_hat.detach().requires_grad_(True)
                t = torch.full((B,), ts, dtype=torch.long, device=x_hat.device)
                s = self.score(x, t)  # (B,T,F)
                # temporal probes: cotangent over all features at output time t_o
                for to in range(0, T, self.probe_stride):
                    m = torch.zeros_like(s)
                    m[:, to, :] = 1.0
                    (g,) = torch.autograd.grad(s, x, grad_outputs=m, retain_graph=True)
                    wt += g.abs().sum(dim=2)  # (B,T) input-time
                # feature probes: cotangent over all times at output feature f_o
                for fo in range(0, Fd, self.probe_stride):
                    m = torch.zeros_like(s)
                    m[:, :, fo] = 1.0
                    (g,) = torch.autograd.grad(s, x, grad_outputs=m, retain_graph=True)
                    wf += g.abs().sum(dim=1)  # (B,F) input-feature
        wt = (wt / (wt.mean(dim=1, keepdim=True) + 1e-8)).detach()
        wf = (wf / (wf.mean(dim=1, keepdim=True) + 1e-8)).detach()
        return wt, wf

    def compute_features(self, x: torch.Tensor):
        """Raw window → (w_t, w_f, h) using the frozen BiN + backbone."""
        if x.dim() == 4:
            x = x.squeeze(1)  # (B,1,T,F) → (B,T,F)
        x_hat = self.bin(x)
        wt, wf = self.extract_saliency(x_hat)
        t = torch.full((x.shape[0],), self.repr_t, dtype=torch.long, device=x.device)
        with torch.no_grad():
            h = self.backbone.bottleneck(x_hat.detach(), t)  # (B, C_b, L_b)
        return wt, wf, h.detach()

    # ---- Stage 4: gated classification head ----

    def head_logits(
        self, wt: torch.Tensor, wf: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        L_b = h.shape[-1]
        g_c = torch.sigmoid(self.gate_feat(wf))  # (B, C_b) channel gate
        g_t = F.interpolate(
            wt.unsqueeze(1), size=L_b, mode="linear", align_corners=False
        ).squeeze(1)  # (B, L_b) temporal gate
        h_gated = h * g_c.unsqueeze(-1) * g_t.unsqueeze(1)  # (B, C_b, L_b)
        return self.head(h_gated.mean(dim=-1))  # (B, 3)

    def freeze_backbone(self) -> None:
        """Phase-2 setup: freeze BiN + diffusion backbone; keep the head trainable."""
        for p in self.bin.parameters():
            p.requires_grad_(False)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.bin.eval()
        self.backbone.eval()

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Trend logits.  Accepts a cached batch (``wt``/``wf``/``h``) or raw ``x``."""
        if "wt" in batch:
            wt = batch["wt"].to(device).float()
            wf = batch["wf"].to(device).float()
            h = batch["h"].to(device).float()
        else:
            x = batch["x"].to(device).float()
            wt, wf, h = self.compute_features(x)
        return self.head_logits(wt, wf, h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
