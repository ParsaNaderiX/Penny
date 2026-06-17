"""The Penny model: a regime-conditioned temporal-UNet diffusion network with
an auxiliary LSTM trend-prediction head.

Conditioning is fused from three sources -- an LSTM context encoder over the
past sequence, an MLP regime encoder, and a sinusoidal timestep encoder -- and
injected into the UNet through FiLM modulation and cross-attention.  After
predicting the noise, the clean trajectory is recovered via Tweedie's formula
and fed to the LSTM trend head.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of scalar diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed ``t`` of shape (B,) into (B, dim)."""
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class FiLM(nn.Module):
    """Feature-wise linear modulation conditioned on the fused vector."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Modulate ``x`` (B, C, L) with ``cond`` (B, cond_dim)."""
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)


class FiLMConvBlock(nn.Module):
    """Conv1d + GroupNorm + FiLM + SiLU, preserving sequence length."""

    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int, groups: int, kernel: int = 3
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.film = FiLM(cond_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.act(self.film(self.norm(self.conv(x)), cond))


class Bottleneck(nn.Module):
    """Self-attention followed by cross-attention with the fused conditioning."""

    def __init__(self, channels: int, cond_dim: int, heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.cond_proj = nn.Linear(cond_dim, channels)
        self.cross_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x`` is (B, C, L); ``cond`` is (B, cond_dim)."""
        h = x.transpose(1, 2)  # (B, L, C)
        sa, _ = self.self_attn(h, h, h)
        h = self.norm1(h + sa)
        kv = self.cond_proj(cond).unsqueeze(1)  # (B, 1, C)
        ca, _ = self.cross_attn(h, kv, kv)
        h = self.norm2(h + ca)
        return h.transpose(1, 2)  # (B, C, L)


class TemporalUNet(nn.Module):
    """1-D UNet over the noisy trajectory with FiLM + attention conditioning."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        f = config["F"]
        base = config["unet_base"]
        mid = config["unet_mid"]
        cond = config["cond_dim"]
        groups = config["gn_groups"]
        heads = config["attn_heads"]

        self.enc_in = FiLMConvBlock(f, base, cond, groups)
        self.down = nn.Conv1d(base, mid, kernel_size=4, stride=2, padding=1)
        self.down_norm = nn.GroupNorm(groups, mid)
        self.down_film = FiLM(cond, mid)
        self.down_act = nn.SiLU()

        self.bottleneck = Bottleneck(mid, cond, heads)

        self.up = nn.ConvTranspose1d(mid, base, kernel_size=4, stride=2, padding=1)
        self.up_norm = nn.GroupNorm(groups, base)
        self.up_film = FiLM(cond, base)
        self.up_act = nn.SiLU()

        self.dec_out = FiLMConvBlock(base, base, cond, groups)
        self.out = nn.Linear(base, f)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x`` is (B, F, T); returns eps prediction (B, T, F)."""
        e1 = self.enc_in(x, cond)  # (B, base, T)
        e2 = self.down_act(self.down_film(self.down_norm(self.down(e1)), cond))
        b = self.bottleneck(e2, cond)  # (B, mid, T/2)
        d = self.up_act(self.up_film(self.up_norm(self.up(b)), cond))  # (B, base, T)
        d = d + e1  # UNet skip
        d = self.dec_out(d, cond)
        return self.out(d.transpose(1, 2))  # (B, T, F)


class Penny(nn.Module):
    """Regime-conditioned diffusion model with an auxiliary trend head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        f = config["F"]
        hidden = config["lstm_hidden"]
        layers = config["lstm_layers"]
        emb = config["emb_dim"]
        cond = config["cond_dim"]
        n_cls = config["num_classes"]

        self.context_lstm = nn.LSTM(f, hidden, layers, batch_first=True)

        self.regime_mlp = nn.Sequential(
            nn.Linear(config["regime_dim"], config["regime_hidden"]),
            nn.SiLU(),
            nn.Linear(config["regime_hidden"], emb),
        )

        self.time_emb = SinusoidalPosEmb(emb)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb)
        )

        self.fuse = nn.Sequential(nn.Linear(hidden + emb + emb, cond), nn.SiLU())

        self.unet = TemporalUNet(config)

        self.trend_lstm = nn.LSTM(
            f, hidden, layers, batch_first=True, dropout=config["trend_dropout"]
        )
        self.trend_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, config["trend_mlp_hidden"]),
            nn.GELU(),
            nn.Linear(config["trend_mlp_hidden"], n_cls),
        )

    def forward(
        self,
        past_seq: torch.Tensor,
        future_noisy: torch.Tensor,
        t: torch.Tensor,
        regime: torch.Tensor,
        alpha_bar_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict noise, recover ``x0`` (Tweedie) and predict the trend class.

        Parameters
        ----------
        past_seq : (B, T, F) context sequence.
        future_noisy : (B, T, F) noised future trajectory ``x_t``.
        t : (B,) diffusion timesteps.
        regime : (B, regime_dim) regime conditioning.
        alpha_bar_t : (B,) cumulative alpha for each sample's timestep.

        Returns
        -------
        eps_pred, trend_logits, x0_hat
        """
        _, (h_ctx, _) = self.context_lstm(past_seq)
        context_emb = h_ctx[-1]  # (B, hidden)
        regime_emb = self.regime_mlp(regime)  # (B, emb)
        t_emb = self.time_mlp(self.time_emb(t))  # (B, emb)
        fused = self.fuse(torch.cat([context_emb, regime_emb, t_emb], dim=1))

        eps_pred = self.unet(future_noisy.transpose(1, 2), fused)

        ab = alpha_bar_t.view(-1, 1, 1)
        x0_hat = (future_noisy - torch.sqrt(1.0 - ab) * eps_pred) / torch.sqrt(ab)

        _, (h_tr, _) = self.trend_lstm(x0_hat)
        trend_logits = self.trend_head(h_tr[-1])
        return eps_pred, trend_logits, x0_hat


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
