"""JumpGate-ScoreGrad: the JumpGate Lévy-jump design on a ScoreGrad backbone.

Same forward process, noise-state estimator and gating as :class:`JumpGateUNet`,
but the 2-D U-Net is replaced by a **ScoreGrad / TimeGrad-style** score network
(Yan et al. 2021, arXiv:2106.10121; https://github.com/yantijin/ScoreGradPred):

* a **GRU encoder** unrolls the LOB window along time, producing a per-timestep
  hidden context ``H (B, T, num_cells)``;
* a **WaveNet ``EpsilonTheta``** denoises each timestep's ``F``-dim feature vector
  with a stack of dilated ``Conv1d`` residual blocks (gated ``sigmoid·tanh``
  activation, per-block diffusion-step + conditioner injection, skip aggregation),
  the conditioner being that timestep's GRU hidden state upsampled to ``F``.

So the diffusion runs over the feature axis (``F`` = ScoreGrad's ``target_dim``)
conditioned on the recurrent context — exactly ScoreGrad's structure — while the
window's whole ``(T, F)`` grid is denoised at once by batching timesteps.

JumpGate additions carried over from :class:`JumpGateUNet`:

* ``g_phi`` (:class:`NoiseStateEstimator`) infers ``(logW_hat, pi_logit)``;
* the per-block **diffusion-step** vector encodes ``(t, logW)`` via ``w_conditioning``
  (``none`` | ``inferred`` | ``oracle``) — this is where W-awareness enters the
  score net (ScoreGrad's ``diffusion_embedding``);
* **gated experts**: two output projections mixed by ``pi = sigmoid(pi_logit)``;
* the **trend head** reads an attention-pool over the GRU context; inference is
  feature-only (GRU + trend head on the clean window, no score net, no sampling).

Padding note: the residual/skip/output convs use *same* padding (vs. ScoreGrad's
asymmetric ``padding=2`` + kernel-3-no-pad scheme) so the network preserves the
feature length for an arbitrary ``F`` — functionally equivalent, more robust.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.jumpgateunet import NoiseStateEstimator
from models.modules import (
    AttentionPool,
    count_parameters as count_parameters,  # re-export
    sinusoidal_embedding,
)


class DiffusionStepMLP(nn.Module):
    """Build the per-block diffusion-step vector from ``t`` (and optionally ``logW``).

    This is ScoreGrad's ``diffusion_embedding`` generalized to be W-aware: it is the
    only place the inferred/true noise state enters the score network.
    """

    def __init__(self, temb_dim: int, hidden: int, w_conditioning: str) -> None:
        super().__init__()
        self.temb_dim = temb_dim
        self.w_conditioning = w_conditioning
        cond_in = temb_dim if w_conditioning == "none" else 2 * temb_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(
        self, t: torch.Tensor, logW_hat: torch.Tensor, logW_oracle: torch.Tensor | None
    ) -> torch.Tensor:
        temb = sinusoidal_embedding(t, self.temb_dim)
        if self.w_conditioning == "none":
            return self.mlp(temb)
        if self.w_conditioning == "oracle":
            logw = logW_hat.detach() if logW_oracle is None else logW_oracle
        else:  # inferred
            logw = logW_hat.detach()
        wemb = sinusoidal_embedding(logw, self.temb_dim)
        return self.mlp(torch.cat([temb, wemb], dim=-1))


class CondUpsampler(nn.Module):
    """Upsample the GRU hidden context (num_cells) to the feature length F."""

    def __init__(self, cond_length: int, target_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2 or 1)
        self.linear2 = nn.Linear(target_dim // 2 or 1, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.linear1(x), 0.4)
        return F.leaky_relu(self.linear2(x), 0.4)


class ResidualBlock(nn.Module):
    """ScoreGrad residual block: dilated conv + gated activation + skip.

    ``x`` and ``conditioner`` are ``(N, C, F)`` / ``(N, 1, F)``; ``diffusion_step`` is
    ``(N, residual_hidden)``.  Same-padding preserves the feature length ``F``.
    """

    def __init__(self, residual_hidden: int, residual_channels: int, dilation: int):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(residual_hidden, residual_channels)
        self.conditioner_projection = nn.Conv1d(1, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = F.leaky_relu(self.output_projection(y), 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class EpsilonTheta(nn.Module):
    """WaveNet score network over the feature axis, conditioned on the GRU context."""

    def __init__(
        self,
        target_dim: int,
        cond_length: int,
        residual_hidden: int,
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        gated_experts: bool = False,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(1, residual_channels, 1)
        self.cond_upsampler = CondUpsampler(cond_length, target_dim)
        self.residual_layers = nn.ModuleList(
            ResidualBlock(
                residual_hidden,
                residual_channels,
                dilation=2 ** (i % dilation_cycle_length),
            )
            for i in range(residual_layers)
        )
        self.skip_projection = nn.Conv1d(
            residual_channels, residual_channels, 3, padding=1
        )
        self.out0 = nn.Conv1d(residual_channels, 1, 3, padding=1)
        self.out1 = (
            nn.Conv1d(residual_channels, 1, 3, padding=1) if gated_experts else None
        )
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.out0.weight)
        if self.out1 is not None:
            nn.init.zeros_(self.out1.weight)

    def _trunk(self, inp: torch.Tensor, cond: torch.Tensor, dstep: torch.Tensor):
        x = F.leaky_relu(self.input_projection(inp), 0.4)  # (N, rc, F)
        cond_up = self.cond_upsampler(cond).unsqueeze(1)  # (N, 1, F)
        skips = []
        for layer in self.residual_layers:
            x, s = layer(x, cond_up, dstep)
            skips.append(s)
        x = torch.stack(skips).sum(0) / math.sqrt(len(self.residual_layers))
        return F.leaky_relu(self.skip_projection(x), 0.4)  # (N, rc, F)

    def forward(self, inp, cond, dstep, pi=None):
        """``inp (N,1,F)``, ``cond (N,cond_length)``, ``dstep (N,residual_hidden)``.

        Returns the predicted noise ``(N, 1, F)`` — a 2-expert mix when ``self.out1``
        exists and ``pi`` is given.
        """
        h = self._trunk(inp, cond, dstep)
        eps0 = self.out0(h)
        if self.out1 is not None and pi is not None:
            eps1 = self.out1(h)
            eps = (1.0 - pi).view(-1, 1, 1) * eps0 + pi.view(-1, 1, 1) * eps1
        else:
            eps = eps0
        return eps


class JumpGateScoreGrad(nn.Module):
    """GRU encoder + WaveNet score net + trend head, with the JumpGate machinery."""

    family = "joint_diffusion"

    def __init__(self, config: dict) -> None:
        super().__init__()
        F_dim = config["n_features"]
        temb_dim = config.get("jdl_time_emb", 128)
        self.temb_dim = temb_dim
        self.F = F_dim

        # JumpGate flags
        self.w_conditioning = config.get("w_conditioning", "none")
        if self.w_conditioning not in ("none", "inferred", "oracle"):
            raise ValueError(
                f"w_conditioning must be none|inferred|oracle, got {self.w_conditioning!r}"
            )
        self.gated_experts = bool(config.get("gated_experts", False))
        self.gate_grad = config.get("gate_grad", "detach")
        if self.gate_grad not in ("detach", "flow"):
            raise ValueError(f"gate_grad must be detach|flow, got {self.gate_grad!r}")

        # ScoreGrad backbone hyperparameters
        num_cells = config.get("sg_num_cells", 64)
        num_layers = config.get("sg_num_layers", 2)
        residual_hidden = config.get("sg_residual_hidden", 64)

        self.gru = nn.GRU(
            input_size=F_dim,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=config.get("sg_rnn_dropout", 0.0) if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.gphi = NoiseStateEstimator(
            F_dim, temb_dim, hidden=config.get("jg_gphi_hidden", 64)
        )
        self.dstep = DiffusionStepMLP(temb_dim, residual_hidden, self.w_conditioning)
        self.score = EpsilonTheta(
            target_dim=F_dim,
            cond_length=num_cells,
            residual_hidden=residual_hidden,
            residual_layers=config.get("sg_residual_layers", 8),
            residual_channels=config.get("sg_residual_channels", 8),
            dilation_cycle_length=config.get("sg_dilation_cycle", 2),
            gated_experts=self.gated_experts,
        )

        # trend head over the GRU context
        self.pool = AttentionPool(num_cells, heads=config.get("jdl_pool_heads", 4))
        self.cls_dropout = nn.Dropout(config.get("cls_dropout", 0.0))
        self.classifier = nn.Linear(num_cells, 3)

    def _encode(self, x_t: torch.Tensor) -> torch.Tensor:
        """GRU context ``H (B, T, num_cells)`` from a window ``(B, 1, T, F)``."""
        H, _ = self.gru(x_t.squeeze(1))  # (B, T, num_cells)
        return H

    def _trend_logits(self, H: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cls_dropout(self.pool(H)))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        logW_oracle: torch.Tensor | None = None,
    ):
        """Return ``(eps_hat (B,1,T,F), logits (B,3), logW_hat (B,), pi_logit (B,))``."""
        b, _, T, Fd = x_t.shape
        temb_t = sinusoidal_embedding(t, self.temb_dim)
        logW_hat, pi_logit = self.gphi(x_t, temb_t)

        H = self._encode(x_t)  # (B, T, num_cells)
        logits = self._trend_logits(H)

        # per-timestep denoising: batch (B,T) rows, feature axis F as conv length
        dstep = self.dstep(t, logW_hat, logW_oracle)  # (B, residual_hidden)
        dstep = dstep.repeat_interleave(T, dim=0)  # (B*T, residual_hidden)
        cond = H.reshape(b * T, -1)  # (B*T, num_cells)
        inp = x_t.squeeze(1).reshape(b * T, 1, Fd)  # (B*T, 1, F)
        pi = None
        if self.gated_experts:
            pi = torch.sigmoid(pi_logit)
            if self.gate_grad == "detach":
                pi = pi.detach()
            pi = pi.repeat_interleave(T, dim=0)  # (B*T,)
        eps = self.score(inp, cond, dstep, pi)  # (B*T, 1, F)
        eps_hat = eps.reshape(b, 1, T, Fd)
        return eps_hat, logits, logW_hat, pi_logit

    @staticmethod
    def recover_score(eps_hat: torch.Tensor, W_hat: torch.Tensor) -> torch.Tensor:
        v = (-1,) + (1,) * (eps_hat.dim() - 1)
        return -eps_hat / W_hat.reshape(v)

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Feature-only inference: GRU + trend head on the clean window (no score net)."""
        x = batch["x"].to(device).float()
        return self._trend_logits(self._encode(x))
