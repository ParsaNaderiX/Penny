"""Training and validation routines for Penny (spec section 7).

Provides the per-epoch training step (mask-weighted diffusion loss plus the
timestep-weighted DeepLOB trend loss), the validation diffusion loss used for
early stopping, and a subset-based validation label accuracy that runs the full
DDIM+RePaint sampler.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import labels as lab
from .diffusion import Diffusion
from .model import painted_future_mid


def build_optimizer_scheduler(unet, trend_head, config, total_steps: int):
    """AdamW over UNet + trend head, with linear warmup then cosine decay (spec 7.2)."""
    optimizer = AdamW(
        list(unet.parameters()) + list(trend_head.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    warmup = config["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, LambdaLR(optimizer, lr_lambda)


def _masked_diffusion_loss(eps_hat, noise, mask) -> torch.Tensor:
    """Mean-squared noise error over the future region only (spec 7.1)."""
    m = mask.expand_as(eps_hat)
    return ((eps_hat - noise) ** 2 * m).sum() / m.sum().clamp(min=1)


def train_one_epoch(
    unet,
    trend_head,
    diffusion: Diffusion,
    loader: DataLoader,
    optimizer,
    scheduler,
    config,
    normalizer,
    level_starts,
    gamma,
    device,
) -> dict:
    """Run one training epoch; returns mean diffusion / trend / total losses."""
    unet.train()
    trend_head.train()
    t_max, k = config["T_max"], config["label_k"]
    lam = config["lambda_trend"]
    tot = dif = trd = 0.0
    n = 0

    for batch in loader:
        image = batch["image"].to(device)  # (B, 2, H, W)
        mask = batch["mask"].to(device)  # (B, 1, H, W)
        label = batch["label"].to(device)
        mid_ref = batch["mid_ref"].to(device).float()
        bwd = batch["bwd_smoothed"].to(device).float()
        history = image * (1.0 - mask)
        b = image.shape[0]

        t = torch.randint(0, t_max, (b,), device=device)
        noise = torch.randn_like(image)
        x_t = diffusion.q_sample(image, t, noise)
        model_in = torch.cat([x_t, history, mask], dim=1)
        eps_hat = unet(model_in, t).sample

        diff_loss = _masked_diffusion_loss(eps_hat, noise, mask)

        x0_hat = diffusion._x0_from_eps(x_t, eps_hat, t)
        fut_mid = painted_future_mid(
            x0_hat, config, normalizer, level_starts, mid_ref, gamma
        )
        fwd = fut_mid[:, :k].mean(dim=1)
        l_pred = (fwd - bwd) / (bwd + 1e-12)  # bwd is real -> detached input
        logits = trend_head(l_pred)
        ce = F.cross_entropy(logits, label, reduction="none")
        w = (1.0 - t.float() / t_max) ** 2
        trend_loss = (w * ce).mean()

        loss = diff_loss + lam * trend_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(unet.parameters()) + list(trend_head.parameters()), config["grad_clip"]
        )
        optimizer.step()
        scheduler.step()

        tot += loss.item()
        dif += diff_loss.item()
        trd += trend_loss.item()
        n += 1

    n = max(n, 1)
    return {"total": tot / n, "diff": dif / n, "trend": trd / n}


@torch.no_grad()
def validate_diffusion(unet, diffusion, loader, config, device) -> float:
    """Mean mask-weighted noise MSE on validation (early-stopping metric, spec 7.3)."""
    unet.eval()
    t_max = config["T_max"]
    total, n = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        history = image * (1.0 - mask)
        b = image.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)
        noise = torch.randn_like(image)
        x_t = diffusion.q_sample(image, t, noise)
        eps_hat = unet(torch.cat([x_t, history, mask], dim=1), t).sample
        total += _masked_diffusion_loss(eps_hat, noise, mask).item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def sample_label(
    unet,
    diffusion,
    image,
    mask,
    config,
    normalizer,
    level_starts,
    gamma,
    alpha,
    mid_ref,
    bwd_smoothed,
    device,
) -> tuple[int, np.ndarray, float, float]:
    """Sample ``n_samples`` futures for one window; return modal label + stats."""
    ns = config["n_samples"]
    k = config["label_k"]
    x0_known = image.unsqueeze(0).repeat(ns, 1, 1, 1).to(device)
    m = mask.unsqueeze(0).repeat(ns, 1, 1, 1).to(device)
    painted = diffusion.sample(unet, x0_known, m, config["ddim_steps"], device)

    ref = torch.full((ns,), float(mid_ref), device=device)
    fut_mid = painted_future_mid(painted, config, normalizer, level_starts, ref, gamma)
    fwd = fut_mid[:, :k].mean(dim=1).cpu().numpy()
    l_vals = (fwd - bwd_smoothed) / (bwd_smoothed + 1e-12)
    labels = np.array([lab.label_from_l(float(x), alpha) for x in l_vals])
    modal = int(np.bincount(labels, minlength=3).argmax())
    return modal, l_vals, float(l_vals.mean()), float(l_vals.std())


@torch.no_grad()
def validate_label_accuracy(
    unet,
    diffusion,
    dataset,
    config,
    normalizer,
    level_starts,
    gamma,
    alpha,
    device,
    n_windows: int,
    seed: int = 0,
) -> dict:
    """Label accuracy on a random subset of validation windows (spec 7.4)."""
    unet.eval()
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(n_windows, len(dataset)), replace=False)
    correct = 0
    cm = np.zeros((3, 3), dtype=np.int64)
    for i in idxs:
        s = dataset[int(i)]
        modal, _, _, _ = sample_label(
            unet,
            diffusion,
            s["image"],
            s["mask"],
            config,
            normalizer,
            level_starts,
            gamma,
            alpha,
            s["mid_ref"],
            s["bwd_smoothed"],
            device,
        )
        true = s["label"]
        cm[true, modal] += 1
        correct += int(modal == true)
    acc = correct / max(len(idxs), 1)
    logger.info("val label acc={:.4f} on {} windows", acc, len(idxs))
    return {"accuracy": acc, "confusion": cm}
