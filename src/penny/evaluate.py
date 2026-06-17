"""Evaluation for Penny: per-epoch validation metrics and the full test suite.

Validation reports diffusion / trend losses and trend-classification metrics.
The end-of-training test suite additionally measures generative quality (KS,
Wasserstein, ACF) and a counterfactual-validity test on the regime conditioning.
All metrics are logged at INFO level.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)


def _trend_weight(t: torch.Tensor, t_max: int) -> torch.Tensor:
    """Timestep weighting ``w_t = (1 - t / T_max)^2``."""
    return (1.0 - t.float() / t_max) ** 2


def compute_losses(model, diffusion, batch, config):
    """Return ``(total, diff_loss, trend_loss, logits, labels)`` for one batch."""
    past, fut_noisy, noise, t, regime, label, _ = batch
    ab_t = diffusion.alpha_bars[t]
    eps_pred, trend_logits, _ = model(past, fut_noisy, t, regime, ab_t)

    diff_loss = F.mse_loss(eps_pred, noise)
    w = _trend_weight(t, config["T_max"])
    ce = F.cross_entropy(trend_logits, label, reduction="none")
    trend_loss = (w * ce).mean()
    total = diff_loss + config["lambda_trend"] * trend_loss
    return total, diff_loss, trend_loss, trend_logits, label


@torch.no_grad()
def run_validation(model, diffusion, loader, config) -> dict:
    """Evaluate validation losses and trend-classification metrics."""
    model.eval()
    tot = dif = trd = 0.0
    n_batches = 0
    preds, gts = [], []
    for batch in loader:
        total, diff_loss, trend_loss, logits, label = compute_losses(
            model, diffusion, batch, config
        )
        if torch.isnan(total):
            logger.warning("NaN loss encountered during validation")
            continue
        tot += total.item()
        dif += diff_loss.item()
        trd += trend_loss.item()
        n_batches += 1
        preds.append(logits.argmax(dim=1).cpu().numpy())
        gts.append(label.cpu().numpy())

    n_batches = max(n_batches, 1)
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(gts) if gts else np.array([])
    metrics = {
        "total_loss": tot / n_batches,
        "diff_loss": dif / n_batches,
        "trend_loss": trd / n_batches,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if len(y_true)
        else 0.0,
        "kappa": float(cohen_kappa_score(y_true, y_pred))
        if len(y_true) and len(np.unique(y_true)) > 1
        else 0.0,
    }
    if len(y_true):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(config["num_classes"])))
        metrics["confusion_matrix"] = cm
    return metrics


def _acf(series: np.ndarray, nlags: int) -> np.ndarray:
    """Autocorrelation of a 1-D series for lags ``1..nlags``."""
    s = series - series.mean()
    denom = np.sum(s * s) + 1e-12
    return np.array([np.sum(s[:-lag] * s[lag:]) / denom for lag in range(1, nlags + 1)])


@torch.no_grad()
def _generate(model, diffusion, loader, config, max_batches):
    """Generate trajectories for up to ``max_batches`` test batches."""
    gen, real, regimes, contexts = [], [], [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        past, _, _, _, regime, _, fut_clean = batch
        shape = (past.shape[0], config["T"], config["F"])
        g = diffusion.sample(model, shape, regime, past, config["ddim_steps"])
        gen.append(g)
        real.append(fut_clean)
        regimes.append(regime)
        contexts.append(past)
    return (torch.cat(gen), torch.cat(real), torch.cat(regimes), torch.cat(contexts))


@torch.no_grad()
def run_test(model, diffusion, loader, config, columns, train_trend_sigma) -> None:
    """Run and log the full test suite (generative + classification metrics)."""
    model.eval()
    mid_idx = columns.index("mid_price")
    spread_idx = columns.index("spread")

    # ---- trend classification ---------------------------------------------------
    preds, gts = [], []
    for batch in loader:
        _, _, _, logits, label = compute_losses(model, diffusion, batch, config)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        gts.append(label.cpu().numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(gts)
    logger.info(
        "TEST trend accuracy=%.4f f1_macro=%.4f kappa=%.4f",
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
        cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config["num_classes"])))
    logger.info("TEST confusion matrix (rows=true up/flat/down):\n%s", cm)

    # ---- generative quality -----------------------------------------------------
    gen, real, regimes, contexts = _generate(
        model, diffusion, loader, config, config["eval_gen_batches"]
    )
    gen_np = gen.cpu().numpy().reshape(-1, config["F"])
    real_np = real.cpu().numpy().reshape(-1, config["F"])

    ks_per_feat = np.array(
        [ks_2samp(real_np[:, j], gen_np[:, j]).statistic for j in range(config["F"])]
    )
    logger.info(
        "TEST KS statistic: mean=%.4f max=%.4f (feature '%s')",
        ks_per_feat.mean(),
        ks_per_feat.max(),
        columns[int(ks_per_feat.argmax())],
    )
    logger.debug("TEST KS per feature: %s", dict(zip(columns, ks_per_feat.round(4))))

    w_spread = wasserstein_distance(real_np[:, spread_idx], gen_np[:, spread_idx])
    logger.info("TEST Wasserstein distance (spread): %.6f", w_spread)

    nlags = 20
    real_mid = real.cpu().numpy()[:, :, mid_idx]
    gen_mid = gen.cpu().numpy()[:, :, mid_idx]
    real_ret = np.diff(real_mid, axis=1)
    gen_ret = np.diff(gen_mid, axis=1)
    real_acf = np.mean([_acf(r, nlags) for r in real_ret], axis=0)
    gen_acf = np.mean([_acf(r, nlags) for r in gen_ret], axis=0)
    logger.info("TEST ACF real (lags1-20): %s", real_acf.round(4).tolist())
    logger.info("TEST ACF gen  (lags1-20): %s", gen_acf.round(4).tolist())
    logger.info(
        "TEST ACF mean abs error: %.4f", float(np.mean(np.abs(real_acf - gen_acf)))
    )

    # ---- counterfactual validity ------------------------------------------------
    sigma_n = train_trend_sigma * config["counterfactual_sigma"]
    shape = (contexts.shape[0], config["T"], config["F"])
    up_reg = regimes.clone()
    up_reg[:, 0] = sigma_n
    down_reg = regimes.clone()
    down_reg[:, 0] = -sigma_n
    gen_up = diffusion.sample(model, shape, up_reg, contexts, config["ddim_steps"])
    gen_down = diffusion.sample(model, shape, down_reg, contexts, config["ddim_steps"])

    def _direction(g):
        m = g[:, :, mid_idx]
        return m[:, -1] - m[:, 0]

    up_ok = (_direction(gen_up) > 0).float().mean().item()
    down_ok = (_direction(gen_down) < 0).float().mean().item()
    logger.info(
        "TEST counterfactual validity: trend=+%.1fσ up=%.3f | "
        "trend=-%.1fσ down=%.3f | overall=%.3f",
        config["counterfactual_sigma"],
        up_ok,
        config["counterfactual_sigma"],
        down_ok,
        (up_ok + down_ok) / 2,
    )
