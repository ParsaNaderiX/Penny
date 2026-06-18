"""Test-set evaluation for Penny (spec section 9).

Runs the full DDIM+RePaint sampler over every test window (``n_samples`` each)
and reports: label accuracy, macro F1, confusion matrix, trend-ratio Pearson
correlation, mid-price MAE (IRT, first ``k`` steps), and the bid-ask spread
Wasserstein distance.  The spread metric needs a price channel, so it is
reported only in ``lob`` mode (the OFI image carries no spread).
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics import confusion_matrix, f1_score

from . import labels as lab
from .model import painted_future_mid


@torch.no_grad()
def _real_future_spread(dataset, idx: int, normalizer, config) -> np.ndarray:
    """Denormalized real best ask-bid spread over the future cols (lob mode)."""
    n = config["n_levels"]
    s = int(dataset.starts[idx])
    fut = slice(s + config["T_past"], s + config["T_total"])
    bb = normalizer.denorm_channel0(dataset.rows[fut, n - 1, 0], n - 1)
    ba = normalizer.denorm_channel0(dataset.rows[fut, n, 0], n)
    return ba - bb  # ask offset - bid offset = spread


@torch.no_grad()
def run_test(
    unet,
    diffusion,
    dataset,
    config,
    normalizer,
    level_starts,
    gamma,
    alpha,
    device,
) -> dict:
    """Evaluate all six test metrics and log them (spec 9)."""

    unet.eval()
    n, k = config["n_levels"], config["label_k"]
    t_past = config["T_past"]
    is_lob = config["feature_mode"] == "lob"

    y_true, y_pred, l_true, l_pred, maes = [], [], [], [], []
    painted_spreads, real_spreads = [], []

    for i in range(len(dataset)):
        s = dataset[i]
        ns = config["n_samples"]
        x0_known = s["image"].unsqueeze(0).repeat(ns, 1, 1, 1).to(device)
        m = s["mask"].unsqueeze(0).repeat(ns, 1, 1, 1).to(device)
        painted = diffusion.sample(unet, x0_known, m, config["ddim_steps"], device)

        ref = torch.full((ns,), float(s["mid_ref"]), device=device)
        fut_mid = painted_future_mid(
            painted, config, normalizer, level_starts, ref, gamma
        )
        fwd = fut_mid[:, :k].mean(dim=1).cpu().numpy()
        l_vals = (fwd - s["bwd_smoothed"]) / (s["bwd_smoothed"] + 1e-12)
        modal = int(
            np.bincount(
                [lab.label_from_l(float(x), alpha) for x in l_vals], minlength=3
            ).argmax()
        )

        y_true.append(s["label"])
        y_pred.append(modal)
        l_true.append(s["l"])
        l_pred.append(float(l_vals.mean()))

        mean_mid = fut_mid.mean(dim=0).cpu().numpy()
        true_future = s["true_mid"].numpy()[t_past : t_past + k]
        maes.append(float(np.mean(np.abs(mean_mid[:k] - true_future))))

        if is_lob:
            bb = painted[:, 0, level_starts[n - 1] : level_starts[n], t_past:].mean(
                dim=1
            )
            ba = painted[:, 0, level_starts[n] : level_starts[n + 1], t_past:].mean(
                dim=1
            )
            bb = normalizer.denorm_channel0(bb.cpu().numpy(), n - 1)
            ba = normalizer.denorm_channel0(ba.cpu().numpy(), n)
            painted_spreads.append((ba - bb).mean(axis=0))
            real_spreads.append(_real_future_spread(dataset, i, normalizer, config))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    corr = float(pearsonr(l_true, l_pred)[0]) if np.std(l_pred) > 0 else 0.0
    mae = float(np.mean(maes))

    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion": cm,
        "trend_corr": corr,
        "mid_mae": mae,
    }
    logger.info("TEST accuracy={:.4f} macro_f1={:.4f}", acc, f1)
    logger.info("TEST confusion (rows=true down/stat/up):\n{}", cm)
    logger.info("TEST trend-ratio Pearson r={:.4f}", corr)
    logger.info("TEST mid MAE (first {} steps, IRT)={:.2f}", k, mae)

    if is_lob and painted_spreads:
        w = float(
            wasserstein_distance(
                np.concatenate(painted_spreads), np.concatenate(real_spreads)
            )
        )
        metrics["spread_wasserstein"] = w
        logger.info("TEST spread Wasserstein={:.4f}", w)
    else:
        metrics["spread_wasserstein"] = None
        logger.info("TEST spread Wasserstein: N/A (ofi mode has no price channel)")

    return metrics
