"""Shared test-set evaluation for all crypto model families.

Every model in this repo exposes ``model.predict(batch, device) → (B, 3) logits``
so the metrics (accuracy, macro-F1, confusion, mean probs) are identical across
DeepLOB, JointDiffusion, LOBTransformer, etc.  Import and use ``run_test`` from here
rather than defining it per-family.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def per_asset_metrics(
    model,
    dataset: Dataset,
    config: dict,
    device: torch.device,
    symbols: list[str],
    split_name: str = "EVAL",
) -> dict:
    """Per-asset accuracy / macro-F1 / confusion for a pooled multi-asset dataset.

    Every sample is classified conditioned on its own (known) asset id (taken from
    ``batch["asset"]``), then results are grouped by asset.  Logs per-asset rows
    plus a pooled ``ALL`` row, and returns a JSON-serialisable dict.

    Args:
        model:      Any model with ``predict(batch, device) → (B, 3) logits``.
        dataset:    PyTorch Dataset; items must include ``"label"`` and ``"asset"``
                    keys.
        config:     Config dict with ``"batch_size"``.
        device:     Target device.
        symbols:    Ordered list of symbol names; index i → ``symbols[i]``.
        split_name: Label shown in log lines (e.g. ``"VAL"``, ``"TEST"``).

    Returns:
        ``{symbol: {"accuracy", "macro_f1", "n", "confusion"}, ..., "ALL": {...}}``.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    bucket: dict = {}
    for batch in loader:
        preds = model.predict(batch, device).argmax(1).cpu().tolist()
        for asset, pred, label in zip(
            batch["asset"].tolist(), preds, batch["label"].tolist()
        ):
            yt, yp = bucket.setdefault(asset, ([], []))
            yt.append(label)
            yp.append(pred)

    logger.info("{}  per-asset metrics:", split_name)
    out: dict = {}
    all_true: list = []
    all_pred: list = []
    for asset in sorted(bucket):
        yt = np.array(bucket[asset][0])
        yp = np.array(bucket[asset][1])
        all_true.extend(yt.tolist())
        all_pred.extend(yp.tolist())
        acc = float((yt == yp).mean()) if len(yt) else 0.0
        f1 = float(f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0))
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
        name = symbols[asset] if asset < len(symbols) else str(asset)
        logger.info(
            "    {:<12} acc={:.4f} macro_f1={:.4f} n={}", name, acc, f1, len(yt)
        )
        out[name] = {
            "accuracy": acc,
            "macro_f1": f1,
            "n": int(len(yt)),
            "confusion": cm.tolist(),
        }

    yt_all = np.array(all_true)
    yp_all = np.array(all_pred)
    acc = float((yt_all == yp_all).mean()) if len(yt_all) else 0.0
    f1 = float(
        f1_score(yt_all, yp_all, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    logger.info(
        "    {:<12} acc={:.4f} macro_f1={:.4f} n={}", "ALL", acc, f1, len(yt_all)
    )
    out["ALL"] = {"accuracy": acc, "macro_f1": f1, "n": int(len(yt_all))}
    return out


@torch.no_grad()
def run_test(model, dataset, config, device) -> dict:
    """Evaluate *model* on *dataset* and log accuracy / F1 / confusion.

    Args:
        model:   Any model with ``predict(batch, device) → (B, 3) logits``.
        dataset: PyTorch Dataset; items must include a ``"label"`` key.
        config:  Config dict; must contain ``"batch_size"``.
        device:  ``torch.device``.

    Returns:
        ``{"accuracy": float, "macro_f1": float, "confusion": ndarray (3,3)}``.
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    y_true, y_pred, probs_all = [], [], []
    for batch in loader:
        logits = model.predict(batch, device)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(batch["label"].tolist())
        y_pred.extend(preds.tolist())
        probs_all.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs_all = np.concatenate(probs_all, axis=0)

    acc = float((y_true == y_pred).mean())
    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    logger.info("TEST  accuracy={:.4f}  macro_f1={:.4f}", acc, f1)
    logger.info("TEST  confusion (rows=true down/stat/up):\n{}", cm)
    logger.info(
        "TEST  mean probs  down={:.3f}  stat={:.3f}  up={:.3f}",
        probs_all[:, 0].mean(),
        probs_all[:, 1].mean(),
        probs_all[:, 2].mean(),
    )
    return {"accuracy": acc, "macro_f1": f1, "confusion": cm}
