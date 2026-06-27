"""Multi-asset windowed LOB dataset: pools every symbol of one exchange.

A single :class:`~models.penny.Penny` model is trained across all coins, so each
window carries an **asset id** (index into the symbol list) in addition to its
trend label.

Per-symbol processing is *identical* to the single-symbol pipeline —
causal rolling-window normalization (:func:`crypto.loader.build_cache`), per-symbol
``alpha`` calibration and chronological train/val/test split
(:func:`crypto.dataset.build_labels` / ``_valid_starts``).  Only the resulting
windows are concatenated.  Each symbol reuses its existing per-symbol cache
(``cache_root/SYMBOL``), so no rebuild is needed if the single-symbol models
were already run.

``config`` keys (in addition to the usual single-symbol ones)
------------------------------------------------------------
symbols      : list[str]        — the coins to pool (e.g. all Binance pairs)
cache_root   : str              — parent cache dir; per-symbol cache is ``cache_root/SYMBOL``
symbol_alphas: dict[str, float] — optional per-symbol alpha overrides, e.g.
               ``{"USDCUSDT": 3e-6}``.  Symbols not listed fall back to
               ``label_alpha`` (auto-calibrate if -1).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from .dataset import _valid_starts
from .features import extract_features, n_features
from .labels import build_labels
from .loader import build_cache


class MultiLOBDataset(Dataset):
    """Windows pooled across symbols; each item carries its asset id.

    Args:
        feats:       per-symbol feature memmaps, ``list[np.memmap]`` ``(N_s, F)``.
        labels_list: per-symbol trend labels, ``list[np.ndarray]``.
        items:       ``(M, 2)`` int array of ``(sym_idx, start)`` window starts.
        t_past:      window length.
    """

    def __init__(
        self,
        feats: list,
        labels_list: list,
        items: np.ndarray,
        t_past: int,
    ) -> None:
        self.feats = feats
        self.labels_list = labels_list
        self.items = items
        self.t_past = t_past

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        sym = int(self.items[idx, 0])
        s = int(self.items[idx, 1])
        window = self.feats[sym][s : s + self.t_past].astype(np.float32)
        x = torch.from_numpy(window.copy()).unsqueeze(0)  # (1, T, F)
        label = int(self.labels_list[sym][s + self.t_past - 1])
        return {"x": x, "label": label, "asset": sym}


def build_multi_datasets(config: dict):
    """Build pooled train/val/test datasets across ``config["symbols"]``.

    Returns ``(train_ds, val_ds, test_ds, meta)`` where ``meta`` includes
    ``n_features``, ``n_assets``, ordered ``symbols``, per-symbol ``alphas`` and
    window ``counts``, and overall ``class_balance``.
    """
    symbols = list(config["symbols"])
    cache_root = Path(config["cache_root"])
    k, t_past, stride = config["label_k"], config["T_past"], config["stride"]

    feats: list = []
    labels_list: list = []
    alphas: dict = {}
    per_asset: dict = {}
    tr_items, va_items, te_items = [], [], []
    F_ref: int | None = None

    symbol_alphas = config.get("symbol_alphas", {})

    for idx, symbol in enumerate(symbols):
        sub = copy.deepcopy(config)
        sub["symbol"] = symbol
        sub["cache_dir"] = str(cache_root / symbol)
        if symbol in symbol_alphas:
            sub["label_alpha"] = float(symbol_alphas[symbol])

        feat, mid, ts = build_cache(sub, extract_features, n_features, tag="lob")
        F = feat.shape[1]
        if F_ref is None:
            F_ref = F
        elif F != F_ref:
            raise ValueError(
                f"feature dim mismatch: {symbol} has {F}, expected {F_ref} "
                "(all pooled symbols must share n_lob_levels and feature_mode)"
            )

        N = len(mid)
        train_end = int(N * config["train_frac"])
        val_end = int(N * (config["train_frac"] + config["val_frac"]))
        labels, alpha = build_labels(mid, sub, train_end)
        alphas[symbol] = alpha

        tr = _valid_starts(0, train_end, t_past, k, labels, ts, stride)
        va = _valid_starts(train_end, val_end, t_past, k, labels, ts, stride)
        te = _valid_starts(val_end, N, t_past, k, labels, ts, stride)

        feats.append(feat)
        labels_list.append(labels)
        for arr, bucket in ((tr, tr_items), (va, va_items), (te, te_items)):
            if len(arr):
                col = np.full((len(arr), 1), idx, dtype=np.int64)
                bucket.append(np.hstack([col, arr.reshape(-1, 1)]))
        per_asset[symbol] = {"train": len(tr), "val": len(va), "test": len(te)}
        logger.info(
            "  {} [{}] — train:{} val:{} test:{}  alpha={:.6f}",
            symbol,
            idx,
            len(tr),
            len(va),
            len(te),
            alpha,
        )

    def _stack(lst) -> np.ndarray:
        return np.vstack(lst) if lst else np.zeros((0, 2), dtype=np.int64)

    tr_items = _stack(tr_items)
    va_items = _stack(va_items)
    te_items = _stack(te_items)

    def _balance(items: np.ndarray) -> dict:
        if len(items) == 0:
            return {"down": 0.0, "stationary": 0.0, "up": 0.0}
        lbl = np.array([labels_list[s][st + t_past - 1] for s, st in items])
        c = np.bincount(lbl, minlength=3) / max(len(lbl), 1)
        return {"down": float(c[0]), "stationary": float(c[1]), "up": float(c[2])}

    train_ds = MultiLOBDataset(feats, labels_list, tr_items, t_past)
    val_ds = MultiLOBDataset(feats, labels_list, va_items, t_past)
    test_ds = MultiLOBDataset(feats, labels_list, te_items, t_past)

    meta = {
        "n_features": F_ref,
        "n_assets": len(symbols),
        "symbols": symbols,
        "alphas": alphas,
        "class_balance": _balance(tr_items),
        "counts": {
            "train": len(tr_items),
            "val": len(va_items),
            "test": len(te_items),
        },
        "per_asset": per_asset,
    }
    logger.info(
        "pooled windows — train:{} val:{} test:{}  across {} assets",
        len(tr_items),
        len(va_items),
        len(te_items),
        len(symbols),
    )
    return train_ds, val_ds, test_ds, meta
