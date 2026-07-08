"""Trend labels from the smoothed mid-price (Binance / DeepLOB formulation).

    bwd         = mean(mid[t - k : t])
    fwd         = mean(mid[t : t + k])
    trend_ratio = (fwd - bwd) / bwd

Label assignment:
    0 (down)       if trend_ratio < -alpha
    1 (stationary) if |trend_ratio| <= alpha
    2 (up)         if trend_ratio > alpha

``alpha`` is calibrated on the **training set** to the 33.3rd percentile of
|trend_ratio|, producing roughly balanced class frequencies.

Shared by ``crypto.deeplob``, ``crypto.jointdiff``, and ``crypto.lobtransformer``.
"""

from __future__ import annotations

import numpy as np


DOWN, STATIONARY, UP = 0, 1, 2


def compute_trend_series(mid: np.ndarray, k: int) -> np.ndarray:
    """Return per-snapshot trend ratio; NaN for the first and last ``k`` positions."""
    N = len(mid)
    trend = np.full(N, np.nan, dtype=np.float64)
    cs = np.cumsum(np.concatenate([[0.0], mid]))
    for t in range(k, N - k):
        bwd = (cs[t] - cs[t - k]) / k
        fwd = (cs[t + k] - cs[t]) / k
        if bwd > 1e-12:
            trend[t] = (fwd - bwd) / bwd
    return trend


def calibrate_alpha(trend_train: np.ndarray) -> float:
    """33rd percentile of |trend_ratio| on training data — yields balanced classes."""
    valid = trend_train[np.isfinite(trend_train)]
    return float(np.percentile(np.abs(valid), 100.0 / 3.0))


def assign_labels(trend: np.ndarray, alpha: float) -> np.ndarray:
    """Map trend ratios → {0, 1, 2}; invalid positions get -1."""
    labels = np.full(len(trend), -1, dtype=np.int64)
    valid = np.isfinite(trend)
    labels[valid & (trend < -alpha)] = DOWN
    labels[valid & (np.abs(trend) <= alpha)] = STATIONARY
    labels[valid & (trend > alpha)] = UP
    return labels


def causal_realized_vol(mid: np.ndarray, window: int) -> np.ndarray:
    """Trailing realized vol of relative returns, aligned per snapshot (causal).

    ``vol[t]`` is the std of the ``window`` most recent returns (using only data up
    to ``t``), NaN before enough history exists.  On the same relative scale as the
    trend ratio, so the two are directly comparable.
    """
    N = len(mid)
    vol = np.full(N, np.nan, dtype=np.float64)
    r = np.zeros(N, dtype=np.float64)
    r[1:] = np.where(mid[:-1] > 1e-12, np.diff(mid) / mid[:-1], 0.0)  # return into t
    cs = np.cumsum(np.concatenate([[0.0], r]))
    csq = np.cumsum(np.concatenate([[0.0], r * r]))
    for t in range(window, N):
        s = cs[t] - cs[t - window]
        sq = csq[t] - csq[t - window]
        mean = s / window
        var = max(sq / window - mean * mean, 0.0)
        vol[t] = np.sqrt(var)
    return vol


def assign_labels_vol_adaptive(
    trend: np.ndarray, vol: np.ndarray, mult: float
) -> np.ndarray:
    """Label by a per-snapshot threshold ``mult * causal_realized_vol`` (not fixed α).

    A move counts as up/down only if it clears ``mult`` local-vol units, so the bar
    scales with volatility.  Invalid where trend or vol is NaN or vol is ~0.
    """
    labels = np.full(len(trend), -1, dtype=np.int64)
    thr = mult * vol
    valid = np.isfinite(trend) & np.isfinite(thr) & (thr > 1e-12)
    labels[valid & (trend < -thr)] = DOWN
    labels[valid & (np.abs(trend) <= thr)] = STATIONARY
    labels[valid & (trend > thr)] = UP
    return labels


def build_labels(
    mid: np.ndarray, config: dict, train_end: int
) -> tuple[np.ndarray, float]:
    """Return ``(labels, alpha)``.  ``labels[t] == -1`` for invalid positions.

    ``label_mode`` (default ``"alpha"``) selects the thresholding scheme:
      * ``"alpha"``       — fixed / 33rd-pct-calibrated global threshold (original).
      * ``"vol_adaptive"``— per-snapshot ``label_vol_mult`` × causal realized vol
        (``label_vol_window`` trailing returns); the returned "alpha" is the
        multiplier (there is no single global threshold).
    """
    k = config["label_k"]
    trend = compute_trend_series(mid, k)
    if config.get("label_mode", "alpha") == "vol_adaptive":
        mult = float(config.get("label_vol_mult", 1.0))
        window = int(config.get("label_vol_window", config["label_k"]))
        vol = causal_realized_vol(mid, window)
        return assign_labels_vol_adaptive(trend, vol, mult), mult
    alpha = (
        float(config["label_alpha"])
        if config.get("label_alpha", -1) > 0
        else calibrate_alpha(trend[:train_end])
    )
    return assign_labels(trend, alpha), alpha
