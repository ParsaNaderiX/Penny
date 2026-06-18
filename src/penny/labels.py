"""DeepLOB trend labels for Penny (spec section 3).

Pure, stateless functions for: computing the smoothed-mid trend ratio ``l`` from
a mid-price series, turning ``l`` into a 3-class label, and calibrating the
threshold ``alpha`` on the training set for balanced classes.

Class encoding (spec 3.1):  ``0 = down``, ``1 = stationary``, ``2 = up``.
"""

from __future__ import annotations

import numpy as np

DOWN, STATIONARY, UP = 0, 1, 2


def smoothed_backward_mid(mid: np.ndarray, t_past: int, k: int) -> float:
    """Mean of the ``k`` mids ending at the boundary — cols ``[t_past-k, t_past)``."""
    return float(np.mean(mid[t_past - k : t_past]))


def smoothed_forward_mid(mid: np.ndarray, t_past: int, k: int) -> float:
    """Mean of the first ``k`` future mids — cols ``[t_past, t_past+k)`` (spec 3.1)."""
    return float(np.mean(mid[t_past : t_past + k]))


def trend_ratio(forward_mid: float, backward_mid: float) -> float:
    """``l = (fwd - bwd) / bwd`` (spec 3.1)."""
    return (forward_mid - backward_mid) / (backward_mid + 1e-12)


def compute_l(mid: np.ndarray, t_past: int, k: int) -> float:
    """Trend ratio ``l`` from a full ground-truth mid window."""
    bwd = smoothed_backward_mid(mid, t_past, k)
    fwd = smoothed_forward_mid(mid, t_past, k)
    return trend_ratio(fwd, bwd)


def label_from_l(trend_value: float, alpha: float) -> int:
    """Map a trend ratio to a class using threshold ``alpha`` (spec 3.1)."""
    if trend_value > alpha:
        return UP
    if trend_value < -alpha:
        return DOWN
    return STATIONARY


def calibrate_alpha(l_values: np.ndarray, stationary_frac: float = 1.0 / 3.0) -> float:
    """Set ``alpha`` so that ~``stationary_frac`` of windows are stationary (spec 3.2).

    ``alpha`` is the ``stationary_frac`` quantile of ``|l|``: a fraction
    ``stationary_frac`` of windows have ``|l| < alpha`` (stationary) and the rest
    split between up / down — giving ~1/3 per class at the default.

    Note: spec 3.2's literal wording ("66.7th percentile of |l|") would instead
    make 2/3 of windows stationary, contradicting its own "one third each class"
    goal and the §12 example (33%/33%/33%); we follow the stated goal.
    """
    return float(np.quantile(np.abs(l_values), stationary_frac))


def ground_truth_label(
    mid: np.ndarray, t_past: int, k: int, alpha: float
) -> tuple[int, float]:
    """Return ``(label, l)`` for a ground-truth mid window."""
    trend_value = compute_l(mid, t_past, k)
    return label_from_l(trend_value, alpha), trend_value
