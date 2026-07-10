"""Render attribution maps as ``(feature × time)`` heatmaps with axis labels
that match the ``crypto/features.py`` column layout, so a bright cell reads as
"level 3 OFI, 12 steps back" rather than "row 3, col 12".
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from xai.attribution import Attribution


def feature_labels(n_lob_levels: int, feature_mode: str) -> list[str]:
    """Column labels matching ``crypto.features.extract_features``'s layout.

    Mirrors the layout documented in ``crypto/features.py``: per-level block(s)
    first, then the fixed 11-wide microstructure/trade/quote tail.
    """
    n = n_lob_levels
    labels: list[str] = []
    if feature_mode == "ofi":
        labels += [f"ofi_L{i}" for i in range(n)]
    else:  # lob
        labels += [f"bid_off_L{i}" for i in range(n)]
        labels += [f"ask_off_L{i}" for i in range(n)]
        labels += [f"bid_vol_L{i}" for i in range(n)]
        labels += [f"ask_vol_L{i}" for i in range(n)]
    labels += [
        "spread_norm",
        "log_depth_imbal",
        "log_ret",
        "log_buy_vol",
        "log_sell_vol",
        "trade_imbal",
        "log_trade_cnt",
        "vwap_dev",
        "log_trade_cnt_2",
        "spread_norm_2",
        "abs_log_ret",
    ]
    return labels


def plot_attribution(
    attr: Attribution,
    n_lob_levels: int,
    feature_mode: str,
    ax: plt.Axes | None = None,
    max_features: int = 40,
) -> plt.Axes:
    """Plot one ``Attribution`` as a ``(feature × time)`` heatmap.

    Features are shown on the y-axis (most LOB windows have far more features
    than are readable at once — ``max_features`` caps rows shown, keeping the
    per-level blocks nearest the top-of-book since those dominate signal in
    the source paper). Time (steps back from the prediction point) is the
    x-axis, matching how a human reads "how far in the past mattered".
    """
    scores = attr.scores.detach().cpu().numpy() if hasattr(attr.scores, "detach") else np.asarray(attr.scores)
    T, F = scores.shape
    labels = feature_labels(n_lob_levels, feature_mode)
    if len(labels) != F:
        labels = [f"f{i}" for i in range(F)]

    keep = min(max_features, F)
    scores = scores[:, :keep]
    labels = labels[:keep]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, keep * 0.18)))

    vmax = np.abs(scores).max() or 1.0
    im = ax.imshow(
        scores.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
    )
    ax.set_yticks(range(keep))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(f"time step (0..{T - 1}, most recent = {T - 1})")
    ax.set_title(f"{attr.model_name} / {attr.method} — target={attr.target_class}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax
