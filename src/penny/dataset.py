"""Dataset construction for Penny.

Builds the feature matrix once, splits it temporally by calendar day, fits the
rolling normalizer on the training split, precomputes per-sample regime vectors
and direction labels, and loads every tensor onto the configured device at
construction time so training never touches the CPU (except for numpy metric
conversion).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import features as feat_mod
from .diffusion import Diffusion

logger = logging.getLogger(__name__)


def _split_bounds(dates: np.ndarray, train_frac: float, val_frac: float) -> tuple[int, int]:
    """Return row indices (train_end, val_end) aligned to calendar-day edges."""
    unique_days = np.array(sorted(pd.unique(dates)))
    n_days = len(unique_days)
    n_train_days = max(int(round(train_frac * n_days)), 1)
    n_val_days = max(int(round(val_frac * n_days)), 1)
    train_last = unique_days[min(n_train_days, n_days) - 1]
    val_last = unique_days[min(n_train_days + n_val_days, n_days) - 1]
    train_end = int(np.searchsorted(dates, train_last, side="right"))
    val_end = int(np.searchsorted(dates, val_last, side="right"))
    return train_end, val_end


class PennyDataset(Dataset):
    """Sliding-window dataset of (past, future) trajectory pairs.

    Each sample spans ``2T`` consecutive snapshots: the first ``T`` form the
    context (``past_seq``) and the next ``T`` form the target trajectory.  The
    regime vector and direction label are derived from the first ``k`` steps of
    the future window.  Diffusion noising is applied on the fly in
    :meth:`__getitem__`.
    """

    def __init__(
        self,
        feats: torch.Tensor,
        regimes: torch.Tensor,
        labels: torch.Tensor,
        starts: list[int],
        config: dict,
        diffusion: Diffusion,
    ) -> None:
        self.feats = feats  # (N, F) on device
        self.regimes = regimes  # (n_samples, regime_dim) on device
        self.labels = labels  # (n_samples,) on device
        self.starts = starts
        self.T = config["T"]
        self.T_max = config["T_max"]
        self.diffusion = diffusion
        self.device = feats.device

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, j: int):
        """Return ``(past, future_noisy, noise, t, regime, label, future_clean)``."""
        i = self.starts[j]
        T = self.T
        past = self.feats[i : i + T]
        future_clean = self.feats[i + T : i + 2 * T]

        t = torch.randint(0, self.T_max, (1,), device=self.device).squeeze(0)
        noise = torch.randn_like(future_clean)
        ab = self.diffusion.alpha_bars[t]
        future_noisy = torch.sqrt(ab) * future_clean + torch.sqrt(1.0 - ab) * noise

        return (past, future_noisy, noise, t, self.regimes[j], self.labels[j], future_clean)


def _regimes_and_labels(
    starts: list[int],
    mid: np.ndarray,
    depth: np.ndarray,
    ofi: np.ndarray,
    T: int,
    k: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw regime vectors and direction labels for each sample start."""
    n = len(starts)
    regimes = np.zeros((n, 4), dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    for idx, i in enumerate(starts):
        f0 = i + T
        seg = mid[f0 : f0 + k]
        m_t = mid[i + T]
        m_bar = np.mean(seg)
        trend = (m_bar - m_t) / (m_t + 1e-12)
        rets = np.diff(seg) / (seg[:-1] + 1e-12)
        vol = float(np.std(rets)) if len(rets) else 0.0
        liquidity = float(np.mean(depth[f0 : f0 + k]))
        flow = float(np.mean(ofi[f0 : f0 + k]))
        regimes[idx] = (trend, vol, liquidity, flow)
        labels[idx] = 0 if trend > alpha else (2 if trend < -alpha else 1)
    return regimes, labels


def _make_starts(lo: int, hi: int, T: int, S: int, drop_first: int) -> list[int]:
    """Valid window starts within ``[lo, hi)`` requiring ``2T`` room, strided ``S``."""
    starts = list(range(lo, hi - 2 * T + 1, S))
    return starts[drop_first:] if drop_first else starts


def build_datasets(config: dict, diffusion: Diffusion, device: torch.device):
    """Build train/val/test datasets sharing one fitted normalizer.

    Returns ``(train_ds, val_ds, test_ds, feature_columns, train_trend_sigma)``.
    """
    feats_df, raw_df, columns = feat_mod.build_features(config)
    feats_df = feats_df.set_index("snapshot_time")
    dates = feats_df.index.normalize().to_numpy()

    train_end, val_end = _split_bounds(dates, config["train_frac"], config["val_frac"])
    n = len(feats_df)
    logger.info(
        "temporal split: train=[0,%d) val=[%d,%d) test=[%d,%d)",
        train_end,
        train_end,
        val_end,
        val_end,
        n,
    )

    normalizer = feat_mod.RollingZScoreNormalizer(config["rolling_window_days"])
    train_df = feats_df.iloc[:train_end]
    normalizer.fit(train_df, columns)

    norm = np.zeros((n, len(columns)), dtype=np.float32)
    norm[:train_end] = normalizer.transform_train(train_df, columns).to_numpy()
    norm[train_end:] = normalizer.transform_frozen(
        feats_df.iloc[train_end:], columns
    ).to_numpy()
    feats_tensor = torch.from_numpy(norm).to(device)

    mid = raw_df["mid_raw"].to_numpy(dtype=np.float64)
    depth = raw_df["depth_raw"].to_numpy(dtype=np.float64)
    ofi = raw_df["ofi_raw"].to_numpy(dtype=np.float64)

    T, S, k = config["T"], config["S"], config["k"]
    if k > T:
        raise ValueError(f"k={k} must be <= T={T}")

    train_starts = _make_starts(0, train_end, T, S, 0)
    val_starts = _make_starts(train_end, val_end, T, S, T)
    test_starts = _make_starts(val_end, n, T, S, T)
    logger.info(
        "samples: train=%d val=%d test=%d",
        len(train_starts),
        len(val_starts),
        len(test_starts),
    )

    tr_reg, tr_lab = _regimes_and_labels(train_starts, mid, depth, ofi, T, k, config["alpha"])
    va_reg, va_lab = _regimes_and_labels(val_starts, mid, depth, ofi, T, k, config["alpha"])
    te_reg, te_lab = _regimes_and_labels(test_starts, mid, depth, ofi, T, k, config["alpha"])

    reg_mean = tr_reg.mean(axis=0)
    reg_std = tr_reg.std(axis=0)
    reg_std[reg_std == 0] = 1.0
    train_trend_sigma = 1.0

    def _norm_reg(r: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((r - reg_mean) / reg_std).astype(np.float32)).to(device)

    for name, lab in (("train", tr_lab), ("val", va_lab), ("test", te_lab)):
        if len(lab):
            counts = np.bincount(lab, minlength=config["num_classes"])
            logger.info(
                "%s label distribution (up/flat/down): %s", name, counts.tolist()
            )

    def _ds(starts, reg, lab):
        return PennyDataset(
            feats_tensor,
            _norm_reg(reg),
            torch.from_numpy(lab).to(device),
            starts,
            config,
            diffusion,
        )

    return (
        _ds(train_starts, tr_reg, tr_lab),
        _ds(val_starts, va_reg, va_lab),
        _ds(test_starts, te_reg, te_lab),
        columns,
        train_trend_sigma,
    )
