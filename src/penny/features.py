"""Feature engineering for the Penny limit-order-book diffusion model.

This module turns raw order-book snapshots and trade ticks (loaded *exclusively*
from the paths in ``config.json``) into a per-snapshot feature matrix of width
``F`` together with the raw helper series needed to derive regime vectors and
direction labels.

Feature layout (F = 56 for the default ``n_levels = 10``)
--------------------------------------------------------
The spec enumerates 52 LOB + trade features; to reach the configured ``F = 56``
we add four additional, well-defined microstructure features (``microprice_tick``,
``weighted_mid_tick``, ``total_depth``, ``mid_return``).  The exact ordering is::

    per level i in 1..n_levels:  bid_price_tick_i, ask_price_tick_i,
                                 bid_vol_i, ask_vol_i                 (4 * n_levels)
    LOB summary:   mid_price, spread, bid_depth, ask_depth,
                   depth_imbalance, level_OFI                         (6)
    extra micro:   microprice_tick, weighted_mid_tick,
                   total_depth, mid_return                            (4)
    trade flow:    OFI, trade_count, vwap_tick_distance,
                   price_momentum, buy_ratio, realized_spread         (6)

All price-distance features are expressed in ticks relative to the mid price.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _level_columns(n_levels: int) -> list[str]:
    """Return the ordered per-level feature column names."""
    cols: list[str] = []
    for i in range(1, n_levels + 1):
        cols += [
            f"bid_price_tick_{i}",
            f"ask_price_tick_{i}",
            f"bid_vol_{i}",
            f"ask_vol_{i}",
        ]
    return cols


def feature_columns(n_levels: int) -> list[str]:
    """Return the full ordered list of feature column names for ``n_levels``."""
    return (
        _level_columns(n_levels)
        + [
            "mid_price",
            "spread",
            "bid_depth",
            "ask_depth",
            "depth_imbalance",
            "level_OFI",
        ]
        + ["microprice_tick", "weighted_mid_tick", "total_depth", "mid_return"]
        + [
            "OFI",
            "trade_count",
            "vwap_tick_distance",
            "price_momentum",
            "buy_ratio",
            "realized_spread",
        ]
    )


def volume_columns(n_levels: int) -> list[str]:
    """Columns that receive a ``log1p`` transform before normalization."""
    cols: list[str] = []
    for i in range(1, n_levels + 1):
        cols += [f"bid_vol_{i}", f"ask_vol_{i}"]
    cols += ["bid_depth", "ask_depth", "total_depth", "trade_count"]
    return cols


def load_orderbook(path: str, n_levels: int) -> pd.DataFrame:
    """Load an order-book CSV, normalising column names to the canonical schema.

    Handles the real-data column names found in this repo (``time`` for the
    snapshot timestamp and ``bid_volume_i`` / ``ask_volume_i`` for sizes) and
    keeps only the first ``n_levels`` levels.
    """
    df = pd.read_csv(path)
    if "time" in df.columns and "snapshot_time" not in df.columns:
        df = df.rename(columns={"time": "snapshot_time"})
    if "snapshot_time" not in df.columns:
        raise ValueError(f"order-book file {path} has no time column")
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])

    rename: dict[str, str] = {}
    for i in range(1, n_levels + 1):
        for side in ("bid", "ask"):
            if f"{side}_volume_{i}" in df.columns:
                rename[f"{side}_volume_{i}"] = f"{side}_vol_raw_{i}"
            elif f"{side}_vol_{i}" in df.columns:
                rename[f"{side}_vol_{i}"] = f"{side}_vol_raw_{i}"
    df = df.rename(columns=rename)

    keep = ["snapshot_time"]
    for i in range(1, n_levels + 1):
        keep += [
            f"bid_price_{i}",
            f"bid_vol_raw_{i}",
            f"ask_price_{i}",
            f"ask_vol_raw_{i}",
        ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"order-book file {path} missing columns: {missing}")

    df = df[keep].sort_values("snapshot_time").reset_index(drop=True)
    return df


def load_trades(path: str) -> pd.DataFrame:
    """Load a trades CSV with ``trade_time``, ``price``, ``volume``, ``direction``."""
    df = pd.read_csv(path)
    if "trade_time" not in df.columns:
        raise ValueError(f"trades file {path} has no trade_time column")
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df.sort_values("trade_time").reset_index(drop=True)
    df["sign"] = np.where(df["direction"].astype(str).str.lower() == "buy", 1.0, -1.0)
    return df


def compute_lob_features(
    ob: pd.DataFrame, n_levels: int, tick_size: float
) -> pd.DataFrame:
    """Compute LOB-derived features (prices in ticks, raw volumes preserved)."""
    out = pd.DataFrame(index=ob.index)
    bid_p1 = ob["bid_price_1"].to_numpy(dtype=np.float64)
    ask_p1 = ob["ask_price_1"].to_numpy(dtype=np.float64)
    mid = (bid_p1 + ask_p1) / 2.0

    bid_depth = np.zeros(len(ob))
    ask_depth = np.zeros(len(ob))
    wmid_num = np.zeros(len(ob))
    for i in range(1, n_levels + 1):
        bp = ob[f"bid_price_{i}"].to_numpy(dtype=np.float64)
        ap = ob[f"ask_price_{i}"].to_numpy(dtype=np.float64)
        bv = ob[f"bid_vol_raw_{i}"].to_numpy(dtype=np.float64)
        av = ob[f"ask_vol_raw_{i}"].to_numpy(dtype=np.float64)
        out[f"bid_price_tick_{i}"] = (bp - mid) / tick_size
        out[f"ask_price_tick_{i}"] = (ap - mid) / tick_size
        out[f"bid_vol_{i}"] = bv
        out[f"ask_vol_{i}"] = av
        bid_depth += bv
        ask_depth += av
        wmid_num += bp * bv + ap * av

    bv1 = ob["bid_vol_raw_1"].to_numpy(dtype=np.float64)
    av1 = ob["ask_vol_raw_1"].to_numpy(dtype=np.float64)
    total_depth = bid_depth + ask_depth
    microprice = (bid_p1 * av1 + ask_p1 * bv1) / (bv1 + av1 + _EPS)
    weighted_mid = wmid_num / (total_depth + _EPS)
    depth_imb = (bid_depth - ask_depth) / (total_depth + _EPS)
    level_ofi = np.diff(bid_depth - ask_depth, prepend=(bid_depth[0] - ask_depth[0]))

    out["mid_price"] = mid
    out["spread"] = ask_p1 - bid_p1
    out["bid_depth"] = bid_depth
    out["ask_depth"] = ask_depth
    out["depth_imbalance"] = depth_imb
    out["level_OFI"] = level_ofi
    out["microprice_tick"] = (microprice - mid) / tick_size
    out["weighted_mid_tick"] = (weighted_mid - mid) / tick_size
    out["total_depth"] = total_depth
    out["mid_return"] = pd.Series(mid).pct_change().fillna(0.0).to_numpy()
    return out


def compute_trade_features(
    ob: pd.DataFrame,
    trades: pd.DataFrame,
    mid: np.ndarray,
    window_seconds: int,
    tick_size: float,
) -> pd.DataFrame:
    """Aggregate trades from the trailing ``window_seconds`` for every snapshot.

    Uses prefix sums + ``searchsorted`` so the cost is ``O(N log M)``.  Snapshots
    with no trades in their window are filled with NaN, a WARNING is logged with
    the count, and the values are forward-filled by the caller.
    """
    snap_ns = ob["snapshot_time"].to_numpy("datetime64[ns]").astype(np.int64)
    if len(trades) == 0:
        logger.warning("no trades available; all trade features will be zero")
        out = pd.DataFrame(
            0.0,
            index=ob.index,
            columns=[
                "OFI",
                "trade_count",
                "vwap_tick_distance",
                "price_momentum",
                "buy_ratio",
                "realized_spread",
            ],
        )
        return out

    tr_ns = trades["trade_time"].to_numpy("datetime64[ns]").astype(np.int64)
    price = trades["price"].to_numpy(dtype=np.float64)
    vol = trades["volume"].to_numpy(dtype=np.float64)
    sign = trades["sign"].to_numpy(dtype=np.float64)
    window_ns = np.int64(window_seconds) * np.int64(1_000_000_000)

    def prefix(arr: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(arr)])

    p_vol = prefix(vol)
    p_signed_vol = prefix(sign * vol)
    p_pv = prefix(price * vol)
    p_buy_vol = prefix(np.where(sign > 0, vol, 0.0))
    p_sign_price = prefix(sign * price)
    p_sign = prefix(sign)

    left = np.searchsorted(tr_ns, snap_ns - window_ns, side="left")
    right = np.searchsorted(tr_ns, snap_ns, side="right")
    count = (right - left).astype(np.float64)

    vol_sum = p_vol[right] - p_vol[left]
    signed_sum = p_signed_vol[right] - p_signed_vol[left]
    pv_sum = p_pv[right] - p_pv[left]
    buy_sum = p_buy_vol[right] - p_buy_vol[left]
    sign_price_sum = p_sign_price[right] - p_sign_price[left]
    sign_sum = p_sign[right] - p_sign[left]

    has = count > 0
    vwap = np.where(has, pv_sum / (vol_sum + _EPS), np.nan)
    buy_ratio = np.where(has, buy_sum / (vol_sum + _EPS), np.nan)
    ofi = np.where(has, signed_sum, np.nan)
    trade_count = np.where(has, count, np.nan)

    first_price = np.where(has, price[np.clip(left, 0, len(price) - 1)], np.nan)
    last_price = np.where(has, price[np.clip(right - 1, 0, len(price) - 1)], np.nan)
    momentum = np.where(has, (last_price - first_price) / (first_price + _EPS), np.nan)

    realized = np.where(
        has,
        (sign_price_sum - mid * sign_sum) / (count + _EPS) / tick_size,
        np.nan,
    )
    vwap_tick = np.where(has, (vwap - mid) / tick_size, np.nan)

    n_empty = int((~has).sum())
    if n_empty:
        logger.warning(
            "empty trade window for %d/%d snapshots; forward-filling",
            n_empty,
            len(ob),
        )

    out = pd.DataFrame(
        {
            "OFI": ofi,
            "trade_count": trade_count,
            "vwap_tick_distance": vwap_tick,
            "price_momentum": momentum,
            "buy_ratio": buy_ratio,
            "realized_spread": realized,
        },
        index=ob.index,
    )
    out = out.ffill().fillna(0.0)
    return out


def build_features(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build the full feature matrix from the configured data paths.

    Returns
    -------
    feats : DataFrame
        ``snapshot_time`` plus the ``F`` feature columns (volumes already
        ``log1p``-transformed; not yet z-scored).
    raw : DataFrame
        ``snapshot_time`` plus raw helper series (``mid_raw``, ``depth_raw``,
        ``ofi_raw``) used for label / regime construction.
    columns : list[str]
        Ordered feature column names.
    """
    n_levels = config["n_levels"]
    tick = config["tick_size"]

    ob = load_orderbook(config["lob_path"], n_levels)
    trades = load_trades(config["trades_path"])
    logger.info(
        "loaded %d order-book snapshots and %d trades", len(ob), len(trades)
    )

    lob = compute_lob_features(ob, n_levels, tick)
    mid = lob["mid_price"].to_numpy(dtype=np.float64)
    tr = compute_trade_features(ob, trades, mid, config["trade_window_seconds"], tick)

    feats = pd.concat([lob, tr], axis=1)
    cols = feature_columns(n_levels)
    feats = feats[cols]

    for c in volume_columns(n_levels):
        feats[c] = np.log1p(feats[c].clip(lower=0))

    expected_f = config["F"]
    if len(cols) != expected_f:
        raise ValueError(
            f"feature width {len(cols)} != config F={expected_f} "
            f"(n_levels={n_levels})"
        )

    feats.insert(0, "snapshot_time", ob["snapshot_time"].to_numpy())

    raw = pd.DataFrame(
        {
            "snapshot_time": ob["snapshot_time"].to_numpy(),
            "mid_raw": mid,
            "depth_raw": lob["total_depth"].to_numpy(dtype=np.float64),
            "ofi_raw": tr["OFI"].to_numpy(dtype=np.float64),
        }
    )

    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return feats, raw, cols


class RollingZScoreNormalizer:
    """Causal rolling z-score normalizer fit on the training split only.

    Training rows are normalized by their own causal rolling mean/std over a
    ``window_days`` time window.  The final training-window statistics are frozen
    and applied as constants to the validation and test splits, preventing any
    look-ahead leakage.
    """

    def __init__(self, window_days: float, eps: float = 1e-8) -> None:
        self.window = f"{window_days}D"
        self.eps = eps
        self.frozen_mean: pd.Series | None = None
        self.frozen_std: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame, columns: list[str]) -> None:
        """Fit frozen statistics from the (time-indexed) training split."""
        roll = train_df[columns].rolling(self.window, min_periods=1)
        mean = roll.mean().iloc[-1]
        std = roll.std().iloc[-1]
        global_std = train_df[columns].std().replace(0, np.nan).fillna(1.0)
        std = std.replace(0, np.nan).fillna(global_std).fillna(1.0)
        self.frozen_mean = mean.fillna(train_df[columns].mean()).fillna(0.0)
        self.frozen_std = std
        logger.info(
            "rolling-normalization frozen stats fit on %d train rows", len(train_df)
        )
        logger.info(
            "frozen mean (head): %s", self.frozen_mean.head().round(4).to_dict()
        )
        logger.info(
            "frozen std  (head): %s", self.frozen_std.head().round(4).to_dict()
        )

    def transform_train(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Normalize training rows with their own causal rolling statistics."""
        roll = df[columns].rolling(self.window, min_periods=1)
        mean = roll.mean()
        std = roll.std().replace(0, np.nan)
        std = std.fillna(self.frozen_std)
        z = (df[columns] - mean) / (std + self.eps)
        return z.fillna(0.0)

    def transform_frozen(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Normalize validation / test rows with frozen training statistics."""
        if self.frozen_mean is None:
            raise RuntimeError("normalizer used before fit()")
        z = (df[columns] - self.frozen_mean) / (self.frozen_std + self.eps)
        return z.fillna(0.0)
