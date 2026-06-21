"""HDF5 cache builder for Feishu A-share equity data.

Expected data_dir contents (two flat multi-asset parquet files)::

    data_dir/
      lob_data_in_sample.parquet    # 5-min LOB snapshots, all symbols
      daily_data_in_sample.parquet  # daily OHLCV, all symbols

LOB file columns: asset_id, trade_day_id, time (HH:MM:SS),
  bid_price_1..10, ask_price_1..10, bid_volume_1..10, ask_volume_1..10.
  Columns are renamed 1-indexed → 0-indexed before passing to features.py.

Daily file columns: asset_id, trade_day_id, open, high, low, close,
  volume, amount, adj_factor, vwap_0930_0935.
  trade_day_id is renamed → "date" before passing to features/labels.

Label (causal pairing)
----------------------
Window of T days ending at day t is paired with the label at day t+1:
  label_{t+1} = (close_{t+2} - vwap_{t+1}) / vwap_{t+1}
By end-of-day t the trader knows all features through day t but NOT the
morning vwap of day t+1 (the trade entry price), so there is zero leakage.

Split
-----
Per-asset chronological split of windows:
  train : first 70 %
  val   : next  15 %  (cumulative 70–85 %)
  test  : last  15 %  (cumulative 85–100 %)

Public API
----------
- discover_symbols(data_dir, config) → sorted list of symbol names (asset_id)
- build_hdf5(config, data_dir, cache_dir, symbols) → (train_path, val_path, test_path)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stocks.feishu.dataset import WindowWriter
from stocks.feishu.features import (
    CLIP_VAL,
    N_LEVELS,
    N_OFI,
    causal_rolling_zscore,
    compute_ofi_tick,
    compute_ohlcv_features,
    extract_lob_day,
    n_features,
    snap_to_slots,
)
from stocks.feishu.labels import assign_labels, compute_forward_returns

_LOB_FILE = "lob_data_in_sample.parquet"
_DAILY_FILE = "daily_data_in_sample.parquet"
_SYM_COL = "asset_id"
_DAY_COL = "trade_day_id"
_TIME_COL = "time"

_TRAIN_FRAC = 0.70
_VAL_CUM = 0.85  # cumulative; val = [70%, 85%), test = [85%, 100%)


def _lob_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 1-indexed LOB columns (bid/ask_price/volume_1..10) to 0-indexed."""
    rename = {}
    for i in range(1, 11):
        for prefix in ("bid_price", "ask_price", "bid_volume", "ask_volume"):
            src = f"{prefix}_{i}"
            if src in df.columns:
                rename[src] = f"{prefix}_{i - 1}"
    return df.rename(columns=rename)


def discover_symbols(data_dir: str | Path, config: dict | None = None) -> list[str]:
    """Return sorted list of asset_id values present in the LOB flat file."""
    if config is None:
        config = {}
    root = Path(data_dir).resolve()
    lob_file = config.get("lob_file", _LOB_FILE)
    sym_col = config.get("symbol_col", _SYM_COL)
    p = root / lob_file
    if not p.exists():
        raise FileNotFoundError(
            f"LOB file not found: {p}\n"
            f"Contents of {root}: {sorted(e.name for e in root.iterdir())}"
        )
    lob = pd.read_parquet(p, columns=[sym_col])
    symbols = sorted(lob[sym_col].dropna().unique().tolist())
    if not symbols:
        raise ValueError(f"No symbols found in column '{sym_col}' of {p}")
    logger.info("Discovered {} symbols from {}", len(symbols), lob_file)
    return symbols


def _cache_key(config: dict) -> str:
    sig = {
        k: config[k]
        for k in ("feature_mode", "n_lob_levels", "T_past", "alpha", "lob_file")
        if k in config
    }
    return hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:8]


def build_hdf5(
    config: dict,
    data_dir: str | Path,
    cache_dir: str | Path,
    symbols: list[str],
) -> tuple[Path, Path, Path]:
    """Build (or return cached) train/val/test HDF5 files.

    Pipeline
    --------
    1. Load flat LOB and daily parquet files once.
    2. Per-symbol: compute LOB/OFI features, apply causal rolling z-score.
    3. Cross-sectional z-score of OHLCV across all assets per day.
    4. Per-symbol: slide T-past windows with causal label, split 70/15/15,
       clip to ±5, write to HDF5.

    Returns:
        (train_path, val_path, test_path)
    """
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(config)
    paths = {s: out_dir / f"{s}_{key}.h5" for s in ("train", "val", "test")}

    if all(p.exists() for p in paths.values()):
        logger.info("Cache hit: {}", out_dir / f"*_{key}.h5")
        return paths["train"], paths["val"], paths["test"]

    mode = config.get("feature_mode", "ofi")
    T = config["T_past"]
    alpha = config["alpha"]
    n_levels = config.get("n_lob_levels", N_LEVELS)
    nf = n_features(config)
    n_lob_feat = N_OFI if mode == "ofi" else 4 * n_levels
    root = Path(data_dir).resolve()
    sym_col = config.get("symbol_col", _SYM_COL)
    day_col = config.get("day_col", _DAY_COL)
    time_col = config.get("time_col", _TIME_COL)
    lob_file = config.get("lob_file", _LOB_FILE)
    daily_file = config.get("daily_file", _DAILY_FILE)

    logger.info("Building HDF5 cache key={} mode={} T={} nf={}", key, mode, T, nf)

    # ── Load flat files; rename columns to match features/labels expectations ─
    lob_all = pd.read_parquet(root / lob_file)
    lob_all = _lob_rename(lob_all)
    lob_all = lob_all.rename(columns={day_col: "date"})

    daily_all = pd.read_parquet(root / daily_file)
    daily_all = daily_all.rename(columns={day_col: "date"})

    # ── Pass 1: per-asset feature extraction ─────────────────────────────────
    # lob_feats_map[sym][date]  → (n_lob_feat,) float32  (causal-z-scored)
    # ohlcv_raw_map[sym][date]  → (19,) float32           (before cs-zscore)
    # labels_map[sym][date]     → int
    # dates_map[sym]            → sorted list[str]
    lob_feats_map: dict[str, dict[str, np.ndarray]] = {}
    ohlcv_raw_map: dict[str, dict[str, np.ndarray]] = {}
    labels_map: dict[str, dict[str, int]] = {}
    dates_map: dict[str, list[str]] = {}

    for sym in symbols:
        lob_sym = (
            lob_all[lob_all[sym_col] == sym]
            .sort_values(["date", time_col])
            .reset_index(drop=True)
        )
        daily_sym = (
            daily_all[daily_all[sym_col] == sym]
            .sort_values("date")
            .reset_index(drop=True)
        )

        sym_dates = sorted(daily_sym["date"].astype(str).unique())
        dates_map[sym] = sym_dates

        # ── LOB / OFI features per day ─────────────────────────────────────
        lob_rows: list[np.ndarray] = []
        for d in sym_dates:
            day_lob = lob_sym[lob_sym["date"].astype(str) == d].reset_index(drop=True)
            if len(day_lob) == 0:
                lob_rows.append(np.zeros(n_lob_feat, dtype=np.float32))
                continue
            if mode == "ofi":
                ofi_df = compute_ofi_tick(day_lob)
                vec = snap_to_slots(ofi_df, day_lob).reshape(-1)  # (240,)
            else:
                vec = extract_lob_day(day_lob, n_levels)  # (4n,)
            lob_rows.append(vec)

        lob_mat = np.stack(lob_rows, axis=0)  # (n_days, n_lob_feat)
        lob_norm = causal_rolling_zscore(lob_mat)
        lob_feats_map[sym] = {d: lob_norm[i] for i, d in enumerate(sym_dates)}

        # ── OHLCV features per day (store raw; cs-zscore in pass 2) ────────
        try:
            ohlcv_df = compute_ohlcv_features(daily_sym)
            for i, d in enumerate(sym_dates):
                ohlcv_raw_map.setdefault(sym, {})[d] = ohlcv_df.iloc[i].values.astype(
                    np.float32
                )
        except Exception as exc:
            logger.warning("OHLCV failed for {}: {}", sym, exc)

        # ── Labels ─────────────────────────────────────────────────────────
        fwd = compute_forward_returns(daily_sym)
        lbl = assign_labels(fwd, alpha)
        labels_map[sym] = {d: int(lbl[i]) for i, d in enumerate(sym_dates)}

    # ── Pass 2: cross-sectional z-score of OHLCV per day ─────────────────────
    all_dates = sorted({d for sym in symbols for d in dates_map[sym]})
    ohlcv_cs: dict[str, dict[str, np.ndarray]] = {sym: {} for sym in symbols}

    for d in all_dates:
        vecs, syms_on_day = [], []
        for sym in symbols:
            if sym in ohlcv_raw_map and d in ohlcv_raw_map[sym]:
                vecs.append(ohlcv_raw_map[sym][d])
                syms_on_day.append(sym)
        if not vecs:
            continue
        mat = np.stack(vecs, axis=0).astype(np.float64)
        mu = mat.mean(axis=0)
        sigma = mat.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        normed = ((mat - mu) / sigma).astype(np.float32)
        for i, sym in enumerate(syms_on_day):
            ohlcv_cs[sym][d] = normed[i]

    # ── Pass 3: per-asset window slide → per-asset 70/15/15 split → HDF5 ─────
    writers = {
        s: WindowWriter(str(paths[s]), T=T, NF=nf) for s in ("train", "val", "test")
    }
    totals = {s: 0 for s in ("train", "val", "test")}

    for asset_idx, sym in enumerate(symbols):
        sym_dates = dates_map[sym]
        n_days = len(sym_dates)

        # need at least T+1 days (window of T + one label day)
        if n_days < T + 1:
            continue

        # Build all valid windows for this asset (causal pairing)
        # window_dates = sym_dates[i:i+T], label = labels_map[sym][sym_dates[i+T]]
        X_wins: list[np.ndarray] = []
        y_wins: list[int] = []

        for i in range(n_days - T):
            label_date = sym_dates[i + T]
            lbl_val = labels_map[sym].get(label_date, -1)
            if lbl_val == -1:
                continue

            window_dates = sym_dates[i : i + T]
            rows = []
            ok = True
            for d in window_dates:
                lob_v = lob_feats_map[sym].get(d)
                ohlcv_v = ohlcv_cs[sym].get(d)
                if lob_v is None or ohlcv_v is None:
                    ok = False
                    break
                feat = np.concatenate([lob_v, ohlcv_v]).astype(np.float32)
                rows.append(feat)
            if not ok:
                continue

            x = np.stack(rows, axis=0)  # (T, nf)
            x = np.clip(x, -CLIP_VAL, CLIP_VAL)
            X_wins.append(x)
            y_wins.append(lbl_val)

        if not X_wins:
            continue

        X = np.stack(X_wins, axis=0)[:, np.newaxis, :, :]  # (N, 1, T, NF)
        Y = np.array(y_wins, dtype=np.int64)
        n_total = len(Y)
        n_train = int(_TRAIN_FRAC * n_total)
        n_val = int(_VAL_CUM * n_total)

        slices = {
            "train": slice(0, n_train),
            "val": slice(n_train, n_val),
            "test": slice(n_val, None),
        }
        for split, sl in slices.items():
            writers[split].write(X[sl], Y[sl], asset_idx)
            totals[split] += len(Y[sl])

    for w in writers.values():
        w.close()

    logger.info(
        "Built cache key={} | train={} val={} test={}",
        key,
        totals["train"],
        totals["val"],
        totals["test"],
    )
    return paths["train"], paths["val"], paths["test"]
