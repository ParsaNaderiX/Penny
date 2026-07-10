# Binance data

USDT-quoted perpetual/spot LOB data for nine symbols, resampled from
[Tardis](https://tardis.dev)-style CSV dumps into the unified parquet schema.

## Symbols

`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT`, `ADAUSDT`, `AVAXUSDT`, `BNBUSDT`,
`DOGEUSDT`, `USDCUSDT`.

## Raw layout

Raw CSVs live under `data/binance/` (DVC-tracked), one file per **day × symbol ×
stream**:

```
binance_book_snapshot_25_{YYYY-MM-DD}_{SYMBOL}.csv.gz    # LOB, up to 25 levels
binance_trades_{YYYY-MM-DD}_{SYMBOL}.csv.gz              # trade ticks
binance_quotes_{YYYY-MM-DD}_{SYMBOL}.csv.gz              # best bid/ask ticks
```

- **book snapshot** — `timestamp` (µs) plus `bids[i].price`, `bids[i].amount`,
  `asks[i].price`, `asks[i].amount` for `i = 0 … 24`.
- **trades** — `timestamp`, `side` (`buy`/`sell`), `price`, `amount`.
- **quotes** — `timestamp`, `bid_price`, `ask_price`, `bid_amount`, `ask_amount`.

## Resampling — `scripts/resample_binance.py`

Bins every stream to a fixed interval and joins them into one row per bin:

- **book** — last snapshot in each bin (last-tick resample); `mid` and `spread`
  derived from level 0.
- **trades** — per-bin `trade_count`, `buy_vol`, `sell_vol`, volume-weighted `vwap`,
  and `trade_imbalance = (buy − sell)/(buy + sell)`.
- **quotes** — last best bid/ask in the bin.

Missing trade/quote bins are left as NaN and handled downstream (treated as zero
activity by the feature extractor).

```bash
# defaults: --interval 0.25 (seconds), --levels 10
uv run python scripts/resample_binance.py --interval 10 --levels 10
uv run python scripts/resample_binance.py --date 2026-06-09          # one day
uv run python scripts/resample_binance.py --data-dir data/binance.bak # alt source
```

Output: `data/resampled/binance/{SYMBOL}.parquet.gz` — all dates for one symbol,
concatenated and sorted by `bin`.

## Config

Binance configs set:

```json
"exchange": "binance",
"symbol": "BTCUSDT",
"data_dir": "data/resampled/binance",
"n_lob_levels": 10
```

With `n_lob_levels = 10`: `ofi` mode → **21** features, `lob` mode → **51**
features (see [features.md](features.md)).
