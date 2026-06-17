# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Penny extends DiffLOB, using an LSTM-integrated diffusion model to generate realistic, directionally consistent crypto limit order books. Trained on BTCIRT/USDTIRT data from Iranian crypto exchanges, it supports synthetic data augmentation and uncertainty-aware signal generation for market microstructure analysis.

## Environment

- Python 3.10, managed via `uv`
- DVC with S3-compatible remote (Cloudflare R2, bucket: `penny`)

## Commands

```bash
# Install dependencies
uv sync

# Train + evaluate the Penny model (reads config.json, or pass a config path)
uv run python scripts/train_penny.py [configs/config.json]

# DVC: pull / push data from the remote
uv run dvc pull
uv run dvc push
```

## Penny Model

`Penny` is a regime-conditioned DDPM diffusion model (temporal 1-D UNet with FiLM
+ cross-attention conditioning) that generates future LOB trajectories, plus an
auxiliary LSTM trend-classification head. Everything is driven by `config.json` —
no hardcoded hyperparameters. Modules:

- `features.py` — builds the per-snapshot feature matrix (F=56 for n_levels=10:
  40 LOB-ladder + 6 LOB-summary + 4 extra microstructure + 6 trade-flow features)
  and the rolling-zscore normalizer (fit on train, frozen for val/test).
- `dataset.py` — temporal calendar-day split, sliding `2T`-snapshot windows
  (past `T` + future `T`), per-sample regime vectors / direction labels. All
  tensors are moved to the configured device at construction.
- `diffusion.py` — linear-beta DDPM schedule, `forward_diffusion`, DDIM `sample`.
- `model.py` — the `Penny` network and conditioning encoders.
- `evaluate.py` — per-epoch validation + full test suite (KS, Wasserstein, ACF,
  counterfactual validity).
- `train.py` — entry point: logging, AdamW + warmup-cosine, early stopping.

**Note:** the spec described the data as `data/nobitext_data/`, `bid_vol_*`, 10
levels; the real files are `data/nobitex_data/`, `bid_volume_*`, 20 levels. The
loader normalizes these names and uses the first `n_levels`. F=56 is reached by
adding 4 microstructure features beyond the 52 enumerated in the spec.

## Data

Raw market data lives under `data/` and is tracked by DVC (not git). Five exchanges are covered:

| Exchange   | Pairs                          |
|------------|-------------------------------|
| Nobitex    | BTCIRT, USDTIRT               |
| Bitpin     | BTC_IRT, USDT_IRT             |
| Wallex     | BTCTMN, USDTTMN               |
| Tabdeal    | BTCIRT, USDTIRT               |
| Ramzinex   | BTC_IRT, USDT_IRT             |

Each exchange directory contains two file types:
- `*_orderbook.csv` — LOB snapshots with up to 20 bid/ask price-volume levels (`time`, `bid_price_N`, `bid_volume_N`, `ask_price_N`, `ask_volume_N`)
- `*_trades.csv` — trade ticks (`snapshot_time`, `trade_time`, `price`, `volume`, `direction`)

IRT and TMN are both Iranian Toman (different naming conventions across exchanges).

## DVC Remote

The DVC remote is an S3-compatible Cloudflare R2 bucket. Connection credentials are stored locally (not in git) in `.dvc/config.local`.
