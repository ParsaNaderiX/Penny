# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Penny is an **inpainting diffusion model** for limit order books, following *Painting the Market* (Backhouse et al., arXiv:2509.05107v1). It concatenates a past + future LOB window into a single 2-channel image and inpaints the future region with a 2D UNet (DDPM + DDIM/RePaint), plus an auxiliary DeepLOB trend-classification loss. Trained on BTCIRT data from Nobitex (Iranian crypto exchange).

## Environment

- Python 3.10, managed via `uv`
- DVC with S3-compatible remote (Cloudflare R2, bucket: `penny`)

## Commands

```bash
# Install dependencies
uv sync

# Train + evaluate (default configs/config.json = OFI mode; configs/config_lob.json = LOB mode)
uv run python scripts/train_penny.py [configs/config.json]

# Inference: trend signal from a past window
uv run python scripts/infer_penny.py --checkpoint <ckpt_dir> \
  --orderbook data/nobitex_data/BTCIRT_orderbook.csv [--trades ...]

# DVC: pull / push data from the remote
uv run dvc pull
uv run dvc push
```

## Penny Model

Penny builds a `(2n+3, T_total, 2)` image per window (rows = LOB levels with the
spread in the middle + 3 trade rows; channel 0 = flow/price, channel 1 = signed
depth), pads it square, and feeds the UNet a 5-channel input (noisy image +
zeroed-future history + inpainting mask). Everything is driven by the JSON config.

A `feature_mode` switch selects channel 0: **`ofi`** (per-level Cont OFI, the
default) or **`lob`** (price offsets from the window mid-anchor). Modules under
`src/penny/`:

- `labels.py` — DeepLOB smoothed-mid trend label (`0=down,1=stationary,2=up`)
  and `alpha` calibration for balanced classes.
- `features.py` — global row stream (OFI/depth/trade features), square padding,
  inpainting mask, and the frozen `RollingNormalizer` (fit on train only).
- `dataset.py` — calendar-day split (6/2/1 days), stride sliding windows that
  skip day-straddlers, `gamma` (OFI→price) fit, `alpha` calibration, `.npz` cache.
- `diffusion.py` — linear-beta DDPM, `q_sample`, DDIM step, RePaint step, sampler.
- `model.py` — `build_unet` (diffusers `UNet2DModel`), `TrendHead` (1→3 linear),
  `painted_future_mid` (price channel for LOB, γ-integrated OFI for OFI mode).
- `train.py` — per-epoch train (masked diffusion loss + timestep-weighted trend
  loss via Tweedie `x0_hat`), val diffusion loss, subset label accuracy.
- `evaluate.py` — test metrics: accuracy, macro-F1, confusion, trend-ratio
  correlation, mid MAE, spread Wasserstein (LOB only).
- `scripts/train_penny.py`, `scripts/infer_penny.py` — entry points.

**OFI mode mid reconstruction:** the OFI image has no price channel, so the
future mid is reconstructed by integrating best-level OFI scaled by a fitted
coefficient `gamma` (OLS slope of Δmid on best-level OFI, frozen from train).
The spread-Wasserstein metric is therefore LOB-only.

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
