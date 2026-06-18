# Penny

Penny is an **inpainting diffusion model** for limit order book (LOB) forecasting,
following *Painting the Market* (Backhouse et al., [arXiv:2509.05107v1](https://arxiv.org/abs/2509.05107)).
A past + future LOB window is concatenated into a single 2-channel image; a 2D
UNet inpaints the future region with DDPM + DDIM/RePaint sampling. An auxiliary
DeepLOB trend-classification loss steers the denoiser toward directionally
consistent futures. Trained on BTC/IRT data from Nobitex (Iranian crypto exchange).

## Adaptations vs. the paper

- **Small data** — 9 days of 10-second snapshots (~77.7k rows). Smaller UNet
  (4 blocks, ~12M params), dropout 0.1, stride-30 overlapping windows.
- **OFI features** — channel 0 carries per-level Order Flow Imbalance (Cont et al.)
  instead of raw prices (`feature_mode` switch; `lob` mode keeps prices).
- **Trend loss** — DeepLOB smoothed-mid label trained jointly via a tiny head.

## Image format (spec §2)

Each window becomes a `(2n+3, T_total, 2)` image:

- **Rows** — `2n` LOB levels with the spread in the middle (best bid at row `n-1`,
  best ask at row `n`), plus 3 trade-feature rows.
- **Columns** — `T_total = T_past + T_future` timesteps (156 + 100 = 256).
- **Channel 0** — flow: per-level OFI (`ofi` mode) or price offset (`lob` mode).
- **Channel 1** — state: signed resting depth + trade features (log volume,
  buy-volume ratio, buy-count ratio).

The UNet receives **5 channels**: noisy image (2) + zeroed-future history (2) +
inpainting mask (1), and predicts noise on the 2 data channels. The image is
padded to a square (256×256) before the UNet and unpadded after.

## Pipeline

```
orderbook + trades CSV
        │  features.build_global_rows   (OFI / depth / trade rows)
        │  RollingNormalizer            (fit on train, frozen)
        ▼
 (N, 2n+3, 2) normalized row stream
        │  dataset: calendar-day split (6/2/1), stride-30 windows,
        │           skip day-straddlers, calibrate alpha + fit gamma
        ▼
 per-window image + mask + DeepLOB label + l + mid anchor/boundary
        │  q_sample → UNet → masked diffusion loss
        │  Tweedie x0_hat → painted mid → trend head → trend loss
        ▼
 DDIM + RePaint sampler (past re-pasted, future generated)
        ▼
 reconstructed future mid → trend signal (label, votes, signal ratio)
```

## Labels (spec §3)

```
bwd = mean(real mid over last k past steps)
fwd = mean(painted mid over first k future steps)
l   = (fwd - bwd) / bwd
label = up (l > alpha) | down (l < -alpha) | stationary
```

`alpha` is calibrated on the training set for ~balanced thirds (the 33.3rd
percentile of `|l|`), then frozen for val/test/inference. Class encoding:
`0 = down, 1 = stationary, 2 = up`.

**OFI mode mid reconstruction:** an OFI image has no price channel, so the future
mid is reconstructed by integrating best-level OFI scaled by a fitted coefficient
`gamma` (OLS slope of Δmid on best-level OFI, frozen from training), anchored at
the real boundary mid. LOB mode reads its price channel directly.

## Training (spec §7)

- **Loss** — masked diffusion MSE (future region only) + `lambda · trend_loss`,
  where the trend cross-entropy is weighted by `(1 - t/T_max)²` (meaningful only
  at low noise, where the Tweedie estimate is a good reconstruction).
- **Optimizer** — AdamW (lr 3e-4, wd 1e-4), linear warmup → cosine decay, grad-clip 1.0.
- **Early stopping** — on validation diffusion loss (patience 20). Validation label
  accuracy is logged each epoch on a 50-window subset via the full sampler.

## Setup

```bash
uv sync
uv run dvc pull        # fetch data from the Cloudflare R2 remote
```

## Usage

```bash
# Train (OFI mode by default; LOB mode via configs/config_lob.json)
uv run python scripts/train_penny.py            # configs/config.json
uv run python scripts/train_penny.py configs/config_lob.json

# Trim the pre-gap segment from nobitex data
uv run python scripts/trim_gap.py --dry-run
```

## Configuration (`configs/config.json`)

| Field | Default | Description |
|---|---|---|
| `feature_mode` | `ofi` | `ofi` (per-level OFI) or `lob` (price offsets) |
| `T_past` / `T_future` | 156 / 100 | Past / future columns (26 / ~17 min @ 10s) |
| `T_total` / `padded_size` | 256 / 256 | Window length / square UNet size |
| `stride` | 30 | 5-min step between windows |
| `n_levels` | 10 | LOB levels per side |
| `label_k` | 10 | Smoothing steps each side of the boundary |
| `label_alpha` | -1 | -1 = auto-calibrate for balanced classes |
| `unet_filters` | `[64,128,256,256]` | UNet block widths (~12M params) |
| `T_max` / `ddim_steps` | 1000 / 50 | Diffusion / sampling steps |
| `n_samples` | 20 | Samples per window at inference/eval |
| `lr` / `lambda_trend` | 3e-4 / 0.5 | Learning rate / trend-loss weight |
| `epochs` / `patience` | 200 / 20 | Max epochs / early-stop patience |

## Evaluation (spec §9)

Reported on the test set: label **accuracy**, **macro F1**, **confusion matrix**,
**trend-ratio Pearson correlation** (mean predicted `l` vs ground truth),
**mid-price MAE** (IRT, first `k` steps), and **spread Wasserstein** (LOB mode only —
the OFI image carries no spread).

## Project structure

```
configs/
  config.json          OFI mode (default)
  config_lob.json      LOB mode
scripts/
  train_penny.py       training entry point
  trim_gap.py          drops pre-gap rows from nobitex CSVs
src/penny/
  labels.py            DeepLOB label + alpha calibration
  features.py          OFI/depth/trade rows, padding, mask, normalizer
  dataset.py           split, windows, gamma + alpha, cache
  diffusion.py         DDPM schedule, q_sample, DDIM, RePaint, sampler
  model.py             UNet2DModel, trend head, painted-mid reconstruction
  train.py             train epoch, val diffusion loss, val label accuracy
  evaluate.py          test metrics
```
