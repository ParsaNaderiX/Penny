# Explainable AI (XAI)

Explanation methods for the four crypto trend classifiers: **CTABL**, **DLA**,
**JointDiT**, **JumpGateLOB**. Every model already shares one inference
contract (`model.predict(batch, device) -> (B, 3)` logits — see
[../models/README.md](../models/README.md)), so the XAI code shares one
contract too: every method, native or gradient-based, ultimately produces
attribution scores on the same input axes — `(T, F)`, the same window shape
the model itself consumes (rows = time, columns = LOB levels + trade/quote
features; see [../data/features.md](../data/features.md) for the column
layout).

Code lives in `src/xai/`; entry points are `scripts/explain_gradient.py`
(one model) and `scripts/compare_xai.py` (all four, side by side).

## Why one method per model, not one method for all

A single off-the-shelf XAI method applied uniformly to all four would be
easier to write about, but would misrepresent at least two of these models.
Each architecture already computes something interpretable internally as
part of its forward pass — using that, instead of a generic post-hoc
approximation, is both cheaper and more faithful:

| Model | Native mechanism already in the forward pass | Native XAI method |
|---|---|---|
| **CTABL** | `TABL`'s own softmax temporal attention + mixing scalar `λ` | Read the attention out directly ([ctabl_attention.py](../../src/xai/native/ctabl_attention.py)) |
| **DLA** | Dual-stage input attention (`α`) + temporal attention (`β`) — the model *is* Dual-Stage Attention RNN | Read both stages out directly ([dla_attention.py](../../src/xai/native/dla_attention.py)) |
| **JointDiT** | Stacked transformer self-attention (DiT blocks) | Attention Rollout across the stack ([jointdit_rollout.py](../../src/xai/native/jointdit_rollout.py)) |
| **JumpGateLOB** | `AttentionPool` over trunk timesteps + the Lévy jump-regime gate `π`/`logŴ` | Pooling attention + gate readout ([jumpgatelob_readout.py](../../src/xai/native/jumpgatelob_readout.py)) |

On top of the four native methods, **Integrated Gradients** and
**GradientSHAP** run identically across all four models as the common,
model-agnostic yardstick for cross-model comparison
([gradient_methods.py](../../src/xai/gradient_methods.py)).

**Why not LIME / plain KernelSHAP here:** both perturb individual input
cells to probe the model, but LOB windows are heavily autocorrelated on both
axes (adjacent price levels, adjacent timesteps) — flipping cells
independently produces off-manifold, physically impossible order books
(crossed spreads, contradictory depth), and the model's response to those
inputs is not informative. IG/GradientSHAP instead interpolate along a
straight path from a real baseline window to the real input, so every
intermediate point stays a smooth combination of two plausible windows. See
[gradient_methods.py](../../src/xai/gradient_methods.py) for the full
reasoning, including why GradientSHAP (not KernelSHAP) is the SHAP-family
method used.

## Module map

| Module | Role |
|---|---|
| `xai/registry.py` | Loads a `best.pt` checkpoint + config for any of the four models. |
| `xai/attribution.py` | `classifier_fn()` — routes each model's `(B,1,T,F) -> (B,3)` classification path (handling `jointdit`/`jumpgatelob`'s joint diffusion+classifier forward, which must be evaluated on the **clean window at t=0**, never the noised denoiser). `Attribution` — the common return type. |
| `xai/baselines.py` | Zero and dataset-mean baseline windows for IG/GradientSHAP. |
| `xai/sampling.py` | Samples matched per-class (down/stationary/up) windows from the shared test split, so every model is explained on identical inputs. |
| `xai/gradient_methods.py` | Integrated Gradients + GradientSHAP via `captum`. |
| `xai/native/` | The four model-specific native explanations (table above). |
| `xai/compare.py` | Normalises native explanations into comparable `(T,F)` maps and shared `(T,)` time profiles; pairwise agreement metrics (cosine similarity, time-profile correlation, top-k overlap). |
| `xai/faithfulness.py` | Attribution-guided deletion curve vs. random-order deletion — a sanity check that an attribution map actually reflects what the model relies on. |
| `xai/aggregate.py` | Corpus-level per-feature-column and per-time-lag importance, aggregated over many explained windows. |
| `xai/visualize.py` | `(feature × time)` heatmap rendering with real column labels. |

## Usage

Explain one model with the shared gradient methods:

```bash
uv run python scripts/explain_gradient.py \
    --model ctabl \
    --checkpoint checkpoints/nobitex/BTCIRT/ctabl_BTCIRT_lob_20260101_000000 \
    --n-per-class 8 \
    --out results/xai/ctabl
```

Compare all four models on the same task (requires one checkpoint per model,
all trained on the same symbol + `feature_mode` + `label_k` — the script
warns if configs don't match, since cross-model agreement is only meaningful
when every model solves the identical task on identical inputs):

```bash
uv run python scripts/compare_xai.py \
    --ctabl checkpoints/.../ctabl_.../best.pt \
    --dla checkpoints/.../dla_.../best.pt \
    --jointdit checkpoints/.../jointdit_.../best.pt \
    --jumpgatelob checkpoints/.../jumpgatelob_.../best.pt \
    --n-per-class 6 \
    --out results/xai/compare_btcirt_lob
```

Output: `comparison_summary.json` (pairwise agreement, faithfulness deletion
gaps, top-10 features per model) + `aggregate_importance.png` (per-lag and
per-feature bar charts, one column per model).

SLURM templates for cluster runs: `slurm/xai/explain_gradient.slurm`,
`slurm/xai/compare_xai.slurm` (both parameterised by env vars — checkpoint
paths are timestamped at train time, so they can't be hardcoded the way
training configs are).

## Interpreting the cross-model comparison

Two of the four models (CTABL, JumpGateLOB) have **no native per-feature
resolution** — their attention lives over a post-projection channel space
(CTABL) or GRU latent context (JumpGateLOB), not the raw `F` input features.
`xai/compare.py` broadcasts their `(T,)` time-only signal across `F` so it
can sit in the same heatmap grid as DLA's and JointDiT's genuine `(T,F)`
maps — this is a display convenience, not a claim that CTABL or JumpGateLOB
have per-level resolution. Prefer `time_profile()` (the shared `(T,)` axis)
over `to_comparable_map()` when the comparison claim is about *time*, and
treat feature-level claims from CTABL/JumpGateLOB's broadcast maps as
unsupported.
