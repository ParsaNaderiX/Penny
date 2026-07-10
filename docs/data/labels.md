# Trend labels

`crypto/labels.py` produces the 3-class direction label used by every model. It is
the DeepLOB smoothed-mid formulation.

## Definition

For each bin `t` and horizon `k` (`label_k` in the config):

```
bwd         = mean(mid[t-k : t])      # mean mid over the previous k bins
fwd         = mean(mid[t : t+k])      # mean mid over the next k bins
trend_ratio = (fwd - bwd) / bwd
```

Smoothing over `k` bins (rather than comparing single snapshots) suppresses
micro-noise so the label reflects a genuine directional move.

## Class assignment

```
0  down        if trend_ratio < -alpha
1  stationary  if |trend_ratio| <= alpha
2  up          if trend_ratio >  alpha
```

The first and last `k` bins have no full backward/forward window and are marked
invalid (`-1`); windows whose label bin is invalid are dropped during windowing.

## `alpha` calibration

`alpha` sets the dead-band that separates "flat" from a real move. It is calibrated
**on the training split only**, to the **33.3rd percentile of `|trend_ratio|`**,
which yields roughly balanced `down / flat / up` frequencies.

- Set `label_alpha` to a positive number in the config to fix `alpha` explicitly.
- Set `label_alpha: -1` (the default in every shipped config) to auto-calibrate from
  the training region.

The resolved `alpha` is logged, stored in each checkpoint, and reported alongside the
realised class balance at the start of every run. Because calibration uses only the
training portion of the series, it introduces no test-set leakage.

## Horizons

Configs are generated per horizon `k ∈ {10, 20, 50, 100}` (the `k10 … k100` suffix in
config/slurm paths). A larger `k` labels a longer-horizon move and generally shifts
the class balance and difficulty.
