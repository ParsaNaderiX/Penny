"""Aggregate per-window attributions into corpus-level book-level / time-lag importance.

A single window's heatmap is anecdotal; the paper-ready claim is "model X
systematically weights level 3 more than level 8" or "model Y looks further
back than model Z". This module averages ``|attribution|`` over many
explained windows, grouped by the same column layout
``xai.visualize.feature_labels`` already renders, and by time-lag (steps back
from the prediction point — lag 0 = most recent).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from xai.visualize import feature_labels


@dataclass
class AggregateImportance:
    model_name: str
    method: str
    feature_importance: np.ndarray  # (F,) mean |score| per feature column, over all windows and time
    lag_importance: np.ndarray  # (T,) mean |score| per time-lag, over all windows and features
    feature_labels: list[str]


def aggregate_maps(
    model_name: str,
    method: str,
    maps: list[torch.Tensor],  # each (T, F), same axes across the list
    n_lob_levels: int,
    feature_mode: str,
) -> AggregateImportance:
    """Average a list of ``(T, F)`` attribution maps into per-feature / per-lag importance.

    ``maps`` should already be on a comparable scale across windows (e.g. all
    normalised via ``xai.compare._normalize`` or all raw ``|IG scores|`` from
    the same model) — mixing normalised and raw maps in one call will bias
    the mean toward whichever windows happen to have larger raw magnitude.
    """
    stacked = torch.stack([m.abs() for m in maps], dim=0)  # (N, T, F)
    T, F = stacked.shape[1], stacked.shape[2]

    feature_importance = stacked.mean(dim=(0, 1)).detach().cpu().numpy()  # (F,)
    lag_importance = stacked.mean(dim=(0, 2)).detach().cpu().numpy()  # (T,)
    # lag 0 = most recent (last row of the window, matching how the dataset
    # is built: `window[-1]` is the prediction point in crypto/dataset.py).
    lag_importance = lag_importance[::-1].copy()

    labels = feature_labels(n_lob_levels, feature_mode)
    if len(labels) != F:
        labels = [f"f{i}" for i in range(F)]

    return AggregateImportance(
        model_name=model_name,
        method=method,
        feature_importance=feature_importance,
        lag_importance=lag_importance,
        feature_labels=labels,
    )


def top_features(agg: AggregateImportance, k: int = 10) -> list[tuple[str, float]]:
    """Return the ``k`` highest-importance feature columns, sorted descending."""
    order = np.argsort(agg.feature_importance)[::-1][:k]
    return [(agg.feature_labels[i], float(agg.feature_importance[i])) for i in order]
