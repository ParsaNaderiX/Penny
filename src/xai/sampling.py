"""Pull representative test windows per trend class for XAI explanation.

Uses the same ``LOBDataset`` produced by ``crypto.dataset.build_datasets`` —
every model trains/evaluates on the identical calendar-day test split, so
sampling once here and reusing the indices across all four models keeps
cross-model comparisons (Task 4) on exactly the same windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

DOWN, STATIONARY, UP = 0, 1, 2
CLASS_NAMES = {DOWN: "down", STATIONARY: "stationary", UP: "up"}


@dataclass
class SampledWindow:
    index: int  # index into the source dataset
    label: int
    x: torch.Tensor  # (1, T, F)


def sample_by_class(
    dataset: Dataset, n_per_class: int = 8, seed: int = 42
) -> dict[int, list[SampledWindow]]:
    """Return up to ``n_per_class`` random windows for each of the 3 trend classes.

    Classes with fewer than ``n_per_class`` available windows return all of
    them (no error) — some symbol/k combinations have thin tails after the
    calendar-day split.
    """
    labels = np.array([dataset[i]["label"] for i in range(len(dataset))])
    rng = np.random.default_rng(seed)

    out: dict[int, list[SampledWindow]] = {}
    for cls in (DOWN, STATIONARY, UP):
        idx = np.flatnonzero(labels == cls)
        pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        out[cls] = [
            SampledWindow(index=int(i), label=cls, x=dataset[int(i)]["x"])
            for i in pick
        ]
    return out


def stack_windows(windows: list[SampledWindow]) -> torch.Tensor:
    """Batch a list of ``SampledWindow`` into a single ``(B, 1, T, F)`` tensor."""
    return torch.stack([w.x for w in windows], dim=0)
