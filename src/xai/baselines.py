"""Reference/baseline windows for gradient-based attribution (IG, GradientSHAP).

Perturbation baselines need to be *plausible* LOB windows, not arbitrary
tensors — the input axes are highly autocorrelated (adjacent book levels,
adjacent timesteps), so an off-manifold baseline (e.g. random noise) makes the
integrated path pass through nonsensical order books. Both baselines here are
built from the same normalised feature space the model was trained on
(``RollingNormalizer``-scaled features in ``ofi``/``lob`` mode from
``crypto/features.py``), just zeroed or averaged, not resampled.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def zero_baseline(x: torch.Tensor) -> torch.Tensor:
    """All-zero window: the natural IG baseline for zero-mean normalised features."""
    return torch.zeros_like(x)


def mean_baseline(dataset: Dataset, n_samples: int = 512, seed: int = 42) -> torch.Tensor:
    """Elementwise mean window over a random sample of the dataset.

    Returns a single ``(1, T, F)`` tensor (unbatched) representing the
    "average" window, used as an alternative baseline to the zero window —
    useful when a feature's zero point isn't a neutral reading (e.g. OFI is
    zero-centered already, but trade-count/volume features are not).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    acc = None
    for i in idx:
        x = dataset[int(i)]["x"]  # (1, T, F)
        acc = x.clone() if acc is None else acc + x
    return acc / len(idx)
