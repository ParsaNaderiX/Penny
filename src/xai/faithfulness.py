"""Attribution-guided deletion curves — a faithfulness sanity check.

An attribution map can look plausible and still not reflect what the model
actually uses. The standard check (Samek et al., 2017; "deletion metric") is:
progressively replace the *most*-attributed cells with a neutral baseline
value and watch the target class's predicted probability fall. A faithful
attribution should cause a fast drop; a random-order deletion is the null
comparison — if attribution-guided deletion doesn't beat random deletion, the
map isn't telling you anything the model cares about.

This reuses ``xai.baselines.zero_baseline`` as the neutral replacement value
(same justification as Task 2: it's the zero-mean point of the trained
feature space, not an arbitrary choice) and ``xai.attribution.classifier_fn``
so it works identically across all four models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from xai.attribution import ClassifierFn


@dataclass
class DeletionCurve:
    fractions: np.ndarray  # (n_steps,) — cumulative fraction of cells deleted
    attribution_guided: np.ndarray  # (n_steps,) target-class prob at each fraction
    random_baseline: np.ndarray  # (n_steps,) mean over `n_random_repeats` random orders
    auc_attribution: float  # lower = attribution finds more damaging cells, faster
    auc_random: float
    faithfulness_gap: float  # auc_random - auc_attribution; >0 means attribution beats random


def _prob_at(fn: ClassifierFn, x: torch.Tensor, target: int) -> float:
    with torch.no_grad():
        probs = torch.softmax(fn(x.unsqueeze(0)), dim=1)
    return float(probs[0, target].item())


def deletion_curve(
    fn: ClassifierFn,
    x: torch.Tensor,  # (1, T, F) single window (channel dim included)
    scores: torch.Tensor,  # (T, F) attribution map, same axes as x[0]
    target: int,
    n_steps: int = 20,
    n_random_repeats: int = 5,
    seed: int = 42,
) -> DeletionCurve:
    """Compute one window's attribution-guided vs. random deletion curve.

    Args:
        fn:      ``classifier_fn(model_name, model)`` output.
        x:       ``(1, T, F)`` window (i.e. ``batch["x"][i]``, channel kept).
        scores:  ``(T, F)`` attribution map for this window (use ``.abs()``
                 first if the method is signed — deletion order should rank
                 by *magnitude* of claimed importance, not sign).
        target:  Class index to track the probability of while deleting.
    """
    device = x.device
    T, F = scores.shape
    n_cells = T * F
    baseline_val = torch.zeros_like(x)  # zero_baseline is just torch.zeros_like

    order = torch.argsort(scores.flatten(), descending=True).cpu().numpy()
    fractions = np.linspace(0.0, 1.0, n_steps)
    cell_counts = (fractions * n_cells).astype(int)

    def _run(order_arr: np.ndarray) -> np.ndarray:
        probs = np.empty(n_steps, dtype=np.float64)
        flat_x = x.clone().flatten()
        flat_base = baseline_val.flatten()
        for step, k in enumerate(cell_counts):
            xk = flat_x.clone()
            if k > 0:
                idx = torch.as_tensor(order_arr[:k], device=device, dtype=torch.long)
                xk[idx] = flat_base[idx]
            probs[step] = _prob_at(fn, xk.view_as(x), target)
        return probs

    attribution_guided = _run(order)

    rng = np.random.default_rng(seed)
    random_curves = []
    for _ in range(n_random_repeats):
        rand_order = order.copy()
        rng.shuffle(rand_order)
        random_curves.append(_run(rand_order))
    random_baseline = np.mean(random_curves, axis=0)

    auc_attr = float(np.trapz(attribution_guided, fractions))
    auc_rand = float(np.trapz(random_baseline, fractions))

    return DeletionCurve(
        fractions=fractions,
        attribution_guided=attribution_guided,
        random_baseline=random_baseline,
        auc_attribution=auc_attr,
        auc_random=auc_rand,
        faithfulness_gap=auc_rand - auc_attr,
    )
