"""T-001 — Positive-Unlabeled (PU) learning for movement-onset labels.

The onset label is ill-defined: we confidently observe SOME onsets (positives),
but the remainder is *unlabeled*, not negative — it is a mix of true non-onsets
and missed onsets. Treating it as negative biases every downstream estimator.
This module frames onset labeling as PU learning and provides the non-negative
PU (nnPU) risk estimator of Kiryo et al. (2017).

Conventions
-----------
- ``positive_mask`` / ``labels``: a positive is an anchor we *confidently* label
  as an onset (e.g. expert ``is_bullish_onset``). ``labels`` uses 1 = labeled
  positive, 0 = unlabeled.
- ``scores``: real-valued decision scores g(x); large => more onset-like.
- ``prior_pi``: the class prior P(y=+1) over the WHOLE population. It is not the
  labeled-positive rate (which underestimates it); pass an estimate. The default
  helper :func:`class_prior` returns the labeled-positive rate as a floor.

CPU-only, numpy. No torch dependency so it stays a pure label/risk utility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["build_pu_sets", "class_prior", "nnpu_risk"]


def build_pu_sets(df: pd.DataFrame, positive_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partition row positions into Positive and Unlabeled index arrays.

    Parameters
    ----------
    df : frame whose rows are the anchors (only its length / index is used).
    positive_mask : boolean array, length len(df); True where the anchor is a
        confidently-labeled positive (onset).

    Returns
    -------
    (positive_idx, unlabeled_idx) : disjoint integer position arrays whose union
    is ``range(len(df))``.
    """
    mask = np.asarray(positive_mask, dtype=bool)
    if mask.shape[0] != len(df):
        raise ValueError(f"positive_mask length {mask.shape[0]} != len(df) {len(df)}")
    all_idx = np.arange(len(df))
    return all_idx[mask], all_idx[~mask]


def class_prior(positive_mask: np.ndarray) -> float:
    """Default prior estimate: the labeled-positive rate (a lower bound on the
    true P(y=+1), since unlabeled data still contains positives)."""
    mask = np.asarray(positive_mask, dtype=bool)
    return float(mask.mean())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable logistic
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _surrogate(scores: np.ndarray, target: int) -> np.ndarray:
    """Sigmoid surrogate loss. target=+1 penalises low scores, target=-1
    penalises high scores. Both in (0, 1)."""
    s = np.asarray(scores, dtype=float)
    return _sigmoid(-s) if target == +1 else _sigmoid(s)


def nnpu_risk(scores: np.ndarray, labels: np.ndarray, prior_pi: float | None = None,
              *, clip: bool = True) -> float:
    """Non-negative PU risk estimator (Kiryo et al., 2017).

        R_pu = pi * R_p^+  +  max(0, R_u^-  -  pi * R_p^-)

    where R_p^+ / R_p^- are the surrogate risks of labelling positives as +1 / -1,
    and R_u^- treats unlabeled data as -1. The ``max(0, .)`` is the non-negative
    correction that stops the empirical risk diverging below zero.

    Parameters
    ----------
    scores : real-valued decision scores, length N.
    labels : 1 = labeled positive, 0 = unlabeled; length N.
    prior_pi : class prior P(y=+1). Defaults to the labeled-positive rate.
    clip : apply the non-negative correction (True = nnPU, False = uPU).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length")

    pos = labels == 1
    unl = ~pos
    if pos.sum() == 0 or unl.sum() == 0:
        raise ValueError("need at least one positive and one unlabeled sample")
    if prior_pi is None:
        prior_pi = float(pos.mean())
    if not (0.0 < prior_pi < 1.0):
        raise ValueError(f"prior_pi must be in (0,1), got {prior_pi}")

    r_p_pos = _surrogate(scores[pos], +1).mean()
    r_p_neg = _surrogate(scores[pos], -1).mean()
    r_u_neg = _surrogate(scores[unl], -1).mean()

    neg_part = r_u_neg - prior_pi * r_p_neg
    if clip:
        neg_part = max(0.0, neg_part)
    return float(prior_pi * r_p_pos + neg_part)
