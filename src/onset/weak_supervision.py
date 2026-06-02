"""T-002 — weak-supervision label model over expert rules.

Turns the V12.31 expert rules (e.g. bottoms_rising / above_5d_low_5pct /
ma_pattern_ok / volume_boost from ``src/onset/expert_pattern.py``) from hard
rules into *labeling functions* (LFs) that vote {-1, 0(abstain), +1}, then
aggregates them with a lightweight generative label model (Snorkel-style):
each LF's reliability is estimated and votes are combined by accuracy-weighted
log-odds. This upgrades contribution C1 (expert knowledge) from a brittle rule
into a learnable weak-supervision source.

CPU-only, numpy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["majority_vote", "label_model"]


def majority_vote(lf_matrix: np.ndarray) -> np.ndarray:
    """Sign of the summed votes per row. Returns {-1, 0, +1}; 0 on ties /
    all-abstain. ``lf_matrix`` is N x K with entries in {-1, 0, +1}."""
    lf = np.asarray(lf_matrix)
    return np.sign(lf.sum(axis=1)).astype(int)


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def label_model(lf_matrix: np.ndarray, *, n_iter: int = 20,
                acc_floor: float = 0.5, acc_cap: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-LF accuracy and produce soft labels in [0, 1].

    A simple EM-like loop: start from majority vote, estimate each LF's accuracy
    as its agreement with the current hard label (over non-abstaining rows),
    weight LFs by accuracy log-odds, recombine into soft labels, repeat.

    Parameters
    ----------
    lf_matrix : N x K array of votes in {-1, 0(abstain), +1}.

    Returns
    -------
    (soft, acc) : soft labels P(y=+1) in [0,1] length N; estimated accuracy per
    LF length K. Rows where every LF abstains get soft = 0.5.
    """
    lf = np.asarray(lf_matrix, dtype=float)
    n, k = lf.shape
    abstain = lf == 0

    hard = majority_vote(lf)
    hard = np.where(hard == 0, 1, hard)  # break ties as +1 for estimation only

    acc = np.full(k, 0.6)
    for _ in range(n_iter):
        # estimate each LF accuracy = agreement with current hard label
        for j in range(k):
            active = ~abstain[:, j]
            if active.sum() == 0:
                acc[j] = acc_floor
                continue
            agree = (lf[active, j] == hard[active]).mean()
            acc[j] = float(np.clip(agree, acc_floor, acc_cap))
        weights = np.log(acc / (1.0 - acc))            # log-odds reliability
        score = (lf * weights[None, :]).sum(axis=1)    # abstains (0) contribute nothing
        soft = _sigmoid(score)
        new_hard = np.where(soft >= 0.5, 1, -1)
        if np.array_equal(new_hard, hard):
            hard = new_hard
            break
        hard = new_hard

    score = (lf * np.log(acc / (1.0 - acc))[None, :]).sum(axis=1)
    soft = _sigmoid(score)
    soft = np.where(abstain.all(axis=1), 0.5, soft)    # no signal -> neutral
    return soft, acc
