"""T-001 — PU-learning onset label builder (hermetic, CPU, synthetic)."""
import numpy as np
import pandas as pd

from src.onset.pu_labels import build_pu_sets, class_prior, nnpu_risk


def test_build_pu_sets_disjoint_and_complete():
    df = pd.DataFrame({"x": range(10)})
    mask = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    P, U = build_pu_sets(df, mask)
    # Positive set is exactly the masked rows; P and U partition all rows.
    assert sorted(P) == [0, 3, 7]
    assert set(P).isdisjoint(set(U))
    assert sorted(np.concatenate([P, U])) == list(range(10))


def test_class_prior_default_is_positive_rate():
    mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    assert abs(class_prior(mask) - 0.2) < 1e-9


def test_nnpu_risk_is_finite_scalar():
    rng = np.random.default_rng(0)
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = rng.normal(size=len(labels))
    r = nnpu_risk(scores, labels, prior_pi=0.3)
    assert np.isscalar(r) or np.ndim(r) == 0
    assert np.isfinite(r)


def test_risk_lower_when_scores_align_with_true_positives():
    # Synthetic: 40 anchors, first 12 are true positives (onset).
    n, n_pos = 40, 12
    true_pos = np.zeros(n, dtype=bool)
    true_pos[:n_pos] = True
    # Only half the true positives are *labeled* (the PU setting); rest unlabeled.
    labels = np.zeros(n, dtype=int)
    labels[:n_pos // 2] = 1  # labeled positives
    pi = n_pos / n

    aligned = np.where(true_pos, 3.0, -3.0)          # high score iff truly onset
    misaligned = -aligned                            # exactly wrong
    r_aligned = nnpu_risk(aligned, labels, prior_pi=pi)
    r_misaligned = nnpu_risk(misaligned, labels, prior_pi=pi)
    assert r_aligned < r_misaligned


def test_nonnegative_correction_keeps_risk_bounded_below():
    # With the nnPU correction the empirical risk should not go arbitrarily
    # negative even on adversarial scores.
    rng = np.random.default_rng(1)
    labels = np.array([1] * 5 + [0] * 25)
    scores = rng.normal(size=len(labels)) * 10
    r = nnpu_risk(scores, labels, prior_pi=0.2)
    assert r >= -1e-6
