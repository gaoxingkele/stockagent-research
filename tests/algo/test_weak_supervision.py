"""T-002 — weak-supervision label model over expert rules (hermetic)."""
import numpy as np

from src.onset.weak_supervision import majority_vote, label_model


def _synth(seed=0):
    rng = np.random.default_rng(seed)
    n = 300
    y = rng.integers(0, 2, n)            # true {0,1}
    vote_true = np.where(y == 1, 1, -1)

    def noisy(acc):
        flip = rng.random(n) > acc
        v = np.where(flip, -vote_true, vote_true)
        abstain = rng.random(n) < 0.1    # 10% abstain
        v[abstain] = 0
        return v

    lf = np.stack([noisy(0.95), noisy(0.60), noisy(0.58)], axis=1)  # LF0 strong
    return lf, y


def test_majority_vote_shape_and_abstain():
    lf = np.array([[1, 1, -1], [0, 0, 0], [-1, -1, 1]])
    mv = majority_vote(lf)
    assert mv.shape == (3,)
    assert mv[0] == 1 and mv[2] == -1
    assert mv[1] == 0          # all-abstain row -> abstain


def test_soft_labels_in_unit_interval():
    lf, _ = _synth()
    soft, acc = label_model(lf)
    assert soft.shape == (lf.shape[0],)
    assert np.all((soft >= 0.0) & (soft <= 1.0))
    assert acc.shape == (lf.shape[1],)


def test_upweights_the_accurate_lf():
    lf, y = _synth()
    soft, acc = label_model(lf)
    # the near-perfect LF (index 0) gets the highest estimated accuracy
    assert acc[0] > acc[1] and acc[0] > acc[2]
    # soft labels track the truth well
    corr = np.corrcoef(soft, y)[0, 1]
    assert corr > 0.6


def test_all_abstain_row_is_neutral():
    lf = np.zeros((5, 3), dtype=int)
    soft, _ = label_model(lf)
    assert np.allclose(soft, 0.5, atol=1e-6)
