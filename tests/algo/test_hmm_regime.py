"""BENCH1 -- hermetic: rolling-HMM recovers regimes and is point-in-time."""
import numpy as np

from src.bench.hmm_regime import rolling_hmm_states


def _two_regime(n=600, seed=0):
    """First half low-mean/low-vol, second half high-mean/high-vol."""
    rng = np.random.default_rng(seed)
    a = -1.0 + 0.3 * rng.standard_normal(n // 2)
    b = 1.5 + 1.2 * rng.standard_normal(n - n // 2)
    return np.concatenate([a, b])


def test_recovers_two_regimes():
    x = _two_regime()
    s = rolling_hmm_states(x, n_states=2, min_train=120, refit_every=40)
    n = len(x)
    # in the steady part of each block (after warmup), the dominant state differs
    early = s[150:n // 2]           # block A (post-warmup)
    late = s[n // 2 + 50:]         # block B
    from collections import Counter
    a_mode = Counter(early).most_common(1)[0][0]
    b_mode = Counter(late).most_common(1)[0][0]
    assert a_mode != b_mode
    # canonical: low-mean regime -> lower state id than high-mean regime
    assert a_mode < b_mode


def test_point_in_time_no_future_leakage():
    x = _two_regime(seed=1)
    s1 = rolling_hmm_states(x, n_states=2, min_train=120, refit_every=40)
    x2 = x.copy()
    x2[-50:] = x2[-50:] + 10.0      # perturb only the future tail
    s2 = rolling_hmm_states(x2, n_states=2, min_train=120, refit_every=40)
    # states well before the perturbed region must be identical (used only obs<=t)
    cut = len(x) - 60
    assert np.array_equal(s1[:cut], s2[:cut])
