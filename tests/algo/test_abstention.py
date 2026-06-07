"""BENCH3 -- hermetic: abstain on high uncertainty/instability; point-in-time."""
import numpy as np

from src.bench.abstention import regime_instability, abstain_mask


def test_regime_instability_switch_rate():
    states = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    inst = regime_instability(states, window=3)
    assert inst[0] == 0.0                 # no switch at start
    assert inst[3] > 0.0                  # a switch just happened
    assert inst[-1] < inst[5]             # settles after switching burst


def test_abstain_on_high_uncertainty():
    rng = np.random.default_rng(0)
    n = 200
    unc = np.abs(rng.standard_normal(n)) * 0.1
    unc[120:130] = 5.0                    # a clear high-uncertainty burst (post-warmup)
    inst = np.zeros(n)                    # calm regime
    trade = abstain_mask(unc, inst, q_unc=0.8, q_reg=0.8, min_warmup=30)
    assert not trade[120:130].any()      # abstain during the burst
    assert trade[40:80].mean() > 0.5     # mostly trade in the calm region


def test_abstain_on_high_instability():
    n = 150
    unc = np.zeros(n)
    inst = np.zeros(n)
    inst[100:110] = 1.0                   # regime thrashing
    trade = abstain_mask(unc, inst, q_unc=0.9, q_reg=0.8, min_warmup=30)
    assert not trade[100:110].any()


def test_point_in_time_thresholds():
    rng = np.random.default_rng(2)
    n = 200
    unc = np.abs(rng.standard_normal(n)) * 0.1
    inst = np.zeros(n)
    t1 = abstain_mask(unc, inst, min_warmup=30)
    unc2 = unc.copy(); unc2[-40:] = 9.0   # perturb only the future tail
    t2 = abstain_mask(unc2, inst, min_warmup=30)
    assert np.array_equal(t1[:150], t2[:150])   # earlier mask unaffected by future
