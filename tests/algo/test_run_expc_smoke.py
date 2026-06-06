"""EXP-C -- hermetic smoke: NMI bounded + positioning logic."""
import numpy as np

from src.bench.run_expc import nmi, conditional_nmi, _positioning


def test_nmi_bounded_and_self_high():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    # NMI(x, copy-with-noise) is high; NMI(x, independent) ~ 0; both in [0,1]
    y_dep = x + 0.01 * rng.standard_normal(4000)
    y_ind = rng.standard_normal(4000)
    hi = nmi(x, y_dep); lo = nmi(x, y_ind)
    assert 0.0 <= lo < 0.2
    assert hi > lo and hi <= 1.0001


def test_conditional_nmi_runs():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(3000); y = rng.standard_normal(3000)
    z = rng.integers(0, 2, 3000)
    v = conditional_nmi(x, y, z)
    assert 0.0 <= v <= 1.0001


def test_positioning_efficient_band_and_interaction_count():
    items = [
        {"marginal_nmi": 0.01, "ours_interaction": {"interaction": 0.004, "interact_p": 0.003}},
        {"marginal_nmi": 0.02, "ours_interaction": {"interaction": 0.002, "interact_p": 0.003}},
        {"marginal_nmi": 0.03, "ours_interaction": {"interaction": -0.001, "interact_p": 0.5}},
    ]
    p = _positioning(items)
    assert p["nmi_efficient_regime"] is True       # max marginal NMI < 0.05
    assert p["n_interaction_significant"] == 2      # two hits II>0 at p<0.05
