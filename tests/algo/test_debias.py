"""DB1 — leakage-calibrated de-biasing estimator. CPU, hermetic."""
from src.identify.debias import debias_accuracy


def test_recovers_reasoning_when_memorization_known():
    chance, reasoning, mem = 0.5, 0.04, 0.23
    us_full = chance + reasoning + mem
    us_nocontext = chance + mem
    clean_full = chance + reasoning          # leakage-free: no memorization
    clean_nocontext = chance                 # at chance -> calibration valid
    r = debias_accuracy(us_full, us_nocontext, clean_full, clean_nocontext, chance)
    assert abs(r["debiased"] - clean_full) < 1e-9     # recovers reasoning-only acc
    assert abs(r["memorization_excess"] - mem) < 1e-9
    assert r["calibration_ok"] is True


def test_flags_invalid_calibration():
    # clean market ALSO above chance -> calibration assumption violated
    r = debias_accuracy(0.77, 0.73, 0.70, 0.65, chance=0.5)
    assert r["calibration_ok"] is False


def test_acl18_like_numbers():
    # ACL18 full 0.672, no-context 0.733; A-share clean full ~0.49, nocontext 0.486
    r = debias_accuracy(0.672, 0.733, 0.49, 0.486, chance=0.5)
    assert r["debiased"] < 0.5                # after removing recall, below chance
    assert r["calibration_ok"] is True
