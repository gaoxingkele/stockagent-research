"""ID1 — leakage-validity test (identification precondition). Hermetic."""
from src.identify.leakage_validity import leakage_validity, from_summary


def test_ashare_holds():
    # A-share: no-context at chance -> memory channel null -> identification holds
    r = leakage_validity(0.486, (0.454, 0.517), chance=0.5)
    assert r["holds"] is True
    assert r["margin"] < 0          # below chance


def test_us_fails():
    # US/ACL18: no-context far above chance -> memorized -> identification fails
    r = leakage_validity(0.733, (0.692, 0.770), chance=0.5)
    assert r["holds"] is False
    assert r["margin"] > 0.2


def test_boundary_ci_lo_at_chance_holds():
    # lower CI bound exactly at chance: cannot reject 'no memory signal' -> holds
    r = leakage_validity(0.55, (0.50, 0.60), chance=0.5)
    assert r["holds"] is True


def test_from_summary_dict():
    summ = {"no_context_accuracy": 0.486, "accuracy_clustered_ci95": [0.454, 0.517]}
    r = from_summary(summ)
    assert r["holds"] is True
    assert "reason" in r
