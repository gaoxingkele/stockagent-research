"""DSYN -- hermetic smoke for the deployability verdict."""
from src.onset.summarize_deploy import deploy_verdict


def test_deployable_when_2025_sig_and_cross_period():
    v = deploy_verdict([0.001, 0.013], 3, True)
    assert v.startswith("DEPLOYABLE")


def test_not_deployable_when_collapses_cross_period():
    v = deploy_verdict([0.001, 0.013], 3, False)
    assert "NOT-DEPLOYABLE" in v


def test_collapsed_when_pooled_spans_zero():
    v = deploy_verdict([-0.005, 0.01], 1, False)
    assert "COLLAPSED" in v or "null" in v
