"""TSYN -- hermetic smoke for the tradability verdict logic."""
from src.onset.summarize_tradability import tradability_verdict


def test_tradable_when_all_pass():
    v = tradability_verdict(any_directional_monotone=True, n_survivors=2,
                            gating_helps=True)
    assert v["build_trading_motif_model"] is True
    assert v["verdict"].startswith("TRADABLE")


def test_information_only_eaten_by_cost():
    v = tradability_verdict(any_directional_monotone=True, n_survivors=0,
                            gating_helps=True)
    assert v["build_trading_motif_model"] is False
    assert "eaten-by-cost" in v["verdict"]


def test_information_only_sign_blind():
    v = tradability_verdict(any_directional_monotone=False, n_survivors=0,
                            gating_helps=False)
    assert v["build_trading_motif_model"] is False
    assert "sign-blind" in v["verdict"]
