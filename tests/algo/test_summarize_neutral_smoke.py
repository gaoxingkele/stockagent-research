"""NB6 — hermetic smoke for the synthesis."""
from src.identify.summarize_neutral import summarize, to_md


def test_summarize_and_md():
    neutral = {
        "leakage_validity": {"holds": True},
        "selection": {"market_neutral": {"raw": {"mean": -0.004, "lo": -0.064, "hi": 0.054}}},
        "timing": {"market_neutral": {"raw": {"mean": 0.085, "lo": -0.055, "hi": 0.221}}},
        "long_short": {"baseline": {"annualized_sharpe": 0.59, "ls_mean_per_period": 0.008, "n_dates": 100}},
    }
    contrastive = {"arms": {
        "raw": {"market_neutral_rank_ic": {"mean": 0.016, "lo": -0.016, "hi": 0.054},
                "long_short": {"annualized_sharpe": 2.18, "ls_mean_per_period": 0.0095, "n_dates": 60}},
        "neutral": {"market_neutral_rank_ic": {"mean": -0.001, "lo": -0.043, "hi": 0.049},
                    "long_short": {"annualized_sharpe": 1.71, "ls_mean_per_period": 0.007, "n_dates": 60}},
    }}
    s = summarize(neutral, contrastive)
    assert s["leakage_validity_holds"] is True
    assert "raw" in s["selection_market_neutral"] and "raw" in s["contrastive_arms"]
    md = to_md(s)
    assert "beta-timing vs alpha-selection" in md and "long-short" in md
