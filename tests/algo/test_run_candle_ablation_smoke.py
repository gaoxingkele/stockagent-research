"""K5 -- hermetic smoke for the ablation summary."""
from src.onset.run_candle_ablation import summarize_ablation


def test_summarize_incremental():
    by_set = {
        "factors": {"net_sharpe": 0.5},
        "factors_plus_candle": {"net_sharpe": 0.7},
        "candle": {"net_sharpe": 0.6},
    }
    s = summarize_ablation(by_set)
    assert abs(s["incremental_net_sharpe_from_candle"] - 0.2) < 1e-9
    assert set(s["net_sharpe"].keys()) == {"factors", "factors_plus_candle", "candle"}
