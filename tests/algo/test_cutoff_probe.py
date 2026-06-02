"""T-006 — cutoff-controlled leakage probe. CPU, hermetic."""
import pandas as pd

from src.eval_e3.cutoff_probe import split_by_cutoff, leakage_flag


def test_split_by_cutoff():
    df = pd.DataFrame({"date": ["2024-01-01", "2025-06-01", "2026-01-01"], "x": [1, 2, 3]})
    pre, post = split_by_cutoff(df, "2025-01-01")
    assert list(pre["x"]) == [1]
    assert list(post["x"]) == [2, 3]


def test_leakage_flag_triggers_on_pre_high_post_chance():
    # pre-cutoff implausibly high, post-cutoff collapses to ~chance -> leakage
    assert leakage_flag(acc_pre=0.73, acc_post=0.51) is True


def test_leakage_flag_silent_when_both_high():
    # genuine skill would stay high post-cutoff -> not flagged
    assert leakage_flag(acc_pre=0.73, acc_post=0.71) is False


def test_leakage_flag_silent_when_pre_at_chance():
    assert leakage_flag(acc_pre=0.51, acc_post=0.50) is False
