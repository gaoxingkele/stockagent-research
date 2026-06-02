"""T-006 — cutoff-controlled temporal-leakage probe.

The high-ROI experiment that upgrades contribution C5 from "suggestive" to
"causal": run the no-context ablation (or any LLM accuracy measurement) split by
whether the target date is BEFORE or AFTER the model's training cutoff. If
accuracy collapses to chance on post-cutoff data while staying high pre-cutoff,
the apparent skill is memorization, not forecasting — proven causally.

CPU-only, numpy/pandas.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["split_by_cutoff", "leakage_flag"]


def split_by_cutoff(df: pd.DataFrame, cutoff_date, date_col: str = "date"):
    """Split into (pre, post) frames at the model training cutoff.

    pre  = target date strictly before the cutoff (potentially memorized);
    post = target date on/after the cutoff (genuinely held out).
    """
    d = pd.to_datetime(df[date_col])
    cut = pd.to_datetime(cutoff_date)
    pre = df[d < cut]
    post = df[d >= cut]
    return pre, post


def leakage_flag(acc_pre: float, acc_post: float, *, chance: float = 0.5,
                 high_margin: float = 0.08, chance_tol: float = 0.05) -> bool:
    """True when pre-cutoff accuracy is well above chance but post-cutoff
    accuracy collapses to ~chance — the signature of temporal leakage.

    Parameters
    ----------
    high_margin : how far above ``chance`` pre-cutoff must be to count as "high".
    chance_tol  : how close to ``chance`` post-cutoff must be to count as "collapsed".
    """
    pre_high = acc_pre >= chance + high_margin
    post_collapsed = abs(acc_post - chance) <= chance_tol
    return bool(pre_high and post_collapsed)
