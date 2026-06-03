"""ID1 — leakage-validity test: the identification precondition.

Identifying the LLM's *reasoning* contribution requires its *memory* channel to
be null on the evaluation data. We operationalise this with the no-context probe
(ticker + date only): if the LLM cannot beat chance with no input, it has no
memorized signal to leak, so any incremental signal it adds over a tabular
baseline on this data is identified as reasoning, not recall.

Decision rule: the assumption HOLDS when the no-context accuracy's lower
clustered-CI bound is at or below chance — i.e. we cannot reject "no
memorization signal". It FAILS when the whole CI lies above chance (the model
demonstrably recalls the future), in which case the data is contaminated and the
LLM contribution is NOT identified.

Examples (measured): A-shares 0.486, CI [0.454, 0.517] -> HOLDS;
ACL18 0.733, CI [0.692, 0.770] -> FAILS.

CPU-only, no deps.
"""
from __future__ import annotations

__all__ = ["leakage_validity", "from_summary"]


def leakage_validity(no_ctx_acc: float, no_ctx_ci: tuple[float, float],
                     chance: float = 0.5) -> dict:
    """Whether the no-leakage / identification precondition holds.

    Parameters
    ----------
    no_ctx_acc : no-context (ticker+date only) directional accuracy.
    no_ctx_ci  : (lo, hi) date-clustered 95% CI of that accuracy.
    chance     : the no-skill baseline (0.5 for balanced Rise/Fall).

    Returns
    -------
    dict(holds, margin, reason) where ``margin = no_ctx_acc - chance`` and
    ``holds`` is True iff the CI lower bound is at or below ``chance``.
    """
    lo, hi = float(no_ctx_ci[0]), float(no_ctx_ci[1])
    margin = float(no_ctx_acc - chance)
    holds = lo <= chance
    if holds:
        reason = (f"no-context CI lower bound {lo:.3f} <= chance {chance:.2f}: "
                  f"cannot reject 'no memorization signal' -> identification holds")
    else:
        reason = (f"no-context CI [{lo:.3f}, {hi:.3f}] lies above chance {chance:.2f}: "
                  f"model recalls the future -> data contaminated, NOT identified")
    return {"holds": holds, "margin": margin, "reason": reason}


def from_summary(summary: dict, chance: float = 0.5) -> dict:
    """Convenience: run the check from an e3 probe summary dict (e.g.
    results/e3_ashare/summary.json), reading no_context_accuracy +
    accuracy_clustered_ci95."""
    acc = summary["no_context_accuracy"]
    ci = summary["accuracy_clustered_ci95"]
    return leakage_validity(acc, (ci[0], ci[1]), chance=chance)
