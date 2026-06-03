"""DB1 — leakage-calibrated de-biasing of contaminated-benchmark accuracy.

A contaminated (pre-cutoff) full-context accuracy mixes three parts:
    us_full ≈ chance + reasoning_effect + memorization_recall
The no-context probe isolates the recall excess: us_nocontext ≈ chance + recall.
Subtracting it yields a recall-corrected estimate of what the model achieves
once memorization is removed:

    debiased = us_full - (us_nocontext - chance)

A leakage-free market validates the estimator: there recall ≈ 0, so
clean_nocontext ≈ chance and debiased(clean) ≈ clean_full. We surface
``calibration_ok`` = whether the clean market's no-context is indeed ~chance
(the assumption under which the correction is valid), and ``reasoning_ref`` =
the clean-market reasoning effect (the only honestly-measurable reasoning gain).

CPU-only, no deps.
"""
from __future__ import annotations

__all__ = ["debias_accuracy"]


def debias_accuracy(us_full: float, us_nocontext: float,
                    clean_full: float, clean_nocontext: float,
                    chance: float = 0.5, clean_tol: float = 0.03) -> dict:
    """Recall-corrected ('reasoning-only') accuracy estimate for a contaminated
    market, calibrated against a leakage-free market.

    Returns dict with debiased, memorization_excess, reasoning_ref,
    calibration_ok.
    """
    memorization_excess = us_nocontext - chance
    debiased = us_full - memorization_excess
    reasoning_ref = clean_full - clean_nocontext
    calibration_ok = abs(clean_nocontext - chance) <= clean_tol
    return {
        "debiased": float(debiased),
        "memorization_excess": float(memorization_excess),
        "reasoning_ref_clean_market": float(reasoning_ref),
        "calibration_ok": bool(calibration_ok),
        "note": ("valid: clean-market no-context ~ chance"
                 if calibration_ok else
                 "INVALID: clean market also shows above-chance no-context"),
    }
