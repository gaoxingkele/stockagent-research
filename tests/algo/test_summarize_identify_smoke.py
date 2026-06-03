"""SYN — hermetic smoke test for the synthesis machinery."""
from src.identify.summarize_identify import summarize, to_md


def test_summarize_and_md():
    ashare = {"leakage_validity": {"holds": True},
              "contribution": {"raw": {"mean": 0.03, "lo": -0.04, "hi": 0.11},
                               "expert": {"mean": 0.01, "lo": -0.06, "hi": 0.08}}}
    distill = {"identified_improvement": -0.15,
               "arm_A_true_labels": {"lo": 0.04, "hi": 0.18},
               "arm_B_llm_weak_refined": {"lo": -0.11, "hi": 0.03}}
    debias = {"corrected": {"acl": {"debiased": 0.44, "memorization_excess": 0.23}}}
    s = summarize(ashare, distill, debias)
    assert s["leakage_validity_holds"] is True
    assert len(s["identified_estimates"]) == 3
    md = to_md(s)
    assert "Identified estimates" in md and "FinBen de-biased" in md
