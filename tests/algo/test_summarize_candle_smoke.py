"""K6 -- hermetic smoke for the alpha verdict logic."""
from src.onset.summarize_candle import alpha_verdict


def test_real_when_pooled_excludes_zero_and_2of3_splits():
    v = alpha_verdict([0.001, 0.01], [0.3, 1.2, -0.1])
    assert v.startswith("REAL")


def test_promising_when_pooled_sig_but_no_per_split():
    v = alpha_verdict([0.001, 0.01], None)
    assert v.startswith("PROMISING")


def test_null_when_pooled_spans_zero():
    v = alpha_verdict([-0.003, 0.009], [0.3, 1.2, -0.1])
    assert "null" in v or "not cost" in v


def test_single_split_overfit():
    v = alpha_verdict([-0.01, 0.01], [-0.2, -0.1, 1.5])
    assert "single-split" in v
