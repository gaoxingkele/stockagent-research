"""WS1 — LLM -> labeling-function adapter. CPU, hermetic."""
import numpy as np
import pandas as pd

from src.identify.llm_lf import llm_to_lf
from src.onset.weak_supervision import label_model


def _df(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    return pd.DataFrame({
        "raw_p_up": rng.uniform(size=n),
        "raw_pump_ratio": rng.uniform(0.5, 1.5, size=n),
        "_exp_onset_score": rng.integers(0, 5, size=n),
    })


def test_lf_matrix_shape_and_values():
    lf = llm_to_lf(_df())
    assert lf.shape == (200, 3)
    assert set(np.unique(lf)).issubset({-1, 0, 1})


def test_feeds_label_model():
    lf = llm_to_lf(_df())
    soft, acc = label_model(lf)
    assert soft.shape == (200,)
    assert np.all((soft >= 0) & (soft <= 1))


def test_missing_columns_skipped():
    df = pd.DataFrame({"raw_p_up": np.linspace(0, 1, 50)})
    lf = llm_to_lf(df, ratio_col=None, score_col=None)
    assert lf.shape == (50, 1)
