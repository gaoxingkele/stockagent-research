"""WS2 — hermetic smoke test for the distillation machinery. CPU, synthetic."""
import numpy as np
import pandas as pd

from src.identify.run_distill import distill_arms


def _df(n, seed):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n); f2 = rng.normal(size=n)
    fwd = f1 + 0.3 * rng.normal(size=n)
    return pd.DataFrame({
        "feat1": f1, "feat2": f2, "_fwd_r5": fwd,
        "raw_p_up": rng.uniform(size=n), "raw_pump_ratio": rng.uniform(0.5, 1.5, n),
        "_exp_onset_score": rng.integers(0, 5, n),
        "trade_date": np.repeat(np.arange(n // 10), 10).astype(str),
    })


def test_distill_arms_runs_and_reports_both():
    tr, te = _df(300, 0), _df(120, 1)
    res = distill_arms(tr, te, ["feat1", "feat2"])
    for k in ("arm_A_true_labels", "arm_B_llm_weak_refined", "identified_improvement"):
        assert k in res
    for arm in ("arm_A_true_labels", "arm_B_llm_weak_refined"):
        assert {"mean", "lo", "hi"} <= set(res[arm].keys())
    assert np.isfinite(res["identified_improvement"])
