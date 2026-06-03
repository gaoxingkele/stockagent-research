"""DB2 — hermetic smoke test for FinBen de-biasing machinery."""
from src.identify.run_debias_finben import debias_finben


def test_debias_finben_keys_and_correction():
    bm = {"acl": (0.672, 0.733), "bigdata": (0.606, 0.719)}
    out = debias_finben(bm, clean_full=0.49, clean_nocontext=0.486)
    assert set(out.keys()) == {"acl", "bigdata"}
    for name in bm:
        r = out[name]
        assert {"debiased", "memorization_excess", "calibration_ok"} <= set(r.keys())
        # removing recall lowers the score below the raw full-context number
        assert r["debiased"] < bm[name][0]
