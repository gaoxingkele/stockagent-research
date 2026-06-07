"""BSYN -- hermetic smoke for the benchmark verdict logic."""
from src.bench.summarize_bench import bench_verdict


def test_none_survive():
    v = bench_verdict(expa_survivors=[], expb_rescues=False,
                      expc_efficient_band=True, expc_interaction_n=6)
    assert v["any_deployable"] is False
    assert "NONE" in v["overall"]
    assert "SUB-COST" in v["expA"]
    assert "does NOT rescue" in v["expB"]


def test_a_survivor_reported():
    v = bench_verdict(expa_survivors=["hmm"], expb_rescues=False,
                      expc_efficient_band=True, expc_interaction_n=6)
    assert v["any_deployable"] is True
    assert "SURVIVED" in v["overall"]


def test_abstention_rescue_reported():
    v = bench_verdict(expa_survivors=[], expb_rescues=True,
                      expc_efficient_band=False, expc_interaction_n=0)
    assert v["any_deployable"] is True
    assert "RESCUES" in v["expB"]
