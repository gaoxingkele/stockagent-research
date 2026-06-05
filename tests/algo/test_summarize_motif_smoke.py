"""MSYN -- hermetic smoke for the motif go/no-go verdict logic."""
from src.onset.summarize_motif import motif_verdict


def test_build_when_stable():
    v = motif_verdict(n_interaction_hits=52, n_stable=6, n_tested=6)
    assert v["build_motif_model"] is True
    assert v["verdict"].startswith("BUILD-THE-MOTIF-MODEL")


def test_present_but_unstable():
    v = motif_verdict(n_interaction_hits=52, n_stable=0, n_tested=6)
    assert v["build_motif_model"] is False
    assert "UNSTABLE" in v["verdict"]


def test_information_exhausted():
    v = motif_verdict(n_interaction_hits=0, n_stable=0, n_tested=0)
    assert v["build_motif_model"] is False
    assert "EXHAUSTED" in v["verdict"]
