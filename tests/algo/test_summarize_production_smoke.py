"""PSYN -- hermetic smoke for the production verdict."""
from src.onset.summarize_production import production_verdict


def test_not_reproducible_when_all_fail():
    v = production_verdict(timing_net_ci=[-0.003, 0.008], selection_pos_years=0,
                           opt_oos_sharpes=[-2.1, -2.6])
    assert v["verdict"].startswith("NOT REPRODUCIBLE")
    assert not v["timing_works"] and not v["selection_works"]


def test_selection_works_path():
    v = production_verdict([-0.001, 0.01], selection_pos_years=3, opt_oos_sharpes=[1.0, 0.8])
    assert "SELECTION" in v["verdict"]


def test_timing_works_path():
    v = production_verdict([0.001, 0.01], selection_pos_years=0, opt_oos_sharpes=[-1.0])
    assert "TIMING" in v["verdict"]
