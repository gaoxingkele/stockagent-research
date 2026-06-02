"""T-008 — end-to-end onset training smoke test. CPU, hermetic, <30s."""
import json

from src.train_onset import train_onset


def test_train_onset_runs_and_writes_stats(tmp_path):
    stats = train_onset(tmp_path / "run", epochs=1, steps=10, device="cpu", seed=0)

    # stats file written with the expected keys
    f = tmp_path / "run" / "stats.json"
    assert f.exists()
    saved = json.loads(f.read_text(encoding="utf-8"))
    for key in ("final_total_loss", "pu_class_prior", "weak_lf_accuracy", "rank_ic", "n_anchors"):
        assert key in saved

    # the pipeline produced sane structural outputs
    assert saved["n_anchors"] > 0
    assert 0.0 <= saved["pu_class_prior"] <= 1.0
    assert {"mean", "lo", "hi"} <= set(saved["rank_ic"].keys())
    import math
    assert math.isfinite(saved["final_total_loss"])
