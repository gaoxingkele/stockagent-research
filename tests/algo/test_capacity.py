"""CAP1 -- capacity estimate. CPU, hermetic."""
import numpy as np

from src.onset.capacity import capacity


def test_capacity_scales_with_adv_and_participation():
    adv = np.array([1e8, 2e8, 3e8])
    r = capacity(adv, participation=0.05)
    assert r["n_names"] == 3
    assert abs(r["capacity_capital"] - 0.05 * 6e8) < 1.0
    assert abs(r["per_name_cap_mean"] - 0.05 * 2e8) < 1.0


def test_handles_nan():
    r = capacity([1e8, np.nan, 3e8], participation=0.1)
    assert r["n_names"] == 2
