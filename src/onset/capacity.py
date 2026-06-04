"""CAP1 -- rough capacity estimate from average daily traded value (ADV).

Max deployable capital before market impact erodes the edge, approximated as a
participation cap on the ADV of the top-K basket names:

    capacity ~ participation * sum(ADV over the basket)

(Pure function; the real path reads `amount` (turnover value) from D1 for the
selected top-K names.)

CPU-only, numpy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["capacity"]


def capacity(basket_adv, participation: float = 0.05) -> dict:
    """basket_adv: array of average daily traded value (currency) for the top-K
    names. Returns rough max capital and per-name participation cap."""
    adv = np.asarray(basket_adv, dtype=float)
    adv = adv[np.isfinite(adv)]
    total = float(adv.sum())
    return {"n_names": int(len(adv)), "participation": participation,
            "basket_adv_sum": total, "capacity_capital": participation * total,
            "per_name_cap_mean": participation * float(adv.mean()) if len(adv) else 0.0}
