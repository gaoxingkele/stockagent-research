"""T-003 — neural onset-intensity head (discrete-time temporal point process).

Reframes "onset" as an event-intensity lambda(t) over the sequence rather than a
fixed-horizon return class. The head sits on top of any sequence encoder that
emits per-timestep features [B, T, D] (e.g. the TCN encoder in
``src/models/tcn_cross_attn.py``) and outputs a non-negative intensity [B, T].

The discrete-time point-process likelihood treats each step as a Bernoulli
event with hazard p_t = 1 - exp(-lambda_t): an onset either fires in the
interval or it does not. This is the natural objective for "when does the move
start" and generalises to self/cross-exciting (sector contagion) intensities.

Runs on CPU; device-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OnsetIntensityHead", "discrete_time_nll"]


class OnsetIntensityHead(nn.Module):
    """Map per-timestep encoder features [B, T, D] to onset intensity [B, T].

    Uses softplus so the intensity lambda(t) >= 0, as a point-process intensity
    must be.
    """

    def __init__(self, d_model: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raw = self.net(feats).squeeze(-1)      # [B, T]
        return F.softplus(raw)                 # lambda >= 0


def discrete_time_nll(intensity: torch.Tensor, event_mask: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """Negative log-likelihood of a discrete-time point process.

    Per step the onset-event hazard is p_t = 1 - exp(-lambda_t). The NLL is the
    Bernoulli cross-entropy between p_t and the observed event indicator:

        -mean[ event * log p_t + (1 - event) * log(1 - p_t) ]

    Parameters
    ----------
    intensity : [B, T] non-negative intensities lambda_t.
    event_mask : [B, T] in {0, 1}; 1 where an onset event occurred.
    """
    lam = intensity.clamp_min(0.0)
    p = (1.0 - torch.exp(-lam)).clamp(eps, 1.0 - eps)
    ev = event_mask.float()
    nll = -(ev * torch.log(p) + (1.0 - ev) * torch.log(1.0 - p))
    return nll.mean()
