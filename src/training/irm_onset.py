"""T-004 — regime-invariant onset training via Invariant Risk Minimization.

Directly targets contribution C4: different methods win in different quarters
(regime heterogeneity), and no naive router captures the oracle headroom. IRM
(Arjovsky et al., 2019) learns a representation whose optimal classifier is the
SAME across environments — here, environments are walk-forward splits/quarters.
The hope is an onset mechanism that is invariant to market regime.

IMPORTANT (SIGN-008): IRM is not a free lunch. It only helps when training
environments exhibit diverse spurious correlations, and can underperform plain
ERM otherwise. Treat the penalty as a regulariser, validate per-regime, and do
not assume superiority.

CPU/GPU agnostic.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F

__all__ = ["irm_penalty", "train_step"]

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def irm_penalty(logits: torch.Tensor, targets: torch.Tensor,
                loss_fn: LossFn = F.binary_cross_entropy_with_logits) -> torch.Tensor:
    """IRMv1 gradient penalty for one environment.

    Multiplies the logits by a dummy scale w=1.0, then returns the squared
    gradient of the environment loss w.r.t. w. A small penalty means the
    classifier is already (locally) optimal for this environment — invariance.
    """
    scale = torch.ones((), requires_grad=True, device=logits.device, dtype=logits.dtype)
    loss = loss_fn(logits * scale, targets)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2)


def train_step(model: torch.nn.Module,
               environments: Sequence[tuple[torch.Tensor, torch.Tensor]],
               optimizer: torch.optim.Optimizer,
               *, loss_fn: LossFn = F.binary_cross_entropy_with_logits,
               irm_lambda: float = 1.0) -> dict:
    """One IRM optimisation step over a list of (X, y) environments.

    total = mean_env ERM_loss + irm_lambda * mean_env IRM_penalty
    """
    optimizer.zero_grad()
    erm = logits_pen = None
    erm_sum = torch.zeros((), dtype=torch.float32)
    pen_sum = torch.zeros((), dtype=torch.float32)
    for X, y in environments:
        logits = model(X).squeeze(-1)
        erm_sum = erm_sum + loss_fn(logits, y)
        pen_sum = pen_sum + irm_penalty(logits, y, loss_fn)
    n = max(1, len(environments))
    erm = erm_sum / n
    pen = pen_sum / n
    total = erm + irm_lambda * pen
    total.backward()
    optimizer.step()
    return {"total": float(total.detach()), "erm": float(erm.detach()), "penalty": float(pen.detach())}
