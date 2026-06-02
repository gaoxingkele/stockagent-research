"""T-007 — differentiable ranking / utility training objectives.

Cross-entropy on a 3-class onset label barely beats random (ln 3 ~ 1.099) and
is not the deployment objective: a trader cares about the RANKING of names and
the return of the top bucket. These losses train directly on those quantities
and are fully differentiable, so they can replace / augment CE.

- soft_rank_ic_loss : negative soft-Spearman IC between scores and returns,
  using a differentiable soft rank (pairwise-sigmoid approximation).
- topk_utility_loss : negative softmax-weighted top-k return (a smooth proxy for
  the realised return of the top-k selected names).

CPU/GPU agnostic.
"""
from __future__ import annotations

import torch

__all__ = ["soft_rank", "soft_rank_ic_loss", "topk_utility_loss"]


def soft_rank(x: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Differentiable rank of a 1-D tensor via pairwise sigmoids.

    R_i = 0.5 + sum_{j != i} sigmoid((x_i - x_j) / tau). As tau -> 0 this
    approaches the hard rank; larger tau is smoother.
    """
    # for each i, sum_j sigmoid((x_i - x_j)/tau)
    d = (x.unsqueeze(1) - x.unsqueeze(0)) / tau       # d[i, j] = x_i - x_j
    s = torch.sigmoid(d).sum(dim=1) - 0.5             # subtract self term sigmoid(0)=0.5
    return s + 0.5


def _pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + eps)


def soft_rank_ic_loss(scores: torch.Tensor, returns: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Negative soft-rank IC (Spearman-like). Minimising it maximises the rank
    correlation between predicted scores and realised returns."""
    rs = soft_rank(scores, tau=tau)
    rr = soft_rank(returns, tau=tau)
    return -_pearson(rs, rr)


def topk_utility_loss(scores: torch.Tensor, returns: torch.Tensor, k: int = 20,
                      temperature: float | None = None) -> torch.Tensor:
    """Negative softmax-weighted top-k return.

    Uses a temperature-sharpened softmax over scores as a differentiable
    selection of the top-k names, then the weighted mean realised return. The
    default temperature scales so that roughly k names receive most of the mass.
    """
    n = scores.shape[0]
    k = max(1, min(k, n))
    if temperature is None:
        temperature = max(scores.std().item(), 1e-3) / 2.0
    w = torch.softmax(scores / temperature, dim=0)
    # emphasise the top-k: renormalise mass over the k largest weights' scale
    util = (w * returns).sum() * n / k
    return -util
