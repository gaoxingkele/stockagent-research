"""NB4 — contrastive stock-vs-reference encoder.

Targets the IDIOSYNCRATIC (market/sector-neutral) component directly by learning
a RELATIVE representation: encode the stock's own sequence and a reference
sequence (the market or sector aggregate), and use their spread to predict the
neutral target. This is the representation-learning instantiation of beta
neutralization. (Method not novel -- asset-embedding contrastive / relative
ranking are prior art; the value here is the clean idiosyncratic target.)

CPU/GPU agnostic; small by design.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ContrastiveEncoder", "neutral_mse"]


class _SeqEncoder(nn.Module):
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):                 # [B,T,F] -> [B,d]
        h, _ = self.gru(torch.relu(self.proj(x)))
        return h[:, -1, :]


class ContrastiveEncoder(nn.Module):
    """Encode stock seq and reference seq; predict the neutral target from the
    stock embedding and the stock-minus-reference spread."""

    def __init__(self, num_features: int, d_model: int = 16):
        super().__init__()
        self.enc = _SeqEncoder(num_features, d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(),
                                  nn.Linear(d_model, 1))

    def forward(self, stock: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        hs = self.enc(stock)
        hr = self.enc(reference)
        spread = hs - hr                  # relative (idiosyncratic) representation
        return self.head(torch.cat([hs, spread], dim=-1)).squeeze(-1)


def neutral_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()
