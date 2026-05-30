"""Barlow Twins SSL pretraining for the TCN+Cross-Attention encoder (Pathway 3).

Per Zbontar et al. 2021 (ICML), the Barlow Twins objective:
    L_BT = sum_i (1 - c_ii)^2  +  lambda * sum_i sum_{j != i} c_ij^2

where c is the empirical cross-correlation matrix between the two batched
view-projected feature embeddings.

Key advantages for finance:
  - No negative samples (avoids the cross-stock cointegration "negative" trap)
  - Small-batch-friendly (feature-axis decorrelation, not sample-axis)
  - High-dim projector encourages fine-grained orthogonal alpha factors

Pretraining flow:
  1. Sample anchor sequences (30-day × 165-feature windows) from D1
  2. For each anchor, generate two augmented views (Gaussian noise, dropout, time-warp)
  3. Forward both through the same encoder (shared weights) + projector
  4. Compute Barlow Twins loss between the two batched embeddings
  5. Backprop, repeat
  6. Save encoder for downstream finetuning on labeled task
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.tcn_cross_attn import (
    TCNCrossAttnPatternCore, TCNCrossAttnConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class BarlowTwinsConfig:
    projector_dim: int = 256       # Output dim of projector (high → fine-grained)
    projector_hidden: int = 128
    lambda_off: float = 5e-3       # Off-diagonal loss weight (per Zbontar 2021)
    aug_noise_std: float = 0.10
    aug_dropout: float = 0.15
    aug_time_warp_max: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    batch_size: int = 256
    max_epochs: int = 30
    early_stopping_patience: int = 5
    seed: int = 42


class BarlowTwinsProjector(nn.Module):
    """3-layer MLP projector with BN, applied to encoder pooled output."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),  # affine=False per Zbontar
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BarlowTwinsEncoder(nn.Module):
    """Wraps TCNCrossAttnPatternCore to expose a poolable encoder output.

    The classifier head is detached; we use a custom pooled representation
    (concat of agg_t + agg_s, same as Phase 2 §4.3.b) as the feature for SSL.
    """

    def __init__(self, base: TCNCrossAttnPatternCore):
        super().__init__()
        self.tcn = base.tcn
        self.space_embed = base.space_embed
        self.st_cross_attn = base.st_cross_attn
        self.feature_dim = base.cfg.d_model * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x_perm = x.transpose(1, 2)
        h_t = self.tcn(x_perm)
        H_T = h_t.transpose(1, 2)
        H_S = self.space_embed(x_perm)
        H_T_out, H_S_out = self.st_cross_attn(H_T, H_S)
        agg_t = H_T_out.mean(dim=1)
        agg_s = H_S_out.mean(dim=1)
        return torch.cat([agg_t, agg_s], dim=-1)  # [B, 2d]


class BarlowTwinsModel(nn.Module):
    """Encoder + projector for SSL pretraining."""

    def __init__(self, encoder: BarlowTwinsEncoder, bt_cfg: BarlowTwinsConfig):
        super().__init__()
        self.encoder = encoder
        self.projector = BarlowTwinsProjector(
            in_dim=encoder.feature_dim,
            hidden_dim=bt_cfg.projector_hidden,
            out_dim=bt_cfg.projector_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.projector(z)


# ─── Augmentations ──────────────────────────────────────────────────────


def augment_view(
    x: torch.Tensor,
    noise_std: float = 0.1,
    dropout_p: float = 0.15,
    time_warp_max: int = 3,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply Gaussian noise + feature dropout + temporal warp to a batch.

    Args:
        x: [B, T, F]
        noise_std: std-multiplier (per-feature; scales with per-feature std).
        dropout_p: fraction of features to zero per anchor.
        time_warp_max: max number of time steps to shift (random direction).

    Returns:
        Augmented tensor of same shape.
    """
    B, T, F_ = x.shape
    device = x.device

    # Gaussian noise (per-feature std-scaled)
    if noise_std > 0:
        per_feat_std = x.std(dim=(0, 1), unbiased=False).clamp(min=1e-6)
        noise = torch.randn(x.shape, device=device) * per_feat_std * noise_std
        x = x + noise

    # Feature dropout (zero a random subset per anchor)
    if dropout_p > 0:
        mask = (torch.rand(B, 1, F_, device=device) > dropout_p).float()
        x = x * mask

    # Temporal warp (cyclic shift per anchor)
    if time_warp_max > 0:
        shifts = torch.randint(-time_warp_max, time_warp_max + 1, (B,), device=device)
        out = torch.empty_like(x)
        for i in range(B):
            s = int(shifts[i])
            out[i] = torch.roll(x[i], shifts=s, dims=0)
        x = out

    return x


# ─── Loss ───────────────────────────────────────────────────────────────


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_off: float = 5e-3) -> torch.Tensor:
    """Cross-correlation Barlow Twins loss.

    z1, z2: [B, D] — BN-normalized projector outputs.
    """
    B, D = z1.shape
    # Empirical cross-correlation (already normalized by BN with affine=False)
    c = (z1.T @ z2) / B  # [D, D]
    on_diag = (1 - torch.diagonal(c)).pow(2).sum()
    off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return on_diag + lambda_off * off_diag


# ─── Training driver ────────────────────────────────────────────────────


def pretrain(
    X_unlabeled: np.ndarray,
    encoder_cfg: TCNCrossAttnConfig,
    bt_cfg: BarlowTwinsConfig | None = None,
    log_every: int = 10,
) -> tuple[BarlowTwinsEncoder, dict]:
    """SSL pretrain the TCN+CrossAttn encoder on unlabeled sequences.

    Args:
        X_unlabeled: [N, T, F] tensor of anchor sequences.
        encoder_cfg: TCN encoder configuration.
        bt_cfg: Barlow Twins configuration.

    Returns:
        pretrained encoder + training stats.
    """
    bt_cfg = bt_cfg or BarlowTwinsConfig()
    torch.manual_seed(bt_cfg.seed)
    np.random.seed(bt_cfg.seed)
    from src.models.tcn_cross_attn import _auto_device
    device = _auto_device()

    # Per-feature normalize the unlabeled data
    mean = X_unlabeled.mean(axis=(0, 1))
    std = X_unlabeled.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    Xn = ((X_unlabeled - mean) / std).clip(-5.0, 5.0).astype(np.float32)

    base = TCNCrossAttnPatternCore(encoder_cfg).to(device)
    encoder = BarlowTwinsEncoder(base).to(device)
    model = BarlowTwinsModel(encoder, bt_cfg).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=bt_cfg.learning_rate, weight_decay=bt_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=bt_cfg.max_epochs, eta_min=1e-5)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xn)),
        batch_size=bt_cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True,
    )

    best_loss, patience_left, history = float("inf"), bt_cfg.early_stopping_patience, []
    t0 = time.time()
    for epoch in range(bt_cfg.max_epochs):
        model.train()
        tot, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            v1 = augment_view(xb, bt_cfg.aug_noise_std, bt_cfg.aug_dropout, bt_cfg.aug_time_warp_max)
            v2 = augment_view(xb, bt_cfg.aug_noise_std, bt_cfg.aug_dropout, bt_cfg.aug_time_warp_max)
            z1 = model(v1)
            z2 = model(v2)
            loss = barlow_twins_loss(z1, z2, lambda_off=bt_cfg.lambda_off)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        avg_loss = tot / max(n, 1)
        history.append(avg_loss)
        logger.info(f"  epoch {epoch+1:>2}/{bt_cfg.max_epochs}  bt_loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss - 1e-3:
            best_loss = avg_loss
            patience_left = bt_cfg.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(f"  early stopping at epoch {epoch+1}, best loss={best_loss:.4f}")
                break

    logger.info(f"  pretrain done in {(time.time()-t0)/60:.1f} min")
    return encoder, {
        "mean": mean.tolist(), "std": std.tolist(),
        "best_loss": best_loss, "history": history,
        "encoder_cfg": encoder_cfg.__dict__,
        "bt_cfg": bt_cfg.__dict__,
    }


def init_pattern_core_from_pretrained(
    encoder: BarlowTwinsEncoder, encoder_cfg: TCNCrossAttnConfig,
) -> TCNCrossAttnPatternCore:
    """Build a fresh TCNCrossAttnPatternCore and load pretrained encoder weights."""
    model = TCNCrossAttnPatternCore(encoder_cfg)
    model.tcn.load_state_dict(encoder.tcn.state_dict())
    model.space_embed.load_state_dict(encoder.space_embed.state_dict())
    model.st_cross_attn.load_state_dict(encoder.st_cross_attn.state_dict())
    # classifier remains randomly initialized
    return model
