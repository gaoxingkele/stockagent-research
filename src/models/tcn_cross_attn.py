"""TCN-Causal + Spatio-Temporal Cross-Attention Pattern Core (Pathway 2).

Architecture (per user white paper, lightly adapted):

  Input:  x_{i,t} ∈ R^{T × F}  (T = 30-day history, F = 165 factor channels)
                          ↓
  TCN-Causal-Dilated layers (kernel=3, dilations=[1, 2, 4, 8], hidden=64)
                          ↓
  Two views:
    H_T ∈ R^{T × d}  (time tokens, derived from TCN output)
    H_S ∈ R^{F × d}  (feature/space tokens, projected from input cols)
                          ↓
  Spatio-Temporal Cross-Attention (bidirectional, multi-head=4):
    H_T' = LN(H_T + CrossAttn(Q=H_T, K=V=H_S))
    H_S' = LN(H_S + CrossAttn(Q=H_S, K=V=H_T))
                          ↓
  Pool: agg_T = mean(H_T') over time, agg_S = mean(H_S') over feature
                          ↓
  Classifier: Linear(2d → 32) → ReLU → Linear(32 → 3)
                          ↓
  Output: 3-class logits {down, neutral, up} (matches LGBM Pattern Core)

This is the Phase 2 Pattern Core; it replaces the LGBM Pattern Core (Phase 1).
All other agents (Macro Regime Monitor, Alpha Factor Explorer, Backtest Verifier)
remain identical for controlled comparison.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def _auto_device() -> torch.device:
    """Auto-select best available accelerator: cuda > xpu (Intel Arc) > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


@dataclass
class TCNCrossAttnConfig:
    num_features: int = 165          # F: factor channels
    time_steps: int = 30             # T: history length
    d_model: int = 64                # hidden dim
    tcn_kernel: int = 3
    tcn_dilations: tuple = (1, 2, 4, 8)
    n_attn_heads: int = 4
    dropout: float = 0.2
    num_classes: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 20
    early_stopping_patience: int = 3
    seed: int = 42


class Chomp1d(nn.Module):
    """Strip the right padding to keep the output causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class CausalTCNLayer(nn.Module):
    """Single causal dilated conv layer + residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        return self.net(x) + res


class SpatioTemporalCrossAttn(nn.Module):
    """Bidirectional cross-attention between time tokens and feature tokens."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.t_x_s = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.s_x_t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_t = nn.LayerNorm(d_model)
        self.ln_s = nn.LayerNorm(d_model)

    def forward(self, H_T: torch.Tensor, H_S: torch.Tensor):
        # H_T: [B, T, d], H_S: [B, F, d]
        att_t, _ = self.t_x_s(query=H_T, key=H_S, value=H_S)
        H_T_out = self.ln_t(H_T + att_t)
        att_s, _ = self.s_x_t(query=H_S, key=H_T, value=H_T)
        H_S_out = self.ln_s(H_S + att_s)
        return H_T_out, H_S_out


class TCNCrossAttnPatternCore(nn.Module):
    """Phase 2 Pattern Core: TCN-Causal + Spatio-Temporal Cross-Attention."""

    def __init__(self, cfg: TCNCrossAttnConfig):
        super().__init__()
        self.cfg = cfg
        # TCN stack
        layers = []
        in_ch = cfg.num_features
        for d in cfg.tcn_dilations:
            layers.append(CausalTCNLayer(in_ch, cfg.d_model, cfg.tcn_kernel, d, cfg.dropout))
            in_ch = cfg.d_model
        self.tcn = nn.Sequential(*layers)

        # Space embedding: project each feature's full time series → d_model
        self.space_embed = nn.Linear(cfg.time_steps, cfg.d_model)

        # Cross-attn
        self.st_cross_attn = SpatioTemporalCrossAttn(cfg.d_model, cfg.n_attn_heads, cfg.dropout)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model * 2, 32),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        # TCN expects [B, F, T]
        x_perm = x.transpose(1, 2)
        h_t = self.tcn(x_perm)        # [B, d, T]
        H_T = h_t.transpose(1, 2)     # [B, T, d]
        # Space tokens: each feature gets its 30-day history → d
        H_S = self.space_embed(x_perm)  # [B, F, d]

        # Cross-attention
        H_T_out, H_S_out = self.st_cross_attn(H_T, H_S)

        # Pool
        agg_t = H_T_out.mean(dim=1)   # [B, d]
        agg_s = H_S_out.mean(dim=1)   # [B, d]
        z = torch.cat([agg_t, agg_s], dim=-1)  # [B, 2d]
        logits = self.classifier(z)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-timestep temporal features [B, T, d] (after TCN + cross-attn),
        for sequence heads such as OnsetIntensityHead (T-008)."""
        x_perm = x.transpose(1, 2)
        H_T = self.tcn(x_perm).transpose(1, 2)   # [B, T, d]
        H_S = self.space_embed(x_perm)           # [B, F, d]
        H_T_out, _ = self.st_cross_attn(H_T, H_S)
        return H_T_out                           # [B, T, d]


# ─── Training utilities ────────────────────────────────────────────────


def label_to_class(y: pd.Series) -> np.ndarray:
    """Map {-1, 0, +1} to {0, 1, 2}."""
    mapping = {-1: 0, 0: 1, 1: 2}
    return y.map(mapping).astype("int64").values


def class_to_signal(prob: np.ndarray) -> np.ndarray:
    """Same mutual-exclusion ratio used in LGBM Pattern Core."""
    eps = 0.01
    p_down, _, p_up = prob[:, 0], prob[:, 1], prob[:, 2]
    return p_up / (p_down + eps)


def build_anchor_sequences(
    panel: pd.DataFrame,
    anchor_keys: pd.DataFrame,
    feature_cols: list,
    time_steps: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """For each anchor (ts_code, trade_date), gather the prior `time_steps` rows.

    Returns:
        X: [N, T, F]
        valid_mask: [N] bool, True if full history available
    """
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    # Build a multiindex for fast lookup
    panel["_row_idx"] = np.arange(len(panel))
    by_code = panel.groupby("ts_code")
    last_idx_per_code = by_code["_row_idx"].agg(list).to_dict()

    N = len(anchor_keys)
    X = np.zeros((N, time_steps, len(feature_cols)), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    # Build (code, date) -> row index lookup (last occurrence)
    panel_lookup = dict(zip(zip(panel["ts_code"], panel["trade_date"]), panel["_row_idx"]))

    feat_arr = panel[feature_cols].values.astype(np.float32)
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

    for i, row in enumerate(anchor_keys.itertuples(index=False)):
        code = row.ts_code
        date = row.trade_date
        end_idx = panel_lookup.get((code, date), None)
        if end_idx is None:
            continue
        # Find the contiguous block for this code containing end_idx
        code_rows = last_idx_per_code.get(code, [])
        if not code_rows or end_idx not in set(code_rows):
            continue
        # Take time_steps rows ending at end_idx (inclusive)
        pos_in_code = code_rows.index(end_idx)
        if pos_in_code + 1 < time_steps:
            continue  # not enough history
        start_pos = pos_in_code + 1 - time_steps
        idxs = code_rows[start_pos:pos_in_code + 1]
        X[i] = feat_arr[idxs]
        valid_mask[i] = True

    logger.info(f"Built sequences: {valid_mask.sum()}/{N} anchors have full {time_steps}-day history")
    return X, valid_mask


def _normalize_train_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std on training data (avoid lookahead)."""
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def train_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray | None, y_val: np.ndarray | None,
    cfg: TCNCrossAttnConfig | None = None,
) -> tuple[TCNCrossAttnPatternCore, dict]:
    cfg = cfg or TCNCrossAttnConfig(num_features=X_train.shape[2])
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _auto_device()

    mean, std = _normalize_train_stats(X_train)
    X_train = (X_train - mean) / std
    X_train = np.clip(X_train, -5.0, 5.0).astype(np.float32)
    if X_val is not None:
        X_val = ((X_val - mean) / std).clip(-5.0, 5.0).astype(np.float32)

    model = TCNCrossAttnPatternCore(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64))),
            batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )

    best_val, best_state, patience_left = float("inf"), None, cfg.early_stopping_patience
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        train_loss = tot / max(n, 1)

        if val_loader is not None:
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                    vt += loss.item() * len(xb)
                    vn += len(xb)
            val_loss = vt / max(vn, 1)
            logger.info(f"  epoch {epoch+1:>2}/{cfg.max_epochs}  train_ce={train_loss:.4f}  val_ce={val_loss:.4f}")
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_left = cfg.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(f"  early stopping at epoch {epoch+1}, best_val={best_val:.4f}")
                    break
        else:
            logger.info(f"  epoch {epoch+1:>2}/{cfg.max_epochs}  train_ce={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"  done in {time.time()-t0:.1f}s")
    return model, {"mean": mean.tolist(), "std": std.tolist(), "best_val": best_val}


def predict(
    model: TCNCrossAttnPatternCore, X: np.ndarray,
    mean: np.ndarray, std: np.ndarray, batch_size: int = 256,
) -> np.ndarray:
    """Predict 3-class probabilities for given anchor sequences."""
    device = next(model.parameters()).device
    Xn = ((X - mean) / std).clip(-5.0, 5.0).astype(np.float32)
    if device.type == "xpu":
        # XPU prefers smaller batch sizes for stability with current driver
        batch_size = min(batch_size, 512)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xn), torch.zeros(len(Xn), dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)
