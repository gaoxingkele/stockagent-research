"""MLP baseline for cross-architecture verification of strength dependence.

Simple 3-layer feedforward classifier with the same multiclass head as LGBM
(p_down, p_neutral, p_up). The ranking signal is the same mutual-exclusion
ratio P_up / (P_down + eps) used in production v12.31.

Architecture (deliberately simple — we want the *strength dependence* finding
to attribute to label/filter, not to backbone capacity):
    Input(165) -> Linear(128) -> GELU -> Dropout
                -> Linear(64)  -> GELU -> Dropout
                -> Linear(3)   (logits)

Training: AdamW + cross-entropy + early stopping on val log-loss.
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


@dataclass
class MLPConfig:
    hidden_dims: tuple = (128, 64)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    max_epochs: int = 30
    early_stopping_patience: int = 3
    seed: int = 42
    device: str = "cpu"


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(128, 64), num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def label_to_class(y: pd.Series) -> np.ndarray:
    """Map {-1, 0, +1} to {0, 1, 2}."""
    mapping = {-1: 0, 0: 1, 1: 2}
    return y.map(mapping).astype("int64").values


def class_to_signal(prob: np.ndarray) -> np.ndarray:
    """Same mutual-exclusion ratio as LGBM (production v12.31)."""
    eps = 0.01
    p_down, _, p_up = prob[:, 0], prob[:, 1], prob[:, 2]
    return p_up / (p_down + eps)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=False, drop_last=False)


def _impute_normalize(X_train: pd.DataFrame, X_val: pd.DataFrame | None, X_test: pd.DataFrame):
    """Fit median + std on train, apply to all splits. Robust to NaN."""
    med = X_train.median(numeric_only=True).fillna(0.0)
    X_tr_filled = X_train.fillna(med)
    std = X_tr_filled.std(ddof=0).replace(0, 1.0).fillna(1.0)

    def transform(X):
        return ((X.fillna(med) - med) / std).clip(-5.0, 5.0)  # winsorize ±5 sigma

    Xt = transform(X_train).values.astype("float32")
    Xv = transform(X_val).values.astype("float32") if X_val is not None else None
    Xtest = transform(X_test).values.astype("float32")
    return Xt, Xv, Xtest


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    X_test: pd.DataFrame,
    cfg: MLPConfig | None = None,
) -> tuple[MLP, np.ndarray]:
    """Train MLP and return (model, test_prediction_probs).

    Note: we hand back the test prediction here too because the normalizer
    (fit on train) is internal — exposing it externally would be brittle.
    """
    cfg = cfg or MLPConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)

    mask_train = y_train != -127
    X_tr = X_train.loc[mask_train]
    y_tr = label_to_class(y_train.loc[mask_train])

    has_val = X_val is not None and y_val is not None
    if has_val:
        mask_val = y_val != -127
        X_va = X_val.loc[mask_val]
        y_va = label_to_class(y_val.loc[mask_val])
    else:
        X_va, y_va = None, None

    # Normalize using train stats
    X_tr_arr, X_va_arr, X_test_arr = _impute_normalize(X_tr, X_va, X_test)

    in_dim = X_tr_arr.shape[1]
    model = MLP(in_dim, hidden_dims=cfg.hidden_dims, num_classes=3, dropout=cfg.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_loader = _make_loader(X_tr_arr, y_tr, cfg.batch_size, shuffle=True)
    val_loader = _make_loader(X_va_arr, y_va, cfg.batch_size, shuffle=False) if has_val else None

    best_val = float("inf")
    best_state = None
    patience_left = cfg.early_stopping_patience
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
    logger.info(f"  training done in {time.time()-t0:.1f}s")

    # Predict on test
    model.eval()
    test_loader = _make_loader(X_test_arr, np.zeros(len(X_test_arr), dtype="int64"),
                                cfg.batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    test_probs = np.concatenate(all_probs, axis=0)
    return model, test_probs
