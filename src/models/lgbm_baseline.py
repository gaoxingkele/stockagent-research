"""LGBM baseline trainer for multiclass stock movement prediction.

This is the workhorse for Phase 1 (B1.1 - B1.4 baseline experiments)
and Phase 2 (E1.x PWC experiments).

Three-class output: {-1=bearish, 0=neutral, +1=bullish}.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


@dataclass
class LGBMConfig:
    num_class: int = 3
    objective: str = "multiclass"
    metric: str = "multi_logloss"
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    num_boost_round: int = 500
    early_stopping_rounds: int = 30
    seed: int = 42
    verbose: int = -1


def label_to_class(y: pd.Series) -> np.ndarray:
    """Map {-1, 0, +1} to {0, 1, 2} for LightGBM multiclass."""
    mapping = {-1: 0, 0: 1, 1: 2}
    return y.map(mapping).astype("int8").values


def class_to_signal(prob: np.ndarray) -> np.ndarray:
    """Map LGBM softmax output to scalar ranking signal.

    Default uses ratio = P_up / (P_down + eps) per v12.31 finding.
    """
    eps = 0.01
    p_down, _, p_up = prob[:, 0], prob[:, 1], prob[:, 2]
    return p_up / (p_down + eps)


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    cfg: LGBMConfig | None = None,
) -> lgb.Booster:
    """Train LGBM multiclass classifier with optional early stopping."""
    cfg = cfg or LGBMConfig()

    mask_train = y_train != -127
    X_tr = X_train.loc[mask_train]
    y_tr = label_to_class(y_train.loc[mask_train])

    train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)

    valid_sets, valid_names = [train_set], ["train"]
    if X_val is not None and y_val is not None:
        mask_val = y_val != -127
        X_va = X_val.loc[mask_val]
        y_va = label_to_class(y_val.loc[mask_val])
        valid_sets.append(lgb.Dataset(X_va, label=y_va, reference=train_set, free_raw_data=False))
        valid_names.append("val")

    params = {
        "objective": cfg.objective,
        "num_class": cfg.num_class,
        "metric": cfg.metric,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l1": cfg.lambda_l1,
        "lambda_l2": cfg.lambda_l2,
        "seed": cfg.seed,
        "verbose": cfg.verbose,
    }

    callbacks = [lgb.log_evaluation(period=50)]
    if X_val is not None:
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=cfg.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    logger.info(f"Training complete. Best iter: {booster.best_iteration}")
    return booster


def predict_signal(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Return ratio-based ranking signal."""
    prob = booster.predict(X, num_iteration=booster.best_iteration)
    return class_to_signal(prob)
