"""T-008 — regime-invariant neural onset model, end-to-end (reference method).

Wires the onset pipeline together:
  - encoder (small built-in; swap in src/models/tcn_cross_attn.py for the real run)
  - OnsetIntensityHead + discrete_time_nll        (T-003, temporal point process)
  - IRM penalty across walk-forward environments  (T-004, regime invariance)
  - PU class prior + weak-supervision soft labels  (T-001 / T-002)
  - soft-rank-IC ranking objective                 (T-007)
  - cluster-robust evaluation                       (T-005)

This is the paper's REFERENCE method, not a production model. The smoke path
runs one tiny epoch on synthetic walk-forward-shaped data on CPU and writes a
stats dict to results/onset/<run>/stats.json.

Run: .venv-xpu\\Scripts\\python.exe -m src.train_onset
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.onset.pu_labels import class_prior
from src.onset.weak_supervision import label_model
from src.models.onset_intensity import OnsetIntensityHead, discrete_time_nll
from src.training.irm_onset import irm_penalty
from src.training.objectives import soft_rank_ic_loss
from src.evaluation.onset_eval import clustered_bootstrap

ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(nn.Module):
    """Minimal per-timestep encoder [B,T,F] -> [B,T,D]. Placeholder for the TCN
    encoder (src/models/tcn_cross_attn.py) in a real run."""

    def __init__(self, num_features: int, d_model: int = 16):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        h, _ = self.gru(torch.relu(self.proj(x)))
        return h


def _synthetic_walkforward(n_env=3, n_per=120, T=8, F=6, seed=0):
    """Walk-forward-shaped synthetic data: list of environments, each with
    sequences X[B,T,F], per-step onset events, anchor returns, and dates."""
    rng = np.random.default_rng(seed)
    envs = []
    for e in range(n_env):
        X = rng.normal(size=(n_per, T, F)).astype("float32")
        # a causal driver: onset more likely when feature 0 trends up
        drive = X[:, :, 0].mean(axis=1)
        onset = (drive + 0.3 * rng.normal(size=n_per)) > 0.4
        events = np.zeros((n_per, T), dtype="float32")
        events[onset, -1] = 1.0                      # onset fires at the last step
        returns = (drive + 0.5 * rng.normal(size=n_per)).astype("float32")
        dates = np.repeat(np.arange(n_per // 10), 10)[:n_per]  # ~10 anchors/date
        envs.append({"X": X, "events": events, "onset": onset.astype(int),
                     "returns": returns, "dates": dates})
    return envs


def _rank_ic(pred, target):
    pr = np.argsort(np.argsort(pred))
    tr = np.argsort(np.argsort(target))
    return float(np.corrcoef(pr, tr)[0, 1])


def train_onset(out_dir: str | Path, *, epochs: int = 1, steps: int = 30,
                device: str = "cpu", seed: int = 0, irm_lambda: float = 1.0,
                rank_lambda: float = 1.0) -> dict:
    torch.manual_seed(seed)
    envs = _synthetic_walkforward(seed=seed)
    F = envs[0]["X"].shape[-1]

    # --- weak-supervision soft labels (T-002) from simple feature LFs ---
    all_X = np.concatenate([e["X"] for e in envs], axis=0)
    lf = np.stack([
        np.sign(all_X[:, :, 0].mean(1) - 0.4),
        np.sign(all_X[:, :, 1].mean(1)),
        np.sign(all_X[:, :, 2].mean(1)),
    ], axis=1).astype(int)
    soft_labels, lf_acc = label_model(lf)
    pu_pi = class_prior(np.concatenate([e["onset"] for e in envs]).astype(bool))  # T-001

    enc = TinyEncoder(F).to(device)
    head = OnsetIntensityHead(d_model=16).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=0.05)

    tens = [{k: torch.as_tensor(v, dtype=torch.float32, device=device) if k != "dates" else v
             for k, v in e.items()} for e in envs]

    last_total = float("nan")
    for _ in range(epochs):
        for _ in range(steps):
            opt.zero_grad()
            nll_sum = torch.zeros((), device=device)
            pen_sum = torch.zeros((), device=device)
            rank_sum = torch.zeros((), device=device)
            for e in tens:
                lam = head(enc(e["X"]))                      # [B,T] intensity (T-003)
                nll_sum = nll_sum + discrete_time_nll(lam, e["events"])
                score = lam.mean(dim=1)                       # anchor score
                pen_sum = pen_sum + irm_penalty(score, e["onset"])         # T-004
                rank_sum = rank_sum + soft_rank_ic_loss(score, e["returns"])  # T-007
            n = len(tens)
            total = nll_sum / n + irm_lambda * pen_sum / n + rank_lambda * rank_sum / n
            total.backward()
            opt.step()
            last_total = float(total.detach())

    # --- cluster-robust evaluation (T-005) on a held-out env ---
    he = tens[-1]
    with torch.no_grad():
        score = head(enc(he["X"])).mean(dim=1).cpu().numpy()
    ic = clustered_bootstrap(_rank_ic, score, he["returns"].cpu().numpy(),
                             np.asarray(envs[-1]["dates"]), n_boot=300)

    stats = {
        "run": "onset_smoke",
        "device": device,
        "n_envs": len(envs),
        "n_anchors": int(sum(len(e["onset"]) for e in envs)),
        "epochs": epochs, "steps": steps,
        "final_total_loss": last_total,
        "pu_class_prior": pu_pi,
        "weak_lf_accuracy": [float(a) for a in lf_acc],
        "weak_soft_label_mean": float(np.mean(soft_labels)),
        "rank_ic": ic,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    stats = train_onset(ROOT / "results/onset/onset_smoke")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
