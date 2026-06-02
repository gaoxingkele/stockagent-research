"""T-004 — regime-invariant (IRM) trainer. CPU, hermetic, STRUCTURAL only.

Per SIGN-008 we do NOT assert IRM beats ERM; we assert the penalty is finite
and the IRM objective decreases over steps.
"""
import torch

from src.training.irm_onset import irm_penalty, train_step


def _two_environments(seed=0, n=200):
    g = torch.Generator().manual_seed(seed)
    envs = []
    for env_id, spurious_sign in enumerate((+1.0, -1.0)):
        y = (torch.rand(n, generator=g) > 0.5).float()
        causal = (2 * y - 1) + 0.3 * torch.randn(n, generator=g)        # invariant
        spurious = spurious_sign * (2 * y - 1) + 0.3 * torch.randn(n, generator=g)  # flips per env
        X = torch.stack([causal, spurious], dim=1)
        envs.append((X, y))
    return envs


def test_irm_penalty_is_finite_scalar():
    torch.manual_seed(0)
    logits = torch.randn(16)
    targets = (torch.rand(16) > 0.5).float()
    p = irm_penalty(logits, targets)
    assert p.ndim == 0 and torch.isfinite(p)


def test_irm_objective_decreases():
    torch.manual_seed(0)
    envs = _two_environments()
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    first = train_step(model, envs, opt, irm_lambda=1.0)
    history = [first]
    for _ in range(60):
        history.append(train_step(model, envs, opt, irm_lambda=1.0))

    assert all(torch.isfinite(torch.tensor(h["penalty"])) for h in history)
    assert history[-1]["total"] < history[0]["total"]
