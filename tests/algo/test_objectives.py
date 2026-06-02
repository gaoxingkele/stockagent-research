"""T-007 — differentiable ranking / utility objectives. CPU, hermetic."""
import torch

from src.training.objectives import soft_rank_ic_loss, topk_utility_loss


def test_soft_rank_ic_differentiable_and_finite():
    torch.manual_seed(0)
    scores = torch.randn(32, requires_grad=True)
    returns = torch.randn(32)
    loss = soft_rank_ic_loss(scores, returns)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None and torch.all(torch.isfinite(scores.grad))


def test_perfect_ranking_is_near_optimal():
    returns = torch.linspace(-1, 1, 40)
    aligned = returns.clone() + 0.01 * torch.randn(40)   # same order as returns
    reversed_ = -returns.clone()                          # opposite order
    assert soft_rank_ic_loss(aligned, returns) < soft_rank_ic_loss(reversed_, returns)
    # aligned IC loss should be close to the -1 floor
    assert soft_rank_ic_loss(aligned, returns).item() < -0.8


def test_topk_utility_differentiable_and_prefers_aligned():
    torch.manual_seed(0)
    returns = torch.randn(50)
    scores = returns.clone().requires_grad_(True)         # high score on high return
    loss = topk_utility_loss(scores, returns, k=5)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None and torch.all(torch.isfinite(scores.grad))

    aligned = topk_utility_loss(returns, returns, k=5)
    misaligned = topk_utility_loss(-returns, returns, k=5)
    assert aligned < misaligned                            # lower loss = higher utility
