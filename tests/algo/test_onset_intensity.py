"""T-003 — neural onset-intensity head (temporal point process). CPU, hermetic."""
import torch

from src.models.onset_intensity import OnsetIntensityHead, discrete_time_nll


def test_intensity_shape_and_positive():
    torch.manual_seed(0)
    B, T, D = 4, 10, 8
    feats = torch.randn(B, T, D)
    head = OnsetIntensityHead(d_model=D)
    lam = head(feats)
    assert lam.shape == (B, T)
    assert torch.all(lam >= 0)            # softplus -> non-negative intensity


def test_nll_is_finite():
    torch.manual_seed(0)
    lam = torch.rand(3, 5) + 0.1
    events = (torch.rand(3, 5) > 0.7).float()
    loss = discrete_time_nll(lam, events)
    assert torch.isfinite(loss)


def test_loss_decreases_on_trivial_fit():
    torch.manual_seed(0)
    B, T, D = 8, 12, 6
    feats = torch.randn(B, T, D)
    # events depend on the first feature channel -> learnable signal
    events = (feats[..., 0] > 0.5).float()

    head = OnsetIntensityHead(d_model=D, hidden=16)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    init = discrete_time_nll(head(feats), events).item()
    for _ in range(50):
        opt.zero_grad()
        loss = discrete_time_nll(head(feats), events)
        loss.backward()
        opt.step()
    final = discrete_time_nll(head(feats), events).item()
    assert final < init
