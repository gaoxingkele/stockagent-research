"""NB4 — contrastive stock-vs-reference encoder. CPU, hermetic."""
import torch

from src.identify.contrastive_encoder import ContrastiveEncoder, neutral_mse


def test_forward_shape():
    torch.manual_seed(0)
    B, T, F = 6, 8, 5
    m = ContrastiveEncoder(F)
    out = m(torch.randn(B, T, F), torch.randn(B, T, F))
    assert out.shape == (B,)


def test_loss_decreases_on_spread_learnable_target():
    torch.manual_seed(0)
    B, T, F = 64, 8, 5
    stock = torch.randn(B, T, F)
    ref = torch.randn(B, T, F)
    # idiosyncratic target depends on the stock-minus-reference spread
    target = (stock[:, -1, 0] - ref[:, -1, 0]) + 0.1 * torch.randn(B)

    m = ContrastiveEncoder(F, d_model=16)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    init = neutral_mse(m(stock, ref), target).item()
    for _ in range(60):
        opt.zero_grad()
        loss = neutral_mse(m(stock, ref), target)
        loss.backward(); opt.step()
    assert torch.isfinite(loss)
    assert loss.item() < init
