import torch
from src.ssl.fixmatch import fixed_threshold_mask, generate_pseudo_labels, compute_fixmatch_loss


def test_fixed_threshold_mask():
    conf = torch.tensor([0.70, 0.90, 0.95, 0.99])
    m = fixed_threshold_mask(conf, 0.95)
    assert m.tolist() == [False, False, True, True]


def test_no_weak_gradient():
    m = torch.nn.Linear(10, 10)
    x = torch.randn(4, 10)
    with torch.no_grad():
        out = m(x)
        conf, pl = out.softmax(1).max(1)
    assert not conf.requires_grad


def test_empty_mask_loss_zero():
    logits = torch.randn(4, 10)
    pl = torch.randint(0, 10, (4,))
    mask = torch.zeros(4, dtype=torch.bool)
    loss = compute_fixmatch_loss(logits, pl, mask)
    assert loss.item() == 0.0


def test_loss_finite():
    logits = torch.randn(8, 10)
    pl = torch.randint(0, 10, (8,))
    mask = torch.tensor([True, False] * 4)
    loss = compute_fixmatch_loss(logits, pl, mask)
    assert torch.isfinite(loss).item()
