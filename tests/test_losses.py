import torch
import torch.nn.functional as F


def test_losses_finite():
    logits = torch.randn(8, 10)
    y = torch.randint(0, 10, (8,))
    ls = F.cross_entropy(logits, y)
    lu_each = F.cross_entropy(logits, y, reduction="none")
    mask = torch.ones(8, dtype=torch.bool)
    lu = (lu_each * mask.float()).mean()
    assert torch.isfinite(ls) and torch.isfinite(lu)
    assert torch.isfinite(ls + lu)
