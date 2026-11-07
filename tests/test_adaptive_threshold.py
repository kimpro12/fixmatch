import torch
from src.ssl.adaptive_threshold import AdaptiveThreshold


def test_ema_update():
    at = AdaptiveThreshold(10, beta=0.9, tau_min=0.8, tau_max=0.95, warmup_steps=0)
    conf = torch.tensor([0.75, 0.75, 0.9])
    pl = torch.tensor([2, 2, 5])
    at.update(conf, pl)
    # class 2: 0.9*0.95+0.1*0.75 = 0.93
    assert abs(at.class_ema_confidence[2].item() - 0.93) < 1e-5
    # class 5: 0.9*0.95+0.1*0.9 = 0.945
    assert abs(at.class_ema_confidence[5].item() - 0.945) < 1e-5


def test_missing_class_unchanged():
    at = AdaptiveThreshold(10, beta=0.9, warmup_steps=0)
    before = at.class_ema_confidence.clone()
    conf = torch.tensor([0.9, 0.9])
    pl = torch.tensor([0, 1])
    at.update(conf, pl)
    assert at.class_ema_confidence[3].item() == before[3].item()


def test_bounds_and_warmup():
    at = AdaptiveThreshold(3, beta=0.9, tau_min=0.8, tau_max=0.95, warmup_steps=10)
    th = at.thresholds(0)
    assert torch.allclose(th, torch.full((3,), 0.95))
    at2 = AdaptiveThreshold(3, beta=0.9, tau_min=0.8, tau_max=0.95, warmup_steps=0)
    at2.class_ema_confidence = torch.tensor([0.5, 0.9, 1.0])
    th2 = at2.thresholds(100)
    assert ((th2 >= 0.8) & (th2 <= 0.95)).all()
    assert torch.isfinite(th2).all()
