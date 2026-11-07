import torch
from src.ssl.adaptive_threshold import AdaptiveThreshold


def test_checkpoint_roundtrip(tmp_path):
    a = AdaptiveThreshold(10, beta=0.9, tau_min=0.8, tau_max=0.95, warmup_steps=5)
    a.update(torch.tensor([0.7, 0.9]), torch.tensor([1, 1]))
    sd = a.state_dict()
    torch.save(sd, str(tmp_path / "at.pt"))
    b = AdaptiveThreshold(10, beta=0.5, tau_min=0.0, tau_max=1.0, warmup_steps=0)
    b.load_state_dict(torch.load(str(tmp_path / "at.pt"), weights_only=False))
    assert torch.allclose(a.class_ema_confidence, b.class_ema_confidence)
    assert torch.allclose(a.thresholds(100), b.thresholds(100))
