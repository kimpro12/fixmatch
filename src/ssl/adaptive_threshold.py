"""EMA-based class-adaptive confidence threshold."""
from __future__ import annotations
import torch


class AdaptiveThreshold:
    """Per-class EMA confidence -> per-class threshold.

    threshold_c = tau_min + (tau_max - tau_min) * s_c / (max_j s_j + eps)
    Warm-up: first `warmup_steps` global steps return tau_max for all classes.
    EMA updated from ALL weak predictions grouped by predicted class.
    """

    def __init__(self, num_classes: int, beta: float = 0.9,
                 tau_min: float = 0.80, tau_max: float = 0.95,
                 warmup_steps: int = 0, device: str = "cpu", eps: float = 1e-12):
        self.num_classes = num_classes
        self.beta = beta
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.warmup_steps = warmup_steps
        self.eps = eps
        self.device = torch.device(device) if isinstance(device, str) else device
        self.class_ema_confidence = torch.full((num_classes,), fill_value=tau_max,
                                               dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, confidences: torch.Tensor, pseudo_labels: torch.Tensor) -> None:
        conf = confidences.detach().to(self.class_ema_confidence.device).float()
        pl = pseudo_labels.detach().to(self.class_ema_confidence.device).long()
        for c in range(self.num_classes):
            m = (pl == c)
            if m.sum().item() > 0:
                batch_mean = conf[m].mean()
                self.class_ema_confidence[c] = (
                    self.beta * self.class_ema_confidence[c] + (1.0 - self.beta) * batch_mean
                )

    def thresholds(self, global_step: int) -> torch.Tensor:
        if global_step < self.warmup_steps:
            return torch.full((self.num_classes,), self.tau_max,
                              dtype=torch.float32, device=self.class_ema_confidence.device)
        s = self.class_ema_confidence
        r = s / (s.max() + self.eps)
        t = self.tau_min + (self.tau_max - self.tau_min) * r
        return t.clamp(min=self.tau_min, max=self.tau_max)

    def sample_thresholds(self, pseudo_labels: torch.Tensor, global_step: int) -> torch.Tensor:
        th = self.thresholds(global_step).to(pseudo_labels.device)
        return th[pseudo_labels]

    def state_dict(self) -> dict:
        return {
            "class_ema_confidence": self.class_ema_confidence.cpu(),
            "beta": self.beta, "tau_min": self.tau_min, "tau_max": self.tau_max,
            "warmup_steps": self.warmup_steps, "num_classes": self.num_classes,
        }

    def load_state_dict(self, state: dict) -> None:
        self.class_ema_confidence = state["class_ema_confidence"].to(self.class_ema_confidence.device).float()
        self.beta = state.get("beta", self.beta)
        self.tau_min = state.get("tau_min", self.tau_min)
        self.tau_max = state.get("tau_max", self.tau_max)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
