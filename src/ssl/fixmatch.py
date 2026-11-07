"""FixMatch core logic: pseudo-labels, masks, unsupervised loss."""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_pseudo_labels(logits_weak: torch.Tensor):
    """Stop-gradient pseudo-label generation. Returns (probs not needed, confidence, pseudo_label)."""
    probs = torch.softmax(logits_weak, dim=1)
    confidence, pseudo_labels = probs.max(dim=1)
    return confidence, pseudo_labels


def fixed_threshold_mask(confidence: torch.Tensor, threshold: float) -> torch.Tensor:
    return confidence.ge(threshold)


def compute_fixmatch_loss(logits_strong: torch.Tensor, pseudo_labels: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    loss_each = F.cross_entropy(logits_strong, pseudo_labels, reduction="none")
    return (loss_each * mask.float()).mean()
