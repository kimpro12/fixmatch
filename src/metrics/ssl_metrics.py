"""SSL analysis metrics (analysis-only, use hidden gt labels)."""
from __future__ import annotations
import torch


def pseudo_label_stats(pseudo_labels: torch.Tensor, confidence: torch.Tensor,
                       mask: torch.Tensor, gt: torch.Tensor) -> dict:
    total = pseudo_labels.numel()
    accepted = mask.sum().item()
    correct = (pseudo_labels == gt)
    accepted_correct = (correct & mask).sum().item()
    precision = accepted_correct / accepted if accepted > 0 else 0.0
    coverage = accepted / total if total > 0 else 0.0
    return {
        "pl_precision": precision,
        "pl_coverage": coverage,
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "total": total,
        "mean_confidence": confidence.mean().item() if total > 0 else 0.0,
        "acceptance_rate": coverage,
    }
