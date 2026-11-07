"""Loss helpers."""
import torch.nn.functional as F


def supervised_loss(logits, targets):
    return F.cross_entropy(logits, targets)
