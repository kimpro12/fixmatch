"""Classification metrics: accuracy, macro P/R/F1, per-class acc, confusion matrix."""
from __future__ import annotations
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_acc = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_model(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        ps.append(pred)
        ys.append(np.asarray(y))
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    model.train()
    return classification_metrics(y_true, y_pred), y_true, y_pred
