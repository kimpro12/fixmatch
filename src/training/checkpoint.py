"""Checkpoint save/resume including adaptive state."""
from __future__ import annotations
import torch


def save_checkpoint(path: str, epoch: int, global_step: int, model, optimizer,
                    scheduler, adaptive_state, best_metric: float, config: dict) -> None:
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "adaptive_threshold_state": adaptive_state,
        "best_metric": best_metric,
        "config": config,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
    return ckpt
