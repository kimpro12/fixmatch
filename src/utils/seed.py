"""Deterministic utilities: seeding, device resolution."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] device=cuda requested but CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)
