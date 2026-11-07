"""CLI: python scripts/evaluate.py --checkpoint <pt> [--data-root ./data]."""
from __future__ import annotations
import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.wideresnet import build_model
from src.data.transforms import eval_transform
from src.metrics.classification import evaluate_model
from src.utils.seed import resolve_device
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg if cfg else {"model": {}})
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    tf = eval_transform()
    test_ds = CIFAR10(root=args.data_root, train=False, download=True, transform=tf)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)
    res, _, _ = evaluate_model(model, loader, device)
    print(json.dumps({k: v for k, v in res.items() if k != "confusion_matrix"}, indent=2))
    out = args.out or os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
