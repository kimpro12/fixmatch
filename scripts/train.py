"""CLI: python scripts/train.py --config <yaml> [k=v overrides] [--resume ckpt] [--debug] [--outdir]."""
from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import load_config
from src.training.trainer import train_one_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--outdir", default=None)
    args, unknown = p.parse_known_args()
    overrides = [a.lstrip("-") if False else a for a in unknown]
    # accept both bare k=v and --k=v
    clean = []
    for a in unknown:
        clean.append(a[2:] if a.startswith("--") and "=" in a else a)
    cfg = load_config(args.config, clean)
    if args.outdir:
        out_dir = args.outdir
    else:
        rid = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{cfg.get('method','run')}_seed{cfg.get('seed',42)}"
        out_dir = os.path.join("outputs", rid)
    r = train_one_run(cfg, out_dir, resume=args.resume, debug=args.debug)
    print("DONE:", r)


if __name__ == "__main__":
    main()
