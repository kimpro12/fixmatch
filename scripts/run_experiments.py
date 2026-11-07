"""Experiment runner: python scripts/run_experiments.py --suite core|thresholds|smoke [--epochs N] [--outbase outputs]."""
from __future__ import annotations
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import load_config
from src.training.trainer import train_one_run

BASE = os.path.join(os.path.dirname(__file__), "..", "configs")


def run_cfg(base_yaml: str, out_name: str, patch: dict, outbase: str):
    cfg = load_config(os.path.join(BASE, base_yaml))
    # apply patch dict with dotted keys
    for k, v in patch.items():
        parts = k.split(".")
        d = cfg
        for pp in parts[:-1]:
            d = d[pp]
        d[parts[-1]] = v
    out_dir = os.path.join(outbase, out_name)
    print(f"=== RUN {out_name} from {base_yaml} patch={patch} ===", flush=True)
    r = train_one_run(cfg, out_dir)
    print(f"=== DONE {out_name}: {r} ===", flush=True)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="core", choices=["core", "thresholds", "smoke", "quick"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--outbase", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    def ep(d):
        if args.epochs is not None:
            d["training.epochs"] = args.epochs
        d["seed"] = args.seed
        return d

    if args.suite == "smoke":
        for method, base in [("supervised", "smoke.yaml"), ("fixmatch", "smoke.yaml"),
                             ("adaptive", "smoke.yaml")]:
            patch = {"method": {"supervised": "supervised", "fixmatch": "fixmatch",
                                "adaptive": "adaptive_fixmatch"}[method]}
            if method == "adaptive":
                patch["adaptive.enabled"] = True
            if args.epochs is not None:
                patch["training.epochs"] = args.epochs
            run_cfg(base, f"smoke_{method}", patch, args.outbase)
    elif args.suite == "quick":  # short but real-data runs for coursework demo
        e = args.epochs or 30
        run_cfg("supervised_cifar10.yaml", f"sup_250_s{args.seed}", ep({"training.epochs": e}), args.outbase)
        run_cfg("fixmatch_cifar10.yaml", f"fm_t095_250_s{args.seed}",
                ep({"training.epochs": e, "fixmatch.threshold": 0.95}), args.outbase)
        run_cfg("fixmatch_cifar10.yaml", f"fm_t090_250_s{args.seed}",
                ep({"training.epochs": e, "fixmatch.threshold": 0.90}), args.outbase)
        run_cfg("fixmatch_cifar10.yaml", f"fm_t085_250_s{args.seed}",
                ep({"training.epochs": e, "fixmatch.threshold": 0.85}), args.outbase)
        run_cfg("adaptive_fixmatch_cifar10.yaml", f"afm_tmin080_tmax095_b090_250_s{args.seed}",
                ep({"training.epochs": e}), args.outbase)
    elif args.suite == "core":
        e = args.epochs
        jobs = [
            ("supervised_cifar10.yaml", f"sup_250_s{args.seed}", {"training.epochs": e} if e else {}),
            ("fixmatch_cifar10.yaml", f"fm_t095_250_s{args.seed}", {"training.epochs": e} if e else {}),
            ("fixmatch_cifar10.yaml", f"fm_t090_250_s{args.seed}", {"fixmatch.threshold": 0.90, **({"training.epochs": e} if e else {})}),
            ("fixmatch_cifar10.yaml", f"fm_t085_250_s{args.seed}", {"fixmatch.threshold": 0.85, **({"training.epochs": e} if e else {})}),
            ("adaptive_fixmatch_cifar10.yaml", f"afm_tmin080_tmax095_b090_250_s{args.seed}", {"training.epochs": e} if e else {}),
        ]
        for base, name, patch in jobs:
            patch = {k: v for k, v in patch.items() if v is not None}
            patch["seed"] = args.seed
            run_cfg(base, name, patch, args.outbase)
    elif args.suite == "thresholds":
        for t in [0.80, 0.85, 0.90, 0.95, 0.99]:
            patch = {"fixmatch.threshold": t, "seed": args.seed}
            if args.epochs is not None:
                patch["training.epochs"] = args.epochs
            run_cfg("fixmatch_cifar10.yaml", f"fm_t{str(t).replace('.','')}_250_s{args.seed}", patch, args.outbase)


if __name__ == "__main__":
    main()
