"""Plot script: python scripts/make_plots.py --runs outputs/a outputs/b ... --outdir report/figures [--results-csv report/results/main_results.csv]."""
from __future__ import annotations
import argparse
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_metrics(run):
    p = os.path.join(run, "metrics.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--outdir", default="report/figures")
    p.add_argument("--results-csv", default="report/results/main_results.csv")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)

    dfs = {}
    for r in args.runs:
        df = load_metrics(r)
        if df is not None:
            dfs[os.path.basename(r.rstrip("/"))] = df

    # P1 test accuracy vs epoch
    plt.figure()
    for name, df in dfs.items():
        if "test_accuracy" in df:
            plt.plot(df["epoch"], df["test_accuracy"] * 100, label=name)
    plt.xlabel("epoch"); plt.ylabel("test accuracy (%)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "P1_test_accuracy_vs_epoch.png"), dpi=150); plt.close()

    # P2 coverage
    plt.figure()
    for name, df in dfs.items():
        if "pl_coverage" in df:
            plt.plot(df["epoch"], df["pl_coverage"] * 100, label=name)
    plt.xlabel("epoch"); plt.ylabel("PL coverage (%)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "P2_coverage_vs_epoch.png"), dpi=150); plt.close()

    # P3 precision
    plt.figure()
    for name, df in dfs.items():
        if "pl_precision" in df:
            plt.plot(df["epoch"], df["pl_precision"] * 100, label=name)
    plt.xlabel("epoch"); plt.ylabel("PL precision (%)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "P3_precision_vs_epoch.png"), dpi=150); plt.close()

    # P4 accepted per class (last epoch of per_class_metrics.csv)
    plt.figure()
    w = 0.8 / max(len(args.runs), 1)
    for i, name in enumerate(args.runs):
        pc = os.path.join(name, "per_class_metrics.csv")
        if os.path.exists(pc):
            dfp = pd.read_csv(pc)
            if len(dfp) == 0:
                continue
            last = dfp["epoch"].max()
            sub = dfp[dfp["epoch"] == last].sort_values("class_id")
            plt.bar([x + i * w for x in range(10)], sub["acceptance_rate"].values, width=w,
                    label=os.path.basename(name.rstrip("/"))[:20])
    plt.xticks(range(10), [c[:4] for c in CLASS_NAMES], rotation=30)
    plt.xlabel("class"); plt.ylabel("acceptance rate"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "P4_accepted_per_class.png"), dpi=150); plt.close()

    # P5 adaptive thresholds vs epoch
    plt.figure()
    for name, df in dfs.items():
        cols = [c for c in df.columns if c.startswith("tau_class_")]
        if cols:
            for c in cols:
                plt.plot(df["epoch"], df[c], label=f"{name[:12]}:{c[-1]}", alpha=0.7)
    plt.xlabel("epoch"); plt.ylabel("threshold"); plt.legend(fontsize=6, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "P5_adaptive_thresholds.png"), dpi=150); plt.close()

    # P6 fixed threshold trade-off (final epoch values across runs)
    rows = []
    for name, df in dfs.items():
        last = df.iloc[-1]
        rows.append({"run": name, "test_accuracy": last.get("test_accuracy", float("nan")),
                     "pl_precision": last.get("pl_precision", float("nan")),
                     "pl_coverage": last.get("pl_coverage", float("nan"))})
    if rows:
        pd.DataFrame(rows).to_csv(args.results_csv, index=False)
    print("saved figures to", args.outdir, "results to", args.results_csv)


if __name__ == "__main__":
    main()
