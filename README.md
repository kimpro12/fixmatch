# FixMatch + Class-Adaptive Threshold on CIFAR-10

Reproduction of **FixMatch** on CIFAR-10 with 250 labeled examples, plus a simple
**EMA-based class-adaptive confidence threshold** extension.

> A simple EMA adaptive-threshold variant inspired by the limitation of using one
> fixed confidence threshold for all classes. Related advanced methods
> (FlexMatch, FreeMatch) are acknowledged; **no novelty is claimed**.
> This repo is a clean, runnable, leakage-free reference implementation.

---

## 1. What this repo does

- **Supervised baseline**: WideresNet trained on 250 labeled CIFAR-10 images only.
- **FixMatch**: consistency regularization + pseudo-labeling with a fixed
  confidence threshold τ (default 0.95).
- **Adaptive FixMatch**: same as FixMatch, but each class `c` gets its own
  threshold `τ_c` updated by EMA of prediction confidence.

Default setting (same as FixMatch paper's low-label regime):

| Item | Value |
|---|---|
| Dataset | CIFAR-10 (50k train / 10k test) |
| Labeled | 250 total = 25 per class, class-balanced, deterministic by seed |
| Unlabeled | full 50k train pool (labeled images included, standard FixMatch setup) |
| Model | WideResNet-28-2 |
| Optimizer | SGD lr=0.03, momentum=0.9, weight_decay=5e-4, Nesterov, cosine schedule |
| Batch | labeled 64 + unlabeled 448 (μ=7) |
| Epochs | 200 |
| Loss | `L = L_sup + λ_u * L_unsup`, `λ_u=1.0` |
| AMP | on when CUDA is available |

### Method formulas

Weak view `x_w` → pseudo-label, strong view `x_s` → consistency target:

```
p = softmax(model(x_w))            # no-grad
conf, y_hat = max(p, dim=1)
mask = conf >= tau                  # FixMatch: scalar tau
L_u = mean( CE(model(x_s), y_hat) * mask )
```

Adaptive version replaces scalar `tau` with per-class thresholds:

```
# EMA per predicted class (all weak predictions, missing classes skipped)
s_c <- beta * s_c + (1 - beta) * mean(conf | y_hat == c)

# normalized threshold in [tau_min, tau_max], 10% warm-up at tau_max
r_c = s_c / max_j(s_j)
tau_c = tau_min + (tau_max - tau_min) * r_c
mask_i = conf_i >= tau_{y_hat_i}
```

Defaults: `beta=0.9, tau_min=0.80, tau_max=0.95, warmup_fraction=0.10`.

Leakage-free guarantee: unlabeled ground-truth labels (`y_gt`) returned by the
loader are used **only** for `pl_precision / pl_coverage / per-class analysis`,
never for loss or threshold updates.

---

## 2. Repo structure

```
.
├── configs/
│   ├── supervised_cifar10.yaml          # supervised baseline, 200ep, WRN-28-2
│   ├── fixmatch_cifar10.yaml            # FixMatch τ=0.95
│   ├── adaptive_fixmatch_cifar10.yaml   # Adaptive τ∈[0.80,0.95], β=0.9
│   └── smoke.yaml                       # tiny CPU smoke (WRN-16-1, 1ep, 5 steps/ep)
├── src/
│   ├── data/
│   │   ├── cifar10.py        # build_loaders(), UnlabeledWrapper -> (weak, strong, gt)
│   │   ├── splits.py         # make_balanced_split(), save_split()
│   │   └── transforms.py     # weak: flip+crop | strong: +RandAugment | eval: normalize
│   ├── models/
│   │   └── wideresnet.py     # WideResNet-28-2 (CIFAR stem), build_model(cfg)
│   ├── ssl/
│   │   ├── fixmatch.py            # generate_pseudo_labels(), fixed_threshold_mask(), compute_fixmatch_loss()
│   │   ├── adaptive_threshold.py  # AdaptiveThreshold EMA class
│   │   └── losses.py              # supervised_loss()
│   ├── training/
│   │   ├── trainer.py      # train_one_run(): split, loop, AMP, cosine, CSVs, checkpoints
│   │   ├── checkpoint.py   # save/load (model, optim, sched, adaptive state, best_acc)
│   │   └── evaluator.py    # test accuracy path
│   ├── metrics/
│   │   ├── classification.py  # accuracy, macro-F1, per-class acc, confusion matrix
│   │   └── ssl_metrics.py     # pseudo_label_stats(): acceptance-rate, mean-conf
│   └── utils/
│       ├── config.py   # YAML + dotted CLI overrides (training.epochs=100 ...)
│       ├── seed.py     # set_seed(), resolve_device()
│       └── logging.py
├── scripts/
│   ├── train.py            # single run: --config + k=v overrides + --resume + --outdir + --debug
│   ├── run_experiments.py  # suites: smoke | quick | core | thresholds
│   ├── evaluate.py         # eval a checkpoint on CIFAR-10 test
│   └── make_plots.py       # P1-P6 figures from metrics.csv
├── tests/
│   ├── test_split.py
│   ├── test_fixmatch_mask.py
│   ├── test_adaptive_threshold.py
│   ├── test_losses.py
│   ├── test_checkpoint.py
│   └── test_smoke_train.py   # 1-epoch end-to-end (needs CIFAR download)
├── aggregate.py        # summarize outputs/full_s*/metrics.csv (best/final/mean±std)
├── make_full_figs.py   # P1-P7 + full_results.csv for the 200-epoch campaign
├── run_full.sh         # full campaign: 3 seeds × core + tau_min ablation (seed 42)
├── fetch/full/         # frozen results + figures from a completed 200ep campaign
└── requirements.txt
```

Per-run output directory contains:

```
config.yaml  metadata.json  split.json  metrics.csv  per_class_metrics.csv
thresholds.csv  checkpoint_last.pt  checkpoint_best.pt  confusion_matrix_last.json
```

---

## 3. Install

```bash
pip install -r requirements.txt
```

Requires: `torch torchvision numpy pandas matplotlib scikit-learn tqdm pyyaml pytest tensorboard`

Python 3.10+ recommended. GPU strongly recommended for 200-epoch runs
(WRN-28-2, B=64, μ=7 → ~448 unlabeled/batch fits in ~8–12 GB).
Reduce `training.batch_size_labeled` / `training.mu` on OOM.

---

## 4. Data

CIFAR-10 auto-downloads to `./data` on first run (torchvision).

```bash
# custom location
python scripts/train.py --config configs/fixmatch_cifar10.yaml data.root=/path/to/data
```

---

## 5. Quickstart

```bash
# 1) supervised baseline (250 labels)
python scripts/train.py --config configs/supervised_cifar10.yaml

# 2) FixMatch τ=0.95
python scripts/train.py --config configs/fixmatch_cifar10.yaml

# 3) Adaptive FixMatch τ∈[0.80, 0.95]
python scripts/train.py --config configs/adaptive_fixmatch_cifar10.yaml

# custom outdir + overrides
python scripts/train.py --config configs/fixmatch_cifar10.yaml \
  --outdir outputs/my_run training.epochs=100 fixmatch.threshold=0.90 seed=123

# resume
python scripts/train.py --config configs/fixmatch_cifar10.yaml \
  --resume outputs/<run>/checkpoint_last.pt

# per-step debug logging
python scripts/train.py --config configs/smoke.yaml --debug
```

Config override syntax: any dotted key `a.b.c=value`
(`training.epochs`, `fixmatch.threshold`, `adaptive.tau_min`, `seed`, `device`, ...).
Both `k=v` and `--k=v` are accepted.

---

## 6. Experiment suites

```bash
# smoke: 1-epoch tiny check for all 3 methods (CPU OK)
python scripts/run_experiments.py --suite smoke

# quick: short real-data demo, 30 epochs × 5 runs (WRN-28-2, full CIFAR-10)
python scripts/run_experiments.py --suite quick --epochs 30

# core: full comparison (default 200 epochs from YAML)
#   sup_250 | fm_t095 | fm_t090 | fm_t085 | afm_tmin080_tmax095
python scripts/run_experiments.py --suite core --epochs 200 --seed 42 --outbase outputs/full_s42

# thresholds: FixMatch τ sweep {0.80, 0.85, 0.90, 0.95, 0.99}
python scripts/run_experiments.py --suite thresholds --epochs 100 --seed 42
```

Full campaign used in this repo (3 seeds + ablation):

```bash
bash run_full.sh
# for SEED in 42 123 2026: suite core, 200ep, outbase outputs/full_s$SEED
# + seed-42 ablation: adaptive tau_min ∈ {0.70, 0.90} (tau_max fixed 0.95)
```

Aggregate printed results:

```bash
python aggregate.py
```

---

## 7. Evaluate & plots

```bash
# evaluate a saved checkpoint
python scripts/evaluate.py --checkpoint outputs/<run>/checkpoint_best.pt
# → prints accuracy/macro-F1, writes eval_results.json next to checkpoint

# figures P1-P3 + per-class P4 + adaptive P5 + summary CSV P6
python scripts/make_plots.py --runs outputs/run1 outputs/run2 --outdir report/figures

# full-campaign figures P1-P7 + report/results/full_results.csv
python make_full_figs.py

# tests (pure unit tests run without data; smoke_train needs CIFAR download)
pytest -q
pytest -q tests/test_split.py tests/test_fixmatch_mask.py \
  tests/test_adaptive_threshold.py tests/test_losses.py tests/test_checkpoint.py
```

Figures:

- P1 test accuracy vs epoch
- P2 pseudo-label coverage vs epoch
- P3 pseudo-label precision vs epoch
- P4 per-class acceptance rate (last epoch)
- P5 adaptive thresholds `τ_c` vs epoch
- P6 fixed-threshold trade-off (acc / precision / coverage)
- P7 confusion matrices

---

## 8. Config reference

| Key | Default (fixmatch) | Meaning |
|---|---|---|
| `method` | `fixmatch` | `supervised` / `fixmatch` / `adaptive_fixmatch` |
| `seed` | `42` | controls balanced 25/class split + shuffling |
| `data.num_labeled` | `250` | must be divisible by 10 |
| `data.include_labeled_in_unlabeled` | `true` | standard FixMatch: unlabeled pool = full 50k |
| `data.subset_train / subset_test` | `null` | set small ints for smoke/debug |
| `model.depth / widen_factor / dropout` | `28 / 2 / 0.0` | WRN-28-2; smoke uses 16-1 |
| `training.batch_size_labeled / mu` | `64 / 7` | unlabeled batch = 64×7 = 448 |
| `training.lr / momentum / weight_decay` | `0.03 / 0.9 / 5e-4` | SGD + Nesterov + cosine |
| `training.epochs / steps_per_epoch` | `200 / null` | null → one pass over unlabeled pool per epoch |
| `training.lambda_u` | `1.0` | unsupervised weight |
| `training.amp` | `true` | mixed precision on CUDA only |
| `fixmatch.threshold` | `0.95` | fixed τ (ignored by adaptive method) |
| `adaptive.enabled / beta` | `false / 0.9` | EMA decay |
| `adaptive.tau_min / tau_max` | `0.80 / 0.95` | clamp range |
| `adaptive.warmup_fraction` | `0.10` | first 10% steps return `tau_max` |
| `logging.log_every_steps / eval_every_epochs / save_every_epochs` | `50 / 1 / 10` | cadence |
| `device` | `cuda` | `cpu` for smoke |

---

## 9. Results (frozen 200-epoch campaign, `fetch/full/full_results.csv`)

Best test accuracy per run (final-epoch values in CSV):

| Method | seed 42 | seed 123 | seed 2026 |
|---|---|---|---|
| sup_250 | 0.4224 | 0.4358 | 0.4024 |
| fm_t095_250 (τ=0.95) | 0.7976 | 0.7923 | 0.8082 |
| fm_t090_250 (τ=0.90) | 0.8209 | 0.8396 | 0.7926 |
| fm_t085_250 (τ=0.85) | 0.8267 | 0.8259 | 0.8274 |
| afm_tmin080 (adaptive [0.80,0.95]) | 0.8023 | 0.7942 | 0.8154 |

Reading: lowering fixed τ 0.95 → 0.85 raises coverage (~81% → ~91%) at a small
precision cost (~87% → ~86%), net test-accuracy gain. The simple EMA adaptive
variant lands between fm_t095 and fm_t090 — it does not beat a well-tuned fixed
threshold in this setup, which is itself an informative result. See
`fetch/full/figures_full/` (P1–P7) and `aggregate.py` for mean±std over seeds.

---

## 10. Implementation notes

- Pseudo-labels from the **weak** view under `torch.no_grad()`; loss on the
  **strong** view (flip+crop+RandAugment).
- Balanced split: 25 labeled images per class via `np.random.RandomState(seed)`;
  saved to `split.json` for reproducibility.
- Supervised baseline uses a fair step budget: one "epoch" = one pass over the
  unlabeled pool (cycled labeled loader), same optimizer steps as SSL methods.
- Adaptive EMA updated from **all** weak predictions grouped by *predicted*
  class; classes with no predictions in a batch are skipped (no decay to zero).
- `metrics.csv` logs per-epoch loss, LR, test acc, macro-F1, PL precision/coverage,
  plus `tau_class_*` / `ema_conf_class_*` for adaptive runs.
- `per_class_metrics.csv` + `thresholds.csv` give class-level acceptance,
  precision, test accuracy, threshold, EMA confidence per epoch.
- Checkpoint `*_best.pt` tracks best test accuracy; resume restores model,
  optimizer, scheduler step, and adaptive EMA state.
- Known limits: single-dataset (CIFAR-10) study; no distribution alignment,
  no contrastive/self-supervised warm-up; RandAugment via torchvision
  (`num_ops=2, magnitude=10`).

---

## 11. Tests

```bash
pytest -q
```

- `test_split`: balanced 25/class, determinism, labeled⊆unlabeled option
- `test_fixmatch_mask`: mask + masked CE correctness
- `test_adaptive_threshold`: warm-up, bounds, EMA movement, per-sample lookup, state_dict round-trip
- `test_losses` / `test_checkpoint`: loss helper + save/load round-trip
- `test_smoke_train`: end-to-end 1-epoch for all 3 methods (downloads CIFAR-10)

---

## 12. Related work

- Sohn et al., *FixMatch: Simplifying Semi-Supervised Learning with Consistency
  and Confidence* (NeurIPS 2020).
- Zhang et al., *FlexMatch* / *FreeMatch* — curriculum / adaptive thresholds;
  acknowledged as the more advanced direction this repo does not attempt to match.

## License

No license file yet — add one (e.g. MIT) before public reuse.
