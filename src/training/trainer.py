"""Main training loop for supervised / fixmatch / adaptive_fixmatch."""
from __future__ import annotations
import copy
import csv
import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from src.data.cifar10 import build_loaders, CIFAR10_CLASSES
from src.data.splits import make_balanced_split, save_split
from src.models.wideresnet import build_model
from src.ssl.fixmatch import generate_pseudo_labels, fixed_threshold_mask, compute_fixmatch_loss
from src.ssl.adaptive_threshold import AdaptiveThreshold
from src.metrics.classification import evaluate_model
from src.metrics.ssl_metrics import pseudo_label_stats
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.utils.seed import set_seed, resolve_device


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def _get_targets(root: str):
    base = CIFAR10(root=root, train=True, download=True, transform=None)
    return list(base.targets)


def train_one_run(cfg: dict, out_dir: str, resume: str | None = None, debug: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = resolve_device(cfg.get("device", "cuda"))
    method = cfg.get("method", "fixmatch")

    # ---- split ----
    root = cfg["data"].get("root", "./data")
    num_labeled = cfg["data"].get("num_labeled", 250)
    include_in_un = cfg["data"].get("include_labeled_in_unlabeled", True)
    targets = _get_targets(root)
    split = make_balanced_split(targets, num_labeled, 10, seed, include_in_un)
    save_split({k: split[k] for k in ("seed", "num_labeled", "indices_by_class")},
               os.path.join(out_dir, "split.json"))

    labeled_loader, unlabeled_loader, test_loader = build_loaders(
        cfg, split["labeled_indices"], split["unlabeled_indices"])

    model = build_model(cfg).to(device)
    opt_cfg = cfg["training"]
    optimizer = torch.optim.SGD(model.parameters(), lr=opt_cfg["lr"],
                                momentum=opt_cfg.get("momentum", 0.9),
                                weight_decay=opt_cfg.get("weight_decay", 5e-4),
                                nesterov=opt_cfg.get("nesterov", True))

    epochs = opt_cfg["epochs"]
    steps_per_epoch = opt_cfg.get("steps_per_epoch", None)
    if steps_per_epoch is None:
        if method == "supervised":
            # Fair budget: same optimizer steps as SSL methods (cycle labeled
            # data), i.e. one "epoch" = one pass over the unlabeled pool.
            import math as _m
            steps_per_epoch = max(1, _m.ceil(len(split["unlabeled_indices"]) /
                                             (opt_cfg["batch_size_labeled"] * opt_cfg.get("mu", 7))))
        else:
            steps_per_epoch = len(unlabeled_loader) if len(unlabeled_loader) > 0 else len(labeled_loader)
    total_steps = epochs * steps_per_epoch

    sched_name = opt_cfg.get("scheduler", "cosine")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps) \
        if sched_name == "cosine" else None

    use_amp = bool(opt_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # adaptive manager
    adapt_cfg = cfg.get("adaptive", {})
    adaptive = None
    if method == "adaptive_fixmatch" or adapt_cfg.get("enabled", False):
        warmup_fraction = adapt_cfg.get("warmup_fraction", 0.10)
        warmup_steps = int(warmup_fraction * total_steps)
        adaptive = AdaptiveThreshold(10, beta=adapt_cfg.get("beta", 0.9),
                                     tau_min=adapt_cfg.get("tau_min", 0.80),
                                     tau_max=adapt_cfg.get("tau_max", 0.95),
                                     warmup_steps=warmup_steps, device="cpu")
        method = "adaptive_fixmatch"

    fixed_threshold = cfg.get("fixmatch", {}).get("threshold", 0.95)
    lambda_u = opt_cfg.get("lambda_u", 1.0)
    log_every = cfg.get("logging", {}).get("log_every_steps", 50)
    eval_every = cfg.get("logging", {}).get("eval_every_epochs", 1)

    # save config + metadata
    import yaml
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    meta = {"seed": seed, "method": method,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_steps": total_steps, "steps_per_epoch": steps_per_epoch}
    try:
        import subprocess
        meta["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        meta["git_commit"] = None
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    perclass_path = os.path.join(out_dir, "per_class_metrics.csv")
    thresholds_path = os.path.join(out_dir, "thresholds.csv")

    start_epoch, global_step, best_acc = 0, 0, 0.0
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_acc = ckpt.get("best_metric", 0.0)
        if adaptive is not None and ckpt.get("adaptive_threshold_state") is not None:
            adaptive.load_state_dict(ckpt["adaptive_threshold_state"])

    lab_iter = _cycle(labeled_loader)
    unl_iter = _cycle(unlabeled_loader) if method != "supervised" else None

    fieldnames = ["epoch", "global_step", "train_loss_total", "train_loss_supervised",
                  "train_loss_unsupervised", "pseudo_label_acceptance_rate",
                  "mean_pseudo_label_confidence", "learning_rate", "test_accuracy",
                  "macro_f1", "pl_precision", "pl_coverage"]
    if adaptive is not None:
        for c in range(10):
            fieldnames.append(f"tau_class_{c}")
        for c in range(10):
            fieldnames.append(f"ema_conf_class_{c}")

    with open(metrics_path, "a" if resume else "w", newline="") as f:
        if not resume or os.path.getsize(metrics_path) == 0:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # per-class + thresholds headers
    if not resume:
        with open(perclass_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["epoch", "class_id", "class_name", "predicted_count",
                                          "accepted_count", "acceptance_rate", "pseudo_label_precision",
                                          "test_accuracy", "threshold", "ema_confidence"]).writeheader()
        with open(thresholds_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["epoch", "global_step"] + [f"tau_{c}" for c in range(10)]).writeheader()

    for epoch in range(start_epoch, epochs):
        model.train()
        sum_ls = sum_lu = sum_tot = 0.0
        sum_acc_rate = sum_conf = 0.0
        # epoch-level PL analysis accumulators (group by predicted class)
        pred_counts = torch.zeros(10, dtype=torch.long)
        acc_counts = torch.zeros(10, dtype=torch.long)
        acc_correct = torch.zeros(10, dtype=torch.long)
        tot_correct_acc = tot_acc = tot_all = 0
        n_batches = 0

        for step_in_ep in range(steps_per_epoch):
            x_l, y_l = next(lab_iter)
            x_l, y_l = x_l.to(device), y_l.to(device)
            if method == "supervised":
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits_l = model(x_l)
                    loss_s = F.cross_entropy(logits_l, y_l)
                    loss = loss_s
                    loss_u_v = 0.0
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                sum_ls += loss_s.item()
                sum_tot += loss.item()
                n_batches += 1
                continue

            x_uw, x_us, y_gt = next(unl_iter)
            x_uw, x_us = x_uw.to(device), x_us.to(device)
            y_gt_d = y_gt.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_l = model(x_l)
                loss_s = F.cross_entropy(logits_l, y_l)
                with torch.no_grad():
                    logits_uw = model(x_uw)
                    confidence, pseudo_labels = generate_pseudo_labels(logits_uw)
                if method == "adaptive_fixmatch":
                    assert adaptive is not None
                    adaptive.update(confidence.detach().cpu(), pseudo_labels.detach().cpu())
                    sample_th = adaptive.sample_thresholds(pseudo_labels, global_step).to(device)
                else:
                    sample_th = torch.full_like(confidence, fixed_threshold)
                mask = confidence.ge(sample_th)
                logits_us = model(x_us)
                loss_u = compute_fixmatch_loss(logits_us, pseudo_labels, mask)
                loss = loss_s + lambda_u * loss_u
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            # analysis-only stats (never influence training)
            with torch.no_grad():
                st = pseudo_label_stats(pseudo_labels, confidence, mask, y_gt_d)
                sum_ls += loss_s.item()
                sum_lu += loss_u.item()
                sum_tot += loss.item()
                sum_acc_rate += st["acceptance_rate"]
                sum_conf += st["mean_confidence"]
                tot_correct_acc += st["accepted_correct"]
                tot_acc += st["accepted"]
                tot_all += st["total"]
                for c in range(10):
                    m_pred = (pseudo_labels == c)
                    pred_counts[c] += m_pred.sum().item()
                    m_acc = (m_pred & mask)
                    acc_counts[c] += m_acc.sum().item()
                    acc_correct[c] += (m_acc & (pseudo_labels == y_gt_d)).sum().item()
                n_batches += 1
                if debug or (global_step % max(log_every, 1) == 0 and step_in_ep == 0):
                    cur_th = adaptive.thresholds(global_step) if adaptive else None
                    print(f"[ep{epoch} st{global_step}] Ls={loss_s.item():.3f} Lu={loss_u.item():.3f} "
                          f"acc_rate={st['acceptance_rate']:.3f} meanconf={st['mean_confidence']:.3f} "
                          f"thr_min={cur_th.min().item():.3f} thr_max={cur_th.max().item():.3f}" if cur_th is not None
                          else f"[ep{epoch} st{global_step}] Ls={loss_s.item():.3f} Lu={loss_u.item():.3f} "
                               f"acc_rate={st['acceptance_rate']:.3f} meanconf={st['mean_confidence']:.3f}",
                          flush=True)

        # epoch aggregates
        avg_ls = sum_ls / max(n_batches, 1)
        avg_lu = sum_lu / max(n_batches, 1) if method != "supervised" else 0.0
        avg_tot = sum_tot / max(n_batches, 1)
        avg_acc = sum_acc_rate / max(n_batches, 1) if method != "supervised" else 0.0
        avg_conf = sum_conf / max(n_batches, 1) if method != "supervised" else 0.0
        pl_prec = tot_correct_acc / tot_acc if tot_acc > 0 else 0.0
        pl_cov = tot_acc / tot_all if tot_all > 0 else 0.0
        lr_now = optimizer.param_groups[0]["lr"]

        test_acc, macro_f1 = 0.0, 0.0
        per_class_test = [0.0] * 10
        cm = None
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            res, _, _ = evaluate_model(model, test_loader, device)
            test_acc, macro_f1 = res["accuracy"], res["macro_f1"]
            per_class_test = res["per_class_accuracy"]
            cm = res["confusion_matrix"]

        row = {"epoch": epoch + 1, "global_step": global_step, "train_loss_total": avg_tot,
               "train_loss_supervised": avg_ls, "train_loss_unsupervised": avg_lu,
               "pseudo_label_acceptance_rate": avg_acc if method != "supervised" else 0.0,
               "mean_pseudo_label_confidence": avg_conf if method != "supervised" else 0.0,
               "learning_rate": lr_now, "test_accuracy": test_acc, "macro_f1": macro_f1,
               "pl_precision": pl_prec if method != "supervised" else 0.0,
               "pl_coverage": pl_cov if method != "supervised" else 0.0}
        if adaptive is not None:
            th = adaptive.thresholds(global_step)
            for c in range(10):
                row[f"tau_class_{c}"] = float(th[c].item())
                row[f"ema_conf_class_{c}"] = float(adaptive.class_ema_confidence[c].item())
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)

        # per-class + thresholds rows
        if method != "supervised":
            th = (adaptive.thresholds(global_step) if adaptive is not None
                  else torch.full((10,), fixed_threshold))
            ema = (adaptive.class_ema_confidence if adaptive is not None
                   else torch.full((10,), float("nan")))
            with open(perclass_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "class_id", "class_name", "predicted_count",
                                                  "accepted_count", "acceptance_rate", "pseudo_label_precision",
                                                  "test_accuracy", "threshold", "ema_confidence"])
                for c in range(10):
                    pc, ac = int(pred_counts[c]), int(acc_counts[c])
                    w.writerow({"epoch": epoch + 1, "class_id": c, "class_name": CIFAR10_CLASSES[c],
                                "predicted_count": pc, "accepted_count": ac,
                                "acceptance_rate": (ac / pc if pc > 0 else 0.0),
                                "pseudo_label_precision": (float(acc_correct[c]) / ac if ac > 0 else 0.0),
                                "test_accuracy": per_class_test[c], "threshold": float(th[c].item()),
                                "ema_confidence": float(ema[c].item()) if torch.isfinite(ema[c]) else ""})
            with open(thresholds_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "global_step"] + [f"tau_{c}" for c in range(10)])
                r = {"epoch": epoch + 1, "global_step": global_step}
                for c in range(10):
                    r[f"tau_{c}"] = float(th[c].item())
                w.writerow(r)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        adapt_state = adaptive.state_dict() if adaptive is not None else None
        save_every = cfg.get("logging", {}).get("save_every_epochs", 10)
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            save_checkpoint(os.path.join(out_dir, "checkpoint_last.pt"), epoch, global_step,
                            model, optimizer, scheduler, adapt_state, best_acc, cfg)
        if is_best:
            save_checkpoint(os.path.join(out_dir, "checkpoint_best.pt"), epoch, global_step,
                            model, optimizer, scheduler, adapt_state, best_acc, cfg)
        print(f"Epoch {epoch+1}/{epochs} loss={avg_tot:.3f} Ls={avg_ls:.3f} Lu={avg_lu:.3f} "
              f"test_acc={test_acc*100:.2f} pl_prec={pl_prec*100:.2f} pl_cov={pl_cov*100:.2f} "
              + (f"thr_min={min(row[f'tau_class_{c}'] for c in range(10)):.3f} "
                 f"thr_max={max(row[f'tau_class_{c}'] for c in range(10)):.3f}" if adaptive else f"thr={fixed_threshold:.3f}"
                 if method != "supervised" else "supervised"),
              flush=True)

    # confusion matrix save
    if cm is not None:
        with open(os.path.join(out_dir, "confusion_matrix_last.json"), "w") as f:
            json.dump(cm, f)
    return {"best_test_accuracy": best_acc, "output_dir": out_dir}
