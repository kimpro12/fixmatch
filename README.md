# FixMatch + Ngưỡng Thích Ứng Theo Lớp trên CIFAR-10

Tái hiện (reproduction) **FixMatch** trên CIFAR-10 với 250 ảnh có nhãn, cộng thêm
biến thể **ngưỡng tin cậy thích ứng theo từng lớp (class-adaptive)** dùng EMA đơn giản.

> Ý tưởng xuất phát từ hạn chế của việc dùng một ngưỡng tin cậy cố định cho mọi lớp.
> Các phương pháp nâng cao liên quan (FlexMatch, FreeMatch) được ghi nhận tham khảo;
> repo này **không claim novelty**. Đây là bản triển khai sạch, chạy được,
> không rò rỉ nhãn (leakage-free).

---

## 1. Repo này làm gì

- **Supervised baseline**: WideresNet chỉ huấn luyện trên 250 ảnh CIFAR-10 có nhãn.
- **FixMatch**: consistency regularization + pseudo-label với một ngưỡng tin cậy
  cố định τ (mặc định 0.95).
- **Adaptive FixMatch**: giống FixMatch, nhưng mỗi lớp `c` có ngưỡng riêng `τ_c`
  được cập nhật bằng EMA theo độ tin cậy dự đoán.

Cấu hình mặc định (đúng regime ít nhãn của bài FixMatch gốc):

| Hạng mục | Giá trị |
|---|---|
| Dataset | CIFAR-10 (50k train / 10k test) |
| Có nhãn | 250 ảnh = 25 ảnh/lớp, chia cân bằng theo lớp, deterministic theo seed |
| Không nhãn | toàn bộ 50k ảnh train (kể cả ảnh đã có nhãn, đúng setup FixMatch chuẩn) |
| Model | WideResNet-28-2 |
| Optimizer | SGD lr=0.03, momentum=0.9, weight_decay=5e-4, Nesterov, lịch cosine |
| Batch | 64 có nhãn + 448 không nhãn (μ=7) |
| Epoch | 200 |
| Loss | `L = L_sup + λ_u * L_unsup`, `λ_u=1.0` |
| AMP | bật khi có CUDA |

### Công thức phương pháp

View yếu `x_w` → sinh pseudo-label, view mạnh `x_s` → ép nhất quán:

```
p = softmax(model(x_w))            # no-grad, không lan gradient
conf, y_hat = max(p, dim=1)
mask = conf >= tau                  # FixMatch: tau là số vô hướng
L_u = mean( CE(model(x_s), y_hat) * mask )
```

Bản Adaptive thay số vô hướng `tau` bằng ngưỡng riêng từng lớp:

```
# EMA trên mọi dự đoán yếu, gom theo lớp DỰ ĐOÁN (lớp nào không xuất hiện thì bỏ qua)
s_c <- beta * s_c + (1 - beta) * mean(conf | y_hat == c)

# chuẩn hoá về đoạn [tau_min, tau_max], 10% step đầu warm-up ở tau_max
r_c = s_c / max_j(s_j)
tau_c = tau_min + (tau_max - tau_min) * r_c
mask_i = conf_i >= tau_{y_hat_i}
```

Mặc định: `beta=0.9, tau_min=0.80, tau_max=0.95, warmup_fraction=0.10`.

Cam kết không rò rỉ: nhãn thật của tập không nhãn (`y_gt`) mà loader trả về
chỉ dùng để tính **phân tích** `pl_precision / pl_coverage / per-class`,
không bao giờ dùng cho loss hay cập nhật ngưỡng.

---

## 2. Cấu trúc thư mục

```
.
├── configs/
│   ├── supervised_cifar10.yaml          # baseline supervised, 200ep, WRN-28-2
│   ├── fixmatch_cifar10.yaml            # FixMatch τ=0.95
│   ├── adaptive_fixmatch_cifar10.yaml   # Adaptive τ∈[0.80,0.95], β=0.9
│   └── smoke.yaml                       # smoke test nhẹ (WRN-16-1, 1ep, 5 step/ep)
├── src/
│   ├── data/
│   │   ├── cifar10.py        # build_loaders(), UnlabeledWrapper -> (weak, strong, gt)
│   │   ├── splits.py         # make_balanced_split(), save_split()
│   │   └── transforms.py     # weak: flip+crop | strong: +RandAugment | eval: normalize
│   ├── models/
│   │   └── wideresnet.py     # WideResNet-28-2 (stem cho CIFAR), build_model(cfg)
│   ├── ssl/
│   │   ├── fixmatch.py            # generate_pseudo_labels(), fixed_threshold_mask(), compute_fixmatch_loss()
│   │   ├── adaptive_threshold.py  # lớp AdaptiveThreshold dùng EMA
│   │   └── losses.py              # supervised_loss()
│   ├── training/
│   │   ├── trainer.py      # train_one_run(): chia split, vòng lặp, AMP, cosine, CSV, checkpoint
│   │   ├── checkpoint.py   # save/load (model, optim, sched, state adaptive, best_acc)
│   │   └── evaluator.py    # đường đánh giá trên tập test
│   ├── metrics/
│   │   ├── classification.py  # accuracy, macro-F1, acc từng lớp, confusion matrix
│   │   └── ssl_metrics.py     # pseudo_label_stats(): acceptance-rate, mean-conf
│   └── utils/
│       ├── config.py   # YAML + override CLI dạng chấm (training.epochs=100 ...)
│       ├── seed.py     # set_seed(), resolve_device()
│       └── logging.py
├── scripts/
│   ├── train.py            # chạy 1 run: --config + k=v override + --resume + --outdir + --debug
│   ├── run_experiments.py  # các suite: smoke | quick | core | thresholds
│   ├── evaluate.py         # đánh giá 1 checkpoint trên test CIFAR-10
│   └── make_plots.py       # vẽ hình P1-P6 từ metrics.csv
├── tests/
│   ├── test_split.py
│   ├── test_fixmatch_mask.py
│   ├── test_adaptive_threshold.py
│   ├── test_losses.py
│   ├── test_checkpoint.py
│   └── test_smoke_train.py   # train end-to-end 1 epoch (cần tải CIFAR)
├── aggregate.py        # tổng hợp outputs/full_s*/metrics.csv (best/final/mean±std)
├── make_full_figs.py   # vẽ P1-P7 + full_results.csv cho chiến dịch 200 epoch
├── run_full.sh         # chiến dịch full: 3 seed × core + ablation tau_min (seed 42)
├── fetch/full/         # kết quả + hình đã đóng băng của chiến dịch 200ep hoàn chỉnh
└── requirements.txt
```

Mỗi thư mục output của 1 run gồm:

```
config.yaml  metadata.json  split.json  metrics.csv  per_class_metrics.csv
thresholds.csv  checkpoint_last.pt  checkpoint_best.pt  confusion_matrix_last.json
```

---

## 3. Cài đặt

```bash
pip install -r requirements.txt
```

Cần: `torch torchvision numpy pandas matplotlib scikit-learn tqdm pyyaml pytest tensorboard`

Nên dùng Python 3.10+. Nên có GPU cho run 200 epoch
(WRN-28-2, B=64, μ=7 → ~448 ảnh không nhãn/batch, tốn ~8–12 GB VRAM).
Nếu tràn VRAM thì giảm `training.batch_size_labeled` / `training.mu`.

---

## 4. Dữ liệu

CIFAR-10 tự tải về `./data` ở lần chạy đầu (qua torchvision).

```bash
# đổi chỗ lưu data
python scripts/train.py --config configs/fixmatch_cifar10.yaml data.root=/path/to/data
```

---

## 5. Chạy nhanh

```bash
# 1) baseline supervised (250 nhãn)
python scripts/train.py --config configs/supervised_cifar10.yaml

# 2) FixMatch τ=0.95
python scripts/train.py --config configs/fixmatch_cifar10.yaml

# 3) Adaptive FixMatch τ∈[0.80, 0.95]
python scripts/train.py --config configs/adaptive_fixmatch_cifar10.yaml

# outdir riêng + override tham số
python scripts/train.py --config configs/fixmatch_cifar10.yaml \
  --outdir outputs/my_run training.epochs=100 fixmatch.threshold=0.90 seed=123

# chạy tiếp từ checkpoint
python scripts/train.py --config configs/fixmatch_cifar10.yaml \
  --resume outputs/<run>/checkpoint_last.pt

# log chi tiết từng step để debug
python scripts/train.py --config configs/smoke.yaml --debug
```

Cú pháp override: khoá dạng chấm `a.b.c=value`
(`training.epochs`, `fixmatch.threshold`, `adaptive.tau_min`, `seed`, `device`, ...).
Chấp nhận cả `k=v` lẫn `--k=v`.

---

## 6. Các suite thí nghiệm

```bash
# smoke: kiểm tra nhanh 1 epoch cho cả 3 phương pháp (chạy CPU được)
python scripts/run_experiments.py --suite smoke

# quick: demo ngắn trên dữ liệu thật, 30 epoch × 5 run (WRN-28-2, full CIFAR-10)
python scripts/run_experiments.py --suite quick --epochs 30

# core: so sánh đầy đủ (mặc định 200 epoch theo YAML)
#   sup_250 | fm_t095 | fm_t090 | fm_t085 | afm_tmin080_tmax095
python scripts/run_experiments.py --suite core --epochs 200 --seed 42 --outbase outputs/full_s42

# thresholds: quét τ của FixMatch {0.80, 0.85, 0.90, 0.95, 0.99}
python scripts/run_experiments.py --suite thresholds --epochs 100 --seed 42
```

Chiến dịch full đã dùng trong repo này (3 seed + ablation):

```bash
bash run_full.sh
# với mỗi SEED trong 42 123 2026: chạy suite core, 200ep, ra outputs/full_s$SEED
# + ablation trên seed 42: adaptive tau_min ∈ {0.70, 0.90} (giữ tau_max=0.95)
```

In bảng tổng hợp kết quả:

```bash
python aggregate.py
```

---

## 7. Đánh giá & vẽ hình

```bash
# đánh giá 1 checkpoint đã lưu
python scripts/evaluate.py --checkpoint outputs/<run>/checkpoint_best.pt
# → in accuracy/macro-F1, ghi file eval_results.json cạnh checkpoint

# hình P1-P3 + P4 theo lớp + P5 adaptive + CSV tổng hợp P6
python scripts/make_plots.py --runs outputs/run1 outputs/run2 --outdir report/figures

# hình P1-P7 của chiến dịch full + report/results/full_results.csv
python make_full_figs.py

# test (unit test thuần chạy không cần data; riêng smoke_train cần tải CIFAR)
pytest -q
pytest -q tests/test_split.py tests/test_fixmatch_mask.py \
  tests/test_adaptive_threshold.py tests/test_losses.py tests/test_checkpoint.py
```

Ý nghĩa các hình:

- P1: test accuracy theo epoch
- P2: độ phủ pseudo-label (coverage) theo epoch
- P3: độ chính xác pseudo-label (precision) theo epoch
- P4: tỉ lệ được chấp nhận (acceptance rate) theo từng lớp (epoch cuối)
- P5: ngưỡng thích ứng `τ_c` theo epoch
- P6: đánh đổi khi đổi ngưỡng cố định (acc / precision / coverage)
- P7: confusion matrix

---

## 8. Bảng tham số config

| Khoá | Mặc định (fixmatch) | Ý nghĩa |
|---|---|---|
| `method` | `fixmatch` | `supervised` / `fixmatch` / `adaptive_fixmatch` |
| `seed` | `42` | quyết định split 25 ảnh/lớp + thứ tự shuffle |
| `data.num_labeled` | `250` | phải chia hết cho 10 |
| `data.include_labeled_in_unlabeled` | `true` | FixMatch chuẩn: pool không nhãn = cả 50k |
| `data.subset_train / subset_test` | `null` | đặt số nhỏ để smoke/debug |
| `model.depth / widen_factor / dropout` | `28 / 2 / 0.0` | WRN-28-2; smoke dùng 16-1 |
| `training.batch_size_labeled / mu` | `64 / 7` | batch không nhãn = 64×7 = 448 |
| `training.lr / momentum / weight_decay` | `0.03 / 0.9 / 5e-4` | SGD + Nesterov + cosine |
| `training.epochs / steps_per_epoch` | `200 / null` | null → 1 epoch = 1 lượt qua pool không nhãn |
| `training.lambda_u` | `1.0` | trọng số loss không nhãn |
| `training.amp` | `true` | mixed precision, chỉ có tác dụng trên CUDA |
| `fixmatch.threshold` | `0.95` | τ cố định (bản adaptive không dùng) |
| `adaptive.enabled / beta` | `false / 0.9` | hệ số suy giảm EMA |
| `adaptive.tau_min / tau_max` | `0.80 / 0.95` | đoạn kẹp ngưỡng |
| `adaptive.warmup_fraction` | `0.10` | 10% step đầu trả về `tau_max` |
| `logging.log_every_steps / eval_every_epochs / save_every_epochs` | `50 / 1 / 10` | nhịp log/đánh giá/lưu |
| `device` | `cuda` | smoke dùng `cpu` |

---

## 9. Kết quả (chiến dịch 200 epoch đã đóng băng, `fetch/full/full_results.csv`)

Accuracy tốt nhất của từng run (giá trị epoch cuối xem trong CSV):

| Phương pháp | seed 42 | seed 123 | seed 2026 |
|---|---|---|---|
| sup_250 | 0.4224 | 0.4358 | 0.4024 |
| fm_t095_250 (τ=0.95) | 0.7976 | 0.7923 | 0.8082 |
| fm_t090_250 (τ=0.90) | 0.8209 | 0.8396 | 0.7926 |
| fm_t085_250 (τ=0.85) | 0.8267 | 0.8259 | 0.8274 |
| afm_tmin080 (adaptive [0.80,0.95]) | 0.8023 | 0.7942 | 0.8154 |

Cách đọc: hạ τ cố định 0.95 → 0.85 thì độ phủ tăng (~81% → ~91%) với chi phí
precision giảm nhẹ (~87% → ~86%), kết quả test accuracy tăng ròng. Bản adaptive
EMA đơn giản nằm giữa fm_t095 và fm_t090 — không vượt được ngưỡng cố định đã
tune kỹ, bản thân nó cũng là một kết quả đáng ghi nhận. Xem hình P1–P7 trong
`fetch/full/figures_full/` và dùng `aggregate.py` để tính mean±std qua các seed.

---

## 10. Ghi chú triển khai

- Pseudo-label sinh từ view **yếu** dưới `torch.no_grad()`; loss tính trên view
  **mạnh** (flip+crop+RandAugment).
- Split cân bằng: 25 ảnh có nhãn mỗi lớp bằng `np.random.RandomState(seed)`;
  lưu vào `split.json` để tái lập.
- Baseline supervised dùng ngân sách step công bằng: một "epoch" = một lượt qua
  pool không nhãn (lặp vòng loader có nhãn), số step optimizer bằng các bản SSL.
- EMA của adaptive cập nhật từ **mọi** dự đoán yếu, gom theo lớp *dự đoán*;
  lớp nào không xuất hiện trong batch thì bỏ qua (không suy giảm về 0).
- `metrics.csv` log loss, LR, test acc, macro-F1, PL precision/coverage mỗi epoch,
  cộng thêm `tau_class_*` / `ema_conf_class_*` cho run adaptive.
- `per_class_metrics.csv` + `thresholds.csv` cho chi tiết từng lớp: acceptance,
  precision, test accuracy, ngưỡng, EMA confidence theo epoch.
- Checkpoint `*_best.pt` bám theo test accuracy tốt nhất; resume khôi phục model,
  optimizer, bước scheduler và state EMA của adaptive.
- Giới hạn đã biết: mới chỉ nghiên cứu trên CIFAR-10; chưa có distribution
  alignment, chưa warm-up contrastive/self-supervised; RandAugment dùng của
  torchvision (`num_ops=2, magnitude=10`).

---

## 11. Test

```bash
pytest -q
```

- `test_split`: đủ 25 ảnh/lớp, deterministic, tuỳ chọn labeled⊆unlabeled
- `test_fixmatch_mask`: đúng mask + CE có mask
- `test_adaptive_threshold`: warm-up, chặn biên, EMA dịch chuyển, tra cứu từng mẫu, vòng state_dict
- `test_losses` / `test_checkpoint`: hàm loss + vòng save/load
- `test_smoke_train`: end-to-end 1 epoch cho cả 3 phương pháp (cần tải CIFAR-10)

---

## 12. Tài liệu liên quan

- Sohn et al., *FixMatch: Simplifying Semi-Supervised Learning with Consistency
  and Confidence* (NeurIPS 2020).
- Zhang et al., *FlexMatch* / *FreeMatch* — ngưỡng thích ứng theo curriculum;
  được ghi nhận là hướng nâng cao mà repo này không cố sánh bằng.

## Giấy phép

Chưa có file license — hãy thêm một file (ví dụ MIT) trước khi cho dùng lại công khai.
