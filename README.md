# FixMatch với ngưỡng thích ứng theo lớp trên CIFAR-10

Repo này tái hiện **FixMatch** trên CIFAR-10 với 250 ảnh có nhãn, rồi thử thêm một biến thể nhỏ: thay vì dùng chung một ngưỡng tin cậy cho mọi lớp, mỗi lớp có ngưỡng riêng và được cập nhật dần bằng EMA.

Động lực khá đơn giản. Một ngưỡng cố định như 0.95 có thể quá khắt khe với những lớp mà mô hình còn chưa tự tin, khiến các lớp đó gần như không nhận được pseudo-label nào. FlexMatch và FreeMatch đã giải quyết vấn đề này bài bản hơn nhiều. Repo này không nhằm cạnh tranh với chúng và cũng không claim novelty; mục tiêu chỉ là có một bản cài đặt gọn, chạy được, không rò rỉ nhãn, để xem một cách làm đơn giản thì đi được tới đâu.

---

## 1. Tổng quan

Repo so sánh ba phương pháp:

- **Supervised baseline**: WideResNet chỉ học trên 250 ảnh có nhãn.
- **FixMatch**: pseudo-label kết hợp consistency regularization, dùng một ngưỡng cố định τ (mặc định 0.95).
- **Adaptive FixMatch**: giống FixMatch, nhưng mỗi lớp `c` có một ngưỡng `τ_c` riêng.

Cấu hình mặc định bám theo setting ít nhãn trong bài FixMatch gốc:

| Hạng mục | Giá trị |
|---|---|
| Dataset | CIFAR-10 (50k ảnh train, 10k ảnh test) |
| Ảnh có nhãn | 250 ảnh, 25 ảnh mỗi lớp, chọn cố định theo seed |
| Ảnh không nhãn | cả 50k ảnh train (gồm luôn 250 ảnh có nhãn, như setup chuẩn của FixMatch) |
| Model | WideResNet-28-2 |
| Optimizer | SGD, lr=0.03, momentum=0.9, weight_decay=5e-4, Nesterov, lịch cosine |
| Batch | 64 ảnh có nhãn + 448 ảnh không nhãn (μ=7) |
| Số epoch | 200 |
| Loss | `L = L_sup + λ_u * L_unsup`, với `λ_u = 1.0` |
| AMP | tự bật khi có CUDA |

### Phương pháp hoạt động thế nào

Với mỗi ảnh không nhãn, mô hình nhìn một bản tăng cường nhẹ `x_w` để đoán nhãn, sau đó bị buộc phải dự đoán đúng nhãn đó trên một bản tăng cường mạnh `x_s`. Chỉ những dự đoán đủ tự tin mới được tính vào loss:

```
p = softmax(model(x_w))            # chạy dưới no_grad, không lan gradient
conf, y_hat = max(p, dim=1)
mask = conf >= tau                  # FixMatch: tau là một số chung
L_u = mean( CE(model(x_s), y_hat) * mask )
```

Ở bản adaptive, `tau` không còn là một số chung mà được tính riêng cho từng lớp. Mỗi lớp giữ một giá trị EMA `s_c`, phản ánh độ tin cậy trung bình của các dự đoán rơi vào lớp đó. Lớp có `s_c` cao nhất nhận ngưỡng `tau_max`; các lớp còn lại được hạ ngưỡng theo tỉ lệ, nhưng không bao giờ thấp hơn `tau_min`:

```
# cập nhật EMA, gom theo lớp được dự đoán (lớp nào không có trong batch thì giữ nguyên)
s_c <- beta * s_c + (1 - beta) * mean(conf | y_hat == c)

# quy về đoạn [tau_min, tau_max]
r_c   = s_c / max_j(s_j)
tau_c = tau_min + (tau_max - tau_min) * r_c
mask_i = conf_i >= tau_{y_hat_i}
```

Trong 10% số step đầu, mọi lớp đều dùng `tau_max`, để tránh nhận pseudo-label kém chất lượng khi mô hình còn chưa ổn định. Giá trị mặc định: `beta=0.9`, `tau_min=0.80`, `tau_max=0.95`, `warmup_fraction=0.10`.

**Về chuyện rò rỉ nhãn:** loader vẫn trả về nhãn thật `y_gt` của ảnh không nhãn, nhưng nhãn này chỉ được dùng để đo chất lượng pseudo-label (precision, coverage, thống kê theo lớp). Nó không bao giờ đi vào loss hay vào việc cập nhật ngưỡng.

---

## 2. Cấu trúc thư mục

```
.
├── configs/
│   ├── supervised_cifar10.yaml          # baseline supervised, 200 epoch, WRN-28-2
│   ├── fixmatch_cifar10.yaml            # FixMatch, τ=0.95
│   ├── adaptive_fixmatch_cifar10.yaml   # Adaptive, τ trong [0.80, 0.95], β=0.9
│   └── smoke.yaml                       # chạy thử nhẹ: WRN-16-1, 1 epoch, 5 step/epoch
├── src/
│   ├── data/
│   │   ├── cifar10.py        # build_loaders(); UnlabeledWrapper trả về (weak, strong, gt)
│   │   ├── splits.py         # make_balanced_split(), save_split()
│   │   └── transforms.py     # weak: flip + crop | strong: thêm RandAugment | eval: chỉ normalize
│   ├── models/
│   │   └── wideresnet.py     # WideResNet-28-2 với stem cho CIFAR, build_model(cfg)
│   ├── ssl/
│   │   ├── fixmatch.py            # generate_pseudo_labels(), fixed_threshold_mask(), compute_fixmatch_loss()
│   │   ├── adaptive_threshold.py  # class AdaptiveThreshold (EMA)
│   │   └── losses.py              # supervised_loss()
│   ├── training/
│   │   ├── trainer.py      # train_one_run(): chia split, vòng train, AMP, cosine, ghi CSV, checkpoint
│   │   ├── checkpoint.py   # lưu/nạp model, optimizer, scheduler, state adaptive, best_acc
│   │   └── evaluator.py    # đánh giá trên tập test
│   ├── metrics/
│   │   ├── classification.py  # accuracy, macro-F1, accuracy từng lớp, confusion matrix
│   │   └── ssl_metrics.py     # pseudo_label_stats(): acceptance rate, độ tin cậy trung bình
│   └── utils/
│       ├── config.py   # đọc YAML + override từ CLI (vd. training.epochs=100)
│       ├── seed.py     # set_seed(), resolve_device()
│       └── logging.py
├── scripts/
│   ├── train.py            # chạy một run: --config, override k=v, --resume, --outdir, --debug
│   ├── run_experiments.py  # các suite: smoke | quick | core | thresholds
│   ├── evaluate.py         # đánh giá một checkpoint trên tập test
│   └── make_plots.py       # vẽ hình P1–P6 từ metrics.csv
├── tests/
│   ├── test_split.py
│   ├── test_fixmatch_mask.py
│   ├── test_adaptive_threshold.py
│   ├── test_losses.py
│   ├── test_checkpoint.py
│   └── test_smoke_train.py   # train thử 1 epoch từ đầu đến cuối (cần tải CIFAR)
├── aggregate.py        # gom outputs/full_s*/metrics.csv thành bảng best/final/mean±std
├── make_full_figs.py   # vẽ P1–P7 và xuất full_results.csv cho đợt chạy 200 epoch
├── run_full.sh         # đợt chạy đầy đủ: 3 seed × suite core + ablation tau_min (seed 42)
├── fetch/full/         # kết quả và hình đã chốt của đợt chạy 200 epoch
└── requirements.txt
```

Mỗi run sẽ tạo một thư mục output gồm:

```
config.yaml  metadata.json  split.json  metrics.csv  per_class_metrics.csv
thresholds.csv  checkpoint_last.pt  checkpoint_best.pt  confusion_matrix_last.json
```

---

## 3. Cài đặt

```bash
pip install -r requirements.txt
```

Các thư viện cần có: `torch torchvision numpy pandas matplotlib scikit-learn tqdm pyyaml pytest tensorboard`.

Nên dùng Python 3.10 trở lên. Với run 200 epoch thì gần như bắt buộc phải có GPU: WRN-28-2 với batch 64 và μ=7 (tức 448 ảnh không nhãn mỗi batch) tốn khoảng 8–12 GB VRAM. Nếu bị tràn bộ nhớ, hãy giảm `training.batch_size_labeled` hoặc `training.mu`.

---

## 4. Dữ liệu

Lần chạy đầu tiên, CIFAR-10 sẽ được torchvision tự tải về `./data`. Muốn lưu ở chỗ khác thì override `data.root`:

```bash
python scripts/train.py --config configs/fixmatch_cifar10.yaml data.root=/path/to/data
```

---

## 5. Chạy thử

```bash
# 1) baseline supervised (250 nhãn)
python scripts/train.py --config configs/supervised_cifar10.yaml

# 2) FixMatch, τ=0.95
python scripts/train.py --config configs/fixmatch_cifar10.yaml

# 3) Adaptive FixMatch, τ trong [0.80, 0.95]
python scripts/train.py --config configs/adaptive_fixmatch_cifar10.yaml

# đặt thư mục output riêng và đổi vài tham số
python scripts/train.py --config configs/fixmatch_cifar10.yaml \
  --outdir outputs/my_run training.epochs=100 fixmatch.threshold=0.90 seed=123

# chạy tiếp từ checkpoint
python scripts/train.py --config configs/fixmatch_cifar10.yaml \
  --resume outputs/<run>/checkpoint_last.pt

# in log chi tiết từng step để debug
python scripts/train.py --config configs/smoke.yaml --debug
```

Override dùng khoá dạng chấm `a.b.c=value`, ví dụ `training.epochs`, `fixmatch.threshold`, `adaptive.tau_min`, `seed`, `device`. Viết `k=v` hay `--k=v` đều được.

---

## 6. Các bộ thí nghiệm

```bash
# smoke: kiểm tra nhanh cả 3 phương pháp trong 1 epoch, chạy được trên CPU
python scripts/run_experiments.py --suite smoke

# quick: bản demo ngắn trên dữ liệu thật, 5 run × 30 epoch (WRN-28-2, đủ CIFAR-10)
python scripts/run_experiments.py --suite quick --epochs 30

# core: so sánh đầy đủ, mặc định 200 epoch theo YAML
#   gồm sup_250 | fm_t095 | fm_t090 | fm_t085 | afm_tmin080_tmax095
python scripts/run_experiments.py --suite core --epochs 200 --seed 42 --outbase outputs/full_s42

# thresholds: quét ngưỡng FixMatch τ ∈ {0.80, 0.85, 0.90, 0.95, 0.99}
python scripts/run_experiments.py --suite thresholds --epochs 100 --seed 42
```

Kết quả trong repo đến từ đợt chạy sau:

```bash
bash run_full.sh
# chạy suite core 200 epoch cho từng seed 42, 123, 2026 (lưu ở outputs/full_s$SEED)
# kèm ablation trên seed 42: adaptive với tau_min ∈ {0.70, 0.90}, giữ tau_max=0.95
```

Sau đó in bảng tổng hợp bằng:

```bash
python aggregate.py
```

---

## 7. Đánh giá và vẽ hình

```bash
# đánh giá một checkpoint: in accuracy, macro-F1 và ghi eval_results.json cạnh checkpoint
python scripts/evaluate.py --checkpoint outputs/<run>/checkpoint_best.pt

# vẽ P1–P5 và xuất CSV tổng hợp cho P6
python scripts/make_plots.py --runs outputs/run1 outputs/run2 --outdir report/figures

# vẽ P1–P7 cho đợt chạy đầy đủ và xuất report/results/full_results.csv
python make_full_figs.py
```

Các hình gồm:

- **P1**: test accuracy theo epoch
- **P2**: coverage của pseudo-label (tỉ lệ ảnh không nhãn được nhận) theo epoch
- **P3**: precision của pseudo-label theo epoch
- **P4**: acceptance rate của từng lớp ở epoch cuối
- **P5**: ngưỡng `τ_c` của từng lớp theo epoch (bản adaptive)
- **P6**: đánh đổi giữa accuracy, precision và coverage khi thay đổi ngưỡng cố định
- **P7**: confusion matrix

---

## 8. Các tham số trong config

| Khoá | Mặc định (fixmatch) | Ý nghĩa |
|---|---|---|
| `method` | `fixmatch` | một trong `supervised`, `fixmatch`, `adaptive_fixmatch` |
| `seed` | `42` | quyết định cách chọn 25 ảnh/lớp và thứ tự shuffle |
| `data.num_labeled` | `250` | phải chia hết cho 10 |
| `data.include_labeled_in_unlabeled` | `true` | như FixMatch chuẩn: tập không nhãn gồm cả 50k ảnh |
| `data.subset_train / subset_test` | `null` | đặt số nhỏ khi muốn chạy thử hoặc debug |
| `model.depth / widen_factor / dropout` | `28 / 2 / 0.0` | WRN-28-2; config smoke dùng WRN-16-1 |
| `training.batch_size_labeled / mu` | `64 / 7` | batch không nhãn = 64 × 7 = 448 |
| `training.lr / momentum / weight_decay` | `0.03 / 0.9 / 5e-4` | SGD + Nesterov + lịch cosine |
| `training.epochs / steps_per_epoch` | `200 / null` | để `null` thì một epoch là một lượt qua toàn bộ tập không nhãn |
| `training.lambda_u` | `1.0` | trọng số của loss không nhãn |
| `training.amp` | `true` | mixed precision, chỉ có tác dụng trên CUDA |
| `fixmatch.threshold` | `0.95` | ngưỡng cố định τ (bản adaptive bỏ qua giá trị này) |
| `adaptive.enabled / beta` | `false / 0.9` | bật adaptive và hệ số EMA |
| `adaptive.tau_min / tau_max` | `0.80 / 0.95` | khoảng giới hạn của ngưỡng |
| `adaptive.warmup_fraction` | `0.10` | trong 10% số step đầu, mọi lớp dùng `tau_max` |
| `logging.log_every_steps / eval_every_epochs / save_every_epochs` | `50 / 1 / 10` | tần suất ghi log, đánh giá và lưu checkpoint |
| `device` | `cuda` | config smoke dùng `cpu` |

---

## 9. Kết quả

Số liệu dưới đây lấy từ đợt chạy 200 epoch đã chốt (`fetch/full/full_results.csv`). Mỗi ô là test accuracy tốt nhất của run đó; accuracy ở epoch cuối có trong file CSV.

| Phương pháp | seed 42 | seed 123 | seed 2026 |
|---|---|---|---|
| sup_250 | 0.4224 | 0.4358 | 0.4024 |
| fm_t095_250 (τ=0.95) | 0.7976 | 0.7923 | 0.8082 |
| fm_t090_250 (τ=0.90) | 0.8209 | 0.8396 | 0.7926 |
| fm_t085_250 (τ=0.85) | 0.8267 | 0.8259 | 0.8274 |
| afm_tmin080 (adaptive, [0.80, 0.95]) | 0.8023 | 0.7942 | 0.8154 |

Một vài nhận xét:

- Hạ ngưỡng cố định từ 0.95 xuống 0.85 giúp nhận thêm nhiều pseudo-label hơn (coverage tăng từ khoảng 81% lên 91%), trong khi precision chỉ giảm chút ít (khoảng 87% xuống 86%). Kết quả là accuracy tăng rõ, và cũng ổn định hơn giữa các seed.
- Bản adaptive dùng EMA đơn giản cho kết quả nằm giữa τ=0.95 và τ=0.90, tức là chưa vượt được một ngưỡng cố định được chọn kỹ. Đây vẫn là một kết quả có ích: nó cho thấy chỉ "hạ ngưỡng cho lớp khó" theo cách đơn giản thì chưa đủ.

Hình P1–P7 nằm trong `fetch/full/figures_full/`. Chạy `aggregate.py` để xem mean ± std qua các seed.

---

## 10. Một số chi tiết cài đặt

- Pseudo-label được sinh từ bản tăng cường nhẹ dưới `torch.no_grad()`; loss được tính trên bản tăng cường mạnh (flip + crop + RandAugment).
- 25 ảnh có nhãn mỗi lớp được chọn bằng `np.random.RandomState(seed)` và lưu vào `split.json`, nên cùng seed sẽ ra cùng một split.
- Để so sánh công bằng, baseline supervised được train với cùng số step như các bản SSL: một "epoch" vẫn là một lượt qua tập không nhãn, còn loader có nhãn được lặp vòng.
- EMA của bản adaptive được cập nhật từ toàn bộ dự đoán trên ảnh yếu, gom theo lớp được dự đoán. Nếu một lớp không xuất hiện trong batch, giá trị của nó giữ nguyên chứ không bị kéo dần về 0.
- `metrics.csv` ghi lại loss, learning rate, test accuracy, macro-F1, precision và coverage của pseudo-label sau mỗi epoch. Với run adaptive còn có thêm các cột `tau_class_*` và `ema_conf_class_*`.
- `per_class_metrics.csv` và `thresholds.csv` chứa số liệu chi tiết từng lớp theo epoch: acceptance rate, precision, test accuracy, ngưỡng và EMA confidence.
- `checkpoint_best.pt` là checkpoint có test accuracy cao nhất. Khi resume, model, optimizer, scheduler và state EMA của adaptive đều được khôi phục.
- Hạn chế hiện tại: mới chỉ thử trên CIFAR-10; chưa có distribution alignment; chưa có bước warm-up bằng self-supervised hay contrastive; RandAugment dùng bản của torchvision (`num_ops=2, magnitude=10`).

---

## 11. Test

```bash
# chạy toàn bộ test
pytest -q

# chỉ chạy các unit test không cần tải dữ liệu
pytest -q tests/test_split.py tests/test_fixmatch_mask.py \
  tests/test_adaptive_threshold.py tests/test_losses.py tests/test_checkpoint.py
```

Các test kiểm tra:

- `test_split`: mỗi lớp đúng 25 ảnh, cùng seed ra cùng kết quả, tuỳ chọn gộp ảnh có nhãn vào tập không nhãn hoạt động đúng
- `test_fixmatch_mask`: mask và cross-entropy có mask tính đúng
- `test_adaptive_threshold`: warm-up, giới hạn biên, EMA thay đổi đúng hướng, tra ngưỡng cho từng mẫu, lưu/nạp state_dict
- `test_losses`, `test_checkpoint`: hàm loss và việc lưu/nạp checkpoint
- `test_smoke_train`: train thử 1 epoch cho cả 3 phương pháp (cần tải CIFAR-10)

---

## 12. Tài liệu tham khảo

- Sohn et al., *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*, NeurIPS 2020.
- Zhang et al., *FlexMatch*; Wang et al., *FreeMatch*: hai hướng ngưỡng thích ứng bài bản hơn, repo này không nhằm so kè với chúng.

## Giấy phép

Repo hiện chưa có file license. Nếu định công khai cho người khác dùng lại, nên thêm một file (ví dụ MIT).
