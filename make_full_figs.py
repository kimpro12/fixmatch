import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

CN = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
      'horse', 'ship', 'truck']
S = 42
runs = {
    'sup': 'outputs/full_s%d/sup_250_s%d' % (S, S),
    'fm095': 'outputs/full_s%d/fm_t095_250_s%d' % (S, S),
    'fm090': 'outputs/full_s%d/fm_t090_250_s%d' % (S, S),
    'fm085': 'outputs/full_s%d/fm_t085_250_s%d' % (S, S),
    'afm080': 'outputs/full_s%d/afm_tmin080_tmax095_b090_250_s%d' % (S, S),
}
OUT = 'report/figures_full'
import os
os.makedirs(OUT, exist_ok=True)
dfs = {k: pd.read_csv(v + '/metrics.csv') for k, v in runs.items()}

plt.figure()
for k, df in dfs.items():
    plt.plot(df['epoch'], df['test_accuracy'] * 100, label=k)
plt.xlabel('epoch')
plt.ylabel('test accuracy (%)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT + '/P1_test_accuracy_200ep.png', dpi=150)
plt.close()

plt.figure()
for k in ['fm095', 'fm090', 'fm085', 'afm080']:
    df = dfs[k]
    plt.plot(df['epoch'], df['pl_coverage'] * 100, label=k)
plt.xlabel('epoch')
plt.ylabel('PL coverage (%)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT + '/P2_coverage_200ep.png', dpi=150)
plt.close()

plt.figure()
for k in ['fm095', 'fm090', 'fm085', 'afm080']:
    df = dfs[k]
    plt.plot(df['epoch'], df['pl_precision'] * 100, label=k)
plt.xlabel('epoch')
plt.ylabel('PL precision (%)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT + '/P3_precision_200ep.png', dpi=150)
plt.close()

# P4 per-class acceptance (fm095 vs afm080, last epoch)
plt.figure()
w = 0.35
for i, k in enumerate(['fm095', 'afm080']):
    dfp = pd.read_csv(runs[k] + '/per_class_metrics.csv')
    sub = dfp[dfp.epoch == dfp.epoch.max()].sort_values('class_id')
    plt.bar([x + i * w for x in range(10)], sub['acceptance_rate'].values,
            width=w, label=k)
plt.xticks(range(10), [c[:4] for c in CN], rotation=30)
plt.xlabel('class')
plt.ylabel('acceptance rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT + '/P4_accepted_per_class.png', dpi=150)
plt.close()

# P5 adaptive thresholds
plt.figure()
df = dfs['afm080']
for c in [c for c in df.columns if c.startswith('tau_class_')]:
    plt.plot(df['epoch'], df[c], label='cls' + c[-1], alpha=0.8)
plt.xlabel('epoch')
plt.ylabel('threshold')
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT + '/P5_adaptive_thresholds.png', dpi=150)
plt.close()

# P6 trade-off + P7 confusion
T, A, P, C = [], [], [], []
for k, t in [('fm095', 0.95), ('fm090', 0.90), ('fm085', 0.85)]:
    last = dfs[k].iloc[-1]
    T.append(t)
    A.append(last.test_accuracy * 100)
    P.append(last.pl_precision * 100)
    C.append(last.pl_coverage * 100)
fig, ax1 = plt.subplots()
ax1.plot(T, A, 'o-', label='test acc')
ax1.plot(T, P, 's-', label='PL precision')
ax1.set_xlabel('threshold')
ax1.set_ylabel('%')
ax2 = ax1.twinx()
ax2.plot(T, C, '^-', color='green', label='PL coverage')
ax2.set_ylabel('coverage %')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1 + h2, l1 + l2, fontsize=8)
plt.title('Fixed-threshold trade-off (200ep, seed 42)')
plt.tight_layout()
plt.savefig(OUT + '/P6_threshold_tradeoff.png', dpi=150)
plt.close()

for k in ['fm085', 'afm080']:
    cm = np.array(json.load(open(runs[k] + '/confusion_matrix_last.json')))
    _, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm,
                           display_labels=[c[:4] for c in CN]).plot(
        ax=ax, cmap='Blues', colorbar=False)
    plt.title(k + ' seed42 200ep')
    plt.tight_layout()
    plt.savefig(OUT + '/P7_confusion_%s.png' % k, dpi=150)
    plt.close()

# summary CSVs
rows = []
for s in [42, 123, 2026]:
    for m in ['sup_250', 'fm_t095_250', 'fm_t090_250', 'fm_t085_250',
              'afm_tmin080_tmax095_b090_250']:
        df = pd.read_csv('outputs/full_s%d/%s_s%d/metrics.csv' % (s, m, s))
        b = df.loc[df.test_accuracy.idxmax()]
        l = df.iloc[-1]
        rows.append({'method': m, 'seed': s,
                     'best_test_accuracy': round(float(b.test_accuracy), 4),
                     'final_test_accuracy': round(float(l.test_accuracy), 4),
                     'macro_f1': round(float(b.macro_f1), 4),
                     'pseudo_label_precision': round(float(l.pl_precision), 4),
                     'pseudo_label_coverage': round(float(l.pl_coverage), 4)})
pd.DataFrame(rows).to_csv('report/results/full_results.csv', index=False)
agg = (pd.DataFrame(rows).groupby('method')['best_test_accuracy']
       .agg(['mean', 'std']).round(4))
agg.to_csv('report/results/aggregate_mean_std.csv')
print(agg)
print('FIGURES+CSVs OK')
