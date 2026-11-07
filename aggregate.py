import json
import pandas as pd
import numpy as np

seeds = [42, 123, 2026]
methods = ['sup_250', 'fm_t095_250', 'fm_t090_250', 'fm_t085_250',
           'afm_tmin080_tmax095_b090_250']
print('run | best_acc | final_acc | best_f1 | PLprec | PLcov | nan | split_ok')
for s in seeds:
    ref = None
    for m in methods:
        d = 'outputs/full_s%d/%s_s%d' % (s, m, s)
        df = pd.read_csv(d + '/metrics.csv')
        sp = json.load(open(d + '/split.json'))
        if ref is None:
            ref = sp['indices_by_class']
        row = df.loc[df.test_accuracy.idxmax()]
        last = df.iloc[-1]
        print('%s_s%d | %.4f | %.4f | %.4f | %.4f | %.4f | nan=%d | split=%s'
              % (m, s, row.test_accuracy, last.test_accuracy, row.macro_f1,
                 last.pl_precision, last.pl_coverage,
                 int(df.isna().sum().sum()),
                 sp['indices_by_class'] == ref))
for m in ['afm_tmin070_tmax095_b090_250_s42',
          'afm_tmin090_tmax095_b090_250_s42']:
    d = 'outputs/full_s42/%s' % m
    df = pd.read_csv(d + '/metrics.csv')
    row = df.loc[df.test_accuracy.idxmax()]
    last = df.iloc[-1]
    print('%s | %.4f | %.4f | %.4f | %.4f | %.4f'
          % (m, row.test_accuracy, last.test_accuracy, row.macro_f1,
             last.pl_precision, last.pl_coverage))
print('--- mean+-std (best acc over 3 seeds) ---')
for m in methods:
    vals = [pd.read_csv('outputs/full_s%d/%s_s%d/metrics.csv' % (s, m, s)
                        ).test_accuracy.max() for s in seeds]
    print('%s: %.4f +- %.4f %s'
          % (m, np.mean(vals), np.std(vals, ddof=1),
             [round(v, 4) for v in vals]))
