cd /root/adaptive-fixmatch
LOG=outputs_full200.log
echo "=== FULL START $(date) ===" >> $LOG
for SEED in 42 123 2026; do
  python3 scripts/run_experiments.py --suite core --epochs 200 --seed $SEED --outbase outputs/full_s$SEED >> $LOG 2>&1
  echo "=== CORE SEED $SEED DONE $(date) ===" >> $LOG
done
# E5 ablation: tau_min sweep, seed 42 only (exploratory)
python3 scripts/train.py --config configs/adaptive_fixmatch_cifar10.yaml --outdir outputs/full_s42/afm_tmin070_tmax095_b090_250_s42 adaptive.tau_min=0.70 training.epochs=200 seed=42 >> $LOG 2>&1
echo "=== ABL tmin070 DONE $(date) ===" >> $LOG
python3 scripts/train.py --config configs/adaptive_fixmatch_cifar10.yaml --outdir outputs/full_s42/afm_tmin090_tmax095_b090_250_s42 adaptive.tau_min=0.90 training.epochs=200 seed=42 >> $LOG 2>&1
echo "=== ABL tmin090 DONE $(date) ===" >> $LOG
echo "=== FULL ALL DONE $(date) ===" >> $LOG
