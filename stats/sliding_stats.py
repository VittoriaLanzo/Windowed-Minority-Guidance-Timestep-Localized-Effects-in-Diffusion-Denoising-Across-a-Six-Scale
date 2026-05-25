"""
Sliding-window experiment Wilcoxon + Bonferroni stats.
Conditions: baseline, full, slide_0..slide_6
Metric: classifier_mean_confidence (higher = more minority-class aligned)
Reference: baseline (no guidance), upper bound: full (guidance t=[0,1000))
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import csv
import json
import math
from pathlib import Path
from scipy.stats import wilcoxon

CSV = r"C:\Users\Vittoria\AppData\Local\Temp\wmg-sliding-output\minority-guidance\data\experiment_runs.csv"

# ── Load data ────────────────────────────────────────────────────────────────
runs = {}  # condition -> {seed: confidence}
with open(CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        cond  = row['condition']
        seed  = int(row['seed'])
        conf  = float(row['classifier_mean_loss'])
        runs.setdefault(cond, {})[seed] = conf

conditions = sorted(runs.keys())
seeds = sorted(runs['baseline'].keys())
n = len(seeds)
print(f"Conditions: {conditions}")
print(f"Seeds: {n}  (seed range {seeds[0]}–{seeds[-1]})")

# Verify all conditions have all seeds
for c in conditions:
    missing = set(seeds) - set(runs[c].keys())
    if missing:
        print(f"WARNING: {c} missing seeds {missing}")

# ── Per-condition paired arrays (aligned by seed) ────────────────────────────
baseline_vals = [runs['baseline'][s] for s in seeds]
full_vals     = [runs['minority-full'][s] for s in seeds]

baseline_mean = sum(baseline_vals) / n
full_mean     = sum(full_vals)     / n

print(f"\nbaseline  mean={baseline_mean:.6f}")
print(f"minority-full mean={full_mean:.6f}   Δrel=1.000 (definition)")

# ── Wilcoxon + stats for each condition vs baseline ──────────────────────────
# Non-baseline conditions (slide_0..slide_6 + full)
test_conditions = [c for c in conditions if c != 'baseline']
# fix condition key for full (CSV uses 'minority-full')
n_comparisons = len(test_conditions)   # 7 slides + full = 8
alpha = 0.05
bonferroni_threshold = alpha / n_comparisons

print(f"\nBonferroni threshold: α/{n_comparisons} = {bonferroni_threshold:.5f}")
print(f"  (α=0.05, {n_comparisons} comparisons vs baseline)\n")

results = {}
print(f"{'Condition':<14} {'t_start':>7} {'t_end':>6}  {'mean':>8}  {'Δrel':>6}  {'W':>8}  {'p':>10}  sig  {'r_rb':>6}")
print("-"*85)

# Slide window ranges from the CSV
WINDOWS = {
    'slide_0': (0,   250),
    'slide_1': (125, 375),
    'slide_2': (250, 500),
    'slide_3': (375, 625),
    'slide_4': (500, 750),
    'slide_5': (625, 875),
    'slide_6': (750, 1000),
    'minority-full': (0, 1000),
}

for cond in test_conditions:
    vals = [runs[cond][s] for s in seeds]
    mean_v = sum(vals) / n

    # Relative effect: lower loss is better; matches original per_scale JSON formula
    # Δrel = (baseline_loss - cond_loss) / (baseline_loss - full_loss)
    delta_rel = (baseline_mean - mean_v) / (baseline_mean - full_mean)

    # Wilcoxon signed-rank (cond vs baseline, paired per seed)
    # Positive diff = baseline > cond (desired: cond has lower loss)
    diffs = [baseline_vals[i] - vals[i] for i in range(n)]
    stat, p = wilcoxon(diffs, zero_method='zsplit')

    # Rank-biserial correlation
    n_pairs = n
    r_rb = 1 - 2 * stat / (n_pairs * (n_pairs + 1) / 2)

    sig = "***" if p < bonferroni_threshold else ("*" if p < 0.05 else "   ")

    t_start, t_end = WINDOWS.get(cond, (0, 0))
    print(f"{cond:<14} {t_start:>7} {t_end:>6}  {mean_v:>8.5f}  {delta_rel:>6.3f}  {stat:>8.1f}  {p:>10.6f}  {sig}  {r_rb:>6.3f}")

    results[cond] = {
        'condition': cond,
        't_start': t_start,
        't_end': t_end,
        'mean_confidence': round(mean_v, 6),
        'delta_rel': round(delta_rel, 4),
        'wilcoxon_W': stat,
        'p_value': round(p, 8),
        'significant_bonferroni': bool(p < bonferroni_threshold),
        'significant_p05': bool(p < 0.05),
        'r_rb': round(r_rb, 4),
    }

print("\nbaseline mean =", round(baseline_mean, 6))
print("full     mean =", round(full_mean, 6))

# ── Peak window detection ─────────────────────────────────────────────────────
slide_only = {k: v for k, v in results.items() if k.startswith('slide_')}
peak_cond = min(slide_only, key=lambda c: slide_only[c]['mean_confidence'])  # lower loss = better
print(f"\nPeak slide window: {peak_cond} "
      f"(t=[{results[peak_cond]['t_start']},{results[peak_cond]['t_end']}), "
      f"mean={results[peak_cond]['mean_confidence']:.5f}, "
      f"Δrel={results[peak_cond]['delta_rel']:.3f})")

# ── Save JSON ─────────────────────────────────────────────────────────────────
OUT = r"C:\Users\Vittoria\Desktop\windowed-minority-guidance\stats\sliding_window_lsun.json"
out = {
    'experiment': 'wmg-sliding-window',
    'dataset': 'lsun_bedroom',
    'scale': 3.5,
    'n_seeds': n,
    'n_comparisons_vs_baseline': n_comparisons,
    'bonferroni_threshold': round(bonferroni_threshold, 6),
    'baseline_mean': round(baseline_mean, 6),
    'full_mean': round(full_mean, 6),
    'window_width': 250,
    'window_step': 125,
    'conditions': results,
}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUT}")
