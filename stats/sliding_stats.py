"""
Sliding-window Wilcoxon + Bonferroni statistics for WMG.

Conditions: baseline, minority-full, slide_0 … slide_6
Metric: classifier_mean_loss (lower = more minority-class aligned)
Reference: baseline (no guidance)
Upper bound: minority-full (guidance t=[0,1000))

Usage (from repo root):
    python stats/sliding_stats.py

Output: stats/results/sliding_window_lsun.json
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import math
from pathlib import Path
from scipy.stats import wilcoxon

# ── portable paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH  = REPO_ROOT / "data" / "sliding_window" / "runs_sliding_window_lsun.csv"
OUT_PATH  = Path(__file__).resolve().parent / "results" / "sliding_window_lsun.json"

# ── window boundaries (t_start, t_end) ────────────────────────────────────────
WINDOWS = {
    "slide_0":       (0,    250),
    "slide_1":       (125,  375),
    "slide_2":       (250,  500),
    "slide_3":       (375,  625),
    "slide_4":       (500,  750),
    "slide_5":       (625,  875),
    "slide_6":       (750, 1000),
    "minority-full": (0,   1000),
}

ALPHA = 0.05


def sig6(x):
    """Round to 6 significant figures."""
    if x == 0.0:
        return 0.0
    mag = math.floor(math.log10(abs(x)))
    fac = 10 ** (5 - mag)
    return round(x * fac) / fac


# ── Load CSV ──────────────────────────────────────────────────────────────────
runs: dict[str, dict[int, float]] = {}
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cond = row["condition"]
        seed = int(row["seed"])
        # accept either column name
        loss = float(row.get("classifier_mean_loss") or row["mean_loss"])
        runs.setdefault(cond, {})[seed] = loss

conditions = sorted(runs.keys())
seeds      = sorted(runs["baseline"].keys())
n          = len(seeds)

print(f"Conditions : {conditions}")
print(f"Seeds      : {n}  (range {seeds[0]}–{seeds[-1]})")

# Sanity check: all conditions have all seeds
for c in conditions:
    missing = set(seeds) - set(runs[c].keys())
    if missing:
        print(f"WARNING: {c} missing seeds {missing}")

# ── Aligned arrays ────────────────────────────────────────────────────────────
baseline_vals = [runs["baseline"][s]       for s in seeds]
full_vals     = [runs["minority-full"][s]  for s in seeds]

baseline_mean = sum(baseline_vals) / n
full_mean     = sum(full_vals)     / n

print(f"\nbaseline      mean = {baseline_mean:.6f}")
print(f"minority-full mean = {full_mean:.6f}   Δrel = 1.000 (definition)")

# ── Wilcoxon tests ────────────────────────────────────────────────────────────
test_conditions    = [c for c in conditions if c != "baseline"]
n_comparisons      = len(test_conditions)   # 7 slides + full = 8
bonferroni_thresh  = ALPHA / n_comparisons

print(f"\nBonferroni threshold: α/{n_comparisons} = {bonferroni_thresh:.5f}")
print(f"  (α={ALPHA}, {n_comparisons} comparisons vs baseline)\n")

header = (
    f"{'Condition':<14} {'t_start':>7} {'t_end':>6}"
    f"  {'mean':>9}  {'Δrel':>6}  {'W':>8}  {'p':>10}  sig  {'r_rb':>6}"
)
print(header)
print("-" * len(header))

results = {}
for cond in sorted(test_conditions):
    vals   = [runs[cond][s] for s in seeds]
    mean_v = sum(vals) / n

    # Δrel: (baseline_loss − cond_loss) / (baseline_loss − full_loss)
    delta_rel = (baseline_mean - mean_v) / (baseline_mean - full_mean)

    # Paired Wilcoxon (positive diff = cond has lower loss than baseline)
    diffs = [baseline_vals[i] - vals[i] for i in range(n)]
    stat, p = wilcoxon(diffs, zero_method="zsplit")

    W    = float(stat)
    p    = float(p)
    r_rb = 1.0 - 2.0 * W / (n * (n + 1) / 2)
    sig  = "***" if p < bonferroni_thresh else ("*" if p < 0.05 else "   ")

    t_start, t_end = WINDOWS.get(cond, (0, 0))
    print(
        f"{cond:<14} {t_start:>7} {t_end:>6}"
        f"  {mean_v:>9.5f}  {delta_rel:>6.3f}"
        f"  {W:>8.1f}  {p:>10.2e}  {sig}  {r_rb:>6.3f}"
    )

    results[cond] = {
        "condition":               cond,
        "t_start":                 t_start,
        "t_end":                   t_end,
        "mean_loss":               sig6(mean_v),
        "delta_rel":               sig6(delta_rel),
        "W":                       sig6(W),
        "p":                       sig6(p),
        "r_rb":                    sig6(r_rb),
        "significant_bonferroni":  bool(p < bonferroni_thresh),
    }

# ── Peak detection ────────────────────────────────────────────────────────────
slide_only = {k: v for k, v in results.items() if k.startswith("slide_")}
peak_cond  = min(slide_only, key=lambda c: slide_only[c]["mean_loss"])
print(
    f"\nPeak: {peak_cond} "
    f"t=[{results[peak_cond]['t_start']},{results[peak_cond]['t_end']})"
    f"  Δrel={results[peak_cond]['delta_rel']:.4f}"
    f"  W={results[peak_cond]['W']}"
    f"  p={results[peak_cond]['p']:.2e}"
)

# ── Save JSON ─────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out_doc = {
    "experiment":                  "wmg-sliding-window",
    "dataset":                     "lsun_bedroom",
    "scale":                       3.5,
    "n_seeds":                     n,
    "window_width":                250,
    "window_step":                 125,
    "n_comparisons_vs_baseline":   n_comparisons,
    "bonferroni_threshold":        sig6(bonferroni_thresh),
    "baseline_mean":               sig6(baseline_mean),
    "full_mean":                   sig6(full_mean),
    "conditions":                  results,
}
with open(OUT_PATH, "w", encoding="utf-8") as fh:
    json.dump(out_doc, fh, indent=2)
print(f"\nSaved → {OUT_PATH}")
