"""
wmg-stats-runner: Compute per-scale and cross-scale statistics for the
Windowed Minority Guidance sweep.

Usage (from repo root):
    python stats/compute_stats.py

Per-scale stats:   stats/results/per_scale_{scale_str}_lsun.json
Cross-scale stats: stats/results/cross_scale_lsun.json
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# ── project paths (portable — resolved relative to this file) ─────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data" / "multi_scale"
STATS_DIR  = Path(__file__).resolve().parent / "results"

# ── scale configs ─────────────────────────────────────────────────────────────
SCALES = [
    {"scale": 0.5, "scale_str": "0p5", "csv": "runs_scale0p5_lsun_bedroom.csv"},
    {"scale": 1.0, "scale_str": "1p0", "csv": "runs_scale1p0_lsun_bedroom.csv"},
    {"scale": 2.0, "scale_str": "2p0", "csv": "runs_scale2p0_lsun_bedroom.csv"},
    {"scale": 3.5, "scale_str": "3p5", "csv": "runs_scale3p5_lsun_bedroom.csv"},
    {"scale": 5.0, "scale_str": "5p0", "csv": "runs_scale5p0_lsun_bedroom.csv"},
    {"scale": 7.0, "scale_str": "7p0", "csv": "runs_scale7p0_lsun_bedroom.csv"},
]

CONDITIONS = ["baseline", "minority-early", "minority-mid", "minority-late", "minority-full"]

N_BOOTSTRAP        = 10_000
BOOTSTRAP_SEED     = 42
ALPHA              = 0.05
N_TESTS_PER_SCALE  = 7
BONFERRONI_PER_SCALE   = ALPHA / N_TESTS_PER_SCALE                       # ≈ 0.00714
BONFERRONI_CROSS_SCALE = ALPHA / (len(SCALES) * N_TESTS_PER_SCALE)       # ≈ 0.00119

WILCOXON_PAIRS = [
    ("mid_vs_baseline",   "minority-mid",   "baseline"),
    ("late_vs_baseline",  "minority-late",  "baseline"),
    ("early_vs_baseline", "minority-early", "baseline"),
    ("full_vs_baseline",  "minority-full",  "baseline"),
    ("mid_vs_late",       "minority-mid",   "minority-late"),
    ("late_vs_early",     "minority-late",  "minority-early"),
    ("full_vs_mid",       "minority-full",  "minority-mid"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def sig6(x):
    """Round to 6 significant figures (float)."""
    if x == 0.0:
        return 0.0
    magnitude = math.floor(math.log10(abs(x)))
    factor = 10 ** (5 - magnitude)
    return round(x * factor) / factor


def load_csv(path):
    """
    Load a per-scale CSV.
    Returns dict[condition -> (seeds_array, loss_array)] sorted by seed.
    Accepts column names 'classifier_mean_loss' or 'mean_loss'.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        col = None
        for row in reader:
            if col is None:
                if "classifier_mean_loss" in row:
                    col = "classifier_mean_loss"
                elif "mean_loss" in row:
                    col = "mean_loss"
                else:
                    raise KeyError(
                        f"No loss column found in {path}. Cols: {list(row.keys())}"
                    )
            rows.append({
                "condition": row["condition"].strip(),
                "seed":      int(row["seed"]),
                "loss":      float(row[col]),
            })

    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append((r["seed"], r["loss"]))

    result = {}
    for cond, pairs in groups.items():
        pairs.sort(key=lambda x: x[0])
        result[cond] = (
            np.array([s for s, _ in pairs]),
            np.array([l for _, l in pairs]),
        )
    return result


def bootstrap_ci(arr, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, ci=0.95):
    """Bootstrap 95 % CI for the mean (fresh rng per call)."""
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = rng.choice(arr, size=(n_resamples, n), replace=True).mean(axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return lo, hi


def cohens_d(baseline_arr, cond_arr):
    """Cohen's d = (baseline_mean - cond_mean) / pooled_std."""
    pooled_std = math.sqrt(
        (np.std(baseline_arr, ddof=1) ** 2 + np.std(cond_arr, ddof=1) ** 2) / 2
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(baseline_arr) - np.mean(cond_arr)) / pooled_std)


def wilcoxon_test(a, b, threshold):
    """
    Paired Wilcoxon signed-rank test on (a - b).
    Returns dict with W, p, r_rb, survives_bonferroni.
    """
    diff   = a - b
    result = wilcoxon(diff, zero_method="zsplit")
    W      = float(result.statistic)
    p      = float(result.pvalue)
    n      = len(diff)
    r_rb   = float(1 - 2 * W / (n * (n + 1) / 2))
    return {
        "W":                    sig6(W),
        "p":                    sig6(p),
        "r_rb":                 sig6(r_rb),
        "survives_bonferroni":  bool(p < threshold),
    }


# ── per-scale computation ─────────────────────────────────────────────────────

def compute_per_scale(scale_cfg):
    csv_path = DATA_DIR / scale_cfg["csv"]
    print(f"\n[scale={scale_cfg['scale']}] Loading {csv_path}")
    data = load_csv(csv_path)

    # Verify seed alignment
    ref_seeds = set(data["baseline"][0].tolist())
    for cond, (seeds, _) in data.items():
        if set(seeds.tolist()) != ref_seeds:
            raise ValueError(
                f"Seed mismatch for '{cond}' vs baseline at scale {scale_cfg['scale']}"
            )

    n_seeds = len(ref_seeds)
    print(f"  -> {n_seeds} seeds, conditions: {list(data.keys())}")

    losses = {cond: arr for cond, (_, arr) in data.items()}

    baseline_mean = float(np.mean(losses["baseline"]))
    full_mean     = float(np.mean(losses["minority-full"]))
    denom_full    = baseline_mean - full_mean

    conditions_out = {}
    for cond in CONDITIONS:
        arr      = losses[cond]
        mean_val = float(np.mean(arr))
        std_val  = float(np.std(arr, ddof=1))
        cv       = float(std_val / mean_val) if mean_val != 0 else float("nan")
        lo, hi   = bootstrap_ci(arr)

        entry = {
            "mean":            sig6(mean_val),
            "std":             sig6(std_val),
            "cv":              sig6(cv),
            "bootstrap_ci_95": [sig6(lo), sig6(hi)],
        }

        if cond != "baseline":
            rel_eff  = float((baseline_mean - mean_val) / denom_full) if denom_full else float("nan")
            win_rate = float(np.mean(arr < losses["baseline"]))
            d        = cohens_d(losses["baseline"], arr)
            entry["relative_effect"] = sig6(rel_eff)
            entry["win_rate"]        = sig6(win_rate)
            entry["cohens_d"]        = sig6(d)

        conditions_out[cond] = entry

    wilcoxon_out = {}
    for key, cond_a, cond_b in WILCOXON_PAIRS:
        wilcoxon_out[key] = wilcoxon_test(
            losses[cond_a], losses[cond_b], BONFERRONI_PER_SCALE
        )
        w = wilcoxon_out[key]
        print(
            f"  Wilcoxon {key}: W={w['W']:.4f}  p={w['p']:.6f}"
            f"  r_rb={w['r_rb']:.4f}  sig={w['survives_bonferroni']}"
        )

    return {
        "scale":    scale_cfg["scale"],
        "dataset":  "lsun_bedroom",
        "n_seeds":  n_seeds,
        "conditions": conditions_out,
        "wilcoxon": wilcoxon_out,
        "bonferroni_threshold_per_scale": BONFERRONI_PER_SCALE,
    }


# ── cross-scale computation ───────────────────────────────────────────────────

def compute_cross_scale(per_scale_results):
    relative_effect_matrix = {}
    mid_bootstrap_ci       = {}
    paths                  = []

    for cfg, res in zip(SCALES, per_scale_results):
        scale_key = str(cfg["scale"])
        conds     = res["conditions"]
        relative_effect_matrix[scale_key] = {
            "early": conds["minority-early"]["relative_effect"],
            "mid":   conds["minority-mid"]["relative_effect"],
            "late":  conds["minority-late"]["relative_effect"],
            "full":  1.0,
        }
        # plain [lo, hi] list — matches audit.py expectation
        mid_bootstrap_ci[scale_key] = conds["minority-mid"]["bootstrap_ci_95"]
        paths.append(f"stats/results/per_scale_{cfg['scale_str']}_lsun.json")

    return {
        "scales":                      [c["scale"] for c in SCALES],
        "dataset":                     "lsun_bedroom",
        "relative_effect_matrix":      relative_effect_matrix,
        "mid_bootstrap_ci_by_scale":   mid_bootstrap_ci,
        "bonferroni_threshold_cross_scale": BONFERRONI_CROSS_SCALE,
        "per_scale_stats_paths":       paths,
    }


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary_table(cross_scale):
    matrix = cross_scale["relative_effect_matrix"]
    scales = cross_scale["scales"]
    print("\n" + "=" * 60)
    print("RELATIVE EFFECT MATRIX (fraction of full-chain effect)")
    print(f"{'Scale':>8}  {'Early':>10}  {'Mid':>10}  {'Late':>10}")
    print("-" * 60)
    for s in scales:
        key = str(s)
        row = matrix[key]
        print(
            f"{key:>8}  {row['early']:>10.4f}"
            f"  {row['mid']:>10.4f}  {row['late']:>10.4f}"
        )
    print("=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Repo root  : {REPO_ROOT}")
    print(f"Data dir   : {DATA_DIR}")
    print(f"Output dir : {STATS_DIR}")

    per_scale_results = []
    for cfg in SCALES:
        res      = compute_per_scale(cfg)
        out_path = STATS_DIR / f"per_scale_{cfg['scale_str']}_lsun.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"  Wrote: {out_path}")
        per_scale_results.append(res)

    cross      = compute_cross_scale(per_scale_results)
    cross_path = STATS_DIR / "cross_scale_lsun.json"
    with open(cross_path, "w", encoding="utf-8") as fh:
        json.dump(cross, fh, indent=2)
    print(f"\nWrote: {cross_path}")

    print_summary_table(cross)

    # Verify all output files
    print("\nVerifying output files …")
    all_files = [STATS_DIR / f"per_scale_{c['scale_str']}_lsun.json" for c in SCALES]
    all_files.append(cross_path)
    ok = True
    for f in all_files:
        if not f.exists():
            print(f"  MISSING: {f}")
            ok = False
            continue
        try:
            with open(f) as fh:
                json.load(fh)
            print(f"  OK: {f}")
        except json.JSONDecodeError as e:
            print(f"  INVALID JSON: {f} — {e}")
            ok = False

    if ok:
        print("\nAll output files valid.")
    else:
        print("\nSome files failed verification!")
        sys.exit(1)


if __name__ == "__main__":
    main()
