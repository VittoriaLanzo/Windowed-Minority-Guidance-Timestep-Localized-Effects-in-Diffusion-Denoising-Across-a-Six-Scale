"""
Independent-replication (robustness) statistics for WMG.

Recomputes the scale=3.5 condition battery on an independent seed set
(seeds 51-100) to test whether the seeds 1-50 findings replicate.
Reuses the exact, audited helpers from compute_stats.py (sig6, load_csv,
bootstrap_ci, cohens_d, wilcoxon_test) so the methodology is identical to
the main per-scale pipeline.

Usage (from repo root):
    python stats/robustness_stats.py

Output: stats/results/robustness_scale3p5_seeds51_100.json
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np

# reuse audited helpers from the main stats runner
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_stats import (
    load_csv, bootstrap_ci, cohens_d, wilcoxon_test, sig6,
    CONDITIONS, WILCOXON_PAIRS, BONFERRONI_PER_SCALE,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH  = REPO_ROOT / "data" / "robustness" / "runs_scale3p5_seeds51_100_lsun.csv"
OUT_PATH  = Path(__file__).resolve().parent / "results" / "robustness_scale3p5_seeds51_100.json"


def main():
    data = load_csv(CSV_PATH)                       # dict[cond] -> (seeds, losses)
    ref_seeds = sorted(data["baseline"][0].tolist())
    for cond, (seeds, _) in data.items():
        if sorted(seeds.tolist()) != ref_seeds:
            raise ValueError(f"Seed mismatch for '{cond}'")
    n_seeds = len(ref_seeds)

    losses = {cond: arr for cond, (_, arr) in data.items()}
    baseline_mean = float(np.mean(losses["baseline"]))
    full_mean     = float(np.mean(losses["minority-full"]))
    denom_full    = baseline_mean - full_mean

    conditions_out = {}
    for cond in CONDITIONS:
        arr = losses[cond]
        mean_val = float(np.mean(arr))
        std_val  = float(np.std(arr, ddof=1))
        cv       = float(std_val / mean_val) if mean_val else float("nan")
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
            entry["relative_effect"] = sig6(rel_eff)
            entry["win_rate"]        = sig6(win_rate)
            entry["cohens_d"]        = sig6(cohens_d(losses["baseline"], arr))
        conditions_out[cond] = entry

    wilcoxon_out = {}
    for key, a, b in WILCOXON_PAIRS:
        wilcoxon_out[key] = wilcoxon_test(losses[a], losses[b], BONFERRONI_PER_SCALE)

    out = {
        "experiment": "wmg-robustness-replication",
        "scale": 3.5,
        "dataset": "lsun_bedroom",
        "seed_range": [min(ref_seeds), max(ref_seeds)],
        "n_seeds": n_seeds,
        "conditions": conditions_out,
        "wilcoxon": wilcoxon_out,
        "bonferroni_threshold_per_scale": BONFERRONI_PER_SCALE,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT_PATH}")
    print(f"  seeds {min(ref_seeds)}-{max(ref_seeds)}, n={n_seeds}")
    for cond in CONDITIONS:
        c = conditions_out[cond]
        re = c.get("relative_effect", "--")
        print(f"  {cond:<18} mean={c['mean']}  rel_eff={re}")
    for key in ["mid_vs_baseline", "late_vs_baseline", "early_vs_baseline"]:
        w = wilcoxon_out[key]
        print(f"  {key:<20} W={w['W']} p={w['p']} r_rb={w['r_rb']} sig={w['survives_bonferroni']}")


if __name__ == "__main__":
    main()
