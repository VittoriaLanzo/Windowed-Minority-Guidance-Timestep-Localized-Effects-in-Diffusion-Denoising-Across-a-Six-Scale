"""
Sample-size sensitivity analysis for WMG.

Repeats the per-scale late-vs-baseline (and all condition-vs-baseline) Wilcoxon
tests on the first n seeds in seed-index order for n in {25, 35, 50}, at every
guidance scale.  Provides the evidence for the paper's claim that the s=1.0
late-window result is "fragile" (passes at n=35, fails at n=50).

Usage (from repo root):
    python stats/sensitivity.py

Outputs:
    stats/results/sensitivity_full.json    per-scale × per-condition × per-n p-values
    stats/results/sensitivity_analysis.json  summary for late-vs-baseline only
"""
import sys, json, csv
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

REPO   = Path(__file__).resolve().parent.parent
DATA   = REPO / "data" / "multi_scale"
STATS  = REPO / "stats" / "results"

SCALE_INFO = [
    ("0p5", 0.5),
    ("1p0", 1.0),
    ("2p0", 2.0),
    ("3p5", 3.5),
    ("5p0", 5.0),
    ("7p0", 7.0),
]
CONDITIONS    = ["minority-early", "minority-mid", "minority-late", "minority-full"]
COND_SHORT    = {"minority-early": "early", "minority-mid": "mid",
                 "minority-late": "late", "minority-full": "full"}
SUBSAMPLE_NS  = [25, 35, 50]
BONFERRONI    = 0.05 / 7


def load(path):
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d.setdefault(r["condition"], {})[int(r["seed"])] = float(r["classifier_mean_loss"])
    return d


def wilcoxon_p(a, b):
    diffs = np.asarray(a) - np.asarray(b)
    return float(wilcoxon(diffs, zero_method="zsplit").pvalue)


def main():
    STATS.mkdir(parents=True, exist_ok=True)
    full_out = {}         # scale_str -> n_str -> condition -> {p, sig}
    summary_out = {}      # scale_str -> n_str -> {W, p, sig, threshold}  (late only)

    for scale_tag, scale_float in SCALE_INFO:
        raw    = load(DATA / f"runs_scale{scale_tag}_lsun_bedroom.csv")
        seeds  = sorted(raw["baseline"])          # all 50, sorted
        scale_key = str(scale_float)
        full_out[scale_key]    = {}
        summary_out[scale_key] = {}

        for n in SUBSAMPLE_NS:
            sub_seeds  = seeds[:n]
            base_sub   = [raw["baseline"][s] for s in sub_seeds]
            full_out[scale_key][str(n)]    = {}
            summary_out[scale_key][str(n)] = {}

            for cond in CONDITIONS:
                cond_sub  = [raw[cond][s] for s in sub_seeds]
                p         = wilcoxon_p(base_sub, cond_sub)
                sig       = bool(p < BONFERRONI)
                short_key = COND_SHORT[cond]
                full_out[scale_key][str(n)][short_key] = {
                    "p":   p,          # full precision — matches committed JSONs
                    "sig": sig,
                }

            # late-specific summary
            late_sub = [raw["minority-late"][s] for s in sub_seeds]
            diffs    = np.array(base_sub) - np.array(late_sub)
            res      = wilcoxon(diffs, zero_method="zsplit")
            W        = float(res.statistic)
            p_late   = float(res.pvalue)
            summary_out[scale_key][str(n)] = {
                "W":         W,
                "p":         round(p_late, 7),
                "sig":       bool(p_late < BONFERRONI),
                "threshold": BONFERRONI,
            }

            print(f"  s={scale_float} n={n:2d}  late p={p_late:.6f}  sig={p_late < BONFERRONI}")

    out_full    = STATS / "sensitivity_full.json"
    out_summary = STATS / "sensitivity_analysis.json"
    out_full.write_text(json.dumps(full_out, indent=2), encoding="utf-8")
    out_summary.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_full}")
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()
