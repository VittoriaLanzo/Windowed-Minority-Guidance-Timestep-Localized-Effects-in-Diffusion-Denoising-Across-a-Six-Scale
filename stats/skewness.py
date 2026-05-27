"""
Confidence-skewness diagnostic for WMG (scale = 1.0).

Reports mean/median of classifier_mean_confidence per condition at the
single-scale operating point (s = 1.0). The mean/median ratio is a
crude right-skew indicator and is reported in the paper as a corroborative
diagnostic; it is not used for any inferential claim.

Usage (from repo root):
    python stats/skewness.py

Output: stats/results/confidence_skewness_s1p0.json
"""
import sys, csv, json
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CSV  = REPO / "data" / "multi_scale" / "runs_scale1p0_lsun_bedroom.csv"
OUT  = Path(__file__).resolve().parent / "results" / "confidence_skewness_s1p0.json"

CONDS = ["baseline", "minority-early", "minority-mid", "minority-late", "minority-full"]


def main():
    by_cond = {}
    with open(CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_cond.setdefault(r["condition"], []).append(float(r["classifier_mean_confidence"]))

    out = {"scale": 1.0, "dataset": "lsun_bedroom", "metric": "classifier_mean_confidence",
           "conditions": {}}
    for c in CONDS:
        arr = np.asarray(by_cond[c])
        mean = float(arr.mean()); med = float(np.median(arr))
        out["conditions"][c] = {
            "n":         int(arr.size),
            "mean":      mean,
            "median":    med,
            "mean_over_median_ratio": float(mean / med) if med != 0 else float("inf"),
            "ratio_rounded": int(round(mean / med)) if med != 0 else None,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT}")
    for c in CONDS:
        r = out["conditions"][c]
        print(f"  {c:<16} n={r['n']:<3} mean={r['mean']:.4e} median={r['median']:.4e} ratio={r['mean_over_median_ratio']:.2f}x (~{r['ratio_rounded']}x)")


if __name__ == "__main__":
    main()
