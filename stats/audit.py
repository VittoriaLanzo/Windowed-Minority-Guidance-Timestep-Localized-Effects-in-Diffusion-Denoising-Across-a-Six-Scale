"""
WMG Stats Auditor -- blind math gate.
Re-derives every numeric value in the per-scale and cross-scale JSON reports
from the raw CSVs, then compares against the stored JSONs.

Usage (from repo root):
    python stats/audit.py
"""

import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO  = Path(__file__).resolve().parent.parent
DATA  = REPO / "data" / "multi_scale"
STATS = REPO / "stats" / "results"

SCALE_INFO = [
    ("0p5", 0.5),
    ("1p0", 1.0),
    ("2p0", 2.0),
    ("3p5", 3.5),
    ("5p0", 5.0),
    ("7p0", 7.0),
]
CONDITIONS = ["baseline", "minority-early", "minority-mid", "minority-late", "minority-full"]

WILCOXON_PAIRS = [
    ("mid_vs_baseline",   "minority-mid",   "baseline"),
    ("late_vs_baseline",  "minority-late",  "baseline"),
    ("early_vs_baseline", "minority-early", "baseline"),
    ("full_vs_baseline",  "minority-full",  "baseline"),
    ("mid_vs_late",       "minority-mid",   "minority-late"),
    ("late_vs_early",     "minority-late",  "minority-early"),
    ("full_vs_mid",       "minority-full",  "minority-mid"),
]

N_SEEDS          = 50
N_BOOT           = 10_000
BOOT_SEED        = 42
BONFERRONI_PER   = 0.05 / 7
BONFERRONI_CROSS = 0.05 / 42   # 6 scales × 7 comparisons

TOL_GENERAL   = 1e-4
TOL_BOOTSTRAP = 1e-3
TOL_P         = 1e-5

n_checked = 0; n_matches = 0; discrepancies = []; notes = []


def rel_close(a, b, tol):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-12) < tol

def abs_close(a, b, tol):
    return abs(float(a) - float(b)) < tol

def _record(scale_label, cond_label, field, stored, computed, ok):
    global n_checked, n_matches
    n_checked += 1
    if ok: n_matches += 1; status = "MATCH"
    else:
        status = "DISCREPANCY"
        discrepancies.append({"scale": scale_label, "condition": cond_label,
                               "field": field, "stored_value": stored,
                               "computed_value": float(computed)})
    print(f"  [{status}] {scale_label}/{cond_label}/{field}: stored={stored}, computed={float(computed):.6g}")

def check_rel(sl, cl, f, s, c):  _record(sl, cl, f, s, c, rel_close(c, s, TOL_GENERAL))
def check_p  (sl, cl, f, s, c):  _record(sl, cl, f, s, c, rel_close(c, s, TOL_P))
def check_abs(sl, cl, f, s, c):  _record(sl, cl, f, s, c, abs_close(c, s, TOL_BOOTSTRAP))
def check_W  (sl, cl, f, s, c):  _record(sl, cl, f, s, c, float(s) == float(c))
def check_bool(sl, cl, f, s, c): _record(sl, cl, f, s, c, bool(s) == bool(c))


# ── per-scale audit ────────────────────────────────────────────────────────────
for scale_tag, scale_float in SCALE_INFO:
    csv_path  = DATA / f"runs_scale{scale_tag}_lsun_bedroom.csv"
    json_path = STATS / f"per_scale_{scale_tag}_lsun.json"
    sl = str(scale_float)

    print(f"\n{'='*70}\nSCALE {scale_float}  --  {csv_path.name}\n{'='*70}")

    df    = pd.read_csv(csv_path)
    pivot = df.pivot(index="seed", columns="condition", values="classifier_mean_loss")
    assert len(pivot) == N_SEEDS, f"Expected {N_SEEDS} seeds, got {len(pivot)}"

    with open(json_path) as fh:
        stored_json = json.load(fh)

    baseline_arr  = pivot["baseline"].values
    full_arr      = pivot["minority-full"].values
    baseline_mean = float(baseline_arr.mean())
    full_mean     = float(full_arr.mean())
    std_b         = float(baseline_arr.std(ddof=1))

    print("\n--- Descriptive stats ---")
    for cond in CONDITIONS:
        arr    = pivot[cond].values
        mean_c = float(arr.mean())
        std_c  = float(arr.std(ddof=1))
        cv_c   = std_c / mean_c

        rng        = np.random.default_rng(BOOT_SEED)
        boot_means = rng.choice(arr, size=(N_BOOT, N_SEEDS), replace=True).mean(axis=1)
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        s = stored_json["conditions"][cond]
        check_rel(sl, cond, "mean",               s["mean"],               mean_c)
        check_rel(sl, cond, "std",                s["std"],                std_c)
        check_rel(sl, cond, "cv",                 s["cv"],                 cv_c)
        check_abs(sl, cond, "bootstrap_ci_95[0]", s["bootstrap_ci_95"][0], ci_lo)
        check_abs(sl, cond, "bootstrap_ci_95[1]", s["bootstrap_ci_95"][1], ci_hi)

        if cond != "baseline":
            rel_eff  = 1.0 if cond == "minority-full" else (baseline_mean - mean_c) / (baseline_mean - full_mean)
            win      = float((arr < baseline_arr).mean())
            pooled   = float(np.sqrt((std_b**2 + std_c**2) / 2.0))
            cohens_d = (baseline_mean - mean_c) / pooled
            check_rel(sl, cond, "relative_effect", s["relative_effect"], rel_eff)
            check_rel(sl, cond, "win_rate",         s["win_rate"],        win)
            check_rel(sl, cond, "cohens_d",         s["cohens_d"],        cohens_d)

    print("\n--- Wilcoxon tests ---")
    denom_rrb = N_SEEDS * (N_SEEDS + 1) / 2.0
    for pair_label, cond_a, cond_b in WILCOXON_PAIRS:
        diffs      = pivot[cond_a].values - pivot[cond_b].values
        stat, pval = wilcoxon(diffs, zero_method="zsplit")
        W = float(stat); p = float(pval)
        r_rb = 1.0 - 2.0 * W / denom_rrb
        surv = bool(p < BONFERRONI_PER)
        sp = stored_json["wilcoxon"][pair_label]
        check_W   (sl, pair_label, "W",                   sp["W"],                   W)
        check_p   (sl, pair_label, "p",                   sp["p"],                   p)
        check_rel (sl, pair_label, "r_rb",                sp["r_rb"],                r_rb)
        check_bool(sl, pair_label, "survives_bonferroni", sp["survives_bonferroni"], surv)

    thresh = stored_json.get("bonferroni_threshold_per_scale")
    if thresh is not None:
        check_rel(sl, "meta", "bonferroni_threshold_per_scale", thresh, BONFERRONI_PER)


# ── cross-scale audit ──────────────────────────────────────────────────────────
print(f"\n{'='*70}\nCROSS-SCALE  --  cross_scale_lsun.json\n{'='*70}")

with open(STATS / "cross_scale_lsun.json") as fh:
    cs = json.load(fh)

COND_LONG = {"early": "minority-early", "mid": "minority-mid", "late": "minority-late"}

print("\n--- relative_effect_matrix ---")
for scale_tag, scale_float in SCALE_INFO:
    scale_key = str(scale_float)
    if scale_key not in cs["relative_effect_matrix"]:
        print(f"  [SKIP] {scale_key} not in cross_scale matrix")
        continue
    with open(STATS / f"per_scale_{scale_tag}_lsun.json") as fh2:
        per = json.load(fh2)
    for short_cond in ["early", "mid", "late"]:
        stored_val   = cs["relative_effect_matrix"][scale_key][short_cond]
        computed_val = per["conditions"][COND_LONG[short_cond]]["relative_effect"]
        check_rel(f"cross/{scale_key}", short_cond, "relative_effect_matrix", stored_val, computed_val)

print("\n--- mid_bootstrap_ci_by_scale ---")
for scale_tag, scale_float in SCALE_INFO:
    scale_key = str(scale_float)
    if scale_key not in cs.get("mid_bootstrap_ci_by_scale", {}):
        continue
    with open(STATS / f"per_scale_{scale_tag}_lsun.json") as fh2:
        per = json.load(fh2)
    per_lo = per["conditions"]["minority-mid"]["bootstrap_ci_95"][0]
    per_hi = per["conditions"]["minority-mid"]["bootstrap_ci_95"][1]
    check_abs(f"cross/{scale_key}", "minority-mid", "ci[0]", cs["mid_bootstrap_ci_by_scale"][scale_key][0], per_lo)
    check_abs(f"cross/{scale_key}", "minority-mid", "ci[1]", cs["mid_bootstrap_ci_by_scale"][scale_key][1], per_hi)

stored_bct = cs.get("bonferroni_threshold_cross_scale")
if stored_bct is not None:
    check_rel("cross", "meta", "bonferroni_threshold_cross_scale", stored_bct, BONFERRONI_CROSS)


# ── baseline identity check ────────────────────────────────────────────────────
print("\n--- baseline means across scales (should be identical — shared draw) ---")
bm_vals = {}
for scale_tag, scale_float in SCALE_INFO:
    with open(STATS / f"per_scale_{scale_tag}_lsun.json") as fh:
        bm_vals[scale_float] = json.load(fh)["conditions"]["baseline"]["mean"]
    print(f"  scale={scale_float}: baseline mean = {bm_vals[scale_float]}")
unique = set(round(v, 6) for v in bm_vals.values())
print("  -> All identical:", len(unique) == 1)


# ── final report ───────────────────────────────────────────────────────────────
verdict = "PASS" if not discrepancies else "DISCREPANCY"
print(f"\n{'='*70}")
print(f"AUDIT COMPLETE: {verdict}")
print(f"  checked={n_checked}  matches={n_matches}  discrepancies={len(discrepancies)}")
if discrepancies:
    print("\nDISCREPANCIES:")
    for d in discrepancies:
        print(f"  {d['scale']}/{d['condition']}/{d['field']}: stored={d['stored_value']}  computed={d['computed_value']}")

out_path = STATS / "audit_result.json"
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump({"verdict": verdict, "n_checked": n_checked, "n_matches": n_matches,
               "n_discrepancies": len(discrepancies), "discrepancies": discrepancies}, fh, indent=2)
print(f"\nResult written to: {out_path}")
if discrepancies:
    sys.exit(1)
