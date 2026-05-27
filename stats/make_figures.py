"""
Regenerate the two paper figures with bootstrap CI error bars.

Fig 1: relative effect Delta_rel vs guidance scale (early/mid/late),
       numeric (true-spacing) x-axis, 95% bootstrap CI error bars.
Fig 2: sliding-window Delta_rel at s=3.5 with 95% bootstrap CI error bars,
       plus the equal-thirds 3-window reference points.

Delta_rel CIs use a paired seed bootstrap (resample seed indices jointly across
baseline/condition/full), 10000 resamples, np.random.default_rng(42) -- the same
seed/resample spec as the mean-loss CIs in compute_stats.py.

Outputs (PDF + PNG) to paper/ and figures/, and the CI values to
stats/results/delta_rel_cis.json for traceability.

Usage (from repo root):  python stats/make_figures.py
"""
import sys, json, csv
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO   = Path(__file__).resolve().parent.parent
MULTI  = REPO / "data" / "multi_scale"
SLIDE  = REPO / "data" / "sliding_window" / "runs_sliding_window_lsun.csv"
PAPER  = REPO / "paper"
FIGS   = REPO / "figures"
RESULTS= REPO / "stats" / "results"

SCALES   = [0.5, 1.0, 2.0, 3.5, 5.0, 7.0]
SCALE_STR= {0.5:"0p5",1.0:"1p0",2.0:"2p0",3.5:"3p5",5.0:"5p0",7.0:"7p0"}
N_BOOT   = 10_000
SEED     = 42

C_EARLY, C_MID, C_LATE = "#1f77b4", "#d62728", "#2ca02c"


def load(path):
    """Return dict[condition] -> np.array of losses ordered by seed."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            col = "classifier_mean_loss" if "classifier_mean_loss" in r else "mean_loss"
            rows.setdefault(r["condition"], {})[int(r["seed"])] = float(r[col])
    seeds = sorted(rows["baseline"])
    return {c: np.array([rows[c][s] for s in seeds]) for c in rows}, seeds


def drel_ci(base, cond, full, n_boot=N_BOOT, seed=SEED):
    """Paired-seed bootstrap 95% CI for Delta_rel = (base-cond)/(base-full)."""
    rng = np.random.default_rng(seed)
    n = len(base)
    point = (base.mean() - cond.mean()) / (base.mean() - full.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    bm = base[idx].mean(axis=1); cm = cond[idx].mean(axis=1); fm = full[idx].mean(axis=1)
    denom = bm - fm
    good = denom != 0
    dr = (bm[good] - cm[good]) / denom[good]
    lo, hi = np.percentile(dr, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ---- compute Delta_rel CIs for the sweep ----
sweep = {}     # scale -> window -> (point, lo, hi)
for sc in SCALES:
    data, _ = load(MULTI / f"runs_scale{SCALE_STR[sc]}_lsun_bedroom.csv")
    base, full = data["baseline"], data["minority-full"]
    sweep[sc] = {}
    for w, cond in [("early","minority-early"),("mid","minority-mid"),("late","minority-late")]:
        sweep[sc][w] = drel_ci(base, data[cond], full)

# ---- compute Delta_rel CIs for the sliding window ----
sdata, _ = load(SLIDE)
sbase, sfull = sdata["baseline"], sdata["minority-full"]
SLIDES = ["slide_0","slide_1","slide_2","slide_3","slide_4","slide_5","slide_6"]
MID_T  = {"slide_0":125,"slide_1":250,"slide_2":375,"slide_3":500,
          "slide_4":625,"slide_5":750,"slide_6":875}
slide_ci = {k: drel_ci(sbase, sdata[k], sfull) for k in SLIDES}

# 3-window equal-thirds reference at s=3.5 (midpoints 166,500,833)
ref = {"early":(166, sweep[3.5]["early"][0]),
       "mid":  (500, sweep[3.5]["mid"][0]),
       "late": (833, sweep[3.5]["late"][0])}

# ---- save CI values for traceability ----
out = {"method": f"paired-seed bootstrap, {N_BOOT} resamples, default_rng({SEED}), 95% percentile",
       "sweep_delta_rel_ci": {str(sc): {w: {"point":p,"lo":lo,"hi":hi}
                                        for w,(p,lo,hi) in sweep[sc].items()} for sc in SCALES},
       "sliding_delta_rel_ci": {k: {"point":p,"lo":lo,"hi":hi} for k,(p,lo,hi) in slide_ci.items()}}
(RESULTS / "delta_rel_cis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("Wrote", RESULTS / "delta_rel_cis.json")

# ======================= FIGURE 1 =======================
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for w, color, marker, label in [
        ("early", C_EARLY, "o", r"Early  $t\in[0,333)$"),
        ("mid",   C_MID,   "s", r"Mid  $t\in[333,667)$"),
        ("late",  C_LATE,  "^", r"Late  $t\in[667,1000)$")]:
    y   = [sweep[sc][w][0] for sc in SCALES]
    lo  = [sweep[sc][w][0]-sweep[sc][w][1] for sc in SCALES]
    hi  = [sweep[sc][w][2]-sweep[sc][w][0] for sc in SCALES]
    ls  = "--" if w == "late" else "-"
    ax.errorbar(SCALES, y, yerr=[lo, hi], marker=marker, color=color, label=label,
                linestyle=ls, capsize=3, markersize=6, linewidth=1.6, elinewidth=1.0)
ax.set_xlabel(r"Guidance scale $s$")
ax.set_ylabel(r"Relative effect $\Delta_{\mathrm{rel}}$")
ax.set_xticks(SCALES); ax.set_xticklabels([str(s) for s in SCALES])
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(frameon=True, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(PAPER / "fig1_relative_effects.pdf"); fig.savefig(PAPER / "fig1_relative_effects.png", dpi=150)
fig.savefig(FIGS / "fig1_relative_effects.pdf"); fig.savefig(FIGS / "fig1_relative_effects.png", dpi=150)
plt.close(fig)
print("Wrote fig1 (numeric x-axis + 95% CI error bars)")

# ======================= FIGURE 2 =======================
fig, ax = plt.subplots(figsize=(6.8, 4.0))
xs  = [MID_T[k] for k in SLIDES]
ys  = [slide_ci[k][0] for k in SLIDES]
lo  = [slide_ci[k][0]-slide_ci[k][1] for k in SLIDES]
hi  = [slide_ci[k][2]-slide_ci[k][0] for k in SLIDES]
ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color="#4c5fd7", capsize=3,
            markersize=7, linewidth=1.8, elinewidth=1.0, label="Sliding window (250 steps)")
ax.axvspan(375, 625, alpha=0.12, color="#4c5fd7", label="Peak window [375,625)")
ax.axhline(1.0, ls="--", color="gray", alpha=0.7, label=r"Full ($\Delta_{\mathrm{rel}}=1.0$)")
# 3-window equal-thirds reference (334 steps wide)
for w, color, lbl in [("early",C_EARLY,"Early"),("mid",C_MID,"Mid"),("late",C_LATE,"Late")]:
    tx, ty = ref[w]
    ax.scatter([tx],[ty], marker="^", s=90, color=color, edgecolor="k", linewidth=0.5, zorder=5)
    ax.annotate(f"{lbl} ({ty:.3f})", (tx, ty), textcoords="offset points",
                xytext=(0,10), ha="center", fontsize=8, color=color, fontweight="bold")
# annotate peak
px, py = MID_T["slide_3"], slide_ci["slide_3"][0]
ax.annotate(f"slide_3\n({py:.3f})", (px, py), textcoords="offset points",
            xytext=(-38,-4), ha="center", fontsize=8, color="#3a4ab0")
ax.set_xlabel(r"Window midpoint  $t$")
ax.set_ylabel(r"$\Delta_{\mathrm{rel}}$")
ax.set_title(r"Sliding window $\Delta_{\mathrm{rel}}$ at scale 3.5 (triangles: equal-thirds 334-step windows)", fontsize=9.5)
ax.set_xlim(0, 1000); ax.set_ylim(0, 1.12)
ax.set_xticks(range(0,1001,125))
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(frameon=True, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(PAPER / "fig2_sliding_window.pdf"); fig.savefig(PAPER / "fig2_sliding_window.png", dpi=150)
fig.savefig(FIGS / "fig2_sliding_window.pdf"); fig.savefig(FIGS / "fig2_sliding_window.png", dpi=150)
plt.close(fig)
print("Wrote fig2 (95% CI error bars + 3-window reference)")

# ---- print the CI table ----
print("\n=== Delta_rel 95% CIs (sweep) ===")
for sc in SCALES:
    s = sweep[sc]
    print(f"  s={sc}: early {s['early'][0]:.3f} [{s['early'][1]:.3f},{s['early'][2]:.3f}]  "
          f"mid {s['mid'][0]:.3f} [{s['mid'][1]:.3f},{s['mid'][2]:.3f}]  "
          f"late {s['late'][0]:.3f} [{s['late'][1]:.3f},{s['late'][2]:.3f}]")
print("=== sliding ===")
for k in SLIDES:
    p,lo2,hi2 = slide_ci[k]
    print(f"  {k}: {p:.3f} [{lo2:.3f},{hi2:.3f}]")
