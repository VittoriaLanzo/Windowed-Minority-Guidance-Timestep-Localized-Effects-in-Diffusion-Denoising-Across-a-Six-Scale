# WMG Reproducibility

Reproducibility package for **Windowed Minority Guidance** (WMG Extended), submitted to EEML 2026.

> **This repo is private until paper submission.**
> It will be merged into [windowed-minority-guidance](https://github.com/VittoriaLanzo/windowed-minority-guidance) at publication.

---

## Quick start

```bash
pip install scipy numpy pandas
```

### Re-run all stats from CSVs

```bash
# Multi-scale Wilcoxon + Bonferroni (6 scales)
python stats/compute_stats.py

# Sliding window Wilcoxon + Bonferroni
python stats/sliding_stats.py

# Audit (verify results match paper)
python stats/audit.py
```

All scripts write JSON to `stats/results/`. The CSVs in `data/` are the single source of truth.

---

## Repository structure

```
data/
  multi_scale/              6 CSVs — one per guidance scale (0.5–7.0)
                            5 conditions × 50 seeds each (250 rows/file)
  robustness/               Scale=3.5 seeds 51–100 (sample-size check)
                            5 conditions × 50 seeds (250 rows)
  sliding_window/           Sliding-window analysis at scale=3.5
                            9 conditions × 50 seeds (450 rows)

stats/
  compute_stats.py          Wilcoxon signed-rank + Bonferroni for multi-scale data
  sliding_stats.py          Same, adapted for sliding-window conditions
  audit.py                  Sanity-checks: verifies JSONs match CSVs
  results/
    per_scale_{scale}_lsun.json   Per-scale stats (Wilcoxon, Δrel, r_rb)
    cross_scale_lsun.json         Cross-scale summary table
    sensitivity_analysis.json     Bootstrap sensitivity check
    sliding_window_lsun.json      Sliding-window stats

experiment/
  windowed_classifier_sample.py   Core WMG sampler — patched cond_fn
  run_experiment.py               Experiment runner (all conditions × N seeds)
  extract_metrics.py              Reads .npz and computes classifier metrics
  guided_diffusion/               Diffusion library (from minority-guidance)

kaggle/
  NOTEBOOKS.md                    Catalog of all Kaggle kernels + links
  windowed-minority-guidance-extended.ipynb   Multi-scale sweep kernel
  wmg-fid-sweep.ipynb             Seeds 51–100 robustness kernel
  wmg-sliding-window.ipynb        Sliding-window kernel

figures/
  fig1_relative_effects.pdf/.png  Cross-scale Δrel figure (Fig. 1 in paper)
```

---

## Conditions

| Label | t_start | t_end | Description |
|-------|---------|-------|-------------|
| baseline | — | — | No classifier guidance |
| minority-early | 0 | 333 | Guidance during low-noise timesteps only |
| minority-mid | 333 | 667 | Guidance during mid timesteps (peak effect) |
| minority-late | 667 | 1000 | Guidance during high-noise timesteps |
| minority-full | 0 | 1000 | Full guidance (upper bound) |
| slide_0–6 | 0–750 | 250–1000 | Sliding 250-step windows, step=125 |

**Timestep convention:** window names follow numerical t-index magnitude (not DDPM chronological order).
`t=0` = final denoising step = lowest noise; `t=1000` = first step = highest noise.

---

## Key results

| Experiment | Finding |
|-----------|---------|
| Multi-scale (s=3.5) | mid Δrel=0.693, Bonferroni p≈6e-14, r_rb=0.986 |
| Seeds 51–100 robustness | mid Δrel=0.693 (identical), all conditions replicate |
| Sliding window (s=3.5) | Peak at slide_3 t=[375,625) Δrel=0.536; all 8 windows significant |

---

## Metric

`classifier_mean_loss` — mean cross-entropy loss of a minority-class classifier on generated images.
**Lower = more minority-class aligned.**

Relative effect: `Δrel = (baseline_loss − cond_loss) / (baseline_loss − full_loss)`

---

## Kaggle dataset

Raw CSVs are also published at:
[vittorialanzo/wmg-sweep-results](https://www.kaggle.com/datasets/vittorialanzo/wmg-sweep-results) (private until submission)
