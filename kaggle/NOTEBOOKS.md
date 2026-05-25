# Kaggle Notebook Catalog

All kernels are private (vittorialanzo workspace). Links go live at paper submission.

---

## windowed-minority-guidance-extended

**URL:** https://www.kaggle.com/code/vittorialanzo/windowed-minority-guidance-extended  
**File:** `windowed-minority-guidance-extended.ipynb`  
**Status:** COMPLETE  

**What it does:**  
Multi-scale sweep — runs all 5 conditions (baseline, early, mid, late, full) across
6 guidance scales (0.5, 1.0, 2.0, 3.5, 5.0, 7.0), seeds 1–50.

**Produces:**  
- `data/multi_scale/runs_scale{s}_lsun_bedroom.csv` (6 files, 250 rows each)

**Dataset input:** `vittorialanzo/mc-lsun`  
**GPU:** T4 / P100  
**Runtime:** ~10–12h

---

## wmg-fid-sweep

**URL:** https://www.kaggle.com/code/vittorialanzo/wmg-fid-sweep  
**File:** `wmg-fid-sweep.ipynb`  
**Title:** WMG Scale=3.5 Robustness: Seeds 51-100 (FID deferred — n=50 uninformative)  
**Status:** COMPLETE (generation); FID cell disabled  

**What it does:**  
Runs all 5 conditions at scale=3.5, seeds 51–100 as an independent replication.
Cell 3 (FID computation) was removed after determining n=50 yields intervals too
wide to be informative (requires ≥2,000 samples per condition; see Heusel et al. 2017).

**Produces:**  
- `data/robustness/runs_scale3p5_seeds51_100_lsun.csv` (250 rows, seeds 51–100)

**Dataset input:** `vittorialanzo/mc-lsun`  
**GPU:** T4  
**Runtime:** ~4h

---

## wmg-sliding-window

**URL:** https://www.kaggle.com/code/vittorialanzo/wmg-sliding-window  
**File:** `wmg-sliding-window.ipynb`  
**Status:** COMPLETE  

**What it does:**  
Fine-grained analysis with a sliding 250-timestep window (step=125) across [0,1000).
Runs 7 sliding windows + baseline + full at scale=3.5, seeds 1–50.

Window schedule:
| Condition | t_start | t_end |
|-----------|---------|-------|
| slide_0 | 0 | 250 |
| slide_1 | 125 | 375 |
| slide_2 | 250 | 500 |
| slide_3 | 375 | 625 |
| slide_4 | 500 | 750 |
| slide_5 | 625 | 875 |
| slide_6 | 750 | 1000 |

**Produces:**  
- `data/sliding_window/runs_sliding_window_lsun.csv` (450 rows, 9 conditions × 50 seeds)

**Key finding:** Peak at slide_3 (t=[375,625)), Δrel=0.536. All 8 non-baseline conditions
pass Bonferroni correction (α/8=0.00625). Confirms mid window as the effective zone,
consistent with the 3-window experiment.

**Dataset input:** `vittorialanzo/mc-lsun`  
**GPU:** T4  
**Runtime:** ~10h

---

## Kaggle dataset

**wmg-sweep-results:** https://www.kaggle.com/datasets/vittorialanzo/wmg-sweep-results  
Contains all CSVs above in one place (updated in sync with each kernel completion).
