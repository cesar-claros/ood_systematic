# Step 4 — Harmonization check

**Date:** 2026-05-02
**Source:** `code/nc_csf_predictivity/data/harmonize.py`

## Worked examples

### Example 1 — z-score within (source, eval_dataset)

Suppose `(source=cifar10, eval_dataset=svhn)` contains four AUGRC entries across CSFs and runs (a small subset for illustration):

| csf    | augrc |
|---|---|
| MSR | 100 |
| Energy | 150 |
| NeCo | 200 |
| CTM | 250 |

Cell mean μ = 175; sample std σ ≈ 64.55.

Apply z = (augrc − μ) / σ:

| csf    | augrc | z |
|---|---|---|
| MSR | 100 | -1.16 |
| Energy | 150 | -0.39 |
| NeCo | 200 | +0.39 |
| CTM | 250 | +1.16 |

Lower z = better detection (since lower AUGRC = better).

### Example 2 — percentile rank within cell

Same 4 entries, ranked from lowest (best):

| csf | augrc | rank | pct rank = rank / n |
|---|---|---|---|
| MSR | 100 | 1 | 0.25 |
| Energy | 150 | 2 | 0.50 |
| NeCo | 200 | 3 | 0.75 |
| CTM | 250 | 4 | 1.00 |

Lower pct = better. Rank harmonization is invariant to monotone transformations of AUGRC, so it is the recommended fallback when the per-cell distribution is far from normal.

### Example 3 — Shapiro-Wilk normality test on a cell

H₀: the cell's AUGRC values are drawn from a normal distribution. W ∈ [0,1], W close to 1 = consistent with normal. We reject at p ≤ 0.05.

| sample | n | W | p | verdict at α=0.05 |
|---|---|---|---|---|
| Normal-ish (mean 150, sd 30) | 200 | 0.996 | 0.883 | PASS |
| Heavy-tailed mixture | 200 | 0.382 | 1.12e-25 | FAIL |

Protocol fallback rule: if more than 50% of cells FAIL at α=0.05, switch the primary harmonization from z-score to percentile rank for downstream training. Note: at large N (thousands), Shapiro-Wilk is over-sensitive and tends to reject normality even for visually normal data — interpret pass rates with that caveat in mind.

### Example 4 — regime-level aggregation across eval_datasets

After harmonization, the per-row z lives at the (architecture, paradigm, source, run, dropout, reward, csf, eval_dataset) granularity. To produce a single regime-level number for a (model, csf) we average z across the eval_datasets in the regime.

For `(source=cifar10, regime=mid)` the eval_datasets per the CLIP groupings are `{isun, lsun resize, lsun cropped, svhn}`. Suppose one model row uses Energy as its CSF and we observe:

| eval_dataset | augrc_z |
|---|---|
| isun | -1.40 |
| lsun resize | -0.90 |
| lsun cropped | -0.60 |
| svhn | +0.50 |

Regime-level z = unweighted mean = -0.60. Equal weighting per eval_dataset prevents high-variance evals from dominating. This regime-level number feeds into per-regime metric and statistical-test aggregations downstream (steps 13–15).

## Track 1

- Output: `outputs/track1/dataset/long_harmonized.parquet`
- Total rows: 67,320
- Columns added: `augrc_z`, `augrc_rank`, `aurc_z`, `aurc_rank`

### Shapiro-Wilk pass rate

- Cells: 36
- Passing α=0.05: 0 (0%)
- Failing share: 100% (fallback triggered if >50%)
- **Verdict:** FALL BACK to rank

### Per-cell stats

| source | eval_dataset | n | mean(augrc) | std(augrc) | W | p | pass |
|---|---|---|---|---|---|---|---|
| cifar10 | cifar100 | 1630 | 171.19 | 15.82 | 0.736 | 4.46e-45 | ✗ |
| cifar10 | isun | 1630 | 143.39 | 18.16 | 0.550 | 1.09e-53 | ✗ |
| cifar10 | lsun cropped | 1630 | 160.19 | 22.10 | 0.549 | 9.63e-54 | ✗ |
| cifar10 | lsun resize | 1630 | 157.94 | 17.37 | 0.562 | 3.10e-53 | ✗ |
| cifar10 | places365 | 1630 | 167.66 | 16.25 | 0.661 | 4.92e-49 | ✗ |
| cifar10 | svhn | 1630 | 294.65 | 15.02 | 0.634 | 2.98e-50 | ✗ |
| cifar10 | test | 1630 | 5.79 | 2.83 | 0.849 | 8.30e-37 | ✗ |
| cifar10 | textures | 1630 | 95.90 | 18.23 | 0.564 | 3.77e-53 | ✗ |
| cifar10 | tinyimagenet | 1630 | 160.61 | 18.61 | 0.572 | 7.79e-53 | ✗ |
| cifar100 | cifar10 | 1870 | 229.10 | 45.45 | 0.753 | 2.16e-46 | ✗ |
| cifar100 | isun | 1870 | 205.16 | 45.71 | 0.682 | 1.64e-50 | ✗ |
| cifar100 | lsun cropped | 1870 | 224.10 | 44.11 | 0.779 | 1.04e-44 | ✗ |
| cifar100 | lsun resize | 1870 | 220.64 | 45.28 | 0.707 | 3.49e-49 | ✗ |
| cifar100 | places365 | 1870 | 229.81 | 41.73 | 0.785 | 2.82e-44 | ✗ |
| cifar100 | svhn | 1870 | 345.37 | 30.01 | 0.866 | 2.88e-37 | ✗ |
| cifar100 | test | 1870 | 81.34 | 65.63 | 0.594 | 1.31e-54 | ✗ |
| cifar100 | textures | 1870 | 151.49 | 46.40 | 0.717 | 1.36e-48 | ✗ |
| cifar100 | tinyimagenet | 1870 | 218.70 | 46.64 | 0.693 | 6.09e-50 | ✗ |
| supercifar100 | cifar10 | 2350 | 282.91 | 31.74 | 0.850 | 1.34e-42 | ✗ |
| supercifar100 | isun | 2350 | 249.30 | 35.47 | 0.822 | 2.96e-45 | ✗ |
| supercifar100 | lsun cropped | 2350 | 266.10 | 36.13 | 0.892 | 9.38e-38 | ✗ |
| supercifar100 | lsun resize | 2350 | 264.31 | 34.71 | 0.828 | 9.11e-45 | ✗ |
| supercifar100 | places365 | 2350 | 282.49 | 36.51 | 0.819 | 1.55e-45 | ✗ |
| supercifar100 | svhn | 2350 | 377.32 | 23.31 | 0.972 | 3.65e-21 | ✗ |
| supercifar100 | test | 2350 | 171.78 | 54.72 | 0.869 | 1.53e-40 | ✗ |
| supercifar100 | textures | 2350 | 190.57 | 40.53 | 0.886 | 1.58e-38 | ✗ |
| supercifar100 | tinyimagenet | 2350 | 259.28 | 37.93 | 0.852 | 1.90e-42 | ✗ |
| tinyimagenet | cifar10 | 1630 | 245.47 | 50.89 | 0.785 | 6.04e-42 | ✗ |
| tinyimagenet | cifar100 | 1630 | 248.70 | 50.89 | 0.781 | 3.29e-42 | ✗ |
| tinyimagenet | isun | 1630 | 225.95 | 49.99 | 0.803 | 1.12e-40 | ✗ |
| tinyimagenet | lsun cropped | 1630 | 245.93 | 60.48 | 0.807 | 2.34e-40 | ✗ |
| tinyimagenet | lsun resize | 1630 | 240.49 | 46.45 | 0.817 | 1.38e-39 | ✗ |
| tinyimagenet | places365 | 1630 | 239.01 | 39.33 | 0.850 | 9.31e-37 | ✗ |
| tinyimagenet | svhn | 1630 | 368.37 | 41.91 | 0.867 | 4.54e-35 | ✗ |
| tinyimagenet | test | 1630 | 112.72 | 43.24 | 0.886 | 5.52e-33 | ✗ |
| tinyimagenet | textures | 1630 | 177.58 | 50.80 | 0.853 | 1.73e-36 | ✗ |

### Sanity: z-score range and rank range per cell (must be ~symmetric and [0,1])

z-score per cell — first 5 cells:

```
                      mean  std    min     max
source  eval_dataset                          
cifar10 cifar100      -0.0  1.0 -1.741  11.464
        isun          -0.0  1.0 -0.912  11.654
        lsun cropped  -0.0  1.0 -0.874   9.895
        lsun resize    0.0  1.0 -0.963  12.198
        places365      0.0  1.0 -1.689  12.595
```

rank per cell — first 5 cells (should be ~0 and ~1):

```
                        min  max
source  eval_dataset            
cifar10 cifar100      0.001  1.0
        isun          0.001  1.0
        lsun cropped  0.001  1.0
        lsun resize   0.001  1.0
        places365     0.001  1.0
```

## Track 2

- Output: `outputs/track2/dataset/long_harmonized.parquet`
- Total rows: 2,844

### Shapiro-Wilk pass rate

- Cells: 36
- Passing α=0.05: 0 (0%)
- Failing share: 100%
- **Verdict:** FALL BACK to rank

### Per-cell stats

| source | eval_dataset | n | mean(augrc) | std(augrc) | W | p | pass |
|---|---|---|---|---|---|---|---|
| cifar10 | cifar100 | 79 | 165.13 | 13.12 | 0.788 | 2.66e-09 | ✗ |
| cifar10 | isun | 79 | 139.18 | 11.57 | 0.628 | 7.77e-13 | ✗ |
| cifar10 | lsun cropped | 79 | 154.79 | 13.77 | 0.587 | 1.42e-13 | ✗ |
| cifar10 | lsun resize | 79 | 153.82 | 11.39 | 0.660 | 3.10e-12 | ✗ |
| cifar10 | places365 | 79 | 162.80 | 13.61 | 0.698 | 1.90e-11 | ✗ |
| cifar10 | svhn | 79 | 290.37 | 10.44 | 0.671 | 5.05e-12 | ✗ |
| cifar10 | test | 79 | 4.53 | 2.22 | 0.902 | 1.59e-05 | ✗ |
| cifar10 | textures | 79 | 90.40 | 13.74 | 0.561 | 5.28e-14 | ✗ |
| cifar10 | tinyimagenet | 79 | 156.37 | 11.93 | 0.619 | 5.28e-13 | ✗ |
| cifar100 | cifar10 | 79 | 208.52 | 29.41 | 0.840 | 8.69e-08 | ✗ |
| cifar100 | isun | 79 | 186.39 | 18.54 | 0.910 | 3.84e-05 | ✗ |
| cifar100 | lsun cropped | 79 | 205.52 | 23.13 | 0.898 | 1.13e-05 | ✗ |
| cifar100 | lsun resize | 79 | 201.78 | 18.55 | 0.920 | 1.08e-04 | ✗ |
| cifar100 | places365 | 79 | 209.54 | 27.30 | 0.854 | 2.52e-07 | ✗ |
| cifar100 | svhn | 79 | 328.81 | 18.08 | 0.882 | 2.61e-06 | ✗ |
| cifar100 | test | 79 | 53.66 | 24.50 | 0.858 | 3.38e-07 | ✗ |
| cifar100 | textures | 79 | 128.69 | 24.70 | 0.828 | 3.72e-08 | ✗ |
| cifar100 | tinyimagenet | 79 | 199.61 | 20.44 | 0.905 | 2.14e-05 | ✗ |
| supercifar100 | cifar10 | 79 | 259.47 | 31.45 | 0.698 | 1.83e-11 | ✗ |
| supercifar100 | isun | 79 | 230.39 | 9.07 | 0.955 | 7.15e-03 | ✗ |
| supercifar100 | lsun cropped | 79 | 244.94 | 22.95 | 0.901 | 1.54e-05 | ✗ |
| supercifar100 | lsun resize | 79 | 245.49 | 9.49 | 0.960 | 1.49e-02 | ✗ |
| supercifar100 | places365 | 79 | 256.10 | 37.64 | 0.650 | 1.96e-12 | ✗ |
| supercifar100 | svhn | 79 | 362.13 | 21.74 | 0.911 | 4.02e-05 | ✗ |
| supercifar100 | test | 79 | 130.08 | 47.57 | 0.665 | 3.89e-12 | ✗ |
| supercifar100 | textures | 79 | 159.49 | 39.37 | 0.743 | 1.90e-10 | ✗ |
| supercifar100 | tinyimagenet | 79 | 235.16 | 16.37 | 0.865 | 6.03e-07 | ✗ |
| tinyimagenet | cifar10 | 79 | 231.54 | 43.90 | 0.820 | 2.15e-08 | ✗ |
| tinyimagenet | cifar100 | 79 | 236.24 | 45.18 | 0.824 | 2.84e-08 | ✗ |
| tinyimagenet | isun | 79 | 212.93 | 44.65 | 0.839 | 8.19e-08 | ✗ |
| tinyimagenet | lsun cropped | 79 | 236.81 | 59.75 | 0.878 | 1.86e-06 | ✗ |
| tinyimagenet | lsun resize | 79 | 227.65 | 40.64 | 0.861 | 4.37e-07 | ✗ |
| tinyimagenet | places365 | 79 | 224.61 | 39.45 | 0.869 | 8.57e-07 | ✗ |
| tinyimagenet | svhn | 79 | 360.09 | 41.77 | 0.897 | 1.07e-05 | ✗ |
| tinyimagenet | test | 79 | 92.25 | 47.38 | 0.847 | 1.51e-07 | ✗ |
| tinyimagenet | textures | 79 | 164.03 | 54.08 | 0.877 | 1.64e-06 | ✗ |

## Notes on harmonization scope

Harmonization pools across all (architecture, paradigm, run, dropout, reward, csf) entries inside a `(source, eval_dataset)` cell. This means the harmonized target for ResNet18 rows is computed using μ, σ that also include ResNet18 rows themselves — a mild train/test leak when the cross-arch held-out evaluation runs in step 13. We accept this because (i) the harmonized values are the *target* the predictor outputs, not features it sees, and (ii) the headline regret metrics in §10 use raw AUGRC, not the harmonized value. If the leakage materially affects results, step 10's regression head can re-fit a per-fold harmonizer on training rows only.
