# Ablation — class-weighted + per-arch standardization (options b, c)

**Date:** 2026-05-05
**Source:** `code/nc_csf_predictivity/ablations/calibration_balanced_perarch.py`
**Model:** `LogisticRegressionCV(Cs=50, cv=5, penalty='l2')` per CSF
**Variants:**
- `baseline_cs50`: pooled standardization, class_weight=None (step 23)
- `class_weighted`: pooled standardization, class_weight='balanced'
- `per_arch_std`: per-architecture standardization, class_weight=None
- `class_weighted_perarch_std`: both fixes

Reference numbers:
- `Cs10_headline` = step 12 (over-regularized; 0% empty by collapsing CTM to constant)
- `Cs50_baseline` = step 23 (clean comparison point)

## xarch

### `side = all` (imputed regret + empty share)

```
regime  Cs10_headline  Cs50_baseline    empty%_baseline  (b)_class_weighted    empty%_(b)  (c)_per_arch_std    empty%_(c)  (b+c)_combined    empty%_(b+c)
  near           1.46          14.51               0.32                3.32           0.0             14.15          0.32            1.88            0.01
   mid           2.22          16.30               0.18                1.72           0.0              4.96          0.02            1.44            0.00
   far           1.49          20.54               0.42                0.93           0.0             25.06          0.50            0.88            0.00
```

### `side = head` (imputed regret + empty share)

```
regime  Cs10_headline  Cs50_baseline    empty%_baseline  (b)_class_weighted    empty%_(b)  (c)_per_arch_std    empty%_(c)  (b+c)_combined    empty%_(b+c)
  near          52.58          52.58               0.73               32.59          0.36             22.33          0.49            4.91            0.09
   mid          41.45          41.40               0.78               20.59          0.24             34.18          0.66           16.96            0.19
   far          39.37          39.84               0.86               18.86          0.26             30.17          0.68           11.27            0.18
```

### `side = feature` (imputed regret + empty share)

```
regime  Cs10_headline  Cs50_baseline    empty%_baseline  (b)_class_weighted    empty%_(b)  (c)_per_arch_std    empty%_(c)  (b+c)_combined    empty%_(b+c)
  near           1.53          24.44               0.57                8.31          0.19             14.69          0.46            6.72            0.15
   mid           1.82           8.79               0.18                3.48          0.00              4.89          0.02            1.85            0.02
   far           1.45          25.16               0.54                3.78          0.10             15.25          0.58            1.11            0.00
```

## lopo

### `side = all` (imputed regret + empty share)

```
regime  Cs10_headline  Cs50_baseline    empty%_baseline  (b)_class_weighted    empty%_(b)  (c)_per_arch_std    empty%_(c)  (b+c)_combined    empty%_(b+c)
  near          18.41          27.01               0.16               38.94          0.27              3.01          0.02           16.05            0.12
   mid           7.98           9.30               0.04               11.20          0.10              4.03          0.02            5.00            0.06
   far          12.71          12.61               0.11               11.64          0.13              8.73          0.13            4.50            0.07
```

### `side = head` (imputed regret + empty share)

```
regime  Cs10_headline  Cs50_baseline    empty%_baseline  (b)_class_weighted    empty%_(b)  (c)_per_arch_std    empty%_(c)  (b+c)_combined    empty%_(b+c)
  near          39.90          41.50               0.50               33.61          0.37             26.06          0.37           14.56            0.20
   mid          28.10          26.68               0.53               16.87          0.28             21.62          0.43           13.96            0.27
   far          28.45          27.49               0.58               20.25          0.35             25.60          0.56           17.67            0.36
```

### `side = feature` (imputed regret + empty share)

```
regime  Cs10_headline  Cs50_baseline    empty%_baseline  (b)_class_weighted    empty%_(b)  (c)_per_arch_std    empty%_(c)  (b+c)_combined    empty%_(b+c)
  near          22.15          30.54               0.22               40.69          0.32              7.34          0.11           19.32            0.16
   mid           8.85          10.78               0.06               12.93          0.13              8.72          0.11            6.06            0.09
   far          12.41          12.45               0.12               12.06          0.15              9.01          0.16            5.52            0.10
```

