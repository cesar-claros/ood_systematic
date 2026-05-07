# Ablation — calibrated (b+c) across three context-feature configs

**Date:** 2026-05-05
**Source:** `code/nc_csf_predictivity/ablations/calibration_features.py`
**Calibration (held constant):** L2 Cs=50, cv=5, class_weight='balanced', per-architecture NC standardization.
**Configs:**
- `source`: NC + source one-hot + regime
- `n_classes`: NC + n_classes (scaled) + regime
- `none`: NC + regime only

## xarch

### `side = all` (imputed regret + empty share)

```
regime  src_regret  src_empty%  ncl_regret  ncl_empty%  none_regret  none_empty%
  near        1.88        0.01        1.09         0.0        14.31         0.23
   mid        1.44        0.00        4.27         0.0        16.08         0.16
   far        0.88        0.00        2.26         0.0         7.23         0.18
```

### `side = head` (imputed regret + empty share)

```
regime  src_regret  src_empty%  ncl_regret  ncl_empty%  none_regret  none_empty%
  near        4.91        0.09        7.47        0.13        18.07         0.41
   mid       16.96        0.19       18.24        0.19        34.35         0.67
   far       11.27        0.18       11.86        0.18        28.90         0.66
```

### `side = feature` (imputed regret + empty share)

```
regime  src_regret  src_empty%  ncl_regret  ncl_empty%  none_regret  none_empty%
  near        6.72        0.15        2.29        0.01         23.4         0.34
   mid        1.85        0.02        4.31        0.00          9.8         0.16
   far        1.11        0.00        2.87        0.06          6.1         0.18
```

### Coefficient stability across configs (xarch)

Pearson correlation between NC coefficient vectors per CSF across config pairs (averaged across folds).

```
                 csf  r(src,ncl)  r(src,none)  r(ncl,none)
          Confidence       0.953        0.479        0.535
                pNML       0.820        0.708        0.911
                 ViM       0.995        0.898        0.886
                 GEN       0.921        0.919        1.000
                 CTM       0.957        0.932        0.961
            GradNorm       0.988        0.947        0.984
                fDBD       0.901        0.957        0.976
                Maha       0.927        0.959        0.945
                 REN       0.970        0.963        0.999
                  PE       0.974        0.975        0.915
            Residual       0.961        0.976        0.950
                 PCE       0.987        0.982        0.996
                 MSR       0.990        0.983        0.998
KPCA RecError global       0.981        0.984        0.999
                 MLS       0.962        0.988        0.965
              Energy       0.978        0.991        0.947
                NeCo       0.989        0.995        0.975
 PCA RecError global       1.000        0.995        0.996
             NNGuide       0.786        0.998        0.778
                  GE       1.000        1.000        0.999
```

## lopo

### `side = all` (imputed regret + empty share)

```
regime  src_regret  src_empty%  ncl_regret  ncl_empty%  none_regret  none_empty%
  near       16.05        0.12       22.33        0.13        12.42         0.12
   mid        5.00        0.06        7.40        0.05         6.04         0.05
   far        4.50        0.07        5.51        0.06         4.71         0.05
```

### `side = head` (imputed regret + empty share)

```
regime  src_regret  src_empty%  ncl_regret  ncl_empty%  none_regret  none_empty%
  near       14.56        0.20       27.85        0.32        13.28         0.24
   mid       13.96        0.27       17.93        0.30        16.39         0.34
   far       17.67        0.36       15.77        0.28        19.05         0.37
```

### `side = feature` (imputed regret + empty share)

```
regime  src_regret  src_empty%  ncl_regret  ncl_empty%  none_regret  none_empty%
  near       19.32        0.16       23.59        0.16        26.53         0.23
   mid        6.06        0.09        6.50        0.05         8.34         0.10
   far        5.52        0.10        7.24        0.09         7.39         0.10
```

### Coefficient stability across configs (lopo)

Pearson correlation between NC coefficient vectors per CSF across config pairs (averaged across folds).

```
                 csf  r(src,ncl)  r(src,none)  r(ncl,none)
          Confidence      -0.218       -0.273        0.773
                 MLS       0.794       -0.223       -0.399
              Energy       0.902       -0.024       -0.162
                NeCo       0.209        0.023       -0.253
                pNML       0.899        0.159        0.379
                 PCE       0.953        0.309        0.409
                 REN       0.985        0.322        0.408
                  GE       0.918        0.548        0.561
                 CTM       0.701        0.562        0.675
                 MSR       0.807        0.624        0.773
            GradNorm       0.995        0.680        0.723
                 ViM       0.961        0.686        0.842
                 GEN       0.753        0.710        0.980
KPCA RecError global       0.816        0.814        0.989
            Residual       0.936        0.919        0.904
                Maha       0.852        0.922        0.937
                  PE       0.993        0.941        0.936
             NNGuide       0.865        0.968        0.773
                fDBD       0.915        0.974        0.938
 PCA RecError global       0.999        0.994        0.996
```

