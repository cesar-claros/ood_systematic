# Ablation — Random forest per CSF (non-linear feature importance)

**Date:** 2026-05-04
**Source:** `code/nc_csf_predictivity/ablations/random_forest.py`
**Model:** RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5)
**Features, label rule, training pool, splits:** identical to the headline `per_csf_binary` from steps 11–12.

## CTM specific check — does RF find non-linear NC signal?

### xarch — CTM RF feature importance (averaged across folds)

```
feature
source_tinyimagenet     0.140
equinorm_uc             0.120
self_duality            0.114
equiangular_wc          0.093
var_collapse            0.087
equiangular_uc          0.082
max_equiangular_wc      0.078
max_equiangular_uc      0.073
equinorm_wc             0.069
regime_near             0.048
source_supercifar100    0.033
regime_far              0.021
source_cifar100         0.021
regime_mid              0.020
```

Group totals (NC vs source vs regime):
- NC features: 0.717
- source one-hots: 0.195
- regime one-hots: 0.089

### lopo — CTM RF feature importance (averaged across folds)

```
feature
source_tinyimagenet     0.128
self_duality            0.115
equinorm_uc             0.110
equiangular_uc          0.089
equiangular_wc          0.089
var_collapse            0.087
equinorm_wc             0.082
max_equiangular_uc      0.076
max_equiangular_wc      0.075
regime_near             0.055
source_supercifar100    0.032
regime_far              0.022
regime_mid              0.021
source_cifar100         0.018
```

Group totals (NC vs source vs regime):
- NC features: 0.724
- source one-hots: 0.178
- regime one-hots: 0.098

## Set-regret comparison (RF vs linear per_csf binary)

### xarch

```
regime    side  linear_imputed  rf_imputed  rf_ci_lo  rf_ci_hi  rf_empty_share   delta
  near     all           1.455       7.965     5.536    10.691           0.176   6.510
   mid     all           2.222      18.840    15.178    22.474           0.320  16.617
   far     all           1.488      21.850    16.749    27.009           0.520  20.362
  near    head          52.579      41.658    34.099    49.530           0.622 -10.920
   mid    head          41.451      36.740    32.679    40.985           0.740  -4.711
   far    head          39.365      34.044    28.700    39.711           0.800  -5.322
  near feature           1.527      11.630     8.566    15.031           0.351  10.103
   mid feature           1.821      18.428    15.261    21.953           0.400  16.607
   far feature           1.454      12.706    10.418    15.099           0.540  11.252
```

### lopo

```
regime    side  linear_imputed  rf_imputed  rf_ci_lo  rf_ci_hi  rf_empty_share   delta
  near     all          18.409       5.518     4.447     6.676           0.072 -12.891
   mid     all           7.980       7.038     6.028     8.146           0.084  -0.942
   far     all          12.709       6.652     5.318     8.345           0.075  -6.057
  near    head          39.897      43.534    40.490    46.646           0.545   3.637
   mid    head          28.100      26.727    25.096    28.465           0.536  -1.373
   far    head          28.448      32.705    29.714    35.801           0.636   4.257
  near feature          22.154       7.892     6.797     9.017           0.154 -14.262
   mid feature           8.848       6.102     5.384     6.845           0.131  -2.746
   far feature          12.407       4.668     4.069     5.344           0.104  -7.739
```

