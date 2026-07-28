# Cross-family predictor transfer (rebuttal experiment E-C)

**Source:** `nc_csf_predictivity/evaluation/vit_cnn_transfer.py`
**Labels:** CNN direction = step-5 cliques; ViT direction = `cliques_vit.parquet` computed here with the same pipeline.
**Regret:** imputed set-regret, bootstrap 95% CI; Wilcoxon two-sided vs the direction's best fixed baseline.

## Direction 1: ViT -> CNN (new)

### vit_to_vgg13 (config=source)

`side = all`

```
regime           predictor  empty%  set_size  best_baseline  bl_regret wilcoxon_p
  near 48.88 [44.52,53.56]    24.1       1.3 Always-NNGuide       4.09   3.74e-71
   mid    5.95 [5.55,6.39]     0.0       1.8     Always-CTM       5.54   0.000235
   far 15.44 [13.01,18.08]     0.0       2.5     Always-CTM       5.54   8.43e-09
```

`side = head`

```
regime           predictor  empty%  set_size        best_baseline  bl_regret wilcoxon_p
  near 55.68 [52.51,59.07]    88.1       0.3 Oracle-on-train (PE)       3.42  4.06e-117
   mid 33.83 [32.13,35.53]    86.0       0.3        Always-Energy       4.78  2.86e-158
   far 24.57 [21.87,27.52]    39.6       0.8 Oracle-on-train (PE)       3.66   4.98e-42
```

`side = feature`

```
regime           predictor  empty%  set_size  best_baseline  bl_regret wilcoxon_p
  near 52.17 [47.81,56.92]    27.6       1.0 Always-NNGuide       3.42   1.01e-80
   mid 11.58 [10.15,13.16]     4.4       1.5     Always-CTM       4.67   4.64e-21
   far 24.08 [21.11,27.24]     2.0       1.6     Always-CTM       5.10   1.14e-28
```

### vit_to_vgg13 (config=none)

`side = all`

```
regime           predictor  empty%  set_size  best_baseline  bl_regret wilcoxon_p
  near 20.26 [18.50,22.05]     9.7       1.7 Always-NNGuide       4.09   5.22e-59
   mid    6.03 [5.63,6.48]     0.0       1.9     Always-CTM       5.54   9.12e-05
   far 12.94 [11.07,14.88]     0.0       2.5     Always-CTM       5.54      1e-09
```

`side = head`

```
regime           predictor  empty%  set_size        best_baseline  bl_regret wilcoxon_p
  near 54.79 [51.53,58.33]    87.3       0.3 Oracle-on-train (PE)       3.42  7.02e-115
   mid 34.02 [32.30,35.69]    87.6       0.4        Always-Energy       4.78  6.76e-160
   far 28.21 [25.48,31.16]    43.8       0.8 Oracle-on-train (PE)       3.66   9.79e-63
```

`side = feature`

```
regime           predictor  empty%  set_size  best_baseline  bl_regret wilcoxon_p
  near 24.12 [22.07,26.15]    18.1       1.4 Always-NNGuide       3.42   3.78e-76
   mid   9.61 [8.41,10.96]     4.0       1.6     Always-CTM       4.67   7.98e-18
   far 22.72 [20.26,25.27]     2.0       1.6     Always-CTM       5.10   4.98e-30
```

### vit_to_resnet18 (config=source)

`side = all`

```
regime           predictor  empty%  set_size best_baseline  bl_regret wilcoxon_p
  near 20.89 [14.85,27.36]     6.1       2.1    Always-CTM       2.07   4.37e-10
   mid    6.08 [4.86,7.41]     0.0       2.0    Always-CTM       3.52    0.00166
   far 17.17 [12.10,23.24]     0.0       2.6    Always-CTM       2.40   1.53e-09
```

`side = head`

```
regime           predictor  empty%  set_size        best_baseline  bl_regret wilcoxon_p
  near 37.54 [31.58,43.96]    64.9       1.0 Oracle-on-train (PE)       3.37   1.45e-19
   mid 54.08 [47.78,60.73]    76.0       0.5        Always-Energy       4.20    1.7e-32
   far 35.03 [26.22,44.70]    29.0       1.0           Always-MLS       3.77   2.75e-09
```

`side = feature`

```
regime           predictor  empty%  set_size best_baseline  bl_regret wilcoxon_p
  near 62.06 [50.67,73.38]    41.2       1.1    Always-CTM       1.55   7.27e-20
   mid 17.11 [12.63,22.26]     7.0       1.5    Always-CTM       2.92   3.46e-08
   far 27.62 [20.66,34.84]     4.0       1.6    Always-CTM       2.19    1.6e-12
```

### vit_to_resnet18 (config=none)

`side = all`

```
regime          predictor  empty%  set_size best_baseline  bl_regret wilcoxon_p
  near  8.77 [6.71,11.07]     0.0       2.6    Always-CTM       2.07   2.93e-08
   mid   7.11 [5.37,9.23]     4.0       2.3    Always-CTM       3.52    0.00852
   far 11.50 [8.21,15.24]     0.0       2.5    Always-CTM       2.40   7.27e-08
```

`side = head`

```
regime           predictor  empty%  set_size        best_baseline  bl_regret wilcoxon_p
  near 29.56 [24.81,34.73]    54.1       1.2 Oracle-on-train (PE)       3.37   4.62e-18
   mid 39.84 [33.83,45.90]    60.0       0.7        Always-Energy       4.20    4.1e-27
   far 30.35 [22.67,38.60]    49.0       0.9           Always-MLS       3.77   1.25e-11
```

`side = feature`

```
regime           predictor  empty%  set_size best_baseline  bl_regret wilcoxon_p
  near 48.60 [38.30,59.34]    28.4       1.5    Always-CTM       1.55   3.98e-16
   mid 17.57 [12.93,22.50]    10.0       1.6    Always-CTM       2.92    1.2e-09
   far 24.43 [18.10,31.31]     2.0       1.6    Always-CTM       2.19   3.18e-12
```

## Direction 2: CNN -> ViT (existing lopo_modelvit fold)

### lopo_modelvit (config=source)

`side = all`

```
regime        predictor  empty%  set_size best_baseline  bl_regret wilcoxon_p
  near 1.90 [1.42,2.43]     0.0       7.3    Always-MLS       6.25   1.21e-15
   mid 3.66 [2.93,4.49]     0.0       4.2   Always-fDBD      12.20   8.67e-11
   far 2.07 [1.26,2.99]     0.0       5.9   Always-fDBD      10.62    2.7e-09
```

`side = head`

```
regime         predictor  empty%  set_size best_baseline  bl_regret wilcoxon_p
  near  3.73 [2.39,5.32]     0.0       4.7    Always-MLS       5.26   6.91e-11
   mid  4.70 [3.54,5.96]     7.1       2.2    Always-MSR       8.09   1.05e-10
   far 8.67 [5.68,12.22]     0.0       3.3 Always-Energy      10.05     0.0991
```

`side = feature`

```
regime        predictor  empty%  set_size  best_baseline  bl_regret wilcoxon_p
  near 3.62 [2.38,5.28]     3.6       2.6 Always-NNGuide       6.19   2.16e-05
   mid 6.30 [4.27,8.69]     8.6       2.0    Always-fDBD      11.48   7.12e-05
   far 3.29 [1.72,5.28]     2.9       2.6    Always-fDBD      10.34   9.69e-06
```

