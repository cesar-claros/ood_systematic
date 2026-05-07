# Ablation — calibrated (b+c) with CLIQUE labels, 3 feature configs

**Date:** 2026-05-05
**Source:** `code/nc_csf_predictivity/ablations/calibration_features_clique.py`
**Model:** L2 Cs=50, cv=5, class_weight='balanced', NC pre-standardized per architecture.
**Label rule:** `clique` (per-(paradigm, source, dropout, reward, regime) Friedman-Conover top cliques from step 5).

## xarch

### `side = all` (imputed regret; clique label rule)

```
regime  clq_src  clq_ncl  clq_none  wer_src best_baseline  bl_regret
  near     1.02     1.24      1.45     1.88    Always-CTM       2.07
   mid     1.18     1.24      1.44     1.44    Always-CTM       3.52
   far     0.39     0.36      0.40     0.88    Always-CTM       2.40
```

Empty-set share (clique):

```
regime  clq_src_empty%  clq_ncl_empty%  clq_none_empty%
  near             0.0             0.0              0.0
   mid             0.0             0.0              0.0
   far             0.0             0.0              0.0
```

### `side = head` (imputed regret; clique label rule)

```
regime  clq_src  clq_ncl  clq_none  wer_src best_baseline  bl_regret
  near     1.41     1.06      1.47     4.91    Always-MLS       3.49
   mid    11.66    13.82     10.02    16.96 Always-Energy       4.20
   far     4.99     3.35      1.56    11.27    Always-MLS       3.77
```

Empty-set share (clique):

```
regime  clq_src_empty%  clq_ncl_empty%  clq_none_empty%
  near            0.00            0.00             0.00
   mid            0.19            0.19             0.15
   far            0.04            0.06             0.00
```

### `side = feature` (imputed regret; clique label rule)

```
regime  clq_src  clq_ncl  clq_none  wer_src best_baseline  bl_regret
  near     4.99    11.41      6.55     6.72    Always-CTM       1.55
   mid     1.95     3.29      1.86     1.85    Always-CTM       2.92
   far     0.75     0.74      0.77     1.11    Always-CTM       2.19
```

Empty-set share (clique):

```
regime  clq_src_empty%  clq_ncl_empty%  clq_none_empty%
  near            0.00            0.03              0.0
   mid            0.02            0.01              0.0
   far            0.00            0.00              0.0
```

## lopo

### `side = all` (imputed regret; clique label rule)

```
regime  clq_src  clq_ncl  clq_none  wer_src  best_baseline  bl_regret
  near    12.42     1.56      1.32    16.05 Always-NNGuide       4.48
   mid     4.22     2.39      2.54     5.00     Always-CTM       6.52
   far     3.31     1.17      1.38     4.50     Always-CTM       5.68
```

Empty-set share (clique):

```
regime  clq_src_empty%  clq_ncl_empty%  clq_none_empty%
  near            0.06             0.0              0.0
   mid            0.01             0.0              0.0
   far            0.01             0.0              0.0
```

### `side = head` (imputed regret; clique label rule)

```
regime  clq_src  clq_ncl  clq_none  wer_src   best_baseline  bl_regret
  near    21.93    11.47      2.10    14.56 Oracle-on-train       3.57
   mid    19.40    17.72     13.99    13.96      Always-MLS       5.34
   far    10.02     8.93      7.99    17.67 Oracle-on-train       4.41
```

Empty-set share (clique):

```
regime  clq_src_empty%  clq_ncl_empty%  clq_none_empty%
  near            0.21            0.10             0.00
   mid            0.31            0.29             0.21
   far            0.09            0.11             0.08
```

### `side = feature` (imputed regret; clique label rule)

```
regime  clq_src  clq_ncl  clq_none  wer_src  best_baseline  bl_regret
  near    15.59     6.65      3.12    19.32 Always-NNGuide       3.74
   mid     7.15     5.29      5.34     6.06     Always-CTM       5.71
   far     5.02     2.92      3.24     5.52     Always-CTM       5.30
```

Empty-set share (clique):

```
regime  clq_src_empty%  clq_ncl_empty%  clq_none_empty%
  near            0.07            0.04             0.00
   mid            0.06            0.03             0.05
   far            0.03            0.02             0.01
```

