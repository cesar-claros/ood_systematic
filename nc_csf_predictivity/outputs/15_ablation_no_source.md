# Ablation — NC-only features (drop `source`)

**Date:** 2026-05-04
**Source:** `code/nc_csf_predictivity/ablations/no_source.py`
**Hypothesis:** NC metrics alone carry enough information to predict CSF competitive sets. If true, removing `source` from the categorical features should not materially hurt the predictor's regret on held-out architectures or paradigms.

**Comparison:** with-source numbers come from the headline step-13 metrics (regression top-1 + within_eps_rank for both binary heads). No-source numbers come from this script's retrained predictors on the same splits.

## xarch

### `side = all`

```
regime                      predictor  with_source  no_source  delta_no_minus_with
  near                     regression        4.049      4.049                0.000
  near     multilabel/within_eps_rank        1.117     40.585               39.469
  near per_csf_binary/within_eps_rank        1.455      1.942                0.487
   mid                     regression        4.792      4.792                0.000
   mid     multilabel/within_eps_rank        2.911      5.072                2.161
   mid per_csf_binary/within_eps_rank        2.222      3.281                1.058
   far                     regression        4.624      4.624                0.000
   far     multilabel/within_eps_rank        1.638     24.489               22.851
   far per_csf_binary/within_eps_rank        1.488      2.221                0.734
```

### `side = head`

```
regime                      predictor  with_source  no_source  delta_no_minus_with
  near                     regression        3.492      3.492                0.000
  near     multilabel/within_eps_rank        1.636     51.814               50.178
  near per_csf_binary/within_eps_rank        1.557     61.237               59.680
   mid                     regression        4.438      4.438                0.000
   mid     multilabel/within_eps_rank        0.752     42.623               41.870
   mid per_csf_binary/within_eps_rank        2.305     43.424               41.118
   far                     regression        3.773      3.773                0.000
   far     multilabel/within_eps_rank        1.878     38.176               36.298
   far per_csf_binary/within_eps_rank        1.554     39.832               38.279
```

### `side = feature`

```
regime                      predictor  with_source  no_source  delta_no_minus_with
  near                     regression        3.530      3.530                0.000
  near     multilabel/within_eps_rank        1.715     31.767               30.052
  near per_csf_binary/within_eps_rank        1.527      1.477               -0.051
   mid                     regression        4.191      4.191                0.000
   mid     multilabel/within_eps_rank        2.985      4.624                1.639
   mid per_csf_binary/within_eps_rank        1.821      2.805                0.984
   far                     regression        4.409      4.409                0.000
   far     multilabel/within_eps_rank        1.481     28.842               27.361
   far per_csf_binary/within_eps_rank        1.454      2.188                0.734
```

### Spearman ρ side-asymmetry — no-source

```
regime    side   n  spearman_rho_mean  spearman_rho_ci_lo  spearman_rho_ci_hi
   far     all 100              0.410               0.330               0.483
   far feature 100              0.546               0.449               0.631
   far    head 100              0.325               0.273               0.377
   mid     all 200              0.454               0.396               0.507
   mid feature 200              0.557               0.488               0.619
   mid    head 200              0.374               0.335               0.411
  near     all 148              0.631               0.589               0.671
  near feature 148              0.753               0.702               0.801
  near    head 148              0.462               0.425               0.498
```

### Spearman ρ side-asymmetry — with-source (reference)

```
regime    side   n  spearman_rho_mean  spearman_rho_ci_lo  spearman_rho_ci_hi
   far     all 100              0.410               0.330               0.483
   far feature 100              0.546               0.449               0.631
   far    head 100              0.325               0.273               0.377
   mid     all 200              0.454               0.396               0.507
   mid feature 200              0.557               0.488               0.619
   mid    head 200              0.374               0.335               0.411
  near     all 148              0.631               0.589               0.671
  near feature 148              0.753               0.702               0.801
  near    head 148              0.462               0.425               0.498
```

## lopo

### `side = all`

```
regime                      predictor  with_source  no_source  delta_no_minus_with
  near                     regression        4.818      4.818                0.000
  near     multilabel/within_eps_rank        1.259     47.130               45.871
  near per_csf_binary/within_eps_rank        4.245     10.023                5.778
   mid                     regression        7.291      7.291                0.000
   mid     multilabel/within_eps_rank        2.410      8.516                6.106
   mid per_csf_binary/within_eps_rank        2.820      6.105                3.285
   far                     regression        6.871      6.871                0.000
   far     multilabel/within_eps_rank        2.193     11.800                9.607
   far per_csf_binary/within_eps_rank        3.036      7.585                4.548
```

### `side = head`

```
regime                      predictor  with_source  no_source  delta_no_minus_with
  near                     regression        4.897      4.897                0.000
  near     multilabel/within_eps_rank        1.771     43.264               41.493
  near per_csf_binary/within_eps_rank        2.214     42.725               40.511
   mid                     regression        7.302      7.302                0.000
   mid     multilabel/within_eps_rank        1.828     27.882               26.054
   mid per_csf_binary/within_eps_rank        1.859     22.948               21.089
   far                     regression        5.707      5.707                0.000
   far     multilabel/within_eps_rank        3.636     28.722               25.086
   far per_csf_binary/within_eps_rank        4.249     26.927               22.678
```

### `side = feature`

```
regime                      predictor  with_source  no_source  delta_no_minus_with
  near                     regression        4.079      4.079                0.000
  near     multilabel/within_eps_rank        2.655     49.480               46.826
  near per_csf_binary/within_eps_rank        6.740     13.508                6.768
   mid                     regression        6.479      6.479                0.000
   mid     multilabel/within_eps_rank        3.418     10.286                6.868
   mid per_csf_binary/within_eps_rank        3.779      8.036                4.257
   far                     regression        6.489      6.489                0.000
   far     multilabel/within_eps_rank        3.141     13.088                9.947
   far per_csf_binary/within_eps_rank        3.859      9.645                5.787
```

### Spearman ρ side-asymmetry — no-source

```
regime    side    n  spearman_rho_mean  spearman_rho_ci_lo  spearman_rho_ci_hi
   far     all  670              0.346               0.316               0.375
   far feature  670              0.433               0.391               0.473
   far    head  670              0.262               0.239               0.285
   mid     all 1340              0.421               0.399               0.442
   mid feature 1340              0.540               0.513               0.566
   mid    head 1340              0.295               0.275               0.313
  near     all  998              0.531               0.513               0.550
  near feature  998              0.684               0.659               0.708
  near    head  998              0.356               0.339               0.373
```

### Spearman ρ side-asymmetry — with-source (reference)

```
regime    side    n  spearman_rho_mean  spearman_rho_ci_lo  spearman_rho_ci_hi
   far     all  670              0.346               0.316               0.375
   far feature  670              0.433               0.391               0.473
   far    head  670              0.262               0.239               0.285
   mid     all 1340              0.421               0.399               0.442
   mid feature 1340              0.540               0.513               0.566
   mid    head 1340              0.295               0.275               0.313
  near     all  998              0.531               0.513               0.550
  near feature  998              0.684               0.659               0.708
  near    head  998              0.356               0.339               0.373
```

