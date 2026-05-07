# Ablation — `n_classes` ordinal instead of `source` one-hot

**Date:** 2026-05-04
**Source:** `code/nc_csf_predictivity/ablations/n_classes_feature.py`
**Mapping:** cifar10 → 10, supercifar100 → 19, cifar100 → 100, tinyimagenet → 200 (Jaeger's scorecard convention).
**Comparison:** vs with-source one-hot (step 11/12) and vs NC-only (step 18 ablation).

Headline metric: `set_regret_raw_imputed` (always-predicts via empty-set imputation; fair vs always-X baselines).

## xarch

### `predictor = per_csf_binary`

`side = all`

```
regime side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near  all            1.455             17.540              1.942                      16.085                    15.598
   mid  all            2.222             26.742              3.281                      24.520                    23.462
   far  all            1.488             10.873              2.221                       9.386                     8.652
```

`side = head`

```
regime side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near head            1.557             60.100             61.237                      58.543                    -1.137
   mid head            2.305             41.493             43.424                      39.188                    -1.931
   far head            1.554             39.834             39.832                      38.280                     0.002
```

`side = feature`

```
regime    side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near feature            1.527             11.611              1.477                      10.083                    10.134
   mid feature            1.821             21.474              2.805                      19.653                    18.669
   far feature            1.454              9.324              2.188                       7.870                     7.136
```

### `predictor = multilabel`

`side = all`

```
regime side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near  all            1.117             12.173             40.585                      11.056                   -28.413
   mid  all            2.911             18.174              5.072                      15.263                    13.102
   far  all            1.638             21.611             24.489                      19.973                    -2.878
```

`side = head`

```
regime side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near head            1.636             41.092             51.814                      39.456                   -10.722
   mid head            0.752             39.133             42.623                      38.381                    -3.490
   far head            1.878             38.194             38.176                      36.316                     0.018
```

`side = feature`

```
regime    side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near feature            1.715             21.086             31.767                      19.371                   -10.681
   mid feature            2.985             15.632              4.624                      12.647                    11.008
   far feature            1.481             23.893             28.842                      22.412                    -4.949
```

### `per_csf_binary` n_classes coefficient per CSF (xarch)

On the standardized n_classes scale (so the coefficient is the change in log-odds per 1 standard deviation increase in n_classes across the training pool).

```
                 csf  n_classes_coef
                Maha          -2.420
          Confidence          -1.041
              Energy          -0.735
                 ViM          -0.733
                NeCo          -0.729
 PCA RecError global          -0.576
                  PE          -0.382
             NNGuide          -0.333
            Residual          -0.257
            GradNorm          -0.239
                 PCE          -0.187
                 MLS          -0.152
                  GE          -0.145
                 MSR          -0.034
KPCA RecError global          -0.008
                pNML          -0.004
                 GEN           0.022
                 REN           0.053
                fDBD           0.067
                 CTM           0.466
```

## lopo

### `predictor = per_csf_binary`

`side = all`

```
regime side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near  all            4.245             24.584             10.023                      20.339                    14.561
   mid  all            2.820             11.802              6.105                       8.981                     5.696
   far  all            3.036             13.955              7.585                      10.919                     6.371
```

`side = head`

```
regime side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near head            2.214             37.765             42.725                      35.551                    -4.960
   mid head            1.859             20.406             22.948                      18.548                    -2.542
   far head            4.249             25.232             26.927                      20.983                    -1.695
```

`side = feature`

```
regime    side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near feature            6.740             36.135             13.508                      29.395                    22.627
   mid feature            3.779             15.567              8.036                      11.789                     7.531
   far feature            3.859             14.682              9.645                      10.824                     5.037
```

### `predictor = multilabel`

`side = all`

```
regime side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near  all            1.259             23.557             47.130                      22.298                   -23.573
   mid  all            2.410              8.470              8.516                       6.060                    -0.046
   far  all            2.193             10.849             11.800                       8.655                    -0.951
```

`side = head`

```
regime side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near head            1.771             40.956             43.264                      39.185                    -2.308
   mid head            1.828             24.353             27.882                      22.525                    -3.529
   far head            3.636             27.135             28.722                      23.500                    -1.586
```

`side = feature`

```
regime    side  with_source_raw  n_classes_imputed  no_source_imputed  Δ(n_classes − with_source)  Δ(n_classes − no_source)
  near feature            2.655             28.997             49.480                      26.342                   -20.483
   mid feature            3.418              9.931             10.286                       6.513                    -0.354
   far feature            3.141              9.485             13.088                       6.344                    -3.603
```

### `per_csf_binary` n_classes coefficient per CSF (lopo)

On the standardized n_classes scale (so the coefficient is the change in log-odds per 1 standard deviation increase in n_classes across the training pool).

```
                 csf  n_classes_coef
          Confidence         -12.761
                  PE          -6.824
                 PCE          -4.795
                 MSR          -4.034
            Residual          -3.335
            Residual          -3.237
                NeCo          -2.614
                Maha          -2.420
                Maha          -2.318
                Maha          -2.295
                 GEN          -2.170
              Energy          -1.872
            GradNorm          -1.708
                 ViM          -1.044
          Confidence          -1.041
          Confidence          -1.009
                pNML          -0.966
                  GE          -0.915
                 MLS          -0.802
             NNGuide          -0.767
                 ViM          -0.747
              Energy          -0.735
                 ViM          -0.733
                NeCo          -0.729
                fDBD          -0.644
                NeCo          -0.584
 PCA RecError global          -0.576
              Energy          -0.576
              Energy          -0.561
                NeCo          -0.553
                 REN          -0.489
          Confidence          -0.411
                  PE          -0.382
             NNGuide          -0.333
                  PE          -0.294
 PCA RecError global          -0.270
 PCA RecError global          -0.270
            Residual          -0.257
            GradNorm          -0.240
            GradNorm          -0.239
            GradNorm          -0.227
KPCA RecError global          -0.213
                 ViM          -0.197
                 PCE          -0.187
                 MLS          -0.152
                  GE          -0.145
                pNML          -0.127
                  GE          -0.113
                  PE          -0.100
                  GE          -0.100
                 PCE          -0.082
                 MLS          -0.080
             NNGuide          -0.056
                 MLS          -0.054
             NNGuide          -0.053
                 MSR          -0.034
                 GEN          -0.012
KPCA RecError global          -0.008
                pNML          -0.004
                pNML          -0.003
KPCA RecError global           0.006
                fDBD           0.014
                fDBD           0.014
                 CTM           0.016
                 PCE           0.020
                 GEN           0.022
                 REN           0.053
                 MSR           0.063
                fDBD           0.067
                 MSR           0.119
                 REN           0.160
                 GEN           0.178
                 REN           0.197
KPCA RecError global           0.203
                 CTM           0.435
                 CTM           0.466
                 CTM           3.807
```

