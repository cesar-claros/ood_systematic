# Ablation — L2 per-CSF binary with Cs=50, cv=5

**Date:** 2026-05-04
**Source:** `code/nc_csf_predictivity/ablations/l2_per_csf_binary_cs50.py`
**Model:** `LogisticRegressionCV(Cs=50, cv=5, penalty='l2')` per CSF
**Comparison:** vs L2 with Cs=10 (step 12 headline).

## xarch — chosen C per CSF (median across folds)

```
                 csf  median_chosen_C
             NNGuide           0.0014
                  GE           0.0043
KPCA RecError global           0.0063
                pNML           0.0091
                 REN           0.0133
                  PE           0.0133
            GradNorm           0.0133
                NeCo           0.0193
                 CTM           0.0193
                 MLS           0.0193
                 MSR           0.0193
                fDBD           0.0409
                 PCE           0.0409
              Energy           0.0596
            Residual           0.0596
                 ViM           0.0869
                 GEN           0.1265
          Confidence           0.2683
 PCA RecError global           0.8286
                Maha           1.7575
```

## lopo — chosen C per CSF (median across folds)

```
                 csf  median_chosen_C
             NNGuide           0.0014
                  GE           0.0091
                 CTM           0.0097
                  PE           0.0112
                pNML           0.0112
            GradNorm           0.0133
KPCA RecError global           0.0142
                 MLS           0.0163
                 MSR           0.0163
                 REN           0.0193
                 PCE           0.0271
                NeCo           0.0281
                fDBD           0.0409
                 ViM           0.0575
              Energy           0.0596
 PCA RecError global           0.0596
                 GEN           0.1067
          Confidence           0.4186
                Maha           2.5595
            Residual           5.4287
```

## Set-regret comparison: L2 Cs=50 vs L2 Cs=10 (headline; both imputed)

### xarch

```
regime    side  Cs10_imputed  Cs50_imputed  Cs50_ci_lo  Cs50_ci_hi  Cs50_empty_share  delta_Cs50_minus_Cs10
  near     all         1.455        14.512      11.116      18.203             0.324                 13.056
   mid     all         2.222        16.297      11.991      20.958             0.180                 14.075
   far     all         1.488        20.538      15.230      26.294             0.420                 19.050
  near    head        52.579        52.579      44.482      60.978             0.730                  0.000
   mid    head        41.451        41.404      37.161      45.808             0.780                 -0.047
   far    head        39.365        39.838      33.775      46.062             0.860                  0.472
  near feature         1.527        24.440      19.244      30.415             0.568                 22.913
   mid feature         1.821         8.789       6.142      11.740             0.180                  6.968
   far feature         1.454        25.156      18.378      32.503             0.540                 23.702
```

### lopo

```
regime    side  Cs10_imputed  Cs50_imputed  Cs50_ci_lo  Cs50_ci_hi  Cs50_empty_share  delta_Cs50_minus_Cs10
  near     all        18.409        27.012      23.476      30.743             0.157                  8.604
   mid     all         7.980         9.296       7.699      11.082             0.045                  1.317
   far     all        12.709        12.609      10.334      15.061             0.113                 -0.100
  near    head        39.897        41.501      38.300      44.663             0.499                  1.604
   mid    head        28.100        26.681      25.083      28.473             0.534                 -1.419
   far    head        28.448        27.492      25.010      30.030             0.582                 -0.956
  near feature        22.154        30.542      27.096      34.251             0.219                  8.388
   mid feature         8.848        10.779       9.197      12.515             0.060                  1.932
   far feature        12.407        12.453      10.312      14.722             0.116                  0.046
```

## Coefficient correlation: L2 Cs=50 vs L2 Cs=10 (per CSF)

### xarch

```
                 csf  n_features  pearson_r_cs50_vs_cs10
                 CTM          14                   0.845
                 MSR          14                   0.958
                 REN          14                   0.963
                 MLS          14                   0.974
                 GEN          14                   0.975
                  PE          14                   0.980
                NeCo          14                   0.991
            GradNorm          14                   0.993
 PCA RecError global          14                   0.993
             NNGuide          14                   0.993
                 ViM          14                   0.996
                Maha          14                   0.998
                pNML          14                   0.998
                  GE          14                   0.999
              Energy          14                   0.999
KPCA RecError global          14                   1.000
                 PCE          14                   1.000
            Residual          14                   1.000
          Confidence          14                   1.000
                fDBD          14                   1.000
```

### lopo

```
                 csf  n_features  pearson_r_cs50_vs_cs10
                 CTM          14                   0.973
                fDBD          14                   0.985
                 REN          14                   0.992
 PCA RecError global          14                   0.993
KPCA RecError global          14                   0.994
                 MLS          14                   0.994
             NNGuide          14                   0.994
            Residual          14                   0.996
              Energy          14                   0.997
                NeCo          14                   0.997
                  PE          14                   0.998
                 ViM          14                   0.999
                 PCE          14                   0.999
                pNML          14                   0.999
            GradNorm          14                   0.999
                 GEN          14                   0.999
          Confidence          14                   0.999
                 MSR          14                   0.999
                  GE          14                   1.000
                Maha          14                   1.000
```

