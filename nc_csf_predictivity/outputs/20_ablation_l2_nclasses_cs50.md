# Ablation — L2 per-CSF binary with n_classes added (Cs=50, cv=5)

**Date:** 2026-05-04
**Source:** `code/nc_csf_predictivity/ablations/l2_per_csf_nclasses_cs50.py`
**Model:** `LogisticRegressionCV(Cs=50, cv=5, penalty='l2')` per CSF
**Features:** 8 NC + n_classes (scaled) + source one-hot + regime one-hot = 15 features
**Comparison:** vs L2 Cs=50 *without* n_classes (step 23 ablation) and vs L2 Cs=10 headline (step 12).

## xarch — chosen C per CSF (median across folds)

```
                 csf  median_chosen_C
KPCA RecError global           0.0029
             NNGuide           0.0029
                  GE           0.0043
                 REN           0.0091
                  PE           0.0091
                pNML           0.0133
                NeCo           0.0133
            GradNorm           0.0133
                 MLS           0.0133
                 MSR           0.0133
                 CTM           0.0193
                 PCE           0.0281
              Energy           0.0281
            Residual           0.0596
                 ViM           0.0596
                fDBD           0.0596
          Confidence           0.0869
                 GEN           0.1265
 PCA RecError global           0.8286
                Maha           1.7575
```

### n_classes coefficient per CSF (averaged across folds)

Standardized-feature scale: positive ⇒ higher class count increases competitive probability.

```
                 csf  n_classes_coef
                Maha          -1.497
          Confidence          -1.228
              Energy          -0.607
 PCA RecError global          -0.537
                 ViM          -0.532
                NeCo          -0.483
            GradNorm          -0.321
                pNML          -0.289
            Residual          -0.272
             NNGuide          -0.210
                 MLS          -0.206
                  PE          -0.188
                  GE          -0.108
                 PCE          -0.084
                 MSR          -0.008
KPCA RecError global          -0.004
                 REN           0.090
                 GEN           0.194
                fDBD           0.347
                 CTM           0.695
```

## lopo — chosen C per CSF (median across folds)

```
                 csf  median_chosen_C
             NNGuide           0.0025
                  PE           0.0077
                  GE           0.0091
                 CTM           0.0118
                 MSR           0.0133
                 MLS           0.0133
                pNML           0.0133
            GradNorm           0.0133
KPCA RecError global           0.0172
                 PCE           0.0207
                 REN           0.0237
                NeCo           0.0281
              Energy           0.0345
                 ViM           0.0395
                fDBD           0.0596
 PCA RecError global           0.0596
          Confidence           0.0869
                 GEN           0.1067
            Residual           5.4287
                Maha           5.4287
```

### n_classes coefficient per CSF (averaged across folds)

Standardized-feature scale: positive ⇒ higher class count increases competitive probability.

```
                 csf  n_classes_coef
                Maha          -1.892
          Confidence          -1.070
            Residual          -0.955
                NeCo          -0.804
              Energy          -0.756
            GradNorm          -0.596
                 PCE          -0.592
                 ViM          -0.507
                 MSR          -0.368
                pNML          -0.353
 PCA RecError global          -0.350
                  PE          -0.325
             NNGuide          -0.317
                 MLS          -0.262
                  GE          -0.258
                 GEN          -0.175
KPCA RecError global           0.002
                 REN           0.068
                fDBD           0.239
                 CTM           0.701
```

## Set-regret comparison (imputed)

Three-way comparison: L2 Cs=10 headline (step 12), L2 Cs=50 without n_classes (step 23), and this run (L2 Cs=50 with n_classes added).

### xarch

```
regime    side  Cs10_headline  Cs50_no_nclasses  Cs50_with_nclasses  ci_lo  ci_hi  empty_share
  near     all          1.455            14.512              16.204 12.208 20.587        0.338
   mid     all          2.222            16.297              23.687 19.101 28.813        0.320
   far     all          1.488            20.538              20.308 14.876 25.988        0.420
  near    head         52.579            52.579              54.041 46.183 62.443        0.743
   mid    head         41.451            41.404              41.377 37.124 45.787        0.780
   far    head         39.365            39.838              39.836 33.775 46.060        0.860
  near feature          1.527            24.440              24.858 19.619 30.798        0.568
   mid feature          1.821             8.789              16.420 13.215 19.966        0.380
   far feature          1.454            25.156              22.787 16.433 29.732        0.520
```

### lopo

```
regime    side  Cs10_headline  Cs50_no_nclasses  Cs50_with_nclasses  ci_lo  ci_hi  empty_share
  near     all         18.409            27.012              22.881 19.686 26.320        0.149
   mid     all          7.980             9.296              10.134  8.671 11.809        0.073
   far     all         12.709            12.609              14.069 11.939 16.369        0.191
  near    head         39.897            41.501              38.021 34.952 41.052        0.470
   mid    head         28.100            26.681              25.569 24.054 27.314        0.512
   far    head         28.448            27.492              27.526 24.964 30.016        0.570
  near feature         22.154            30.542              33.427 29.883 37.018        0.321
   mid feature          8.848            10.779              12.821 11.162 14.638        0.128
   far feature         12.407            12.453              13.089 11.213 15.130        0.218
```

