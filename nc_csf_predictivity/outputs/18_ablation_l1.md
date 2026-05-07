# Ablation — L1-regularized per-CSF binary head

**Date:** 2026-05-04
**Source:** `code/nc_csf_predictivity/ablations/l1_per_csf_binary.py`
**Model:** `LogisticRegressionCV(Cs=100, cv=5, penalty='l1', solver='saga')` per CSF
**Comparison:** vs L2 per_csf binary (step 12).

## xarch — sparsity per CSF

Per-CSF: share of features with |coef| < 1e-6 (i.e., L1 zeroed them out), and the median chosen C across folds.

```
                 csf  zero_share  median_chosen_C
                Maha       0.357            0.911
                NeCo       0.500            0.067
                 CTM       0.500            0.056
          Confidence       0.500            0.081
              Energy       0.500            0.081
                 MLS       0.571            0.067
 PCA RecError global       0.571            0.248
                 ViM       0.643            0.206
                  PE       0.643            0.046
                 PCE       0.643            0.056
                 MSR       0.714            0.046
                  GE       0.786            0.022
                fDBD       0.786            0.032
                 GEN       0.786            0.046
             NNGuide       0.786            0.027
                 REN       0.857            0.022
                pNML       0.857            0.098
            GradNorm       0.929            0.027
KPCA RecError global       1.000            0.003
            Residual       1.000            0.142
```

## lopo — sparsity per CSF

Per-CSF: share of features with |coef| < 1e-6 (i.e., L1 zeroed them out), and the median chosen C across folds.

```
                 csf  zero_share  median_chosen_C
                Maha       0.310            1.322
          Confidence       0.500            0.099
              Energy       0.536            0.143
                NeCo       0.536            0.093
                 MSR       0.554            0.046
            Residual       0.571            3.352
                  PE       0.571            0.046
                 PCE       0.589            0.047
                 MLS       0.625            0.062
                 CTM       0.661            0.039
                 GEN       0.679            0.051
             NNGuide       0.714            0.024
                  GE       0.714            0.027
                 ViM       0.714            0.136
                fDBD       0.714            0.032
 PCA RecError global       0.810            0.032
                 REN       0.821            0.039
                pNML       0.839            0.083
            GradNorm       0.857            0.027
KPCA RecError global       0.911            0.029
```

## Set-regret comparison: L1 vs L2 per_csf binary (imputed)

### xarch

```
regime    side  L2_imputed  L1_imputed  L1_ci_lo  L1_ci_hi  L1_empty_share  delta_L1_minus_L2
  near     all       1.455      19.577    15.123    24.173           0.405             18.122
   mid     all       2.222      16.527    12.671    20.696           0.240             14.305
   far     all       1.488      19.285    14.101    24.974           0.260             17.797
  near    head      52.579      60.879    53.325    68.771           0.892              8.300
   mid    head      41.451      42.813    38.727    46.969           0.840              1.362
   far    head      39.365      39.836    33.775    46.060           0.860              0.471
  near feature       1.527      21.619    16.252    27.748           0.500             20.092
   mid feature       1.821      12.863     9.704    16.418           0.260             11.042
   far feature       1.454      24.582    17.918    32.020           0.380             23.128
```

### lopo

```
regime    side  L2_imputed  L1_imputed  L1_ci_lo  L1_ci_hi  L1_empty_share  delta_L1_minus_L2
  near     all      18.409      28.626    24.858    32.346           0.204             10.217
   mid     all       7.980      12.895    11.101    14.790           0.134              4.915
   far     all      12.709      14.214    11.857    16.604           0.216              1.505
  near    head      39.897      42.568    39.369    45.822           0.515              2.671
   mid    head      28.100      26.194    24.505    28.092           0.516             -1.906
   far    head      28.448      27.754    25.201    30.198           0.585             -0.694
  near feature      22.154      33.153    29.656    36.862           0.323             10.999
   mid feature       8.848      14.017    12.206    15.870           0.158              5.169
   far feature      12.407      14.336    12.097    16.629           0.240              1.929
```

