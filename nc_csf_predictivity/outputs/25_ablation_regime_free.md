# Regime-free predictor ablation (rebuttal experiment E-B)

**Source:** `nc_csf_predictivity/ablations/calibration_regime_free.py`
**Protocol:** identical to `calibration_features_clique.py` except the regime input is removed (marginal/pooled label variants). Regret = imputed set-regret, bootstrap 95% CI.

## xarch

### side = all

```
regime best_baseline  bl_regret with-regime (source)  with-regime (source)|empty% source_nr_marginal  source_nr_marginal|empty% source_nr_marginal|p_vs_regime source_nr_pooled  source_nr_pooled|empty% source_nr_pooled|p_vs_regime none_nr_marginal  none_nr_marginal|empty% none_nr_marginal|p_vs_regime   none_nr_pooled  none_nr_pooled|empty% none_nr_pooled|p_vs_regime
  near    Always-CTM       2.07     1.02 [0.73,1.34]                          0.0   1.06 [0.78,1.37]                        0.0                 0.0428 (n=148) 2.20 [1.47,3.15]                      1.4             6.06e-08 (n=148) 1.23 [0.94,1.53]                      0.0             0.000296 (n=148) 2.34 [1.40,3.51]                    2.7           3.59e-06 (n=148)
   mid    Always-CTM       3.52     1.18 [0.94,1.45]                          0.0   0.96 [0.73,1.23]                        0.0               7.83e-06 (n=200) 2.37 [1.48,3.50]                      2.0               0.0127 (n=200) 1.16 [0.89,1.45]                      0.0               0.0109 (n=200) 3.16 [1.86,4.76]                    4.0             0.0148 (n=200)
   far    Always-CTM       2.40     0.39 [0.22,0.61]                          0.0   0.64 [0.43,0.87]                        0.0                0.00223 (n=100) 1.63 [1.06,2.36]                      2.0             1.42e-06 (n=100) 0.80 [0.51,1.14]                      0.0              0.00351 (n=100) 1.94 [1.05,3.08]                    4.0           4.61e-05 (n=100)
```

### side = head

```
regime best_baseline  bl_regret with-regime (source)  with-regime (source)|empty% source_nr_marginal  source_nr_marginal|empty% source_nr_marginal|p_vs_regime    source_nr_pooled  source_nr_pooled|empty% source_nr_pooled|p_vs_regime none_nr_marginal  none_nr_marginal|empty% none_nr_marginal|p_vs_regime      none_nr_pooled  none_nr_pooled|empty% none_nr_pooled|p_vs_regime
  near    Always-MLS       3.49     1.41 [0.97,1.92]                          0.0  9.58 [6.42,13.15]                       10.1               1.09e-09 (n=148) 20.22 [16.22,24.60]                     32.4             3.61e-17 (n=148) 3.33 [2.15,4.92]                      0.0             5.34e-10 (n=148) 24.80 [20.21,29.89]                   50.7           1.76e-18 (n=148)
   mid Always-Energy       4.20   11.66 [9.20,14.34]                         19.0   6.65 [4.48,9.16]                        3.0               6.49e-11 (n=200) 24.86 [21.16,29.04]                     32.0             6.44e-11 (n=200) 5.29 [3.62,7.38]                      0.0             1.68e-13 (n=200) 32.61 [28.61,36.90]                   63.0           1.85e-15 (n=200)
   far    Always-MLS       3.77     4.99 [3.07,7.22]                          4.0   4.10 [2.45,5.91]                        3.0               0.000643 (n=100) 18.96 [15.16,22.99]                     32.0             1.79e-09 (n=100) 3.04 [1.94,4.37]                      0.0               0.0236 (n=100) 27.88 [22.71,33.53]                   63.0           5.17e-11 (n=100)
```

### side = feature

```
regime best_baseline  bl_regret with-regime (source)  with-regime (source)|empty% source_nr_marginal  source_nr_marginal|empty% source_nr_marginal|p_vs_regime source_nr_pooled  source_nr_pooled|empty% source_nr_pooled|p_vs_regime none_nr_marginal  none_nr_marginal|empty% none_nr_marginal|p_vs_regime    none_nr_pooled  none_nr_pooled|empty% none_nr_pooled|p_vs_regime
  near    Always-CTM       1.55     4.99 [3.03,7.18]                          0.0   2.78 [1.81,3.86]                        0.0               0.000292 (n=148) 5.95 [3.93,8.24]                      2.7             0.000646 (n=148) 6.03 [3.99,8.35]                      0.0               0.0797 (n=148) 7.59 [5.20,10.22]                    4.1            0.00338 (n=148)
   mid    Always-CTM       2.92     1.95 [1.34,2.60]                          2.0   1.09 [0.72,1.58]                        0.0               1.68e-07 (n=200) 2.99 [2.07,4.09]                      4.0              0.00118 (n=200) 1.64 [1.10,2.27]                      0.0               0.0145 (n=200)  4.28 [2.76,6.10]                    6.0             0.0903 (n=200)
   far    Always-CTM       2.19     0.75 [0.35,1.27]                          0.0   1.06 [0.64,1.59]                        0.0               0.000293 (n=100) 2.92 [1.86,4.21]                      4.0             3.65e-07 (n=100) 1.89 [0.90,3.19]                      0.0              0.00406 (n=100)  3.22 [1.83,4.78]                    6.0           7.98e-05 (n=100)
```

## lopo

### side = all

```
regime  best_baseline  bl_regret with-regime (source)  with-regime (source)|empty% source_nr_marginal  source_nr_marginal|empty% source_nr_marginal|p_vs_regime source_nr_pooled  source_nr_pooled|empty% source_nr_pooled|p_vs_regime none_nr_marginal  none_nr_marginal|empty% none_nr_marginal|p_vs_regime   none_nr_pooled  none_nr_pooled|empty% none_nr_pooled|p_vs_regime
  near Always-NNGuide       4.48   12.42 [9.87,15.32]                          6.0   1.12 [0.95,1.33]                        0.0               9.45e-20 (n=998) 2.01 [1.69,2.39]                      0.4                0.731 (n=998) 1.93 [1.40,2.57]                      0.0              1.9e-09 (n=998) 5.71 [4.25,7.25]                    2.7              0.719 (n=998)
   mid     Always-CTM       6.52     4.22 [3.38,5.17]                          1.2   1.69 [1.50,1.91]                        0.0              7.51e-35 (n=1340) 3.62 [3.24,4.05]                      0.6            9.03e-09 (n=1340) 1.96 [1.74,2.24]                      0.0            1.75e-20 (n=1340) 4.60 [3.96,5.27]                    1.0          5.16e-05 (n=1340)
   far     Always-CTM       5.68     3.31 [2.19,4.68]                          1.5   1.47 [1.28,1.68]                        0.0                 0.0181 (n=670) 3.08 [2.65,3.54]                      0.6              3.7e-15 (n=670) 1.82 [1.39,2.35]                      0.0                0.487 (n=670) 3.96 [3.15,4.83]                    1.0           5.99e-14 (n=670)
```

### side = head

```
regime        best_baseline  bl_regret with-regime (source)  with-regime (source)|empty%  source_nr_marginal  source_nr_marginal|empty% source_nr_marginal|p_vs_regime    source_nr_pooled  source_nr_pooled|empty% source_nr_pooled|p_vs_regime  none_nr_marginal  none_nr_marginal|empty% none_nr_marginal|p_vs_regime      none_nr_pooled  none_nr_pooled|empty% none_nr_pooled|p_vs_regime
  near Oracle-on-train (PE)       3.57  21.93 [19.34,24.66]                         21.0 31.12 [28.12,34.32]                       30.0               5.58e-21 (n=998) 39.27 [36.42,42.36]                     51.1             1.31e-69 (n=998) 9.67 [8.09,11.26]                      0.0             1.55e-14 (n=998) 38.91 [35.87,41.92]                   50.0           2.09e-67 (n=998)
   mid           Always-MLS       5.34  19.40 [17.95,20.95]                         31.5    8.26 [7.22,9.41]                        9.4              4.56e-71 (n=1340) 20.66 [19.25,22.12]                     36.7            9.83e-06 (n=1340)  4.30 [3.75,4.88]                      0.0            1.88e-84 (n=1340) 21.40 [19.93,22.89]                   36.9          1.05e-11 (n=1340)
   far Oracle-on-train (PE)       4.41   10.02 [8.21,11.97]                          9.0   9.82 [7.96,11.75]                        9.4               1.73e-06 (n=670) 19.75 [17.59,21.97]                     36.7             8.96e-43 (n=670)  4.04 [3.11,5.07]                      0.0             8.59e-18 (n=670) 20.75 [18.50,23.04]                   36.9           6.01e-37 (n=670)
```

### side = feature

```
regime  best_baseline  bl_regret with-regime (source)  with-regime (source)|empty% source_nr_marginal  source_nr_marginal|empty% source_nr_marginal|p_vs_regime source_nr_pooled  source_nr_pooled|empty% source_nr_pooled|p_vs_regime none_nr_marginal  none_nr_marginal|empty% none_nr_marginal|p_vs_regime     none_nr_pooled  none_nr_pooled|empty% none_nr_pooled|p_vs_regime
  near Always-NNGuide       3.74  15.59 [12.94,18.64]                          6.9   4.29 [3.25,5.43]                        2.5               1.63e-17 (n=998) 5.52 [4.48,6.69]                      4.6             0.000371 (n=998) 5.15 [4.05,6.36]                      2.0             7.49e-13 (n=998) 10.99 [9.16,12.90]                    8.8              0.297 (n=998)
   mid     Always-CTM       5.71     7.15 [6.00,8.32]                          5.8   4.13 [3.44,4.85]                        3.1              8.99e-24 (n=1340) 7.79 [6.87,8.77]                      6.9            1.57e-09 (n=1340) 4.26 [3.62,5.02]                      1.8            7.71e-23 (n=1340)   8.82 [7.84,9.87]                    7.2           2.5e-11 (n=1340)
   far     Always-CTM       5.30     5.02 [3.72,6.47]                          3.3   3.34 [2.69,4.07]                        3.1                  0.292 (n=670) 6.47 [5.50,7.53]                      6.9             3.76e-15 (n=670) 3.89 [3.11,4.75]                      1.8                0.799 (n=670)  8.74 [7.26,10.37]                    7.2           2.01e-19 (n=670)
```

