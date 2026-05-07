# Full metrics suite — clique-rule (b+c) headline predictor

**Date:** 2026-05-05
**Source:** `code/nc_csf_predictivity/evaluation/clique_bc_full_metrics.py`
**Predictor:** L2 LogisticRegressionCV(Cs=50, cv=5, class_weight='balanced'), NC pre-standardized per architecture
**Label rule:** clique (per-(paradigm, source, dropout, reward, regime) Friedman-Conover top cliques)
**Bootstrap:** n=2000, seed=0; **Holm-Bonferroni** within (regime, side)

## xarch

### config = `source`

**Predictor regret (imputed) per (regime, side):**

```
regime    side   n  set_regret_imputed_mean  set_regret_imputed_ci_lo  set_regret_imputed_ci_hi  set_size_mean  empty_set_share
   far     all 100                    0.391                     0.217                     0.607          6.070             0.00
   mid     all 200                    1.177                     0.941                     1.450          4.240             0.00
  near     all 148                    1.017                     0.732                     1.339          6.527             0.00
   far feature 100                    0.746                     0.355                     1.274          3.510             0.00
   mid feature 200                    1.946                     1.341                     2.601          2.270             0.02
  near feature 148                    4.989                     3.033                     7.175          2.236             0.00
   far    head 100                    4.991                     3.071                     7.219          2.560             0.04
   mid    head 200                   11.661                     9.198                    14.337          1.970             0.19
  near    head 148                    1.415                     0.967                     1.920          4.291             0.00
```

**Holm-corrected Wilcoxon wins** (predictor regret < baseline at α=0.05):

```
regime    side            comparator   n  median_diff      W      p  p_holm
   far     all            Always-CTM 100      -1.1768  360.0 0.0000  0.0000
   far     all         Always-Energy 100      -5.2228    0.0 0.0000  0.0000
   far     all            Always-MLS 100      -5.0987    0.0 0.0000  0.0000
   far     all            Always-MSR 100      -6.2708    2.0 0.0000  0.0000
   far     all        Always-NNGuide 100      -3.6476  231.0 0.0000  0.0000
   far     all           Always-fDBD 100      -4.3716    0.0 0.0000  0.0000
   far     all Oracle-on-train (CTM) 100      -1.1768  360.0 0.0000  0.0000
   far     all            Random-CSF 100     -10.1237    0.0 0.0000  0.0000
   far feature            Always-CTM 100      -0.6086  575.0 0.0000  0.0000
   far feature        Always-NNGuide 100      -3.2620  527.0 0.0000  0.0000
   far feature           Always-fDBD 100      -3.8460   19.0 0.0000  0.0000
   far feature Oracle-on-train (CTM) 100      -0.6086  575.0 0.0000  0.0000
   far feature            Random-CSF 100      -9.0165    0.0 0.0000  0.0000
   far    head         Always-Energy 100      -0.5597 1381.0 0.0055  0.0055
   far    head            Always-MLS 100      -0.3875 1452.0 0.0016  0.0049
   far    head            Always-MSR 100      -0.6386 1454.0 0.0001  0.0005
   far    head Oracle-on-train (MLS) 100      -0.3875 1452.0 0.0016  0.0049
   far    head            Random-CSF 100      -5.3516  953.0 0.0000  0.0000
   mid     all            Always-CTM 200      -1.0056 1363.0 0.0000  0.0000
   mid     all         Always-Energy 200      -2.0920 1255.0 0.0000  0.0000
   mid     all            Always-MLS 200      -2.4697  480.0 0.0000  0.0000
   mid     all            Always-MSR 200      -5.4839    3.0 0.0000  0.0000
   mid     all        Always-NNGuide 200      -1.2431 1169.0 0.0000  0.0000
   mid     all           Always-fDBD 200      -6.1044  105.0 0.0000  0.0000
   mid     all Oracle-on-train (CTM) 200      -1.0056 1363.0 0.0000  0.0000
   mid     all            Random-CSF 200     -13.7794    0.0 0.0000  0.0000
   mid feature            Always-CTM 200      -0.5133 3222.0 0.0000  0.0000
   mid feature        Always-NNGuide 200      -0.2055 3423.0 0.0000  0.0000
   mid feature           Always-fDBD 200      -5.4952 1362.0 0.0000  0.0000
   mid feature Oracle-on-train (CTM) 200      -0.5133 3222.0 0.0000  0.0000
   mid feature            Random-CSF 200     -13.3462  329.0 0.0000  0.0000
   mid    head            Always-MSR 200      -0.8402 8131.0 0.0096  0.0384
   mid    head            Random-CSF 200      -4.9439 6618.0 0.0000  0.0001
  near     all            Always-CTM 148       0.0000 1589.0 0.0003  0.0005
  near     all         Always-Energy 148      -3.6813  273.0 0.0000  0.0000
  near     all            Always-MLS 148      -2.8845    0.0 0.0000  0.0000
  near     all            Always-MSR 148      -3.9584    2.0 0.0000  0.0000
  near     all        Always-NNGuide 148      -2.2520  164.0 0.0000  0.0000
  near     all           Always-fDBD 148      -3.7267  237.0 0.0000  0.0000
  near     all Oracle-on-train (CTM) 148       0.0000 1589.0 0.0003  0.0005
  near     all            Random-CSF 148     -20.4935    0.0 0.0000  0.0000
  near feature        Always-NNGuide 148      -0.7002 2880.0 0.0032  0.0096
  near feature           Always-fDBD 148      -2.1752 2782.0 0.0000  0.0000
  near feature            Random-CSF 148     -19.3867    2.0 0.0000  0.0000
  near    head         Always-Energy 148      -2.7373  741.0 0.0000  0.0000
  near    head            Always-MLS 148      -1.2757  330.0 0.0000  0.0000
  near    head            Always-MSR 148      -2.0690   55.0 0.0000  0.0000
  near    head Oracle-on-train (MLS) 148      -1.2757  330.0 0.0000  0.0000
  near    head            Random-CSF 148     -10.0394    0.0 0.0000  0.0000
```

**Comparators not significantly beaten** (after Holm): 5 of 54 pairs.

### config = `n_classes`

**Predictor regret (imputed) per (regime, side):**

```
regime    side   n  set_regret_imputed_mean  set_regret_imputed_ci_lo  set_regret_imputed_ci_hi  set_size_mean  empty_set_share
   far     all 100                    0.361                     0.192                     0.567          7.680            0.000
   mid     all 200                    1.240                     0.909                     1.594          4.780            0.000
  near     all 148                    1.242                     0.897                     1.618          6.534            0.000
   far feature 100                    0.742                     0.348                     1.272          4.170            0.000
   mid feature 200                    3.288                     1.636                     5.466          2.560            0.010
  near feature 148                   11.409                     7.118                    16.587          2.169            0.034
   far    head 100                    3.352                     1.522                     5.550          3.510            0.060
   mid    head 200                   13.819                    10.600                    17.300          2.220            0.190
  near    head 148                    1.057                     0.700                     1.499          4.365            0.000
```

**Holm-corrected Wilcoxon wins** (predictor regret < baseline at α=0.05):

```
regime    side            comparator   n  median_diff      W      p  p_holm
   far     all            Always-CTM 100      -1.1657  372.0 0.0000  0.0000
   far     all         Always-Energy 100      -5.2662    0.0 0.0000  0.0000
   far     all            Always-MLS 100      -5.2744    0.0 0.0000  0.0000
   far     all            Always-MSR 100      -6.3143    1.0 0.0000  0.0000
   far     all        Always-NNGuide 100      -3.7711  217.0 0.0000  0.0000
   far     all           Always-fDBD 100      -4.6145    0.0 0.0000  0.0000
   far     all Oracle-on-train (CTM) 100      -1.1657  372.0 0.0000  0.0000
   far     all            Random-CSF 100      -9.9691    0.0 0.0000  0.0000
   far feature            Always-CTM 100      -0.6086  595.0 0.0000  0.0000
   far feature        Always-NNGuide 100      -3.4738  522.0 0.0000  0.0000
   far feature           Always-fDBD 100      -3.8460   18.0 0.0000  0.0000
   far feature Oracle-on-train (CTM) 100      -0.6086  595.0 0.0000  0.0000
   far feature            Random-CSF 100      -8.9444    0.0 0.0000  0.0000
   far    head         Always-Energy 100      -1.6408  838.0 0.0000  0.0000
   far    head            Always-MLS 100      -0.8369  630.0 0.0000  0.0000
   far    head            Always-MSR 100      -2.0025  672.0 0.0000  0.0000
   far    head Oracle-on-train (MLS) 100      -0.8369  630.0 0.0000  0.0000
   far    head            Random-CSF 100      -6.7917  585.0 0.0000  0.0000
   mid     all            Always-CTM 200      -1.1909 2337.0 0.0000  0.0000
   mid     all         Always-Energy 200      -2.1058  859.0 0.0000  0.0000
   mid     all            Always-MLS 200      -2.4915   11.0 0.0000  0.0000
   mid     all            Always-MSR 200      -5.4776    7.0 0.0000  0.0000
   mid     all        Always-NNGuide 200      -1.4815 1225.0 0.0000  0.0000
   mid     all           Always-fDBD 200      -5.6604  503.0 0.0000  0.0000
   mid     all Oracle-on-train (CTM) 200      -1.1909 2337.0 0.0000  0.0000
   mid     all            Random-CSF 200     -13.7868    3.0 0.0000  0.0000
   mid feature            Always-CTM 200      -0.6875 3916.0 0.0000  0.0000
   mid feature        Always-NNGuide 200      -0.0052 3020.0 0.0000  0.0000
   mid feature           Always-fDBD 200      -5.1344 1422.0 0.0000  0.0000
   mid feature Oracle-on-train (CTM) 200      -0.6875 3916.0 0.0000  0.0000
   mid feature            Random-CSF 200     -13.0377  563.0 0.0000  0.0000
   mid    head            Always-MSR 200      -1.2336 7678.0 0.0019  0.0076
   mid    head            Random-CSF 200      -5.2907 6875.0 0.0001  0.0003
  near     all            Always-CTM 148       0.0000 2485.0 0.0243  0.0487
  near     all         Always-Energy 148      -3.1607  324.0 0.0000  0.0000
  near     all            Always-MLS 148      -2.0854    0.0 0.0000  0.0000
  near     all            Always-MSR 148      -3.9584   71.0 0.0000  0.0000
  near     all        Always-NNGuide 148      -2.0497  736.0 0.0000  0.0000
  near     all           Always-fDBD 148      -3.8793  445.0 0.0000  0.0000
  near     all Oracle-on-train (CTM) 148       0.0000 2485.0 0.0243  0.0487
  near     all            Random-CSF 148     -20.1732    0.0 0.0000  0.0000
  near feature           Always-fDBD 148      -2.0441 3877.0 0.0037  0.0148
  near feature            Random-CSF 148     -16.3938  736.0 0.0000  0.0000
  near    head         Always-Energy 148      -2.8999  524.0 0.0000  0.0000
  near    head            Always-MLS 148      -1.4707   92.0 0.0000  0.0000
  near    head            Always-MSR 148      -2.2867   83.0 0.0000  0.0000
  near    head Oracle-on-train (MLS) 148      -1.4707   92.0 0.0000  0.0000
  near    head            Random-CSF 148     -10.7230    0.0 0.0000  0.0000
```

**Comparators not significantly beaten** (after Holm): 6 of 54 pairs.

### config = `none`

**Predictor regret (imputed) per (regime, side):**

```
regime    side   n  set_regret_imputed_mean  set_regret_imputed_ci_lo  set_regret_imputed_ci_hi  set_size_mean  empty_set_share
   far     all 100                    0.399                     0.204                     0.647          7.790             0.00
   mid     all 200                    1.440                     1.099                     1.786          4.940             0.00
  near     all 148                    1.454                     1.129                     1.840          7.223             0.00
   far feature 100                    0.766                     0.356                     1.308          4.590             0.00
   mid feature 200                    1.856                     1.301                     2.498          2.790             0.00
  near feature 148                    6.546                     4.482                     8.901          2.291             0.00
   far    head 100                    1.559                     1.013                     2.249          3.200             0.00
   mid    head 200                   10.020                     7.758                    12.525          2.150             0.15
  near    head 148                    1.467                     1.011                     1.972          4.932             0.00
```

**Holm-corrected Wilcoxon wins** (predictor regret < baseline at α=0.05):

```
regime    side            comparator   n  median_diff      W      p  p_holm
   far     all            Always-CTM 100      -1.2872  361.0 0.0000  0.0000
   far     all         Always-Energy 100      -5.2228    0.0 0.0000  0.0000
   far     all            Always-MLS 100      -4.9761    0.0 0.0000  0.0000
   far     all            Always-MSR 100      -6.2708    2.0 0.0000  0.0000
   far     all        Always-NNGuide 100      -3.6476  215.0 0.0000  0.0000
   far     all           Always-fDBD 100      -4.3716   35.0 0.0000  0.0000
   far     all Oracle-on-train (CTM) 100      -1.2872  361.0 0.0000  0.0000
   far     all            Random-CSF 100     -10.0371    0.0 0.0000  0.0000
   far feature            Always-CTM 100      -0.6086  580.0 0.0000  0.0000
   far feature        Always-NNGuide 100      -3.4738  506.0 0.0000  0.0000
   far feature           Always-fDBD 100      -3.8460   57.0 0.0000  0.0000
   far feature Oracle-on-train (CTM) 100      -0.6086  580.0 0.0000  0.0000
   far feature            Random-CSF 100      -8.9444    0.0 0.0000  0.0000
   far    head         Always-Energy 100      -1.6852  585.0 0.0000  0.0000
   far    head            Always-MLS 100      -0.9496  442.0 0.0000  0.0000
   far    head            Always-MSR 100      -1.7457  650.0 0.0000  0.0000
   far    head Oracle-on-train (MLS) 100      -0.9496  442.0 0.0000  0.0000
   far    head            Random-CSF 100      -7.5285  143.0 0.0000  0.0000
   mid     all            Always-CTM 200      -1.0310 2167.0 0.0000  0.0000
   mid     all         Always-Energy 200      -2.0866 1430.0 0.0000  0.0000
   mid     all            Always-MLS 200      -2.3383  509.0 0.0000  0.0000
   mid     all            Always-MSR 200      -5.0099    3.0 0.0000  0.0000
   mid     all        Always-NNGuide 200      -1.1596 1122.0 0.0000  0.0000
   mid     all           Always-fDBD 200      -5.6060  343.0 0.0000  0.0000
   mid     all Oracle-on-train (CTM) 200      -1.0310 2167.0 0.0000  0.0000
   mid     all            Random-CSF 200     -13.1758    3.0 0.0000  0.0000
   mid feature            Always-CTM 200      -0.7754 3545.0 0.0000  0.0000
   mid feature        Always-NNGuide 200      -0.0530 3046.0 0.0000  0.0000
   mid feature           Always-fDBD 200      -5.1123 1017.0 0.0000  0.0000
   mid feature Oracle-on-train (CTM) 200      -0.7754 3545.0 0.0000  0.0000
   mid feature            Random-CSF 200     -13.3462  111.0 0.0000  0.0000
   mid    head            Always-MSR 200      -0.8402 7376.0 0.0006  0.0022
   mid    head            Random-CSF 200      -5.3114 5295.0 0.0000  0.0000
  near     all         Always-Energy 148      -3.3643  316.0 0.0000  0.0000
  near     all            Always-MLS 148      -2.6071    0.0 0.0000  0.0000
  near     all            Always-MSR 148      -3.9761  103.0 0.0000  0.0000
  near     all        Always-NNGuide 148      -1.8855  171.0 0.0000  0.0000
  near     all           Always-fDBD 148      -3.6352  336.0 0.0000  0.0000
  near     all            Random-CSF 148     -18.8516    0.0 0.0000  0.0000
  near feature           Always-fDBD 148      -1.9310 3484.0 0.0006  0.0024
  near feature            Random-CSF 148     -18.0839   12.0 0.0000  0.0000
  near    head         Always-Energy 148      -2.6361  752.0 0.0000  0.0000
  near    head            Always-MLS 148      -1.2071  347.0 0.0000  0.0000
  near    head            Always-MSR 148      -1.9704  149.0 0.0000  0.0000
  near    head Oracle-on-train (MLS) 148      -1.2071  347.0 0.0000  0.0000
  near    head            Random-CSF 148     -10.0394    0.0 0.0000  0.0000
```

**Comparators not significantly beaten** (after Holm): 8 of 54 pairs.

## lopo

### config = `source`

**Predictor regret (imputed) per (regime, side):**

```
regime    side    n  set_regret_imputed_mean  set_regret_imputed_ci_lo  set_regret_imputed_ci_hi  set_size_mean  empty_set_share
   far     all  670                    3.308                     2.185                     4.683          4.490            0.015
   mid     all 1340                    4.223                     3.381                     5.170          3.169            0.012
  near     all  998                   12.416                     9.872                    15.321          4.492            0.060
   far feature  670                    5.021                     3.722                     6.469          2.248            0.033
   mid feature 1340                    7.145                     6.001                     8.319          1.649            0.058
  near feature  998                   15.589                    12.944                    18.638          1.594            0.069
   far    head  670                   10.024                     8.207                    11.972          2.242            0.090
   mid    head 1340                   19.402                    17.949                    20.949          1.519            0.315
  near    head  998                   21.935                    19.339                    24.660          2.898            0.210
```

**Holm-corrected Wilcoxon wins** (predictor regret < baseline at α=0.05):

```
regime    side            comparator    n  median_diff        W      p  p_holm
   far     all            Always-CTM  670      -1.9563  32042.0 0.0000  0.0000
   far     all         Always-Energy  670      -3.6375   7426.0 0.0000  0.0000
   far     all            Always-MLS  670      -3.6472   7949.0 0.0000  0.0000
   far     all            Always-MSR  670      -4.5763   8286.0 0.0000  0.0000
   far     all        Always-NNGuide  670      -3.7578  12158.0 0.0000  0.0000
   far     all           Always-fDBD  670      -5.1004   6530.0 0.0000  0.0000
   far     all Oracle-on-train (CTM)  670      -1.9563  32042.0 0.0000  0.0000
   far     all            Random-CSF  670     -10.4812   6674.0 0.0000  0.0000
   far feature            Always-CTM  670      -1.0717  43799.0 0.0000  0.0000
   far feature        Always-NNGuide  670      -3.0836  32969.0 0.0000  0.0000
   far feature           Always-fDBD  670      -4.2102  11030.0 0.0000  0.0000
   far feature Oracle-on-train (CTM)  670      -1.0717  43799.0 0.0000  0.0000
   far feature            Random-CSF  670     -10.0343  13061.0 0.0000  0.0000
   far    head         Always-Energy  670      -0.4493  55638.0 0.0000  0.0000
   far    head            Always-MLS  670      -0.4972  62706.0 0.0000  0.0000
   far    head            Always-MSR  670      -0.9225  66827.0 0.0000  0.0000
   far    head  Oracle-on-train (PE)  670      -0.5258  58147.0 0.0000  0.0000
   far    head            Random-CSF  670      -4.1839  52050.0 0.0000  0.0000
   mid     all            Always-CTM 1340      -1.2205 140574.0 0.0000  0.0000
   mid     all         Always-Energy 1340      -1.4235  31674.0 0.0000  0.0000
   mid     all            Always-MLS 1340      -2.3836  39608.0 0.0000  0.0000
   mid     all            Always-MSR 1340      -5.2528  24660.0 0.0000  0.0000
   mid     all        Always-NNGuide 1340      -1.5721  81234.0 0.0000  0.0000
   mid     all           Always-fDBD 1340      -4.6222  47427.0 0.0000  0.0000
   mid     all Oracle-on-train (CTM) 1340      -1.2205 140574.0 0.0000  0.0000
   mid     all            Random-CSF 1340     -11.9878  25880.0 0.0000  0.0000
   mid feature            Always-CTM 1340       0.0000 225068.0 0.0000  0.0000
   mid feature        Always-NNGuide 1340      -0.5799 179461.0 0.0000  0.0000
   mid feature           Always-fDBD 1340      -3.6598 104576.0 0.0000  0.0000
   mid feature Oracle-on-train (CTM) 1340       0.0000 225068.0 0.0000  0.0000
   mid feature            Random-CSF 1340     -12.0989  87679.0 0.0000  0.0000
  near     all            Always-CTM  998       0.0000  75135.0 0.0000  0.0000
  near     all         Always-Energy  998      -2.9246  55855.0 0.0000  0.0000
  near     all            Always-MLS  998      -2.1845  59274.0 0.0000  0.0000
  near     all            Always-MSR  998      -3.3120  59487.0 0.0000  0.0000
  near     all        Always-NNGuide  998      -1.8750  69012.0 0.0000  0.0000
  near     all           Always-fDBD  998      -3.9078  66987.0 0.0000  0.0000
  near     all Oracle-on-train (CTM)  998       0.0000  75135.0 0.0000  0.0000
  near     all            Random-CSF  998     -13.5207  59436.0 0.0000  0.0000
  near feature        Always-NNGuide  998      -0.8102 108294.0 0.0000  0.0000
  near feature           Always-fDBD  998      -2.7501  97990.0 0.0000  0.0000
  near feature            Random-CSF  998     -17.9305  73446.0 0.0000  0.0000
  near    head         Always-Energy  998      -0.5801 173203.0 0.0051  0.0152
  near    head            Always-MLS  998      -0.2999 191076.0 0.0058  0.0152
  near    head            Always-MSR  998      -0.9466 201471.0 0.0000  0.0000
  near    head            Random-CSF  998      -4.9961 188106.0 0.0000  0.0000
```

**Comparators not significantly beaten** (after Holm): 8 of 54 pairs.

### config = `n_classes`

**Predictor regret (imputed) per (regime, side):**

```
regime    side    n  set_regret_imputed_mean  set_regret_imputed_ci_lo  set_regret_imputed_ci_hi  set_size_mean  empty_set_share
   far     all  670                    1.166                     0.988                     1.341          5.642            0.000
   mid     all 1340                    2.385                     2.176                     2.624          4.133            0.000
  near     all  998                    1.560                     1.395                     1.723          5.325            0.000
   far feature  670                    2.924                     2.302                     3.558          2.834            0.018
   mid feature 1340                    5.287                     4.463                     6.125          2.173            0.025
  near feature  998                    6.653                     5.230                     8.037          1.890            0.037
   far    head  670                    8.932                     7.352                    10.692          2.807            0.106
   mid    head 1340                   17.717                    16.246                    19.188          1.960            0.285
  near    head  998                   11.473                     9.570                    13.329          3.435            0.100
```

**Holm-corrected Wilcoxon wins** (predictor regret < baseline at α=0.05):

```
regime    side            comparator    n  median_diff        W      p  p_holm
   far     all            Always-CTM  670      -2.0873  23251.0 0.0000  0.0000
   far     all         Always-Energy  670      -4.2764   1713.0 0.0000  0.0000
   far     all            Always-MLS  670      -4.0052   3675.0 0.0000  0.0000
   far     all            Always-MSR  670      -4.6714    917.0 0.0000  0.0000
   far     all        Always-NNGuide  670      -3.9042   6369.0 0.0000  0.0000
   far     all           Always-fDBD  670      -5.7406   2096.0 0.0000  0.0000
   far     all Oracle-on-train (CTM)  670      -2.0873  23251.0 0.0000  0.0000
   far     all            Random-CSF  670     -11.0523      0.0 0.0000  0.0000
   far feature            Always-CTM  670      -1.4508  37284.0 0.0000  0.0000
   far feature        Always-NNGuide  670      -3.4309  25863.0 0.0000  0.0000
   far feature           Always-fDBD  670      -4.8117   5658.0 0.0000  0.0000
   far feature Oracle-on-train (CTM)  670      -1.4508  37284.0 0.0000  0.0000
   far feature            Random-CSF  670     -10.6630   7322.0 0.0000  0.0000
   far    head         Always-Energy  670      -0.3078  59995.0 0.0000  0.0000
   far    head            Always-MLS  670      -0.4466  76348.0 0.0000  0.0000
   far    head            Always-MSR  670      -1.0322  68010.0 0.0000  0.0000
   far    head  Oracle-on-train (PE)  670      -0.6072  67128.0 0.0000  0.0000
   far    head            Random-CSF  670      -4.0918  56773.0 0.0000  0.0000
   mid     all            Always-CTM 1340      -1.1665 118127.0 0.0000  0.0000
   mid     all         Always-Energy 1340      -1.7973  14818.0 0.0000  0.0000
   mid     all            Always-MLS 1340      -2.6597  25103.0 0.0000  0.0000
   mid     all            Always-MSR 1340      -5.4848   1992.0 0.0000  0.0000
   mid     all        Always-NNGuide 1340      -1.8021  67254.0 0.0000  0.0000
   mid     all           Always-fDBD 1340      -4.4752  15729.0 0.0000  0.0000
   mid     all Oracle-on-train (CTM) 1340      -1.1665 118127.0 0.0000  0.0000
   mid     all            Random-CSF 1340     -12.4102    957.0 0.0000  0.0000
   mid feature            Always-CTM 1340      -0.4066 176439.0 0.0000  0.0000
   mid feature        Always-NNGuide 1340      -0.7493 128575.0 0.0000  0.0000
   mid feature           Always-fDBD 1340      -3.8580  39494.0 0.0000  0.0000
   mid feature Oracle-on-train (CTM) 1340      -0.4066 176439.0 0.0000  0.0000
   mid feature            Random-CSF 1340     -12.7113  49105.0 0.0000  0.0000
  near     all            Always-CTM  998       0.0000  69860.0 0.0000  0.0000
  near     all         Always-Energy  998      -3.0532  11388.0 0.0000  0.0000
  near     all            Always-MLS  998      -2.0157  21621.0 0.0000  0.0000
  near     all            Always-MSR  998      -3.0398    505.0 0.0000  0.0000
  near     all        Always-NNGuide  998      -1.8843  36016.0 0.0000  0.0000
  near     all           Always-fDBD  998      -3.9987  15880.0 0.0000  0.0000
  near     all Oracle-on-train (CTM)  998       0.0000  69860.0 0.0000  0.0000
  near     all            Random-CSF  998     -15.1462     11.0 0.0000  0.0000
  near feature            Always-CTM  998       0.0000 102026.0 0.0003  0.0007
  near feature        Always-NNGuide  998      -0.4166  91845.0 0.0000  0.0000
  near feature           Always-fDBD  998      -2.7649  35881.0 0.0000  0.0000
  near feature Oracle-on-train (CTM)  998       0.0000 102026.0 0.0003  0.0007
  near feature            Random-CSF  998     -20.0312  35295.0 0.0000  0.0000
  near    head         Always-Energy  998      -1.0562 110572.0 0.0000  0.0000
  near    head            Always-MLS  998      -0.4958 146649.0 0.0000  0.0000
  near    head            Always-MSR  998      -0.8998  88360.0 0.0000  0.0000
  near    head  Oracle-on-train (PE)  998      -0.4653 106607.0 0.0000  0.0000
  near    head            Random-CSF  998      -6.6573  95263.0 0.0000  0.0000
```

**Comparators not significantly beaten** (after Holm): 5 of 54 pairs.

### config = `none`

**Predictor regret (imputed) per (regime, side):**

```
regime    side    n  set_regret_imputed_mean  set_regret_imputed_ci_lo  set_regret_imputed_ci_hi  set_size_mean  empty_set_share
   far     all  670                    1.378                     1.183                     1.585          5.560            0.000
   mid     all 1340                    2.542                     2.273                     2.818          4.149            0.000
  near     all  998                    1.317                     1.157                     1.493          6.226            0.000
   far feature  670                    3.239                     2.625                     3.867          2.913            0.009
   mid feature 1340                    5.338                     4.706                     6.024          2.173            0.048
  near feature  998                    3.121                     2.501                     3.808          2.349            0.002
   far    head  670                    7.995                     6.592                     9.517          2.646            0.081
   mid    head 1340                   13.992                    12.703                    15.290          1.976            0.212
  near    head  998                    2.096                     1.857                     2.343          3.878            0.000
```

**Holm-corrected Wilcoxon wins** (predictor regret < baseline at α=0.05):

```
regime    side            comparator    n  median_diff        W      p  p_holm
   far     all            Always-CTM  670      -2.0563  25581.0 0.0000  0.0000
   far     all         Always-Energy  670      -3.9575   2855.0 0.0000  0.0000
   far     all            Always-MLS  670      -3.8431   3475.0 0.0000  0.0000
   far     all            Always-MSR  670      -4.5763   1019.0 0.0000  0.0000
   far     all        Always-NNGuide  670      -4.0031   5402.0 0.0000  0.0000
   far     all           Always-fDBD  670      -5.6530    845.0 0.0000  0.0000
   far     all Oracle-on-train (CTM)  670      -2.0563  25581.0 0.0000  0.0000
   far     all            Random-CSF  670     -10.4819      0.0 0.0000  0.0000
   far feature            Always-CTM  670      -1.0951  37655.0 0.0000  0.0000
   far feature        Always-NNGuide  670      -3.4525  31178.0 0.0000  0.0000
   far feature           Always-fDBD  670      -4.6289   4306.0 0.0000  0.0000
   far feature Oracle-on-train (CTM)  670      -1.0951  37655.0 0.0000  0.0000
   far feature            Random-CSF  670     -10.1135   6953.0 0.0000  0.0000
   far    head         Always-Energy  670      -0.3440  60440.0 0.0000  0.0000
   far    head            Always-MLS  670      -0.4213  69607.0 0.0000  0.0000
   far    head            Always-MSR  670      -0.9125  64684.0 0.0000  0.0000
   far    head  Oracle-on-train (PE)  670      -0.6503  64476.0 0.0000  0.0000
   far    head            Random-CSF  670      -4.1250  48504.0 0.0000  0.0000
   mid     all            Always-CTM 1340      -1.4033  91191.0 0.0000  0.0000
   mid     all         Always-Energy 1340      -1.9375  16067.0 0.0000  0.0000
   mid     all            Always-MLS 1340      -2.7707  22128.0 0.0000  0.0000
   mid     all            Always-MSR 1340      -5.5556   4033.0 0.0000  0.0000
   mid     all        Always-NNGuide 1340      -2.1320  60847.0 0.0000  0.0000
   mid     all           Always-fDBD 1340      -4.7645   5550.0 0.0000  0.0000
   mid     all Oracle-on-train (CTM) 1340      -1.4033  91191.0 0.0000  0.0000
   mid     all            Random-CSF 1340     -12.0655   2847.0 0.0000  0.0000
   mid feature            Always-CTM 1340      -0.1704 187387.0 0.0000  0.0000
   mid feature        Always-NNGuide 1340      -1.0738 201300.0 0.0000  0.0000
   mid feature           Always-fDBD 1340      -3.5248  54384.0 0.0000  0.0000
   mid feature Oracle-on-train (CTM) 1340      -0.1704 187387.0 0.0000  0.0000
   mid feature            Random-CSF 1340     -12.8079  66848.0 0.0000  0.0000
   mid    head            Always-MSR 1340      -0.9974 365708.0 0.0025  0.0101
   mid    head            Random-CSF 1340      -3.6964 351888.0 0.0000  0.0000
  near     all            Always-CTM  998      -0.0085  36499.0 0.0000  0.0000
  near     all         Always-Energy  998      -3.3643   5113.0 0.0000  0.0000
  near     all            Always-MLS  998      -2.1886   9704.0 0.0000  0.0000
  near     all            Always-MSR  998      -3.3971   1125.0 0.0000  0.0000
  near     all        Always-NNGuide  998      -1.9815  12449.0 0.0000  0.0000
  near     all           Always-fDBD  998      -4.0940   2497.0 0.0000  0.0000
  near     all Oracle-on-train (CTM)  998      -0.0085  36499.0 0.0000  0.0000
  near     all            Random-CSF  998     -16.3597    120.0 0.0000  0.0000
  near feature            Always-CTM  998       0.0000  58870.0 0.0000  0.0000
  near feature        Always-NNGuide  998      -1.1031  62211.0 0.0000  0.0000
  near feature           Always-fDBD  998      -3.1029  13875.0 0.0000  0.0000
  near feature Oracle-on-train (CTM)  998       0.0000  58870.0 0.0000  0.0000
  near feature            Random-CSF  998     -21.3035   5312.0 0.0000  0.0000
  near    head         Always-Energy  998      -1.8522  21435.0 0.0000  0.0000
  near    head            Always-MLS  998      -0.6583  56525.0 0.0000  0.0000
  near    head            Always-MSR  998      -1.2340  23610.0 0.0000  0.0000
  near    head  Oracle-on-train (PE)  998      -0.5109   9223.0 0.0000  0.0000
  near    head            Random-CSF  998      -7.6079    423.0 0.0000  0.0000
```

**Comparators not significantly beaten** (after Holm): 3 of 54 pairs.

## Run summary

```
split    config  n_metrics_rows  n_baseline_rows  n_wilcoxon_tests
xarch    source            1344             8064                54
xarch n_classes            1344             8064                54
xarch      none            1344             8064                54
 lopo    source            9024            54144                54
 lopo n_classes            9024            54144                54
 lopo      none            9024            54144                54
```
