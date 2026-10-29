# Regime-free predictor: Holm-corrected paired Wilcoxon (VGG13 -> ResNet18)

**Source:** `nc_csf_predictivity/evaluation/regimefree_wilcoxon.py`
**Protocol:** per (regime, side) cell, each predictor is compared with each NC-free baseline on per-row imputed set-regret, Holm within the cell (alpha 0.05). `wil_*` = the paper's Sec 4.4 test, rebuilt (one-sided paired Wilcoxon signed-rank). `mean_*` = paired sign-flip permutation test on the mean difference, clustered by model (100,000 draws, seed 0). `better` / `worse` = significant in that direction. Gates G1 (stored paper stats reproduced: 49/48/46 of 54) and G2 (ablation scoring path reproduces the paper per-row regret) passed before these numbers were computed. Arms `paper_*` are the published with-regime configs; the others are regime-free.

Why two tests: Wilcoxon ranks the paired differences, so it tests whether the predictor is usually better, not whether its mean regret is lower. With a few large misses (empty shortlists imputed at the worst AUGRC, or rare large regrets) the two can disagree; the paper reports means.

## paper_source

Wilcoxon: better 49, worse 0 of 54. Mean test: better 39, worse 4 of 54.

```
regime    side  pred_mean  wil_better  wil_worse  mean_better  mean_worse  tests best_baseline  baseline_mean  reduction_%
  near     all       1.02           8          0            8           0      8    Always-CTM           2.07         51.0
   mid     all       1.18           8          0            8           0      8    Always-CTM           3.52         67.0
   far     all       0.39           8          0            8           0      8    Always-CTM           2.40         84.0
  near    head       1.41           5          0            5           0      5    Always-MLS           3.49         59.0
   mid    head      11.66           2          0            0           4      5 Always-Energy           4.20       -178.0
   far    head       4.99           5          0            1           0      5    Always-MLS           3.77        -32.0
  near feature       4.99           3          0            1           0      5    Always-CTM           1.55       -222.0
   mid feature       1.95           5          0            3           0      5    Always-CTM           2.92         33.0
   far feature       0.75           5          0            5           0      5    Always-CTM           2.19         66.0
```

Comparisons where the two tests disagree:

```
regime    side            comparator  pred_mean  baseline_mean  median_diff  p_holm  p_mean_holm  p_mean_worse_holm
   far    head         Always-Energy     4.9911         4.0927      -0.5597  0.0055       1.0000             0.8758
   far    head            Always-MLS     4.9911         3.7728      -0.3875  0.0049       1.0000             0.8758
   far    head            Always-MSR     4.9911         4.5299      -0.6386  0.0005       1.0000             0.8758
   far    head Oracle-on-train (MLS)     4.9911         3.7728      -0.3875  0.0049       1.0000             0.8758
   mid feature            Always-CTM     1.9463         2.9187      -0.5133  0.0000       0.1556             1.0000
   mid feature Oracle-on-train (CTM)     1.9463         2.9187      -0.5133  0.0000       0.1556             1.0000
   mid    head            Always-MSR    11.6609         6.3073      -0.8402  0.0384       1.0000             0.0162
   mid    head            Random-CSF    11.6609        11.8230      -4.9439  0.0001       1.0000             0.5248
  near feature        Always-NNGuide     4.9894         3.5305      -0.7002  0.0096       1.0000             1.0000
  near feature           Always-fDBD     4.9894         6.5146      -2.1752  0.0000       1.0000             1.0000
```

## paper_n_classes

Wilcoxon: better 48, worse 2 of 54. Mean test: better 36, worse 4 of 54.

```
regime    side  pred_mean  wil_better  wil_worse  mean_better  mean_worse  tests best_baseline  baseline_mean  reduction_%
  near     all       1.24           8          0            6           0      8    Always-CTM           2.07         40.0
   mid     all       1.24           8          0            8           0      8    Always-CTM           3.52         65.0
   far     all       0.36           8          0            8           0      8    Always-CTM           2.40         85.0
  near    head       1.06           5          0            5           0      5    Always-MLS           3.49         70.0
   mid    head      13.82           2          0            0           4      5 Always-Energy           4.20       -229.0
   far    head       3.35           5          0            1           0      5    Always-MLS           3.77         11.0
  near feature      11.41           2          2            1           0      5    Always-CTM           1.55       -637.0
   mid feature       3.29           5          0            2           0      5    Always-CTM           2.92        -13.0
   far feature       0.74           5          0            5           0      5    Always-CTM           2.19         66.0
```

Comparisons where the two tests disagree:

```
regime    side            comparator  pred_mean  baseline_mean  median_diff  p_holm  p_mean_holm  p_mean_worse_holm
   far    head         Always-Energy     3.3520         4.0927      -1.6408  0.0000       0.8666             1.0000
   far    head            Always-MLS     3.3520         3.7728      -0.8369  0.0000       0.8666             1.0000
   far    head            Always-MSR     3.3520         4.5299      -2.0025  0.0000       0.7947             1.0000
   far    head Oracle-on-train (MLS)     3.3520         3.7728      -0.8369  0.0000       0.8666             1.0000
   mid feature            Always-CTM     3.2877         2.9187      -0.6875  0.0000       1.0000             1.0000
   mid feature        Always-NNGuide     3.2877         4.1911      -0.0052  0.0000       1.0000             1.0000
   mid feature Oracle-on-train (CTM)     3.2877         2.9187      -0.6875  0.0000       1.0000             1.0000
   mid    head            Always-MSR    13.8190         6.3073      -1.2336  0.0076       1.0000             0.0210
   mid    head            Random-CSF    13.8190        11.8230      -5.2907  0.0003       1.0000             0.2785
  near     all            Always-CTM     1.2416         2.0663       0.0000  0.0487       0.0983             1.0000
  near     all Oracle-on-train (CTM)     1.2416         2.0663       0.0000  0.0487       0.0983             1.0000
  near feature           Always-fDBD    11.4088         6.5146      -2.0441  0.0148       1.0000             0.5435
```

## paper_none

Wilcoxon: better 46, worse 2 of 54. Mean test: better 41, worse 3 of 54.

```
regime    side  pred_mean  wil_better  wil_worse  mean_better  mean_worse  tests best_baseline  baseline_mean  reduction_%
  near     all       1.45           6          0            6           0      8    Always-CTM           2.07         30.0
   mid     all       1.44           8          0            8           0      8    Always-CTM           3.52         59.0
   far     all       0.40           8          0            8           0      8    Always-CTM           2.40         83.0
  near    head       1.47           5          0            5           0      5    Always-MLS           3.49         58.0
   mid    head      10.02           2          0            0           3      5 Always-Energy           4.20       -139.0
   far    head       1.56           5          0            5           0      5    Always-MLS           3.77         59.0
  near feature       6.55           2          2            1           0      5    Always-CTM           1.55       -323.0
   mid feature       1.86           5          0            3           0      5    Always-CTM           2.92         36.0
   far feature       0.77           5          0            5           0      5    Always-CTM           2.19         65.0
```

Comparisons where the two tests disagree:

```
regime    side            comparator  pred_mean  baseline_mean  median_diff  p_holm  p_mean_holm  p_mean_worse_holm
   mid feature            Always-CTM     1.8562         2.9187      -0.7754  0.0000       0.1123             1.0000
   mid feature Oracle-on-train (CTM)     1.8562         2.9187      -0.7754  0.0000       0.1123             1.0000
   mid    head            Always-MSR    10.0202         6.3073      -0.8402  0.0022       1.0000             0.0858
   mid    head            Random-CSF    10.0202        11.8230      -5.3114  0.0000       1.0000             0.7715
  near feature           Always-fDBD     6.5456         6.5146      -1.9310  0.0024       1.0000             0.9911
```

## source_nr_marginal

Wilcoxon: better 52, worse 0 of 54. Mean test: better 38, worse 0 of 54.

```
regime    side  pred_mean  wil_better  wil_worse  mean_better  mean_worse  tests best_baseline  baseline_mean  reduction_%
  near     all       1.06           8          0            8           0      8    Always-CTM           2.07         49.0
   mid     all       0.96           8          0            8           0      8    Always-CTM           3.52         73.0
   far     all       0.64           8          0            8           0      8    Always-CTM           2.40         73.0
  near    head       9.58           5          0            0           0      5    Always-MLS           3.49       -174.0
   mid    head       6.65           5          0            1           0      5 Always-Energy           4.20        -58.0
   far    head       4.10           5          0            1           0      5    Always-MLS           3.77         -9.0
  near feature       2.78           3          0            2           0      5    Always-CTM           1.55        -80.0
   mid feature       1.09           5          0            5           0      5    Always-CTM           2.92         63.0
   far feature       1.06           5          0            5           0      5    Always-CTM           2.19         51.0
```

Comparisons where the two tests disagree:

```
regime    side            comparator  pred_mean  baseline_mean  median_diff  p_holm  p_mean_holm  p_mean_worse_holm
   far    head         Always-Energy     4.1031         4.0927      -0.6294  0.0003        1.000             1.0000
   far    head            Always-MLS     4.1031         3.7728      -0.5034  0.0000        1.000             1.0000
   far    head            Always-MSR     4.1031         4.5299      -0.8107  0.0000        1.000             1.0000
   far    head Oracle-on-train (MLS)     4.1031         3.7728      -0.5034  0.0000        1.000             1.0000
   mid    head         Always-Energy     6.6459         4.2013      -0.0457  0.0103        1.000             0.6968
   mid    head            Always-MLS     6.6459         4.4379      -0.4013  0.0001        1.000             0.6968
   mid    head            Always-MSR     6.6459         6.3073      -2.2204  0.0000        1.000             0.8756
   mid    head Oracle-on-train (MLS)     6.6459         4.4379      -0.4013  0.0001        1.000             0.6968
  near feature        Always-NNGuide     2.7848         3.5305      -1.4217  0.0001        0.728             1.0000
  near    head         Always-Energy     9.5759         4.4173      -1.1156  0.0056        1.000             0.2853
  near    head            Always-MLS     9.5759         3.4925      -0.4389  0.0087        1.000             0.2853
  near    head            Always-MSR     9.5759         4.0941      -0.7641  0.0054        1.000             0.2853
  near    head Oracle-on-train (MLS)     9.5759         3.4925      -0.4389  0.0087        1.000             0.2853
  near    head            Random-CSF     9.5759        12.5739      -6.3934  0.0000        1.000             0.7939
```

## none_nr_marginal

Wilcoxon: better 49, worse 2 of 54. Mean test: better 32, worse 0 of 54.

```
regime    side  pred_mean  wil_better  wil_worse  mean_better  mean_worse  tests best_baseline  baseline_mean  reduction_%
  near     all       1.23           6          0            6           0      8    Always-CTM           2.07         41.0
   mid     all       1.16           8          0            8           0      8    Always-CTM           3.52         67.0
   far     all       0.80           8          0            8           0      8    Always-CTM           2.40         67.0
  near    head       3.33           5          0            1           0      5    Always-MLS           3.49          5.0
   mid    head       5.29           5          0            1           0      5 Always-Energy           4.20        -26.0
   far    head       3.04           5          0            1           0      5    Always-MLS           3.77         19.0
  near feature       6.03           2          2            1           0      5    Always-CTM           1.55       -290.0
   mid feature       1.64           5          0            3           0      5    Always-CTM           2.92         44.0
   far feature       1.89           5          0            3           0      5    Always-CTM           2.19         14.0
```

Comparisons where the two tests disagree:

```
regime    side            comparator  pred_mean  baseline_mean  median_diff  p_holm  p_mean_holm  p_mean_worse_holm
   far feature            Always-CTM     1.8893         2.1876       0.0000  0.0121       0.7291                1.0
   far feature Oracle-on-train (CTM)     1.8893         2.1876       0.0000  0.0121       0.7291                1.0
   far    head         Always-Energy     3.0420         4.0927      -1.3264  0.0000       0.3259                1.0
   far    head            Always-MLS     3.0420         3.7728      -0.8369  0.0000       0.3919                1.0
   far    head            Always-MSR     3.0420         4.5299      -0.9224  0.0000       0.0838                1.0
   far    head Oracle-on-train (MLS)     3.0420         3.7728      -0.8369  0.0000       0.3919                1.0
   mid feature            Always-CTM     1.6371         2.9187      -0.8999  0.0000       0.0573                1.0
   mid feature Oracle-on-train (CTM)     1.6371         2.9187      -0.8999  0.0000       0.0573                1.0
   mid    head         Always-Energy     5.2929         4.2013      -0.0439  0.0008       1.0000                1.0
   mid    head            Always-MLS     5.2929         4.4379      -0.4493  0.0000       1.0000                1.0
   mid    head            Always-MSR     5.2929         6.3073      -1.7796  0.0000       1.0000                1.0
   mid    head Oracle-on-train (MLS)     5.2929         4.4379      -0.4493  0.0000       1.0000                1.0
  near feature           Always-fDBD     6.0321         6.5146      -1.9310  0.0004       1.0000                1.0
  near    head         Always-Energy     3.3299         4.4173      -1.7388  0.0000       0.5497                1.0
  near    head            Always-MLS     3.3299         3.4925      -0.6192  0.0000       0.9088                1.0
  near    head            Always-MSR     3.3299         4.0941      -0.7641  0.0000       0.6463                1.0
  near    head Oracle-on-train (MLS)     3.3299         3.4925      -0.6192  0.0000       0.9088                1.0
```

## source_nr_pooled

Wilcoxon: better 32, worse 17 of 54. Mean test: better 22, worse 17 of 54.

```
regime    side  pred_mean  wil_better  wil_worse  mean_better  mean_worse  tests best_baseline  baseline_mean  reduction_%
  near     all       2.20           6          0            6           0      8    Always-CTM           2.07         -6.0
   mid     all       2.37           8          0            5           0      8    Always-CTM           3.52         33.0
   far     all       1.63           8          0            6           0      8    Always-CTM           2.40         32.0
  near    head      20.22           0          5            0           5      5    Always-MLS           3.49       -479.0
   mid    head      24.86           0          5            0           5      5 Always-Energy           4.20       -492.0
   far    head      18.96           0          5            0           5      5    Always-MLS           3.77       -402.0
  near feature       5.95           2          2            1           2      5    Always-CTM           1.55       -285.0
   mid feature       2.99           5          0            2           0      5    Always-CTM           2.92         -2.0
   far feature       2.92           3          0            2           0      5    Always-CTM           2.19        -34.0
```

Comparisons where the two tests disagree:

```
regime    side            comparator  pred_mean  baseline_mean  median_diff  p_holm  p_mean_holm  p_mean_worse_holm
   far     all            Always-CTM     1.6325         2.4033       0.0000  0.0159       0.2755                1.0
   far     all Oracle-on-train (CTM)     1.6325         2.4033       0.0000  0.0159       0.2755                1.0
   far feature        Always-NNGuide     2.9243         4.4086      -0.4731  0.0007       0.1941                1.0
   mid     all            Always-CTM     2.3720         3.5191      -1.0056  0.0000       0.2753                1.0
   mid     all        Always-NNGuide     2.3720         4.7916      -0.6306  0.0000       0.0506                1.0
   mid     all Oracle-on-train (CTM)     2.3720         3.5191      -1.0056  0.0000       0.2753                1.0
   mid feature            Always-CTM     2.9851         2.9187      -0.3421  0.0034       1.0000                1.0
   mid feature        Always-NNGuide     2.9851         4.1911       0.0000  0.0034       0.5528                1.0
   mid feature Oracle-on-train (CTM)     2.9851         2.9187      -0.3421  0.0034       1.0000                1.0
  near feature           Always-fDBD     5.9543         6.5146      -1.8426  0.0001       1.0000                1.0
```

## none_nr_pooled

Wilcoxon: better 32, worse 17 of 54. Mean test: better 20, worse 17 of 54.

```
regime    side  pred_mean  wil_better  wil_worse  mean_better  mean_worse  tests best_baseline  baseline_mean  reduction_%
  near     all       2.34           6          0            6           0      8    Always-CTM           2.07        -13.0
   mid     all       3.16           8          0            3           0      8    Always-CTM           3.52         10.0
   far     all       1.94           8          0            6           0      8    Always-CTM           2.40         19.0
  near    head      24.80           0          5            0           5      5    Always-MLS           3.49       -610.0
   mid    head      32.61           0          5            0           5      5 Always-Energy           4.20       -676.0
   far    head      27.88           0          5            0           5      5    Always-MLS           3.77       -639.0
  near feature       7.59           2          2            1           2      5    Always-CTM           1.55       -390.0
   mid feature       4.28           5          0            2           0      5    Always-CTM           2.92        -47.0
   far feature       3.22           3          0            2           0      5    Always-CTM           2.19        -47.0
```

Comparisons where the two tests disagree:

```
regime    side            comparator  pred_mean  baseline_mean  median_diff  p_holm  p_mean_holm  p_mean_worse_holm
   far     all            Always-CTM     1.9395         2.4033      -0.3243  0.0097       0.6057             1.0000
   far     all Oracle-on-train (CTM)     1.9395         2.4033      -0.3243  0.0097       0.6057             1.0000
   far feature        Always-NNGuide     3.2213         4.4086      -0.9596  0.0003       0.3919             1.0000
   mid     all            Always-CTM     3.1585         3.5191      -1.0056  0.0000       0.7376             1.0000
   mid     all         Always-Energy     3.1585         6.5797      -2.0750  0.0000       0.0905             1.0000
   mid     all            Always-MLS     3.1585         6.8164      -2.2291  0.0000       0.0505             1.0000
   mid     all        Always-NNGuide     3.1585         4.7916      -0.7967  0.0000       0.3986             1.0000
   mid     all Oracle-on-train (CTM)     3.1585         3.5191      -1.0056  0.0000       0.7376             1.0000
   mid feature            Always-CTM     4.2819         2.9187      -0.5845  0.0031       1.0000             0.9117
   mid feature        Always-NNGuide     4.2819         4.1911       0.0000  0.0053       1.0000             1.0000
   mid feature Oracle-on-train (CTM)     4.2819         2.9187      -0.5845  0.0031       1.0000             0.9117
  near feature           Always-fDBD     7.5914         6.5146      -1.9932  0.0113       1.0000             0.6913
```

## Size-matched random-set control

Per row, the exact expected regret of a uniformly random shortlist with the predictor's own set size, drawn from the same side (empty rows keep the imputed regret). One-sided tests, predictor better; Holm over the nine cells of each arm. Separate from the 54-comparison family above.

```
regime    side  pred_mean  random_mean  set_size  p_wilcoxon  p_mean  p_wilcoxon_holm  p_mean_holm                arm
   far     all     0.8015       1.6703    7.2900      0.0000  0.0000           0.0000       0.0001   none_nr_marginal
   far feature     1.8893       5.8528    3.5400      0.0000  0.0000           0.0000       0.0001   none_nr_marginal
   far    head     3.0420       2.6472    3.7500      0.0622  0.7054           0.1865       1.0000   none_nr_marginal
   mid     all     1.1617       1.6789    7.2900      0.0000  0.0001           0.0000       0.0004   none_nr_marginal
   mid feature     1.6371       7.0205    3.5400      0.0000  0.0000           0.0000       0.0001   none_nr_marginal
   mid    head     5.2929       3.2408    3.7500      0.2919  0.9480           0.5838       1.0000   none_nr_marginal
  near     all     1.2267       1.6236    7.4932      0.0003  0.0689           0.0014       0.2758   none_nr_marginal
  near feature     6.0321      10.1459    3.2027      0.0000  0.0119           0.0000       0.0595   none_nr_marginal
  near    head     3.3299       2.2756    4.2905      0.4992  0.9657           0.5838       1.0000   none_nr_marginal
   far     all     0.6386       2.0934    5.8400      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
   far feature     1.0616       9.2905    3.0900      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
   far    head     4.1031       5.2687    2.7500      0.0000  0.0528           0.0001       0.1583 source_nr_marginal
   mid     all     0.9564       1.9795    5.8400      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
   mid feature     1.0936      10.8276    3.0900      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
   mid    head     6.6459       5.8779    2.7500      0.0000  0.6633           0.0000       0.7498 source_nr_marginal
  near     all     1.0637       2.3564    5.6216      0.0000  0.0007           0.0000       0.0030 source_nr_marginal
  near feature     2.7848      16.1189    2.7635      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
  near    head     9.5759       9.8918    2.8581      0.0003  0.3749           0.0003       0.7498 source_nr_marginal
   far     all     0.3915       2.0765    6.0700      0.0000  0.0000           0.0000       0.0001       paper_source
   far feature     0.7460       8.2999    3.5100      0.0000  0.0000           0.0000       0.0001       paper_source
   far    head     4.9911       6.5293    2.5600      0.0000  0.0221           0.0000       0.0221       paper_source
   mid     all     1.1767       3.3269    4.2400      0.0000  0.0000           0.0000       0.0001       paper_source
   mid feature     1.9463      13.8385    2.2700      0.0000  0.0000           0.0000       0.0001       paper_source
   mid    head    11.6609      14.1250    1.9700      0.0000  0.0000           0.0000       0.0001       paper_source
  near     all     1.0171       1.8827    6.5270      0.0000  0.0029           0.0000       0.0058       paper_source
  near feature     4.9894      19.6210    2.2365      0.0000  0.0000           0.0000       0.0001       paper_source
  near    head     1.4147       2.0718    4.2905      0.0000  0.0014           0.0000       0.0041       paper_source
```
