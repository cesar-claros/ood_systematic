# Step 15 — Statistical tests menu

**Date:** 2026-05-04
**Source:** `code/nc_csf_predictivity/stats/tests.py`
**Permutation R:** 5000; **Mantel R:** 999; **seed:** 0

## Headline — xarch paired Wilcoxon (predictor vs Always-CTM)

```
regime    side                               predictor   n  median_diff       W      p  p_holm  reject_holm_05
   far     all      imp/per_csf_binary/within_eps_rank 100       0.0000     0.0 0.0000  0.0024            True
   far     all       imp/per_csf_binary/within_eps_raw 100       0.0000     0.0 0.0001  0.0051            True
   far     all  imp/per_csf_binary/within_eps_majority 100       0.0000     0.0 0.0003  0.0154            True
   far     all                   imp/multilabel/clique 100       0.0000   644.0 0.1045  1.0000           False
   far     all      imp/multilabel/within_eps_majority 100       0.0000   687.0 0.5890  1.0000           False
   far     all               imp/per_csf_binary/clique 100       0.0000   812.0 0.9187  1.0000           False
   far     all          imp/multilabel/within_eps_rank 100       0.9516  2214.0 1.0000  1.0000           False
   far     all           imp/multilabel/within_eps_raw 100       0.0000  1585.0 1.0000  1.0000           False
   far     all     imp/multilabel/within_eps_unanimous 100      23.4924  3712.0 1.0000  1.0000           False
   far     all imp/per_csf_binary/within_eps_unanimous 100      25.8250  3815.0 1.0000  1.0000           False
   far     all                          raw/regression 100       2.0768  4044.0 1.0000  1.0000           False
   far feature      imp/per_csf_binary/within_eps_rank 100       0.0000     0.0 0.0005  0.0216            True
   far feature       imp/per_csf_binary/within_eps_raw 100       0.0000     0.0 0.0011  0.0466            True
   far feature  imp/per_csf_binary/within_eps_majority 100       0.0000     0.0 0.0038  0.1499           False
   far feature      imp/multilabel/within_eps_majority 100       0.0000   715.0 0.9036  1.0000           False
   far feature                   imp/multilabel/clique 100       0.0000  1144.0 0.9991  1.0000           False
   far feature               imp/per_csf_binary/clique 100       0.0000  1008.0 0.9994  1.0000           False
   far feature          imp/multilabel/within_eps_rank 100       1.0437  2237.0 1.0000  1.0000           False
   far feature           imp/multilabel/within_eps_raw 100       0.0000  1696.0 1.0000  1.0000           False
   far feature     imp/multilabel/within_eps_unanimous 100      16.1411  3815.0 1.0000  1.0000           False
   far feature imp/per_csf_binary/within_eps_unanimous 100      16.1411  3815.0 1.0000  1.0000           False
   far feature                          raw/regression 100       2.0768  4044.0 1.0000  1.0000           False
   mid     all      imp/multilabel/within_eps_majority 200       0.0000   774.0 0.0000  0.0000            True
   mid     all  imp/per_csf_binary/within_eps_majority 200       0.0000     0.0 0.0000  0.0000            True
   mid     all      imp/per_csf_binary/within_eps_rank 200       0.0000     0.0 0.0000  0.0000            True
   mid     all       imp/per_csf_binary/within_eps_raw 200       0.0000     0.0 0.0000  0.0000            True
   mid     all           imp/multilabel/within_eps_raw 200       0.0000  1466.0 0.0044  0.1361           False
   mid     all          imp/multilabel/within_eps_rank 200       0.0000  2371.0 0.2442  1.0000           False
   mid     all                   imp/multilabel/clique 200       0.0000  3383.0 0.7908  1.0000           False
   mid     all                          raw/regression 200       1.1510 12724.0 0.9994  1.0000           False
   mid     all     imp/multilabel/within_eps_unanimous 200      45.8902 15383.0 1.0000  1.0000           False
   mid     all               imp/per_csf_binary/clique 200       0.0000  3513.0 1.0000  1.0000           False
   mid     all imp/per_csf_binary/within_eps_unanimous 200      40.0080 15524.0 1.0000  1.0000           False
   mid feature  imp/per_csf_binary/within_eps_majority 200       0.0000     0.0 0.0000  0.0000            True
   mid feature      imp/per_csf_binary/within_eps_rank 200       0.0000     0.0 0.0000  0.0000            True
   mid feature       imp/per_csf_binary/within_eps_raw 200       0.0000     0.0 0.0000  0.0000            True
   mid feature      imp/multilabel/within_eps_majority 200       0.0000  2318.0 0.0277  0.9340           False
   mid feature                          raw/regression 200       1.1510 12724.0 0.9994  1.0000           False
   mid feature                   imp/multilabel/clique 200       0.0000  5640.0 1.0000  1.0000           False
   mid feature          imp/multilabel/within_eps_rank 200       0.0000  2658.0 1.0000  1.0000           False
   mid feature           imp/multilabel/within_eps_raw 200       0.0000  3484.0 1.0000  1.0000           False
   mid feature     imp/multilabel/within_eps_unanimous 200      26.3787 15576.0 1.0000  1.0000           False
   mid feature               imp/per_csf_binary/clique 200       0.0000  4272.0 1.0000  1.0000           False
   mid feature imp/per_csf_binary/within_eps_unanimous 200      19.1843 15525.0 1.0000  1.0000           False
  near     all  imp/per_csf_binary/within_eps_majority 148       0.0000     0.0 0.0000  0.0002            True
  near     all      imp/per_csf_binary/within_eps_rank 148       0.0000     0.0 0.0001  0.0046            True
  near     all       imp/per_csf_binary/within_eps_raw 148       0.0000     0.0 0.0001  0.0046            True
  near     all      imp/multilabel/within_eps_majority 148       0.0000   575.0 0.1496  1.0000           False
  near     all                   imp/multilabel/clique 148       0.0000   838.0 0.6279  1.0000           False
  near     all           imp/multilabel/within_eps_raw 148       0.0000  2094.0 0.9999  1.0000           False
  near     all          imp/multilabel/within_eps_rank 148       0.0000  3490.0 1.0000  1.0000           False
  near     all     imp/multilabel/within_eps_unanimous 148       0.0000  3824.0 1.0000  1.0000           False
  near     all               imp/per_csf_binary/clique 148       0.0000  1075.0 1.0000  1.0000           False
  near     all imp/per_csf_binary/within_eps_unanimous 148       3.3819  3778.0 1.0000  1.0000           False
  near     all                          raw/regression 148       2.0890  8653.0 1.0000  1.0000           False
  near feature       imp/per_csf_binary/within_eps_raw 148       0.0000     0.0 0.0003  0.0118            True
  near feature  imp/per_csf_binary/within_eps_majority 148       0.0000     0.0 0.0005  0.0167            True
  near feature      imp/per_csf_binary/within_eps_rank 148       0.0000     0.0 0.0544  1.0000           False
  near feature      imp/multilabel/within_eps_majority 148       0.0000   749.0 0.9749  1.0000           False
  near feature                   imp/multilabel/clique 148       0.0000  1230.0 1.0000  1.0000           False
  near feature          imp/multilabel/within_eps_rank 148       0.2391  3537.0 1.0000  1.0000           False
  near feature           imp/multilabel/within_eps_raw 148       0.0000  2147.0 1.0000  1.0000           False
  near feature     imp/multilabel/within_eps_unanimous 148      11.1080  3741.0 1.0000  1.0000           False
  near feature               imp/per_csf_binary/clique 148       0.0000  1176.0 1.0000  1.0000           False
  near feature imp/per_csf_binary/within_eps_unanimous 148      11.1080  3741.0 1.0000  1.0000           False
  near feature                          raw/regression 148       2.0890  8653.0 1.0000  1.0000           False
```

Reading: `predictor` is `imp/<predictor_name>` (always-predicts via empty-set imputation) or `raw/regression`. `median_diff` is `regret(predictor) − regret(Always-CTM)`; negative = NC predictor wins. `p_holm` is Holm-Bonferroni corrected within (regime, side). `reject_holm_05` flags predictors that significantly beat Always-CTM at α=0.05 after correction.

## Permutation test (NC ↔ oracle association)

On xarch test set (n=448): observed top-1 acc = **0.1429**. Under permuted oracle labels: mean acc = 0.1429, 95% percentile = 0.1429. Permutation p-value = **1.0000**.

## Multinomial logistic LR test (NC features → oracle CSF)

On xarch test rows (n=448, K=20 oracle classes, 8 NC features): LR statistic = **534.80** on df = 152, p = **4.51e-44**.

## Mantel test (NC pairwise distance vs AUGRC-vector cosine)

On 56 sampled models from xarch test pool (1,540 pairs): observed Mantel r = **0.5858**. Mean perm r = 0.0019, perm p = **0.0010**.

## Conditional Friedman within NC bins

k=3 KMeans on standardized NC for 56 models. Unconditional χ² = **2842.75**, avg per-bin χ² = **1175.33**, ratio = 0.413.

- bin 0 (n=16): χ² = 1531.99, p = 0.00e+00
- bin 1 (n=39): χ² = 1862.07, p = 0.00e+00
- bin 2 (n=1): χ² = 131.94, p = 6.27e-19

Per-bin χ² < unconditional χ² ⇒ conditioning on NC reduces CSF-rank disagreement, consistent with NC being a relevant covariate.

## Spearman / Kendall ρ (regression per-row rankings)

Spearman ρ on xarch regression per-row CSF orderings, with bootstrap 95% CI:

```
regime    side   n  mean  ci_lo  ci_hi
   far     all 100 0.410  0.330  0.483
   far feature 100 0.546  0.449  0.631
   far    head 100 0.325  0.273  0.377
   mid     all 200 0.454  0.396  0.507
   mid feature 200 0.557  0.488  0.619
   mid    head 200 0.374  0.335  0.411
  near     all 148 0.631  0.589  0.671
  near feature 148 0.753  0.702  0.801
  near    head 148 0.462  0.425  0.498
```

## per_csf_binary vs multilabel (paired Wilcoxon, two-sided)

```
regime    side           label_rule comparator_kind   n  median_diff_pc_minus_ml      W      p
   far     all               clique         imputed 100                   0.0000  309.0 0.0025
   far     all  within_eps_majority         imputed 100                   0.0000  293.0 0.1759
   far     all      within_eps_rank         imputed 100                  -0.9516   46.0 0.0000
   far     all       within_eps_raw         imputed 100                   0.0000  166.0 0.0000
   far     all within_eps_unanimous         imputed 100                   0.0000    0.0 0.0010
   far feature               clique         imputed 100                   0.0000  319.0 0.6239
   far feature  within_eps_majority         imputed 100                   0.0000  237.0 0.0328
   far feature      within_eps_rank         imputed 100                  -1.0437   23.0 0.0000
   far feature       within_eps_raw         imputed 100                   0.0000   87.0 0.0000
   far feature within_eps_unanimous         imputed 100                   0.0000    NaN 1.0000
   far    head               clique         imputed 100                   0.0000    0.0 0.0000
   far    head  within_eps_majority         imputed 100                   0.0000    1.0 0.0012
   far    head      within_eps_rank         imputed 100                   0.0000   22.0 0.0057
   far    head       within_eps_raw         imputed 100                   0.0000   21.0 0.0050
   far    head within_eps_unanimous         imputed 100                   0.0000    0.0 0.0010
   mid     all               clique         imputed 200                   0.0000  352.0 0.0000
   mid     all  within_eps_majority         imputed 200                   0.0000  989.0 0.0141
   mid     all      within_eps_rank         imputed 200                   0.0000 1429.0 0.0000
   mid     all       within_eps_raw         imputed 200                   0.0000 1409.0 0.0102
   mid     all within_eps_unanimous         imputed 200                   0.0000  393.0 0.0290
   mid feature               clique         imputed 200                   0.0000  325.0 0.2532
   mid feature  within_eps_majority         imputed 200                   0.0000 1962.0 0.3077
   mid feature      within_eps_rank         imputed 200                   0.0000  777.0 0.0000
   mid feature       within_eps_raw         imputed 200                   0.0000   71.0 0.0000
   mid feature within_eps_unanimous         imputed 200                   0.0000    0.0 0.0000
   mid    head               clique         imputed 200                   0.0000    0.0 0.0000
   mid    head  within_eps_majority         imputed 200                   0.0000   42.0 0.8068
   mid    head      within_eps_rank         imputed 200                   0.0000  189.0 0.0000
   mid    head       within_eps_raw         imputed 200                   0.0000    1.0 0.0000
   mid    head within_eps_unanimous         imputed 200                   0.0000    1.0 0.0001
  near     all               clique         imputed 148                   0.0000  183.0 0.0002
  near     all  within_eps_majority         imputed 148                   0.0000  139.0 0.0115
  near     all      within_eps_rank         imputed 148                   0.0000  240.0 0.0000
  near     all       within_eps_raw         imputed 148                   0.0000  186.0 0.0000
  near     all within_eps_unanimous         imputed 148                   0.0000    0.0 0.0002
  near feature               clique         imputed 148                   0.0000   26.0 0.0032
  near feature  within_eps_majority         imputed 148                   0.0000  187.0 0.0027
  near feature      within_eps_rank         imputed 148                  -0.2391  291.0 0.0000
  near feature       within_eps_raw         imputed 148                   0.0000   46.0 0.0000
  near feature within_eps_unanimous         imputed 148                   0.0000    NaN 1.0000
  near    head               clique         imputed 148                   0.0000  323.0 0.0000
  near    head  within_eps_majority         imputed 148                   0.0000   21.0 0.2858
  near    head      within_eps_rank         imputed 148                   0.0000   98.0 0.1373
  near    head       within_eps_raw         imputed 148                   0.0000   30.0 0.0000
  near    head within_eps_unanimous         imputed 148                   0.0000    0.0 0.0000
```

`median_diff_pc_minus_ml > 0` ⇒ per_csf has higher regret (worse). Two-sided p tests whether the two predictors give systematically different set-regret on the same rows.

