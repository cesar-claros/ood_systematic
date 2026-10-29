# 22-CSF predictor (Mahalanobis++ and NCI heads), VGG13 -> ResNet18

**Source:** `nc_csf_predictivity/evaluation/newcsf_heads.py`
**Protocol:** regime-free marginal heads (NC only, NC + source) on the VGG13 pool with per-config 22-CSF top-clique labels; evaluation on the paper's 56 ResNet18 models with a 22-CSF oracle per side. Baselines: Always-X for MSR, Energy, MLS, CTM, fDBD, NNGuide, MahaPP, NCI, Oracle-on-train, Random-CSF (all over 22 CSFs). Tests as in 48_regimefree_wilcoxon.md (Wilcoxon and model-clustered sign-flip mean test, 100,000 draws, seed 0, Holm within each cell).

## Gates

- GA passed: aggregate reproduces the 20-CSF AUGRC table exactly
- GB passed: oracle builder reproduces oracle.parquet
- GC passed: baseline builder reproduces the stored clique_bc baselines
- New-CSF coverage complete: 5,376 MahaPP/NCI rows

## Labels

0.9812 of 3,360 (cell, CSF) labels agree; 144 of 168 cells have identical cliques on the 20 legacy CSFs.

Positive (cell, regime) labels for the new CSFs on VGG13: MahaPP 7 of 168, NCI 40 of 168.

## Joint-side mean regret, 20 vs 22 CSFs

```
               arm  roster  near  mid  far
  none_nr_marginal      20  1.23 1.16 0.80
  none_nr_marginal      22  0.93 0.88 0.50
source_nr_marginal      20  1.06 0.96 0.64
source_nr_marginal      22  0.78 0.82 0.63
```

## none_nr_marginal

Wilcoxon: better 65, worse 0 of 66. Mean test: better 65, worse 0 of 66.

```
regime    side  pred_mean  wil_better  wil_worse  mean_better  mean_worse  tests best_baseline  baseline_mean  reduction_%
  near     all       0.93          10          0           10           0     10    Always-CTM           2.08         55.0
   mid     all       0.88          10          0           10           0     10    Always-CTM           3.56         75.0
   far     all       0.50          10          0           10           0     10    Always-CTM           2.57         80.0
  near    head       2.41           5          0            5           0      5    Always-MLS           3.49         31.0
   mid    head       3.64           5          0            5           0      5 Always-Energy           4.20         13.0
   far    head       2.75           5          0            5           0      5    Always-MLS           3.77         27.0
  near feature       1.76           6          0            6           0      7    Always-CTM           1.58        -11.0
   mid feature       1.10           7          0            7           0      7    Always-CTM           2.98         63.0
   far feature       0.87           7          0            7           0      7    Always-CTM           2.45         65.0
```

Shortlist size and empty rate:

```
regime    side  set_size_mean  empty_set_share
   far     all          8.380              0.0
   far feature          4.820              0.0
   far    head          3.560              0.0
   mid     all          8.380              0.0
   mid feature          4.820              0.0
   mid    head          3.560              0.0
  near     all          8.095              0.0
  near feature          4.122              0.0
  near    head          3.973              0.0
```

## source_nr_marginal

Wilcoxon: better 65, worse 0 of 66. Mean test: better 60, worse 0 of 66.

```
regime    side  pred_mean  wil_better  wil_worse  mean_better  mean_worse  tests best_baseline  baseline_mean  reduction_%
  near     all       0.78          10          0           10           0     10    Always-CTM           2.08         63.0
   mid     all       0.82          10          0           10           0     10    Always-CTM           3.56         77.0
   far     all       0.63          10          0           10           0     10    Always-CTM           2.57         76.0
  near    head       2.06           5          0            5           0      5    Always-MLS           3.49         41.0
   mid    head       2.82           5          0            5           0      5 Always-Energy           4.20         33.0
   far    head       2.48           5          0            5           0      5    Always-MLS           3.77         34.0
  near feature       5.35           6          0            2           0      7    Always-CTM           1.58       -238.0
   mid feature       1.46           7          0            7           0      7    Always-CTM           2.98         51.0
   far feature       1.73           7          0            6           0      7    Always-CTM           2.45         29.0
```

Shortlist size and empty rate:

```
regime    side  set_size_mean  empty_set_share
   far     all          6.990              0.0
   far feature          4.120              0.0
   far    head          2.870              0.0
   mid     all          6.990              0.0
   mid feature          4.120              0.0
   mid    head          2.870              0.0
  near     all          6.372              0.0
  near feature          3.541              0.0
  near    head          2.831              0.0
```

## Size-matched random-set control

```
regime    side  pred_mean  random_mean  set_size  p_wilcoxon  p_mean  p_wilcoxon_holm  p_mean_holm                arm
   far     all     0.5047       1.4289    8.3800      0.0000  0.0000           0.0000       0.0001   none_nr_marginal
   far feature     0.8659       5.3235    4.8200      0.0000  0.0000           0.0000       0.0001   none_nr_marginal
   far    head     2.7469       2.9501    3.5600      0.1240  0.2089           0.3720       0.6267   none_nr_marginal
   mid     all     0.8811       1.3792    8.3800      0.0000  0.0000           0.0000       0.0001   none_nr_marginal
   mid feature     1.0997       6.2275    4.8200      0.0000  0.0000           0.0000       0.0001   none_nr_marginal
   mid    head     3.6435       3.6535    3.5600      0.6180  0.4875           1.0000       0.9579   none_nr_marginal
  near     all     0.9306       1.4599    8.0946      0.0000  0.0019           0.0000       0.0077   none_nr_marginal
  near feature     1.7602       8.5878    4.1216      0.0000  0.0000           0.0000       0.0001   none_nr_marginal
  near    head     2.4146       2.4263    3.9730      0.6042  0.4789           1.0000       0.9579   none_nr_marginal
   far     all     0.6259       1.7455    6.9900      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
   far feature     1.7297       8.3914    4.1200      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
   far    head     2.4841       3.4511    2.8700      0.0001  0.0001           0.0001       0.0001 source_nr_marginal
   mid     all     0.8204       1.6967    6.9900      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
   mid feature     1.4567       9.6363    4.1200      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
   mid    head     2.8154       4.2519    2.8700      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
  near     all     0.7785       1.9881    6.3716      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
  near feature     5.3467      15.0781    3.5405      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
  near    head     2.0578       4.2217    2.8311      0.0000  0.0000           0.0000       0.0001 source_nr_marginal
```

## Anticipation of the new CSFs (target: CSF in the top 3 of 22 on the row)

```
   csf regime  rows  target_rate  shortlist_rate  auroc  p_perm                arm
MahaPP pooled   448       0.1183          0.5536 0.6431  0.0055   none_nr_marginal
MahaPP    far   100       0.2500          0.5536 0.5773  0.1230   none_nr_marginal
MahaPP    mid   200       0.0900          0.5536 0.6722  0.0128   none_nr_marginal
MahaPP   near   148       0.0676          0.5536 0.6797  0.0337   none_nr_marginal
   NCI pooled   448       0.1674          0.4643 0.6457  0.0137   none_nr_marginal
   NCI    far   100       0.1600          0.4643 0.6235  0.0675   none_nr_marginal
   NCI    mid   200       0.1450          0.4643 0.6639  0.0078   none_nr_marginal
   NCI   near   148       0.2027          0.4643 0.6331  0.0684   none_nr_marginal
MahaPP pooled   448       0.1183          0.5714 0.7390  0.0001 source_nr_marginal
MahaPP    far   100       0.2500          0.5714 0.7059  0.0006 source_nr_marginal
MahaPP    mid   200       0.0900          0.5714 0.7283  0.0015 source_nr_marginal
MahaPP   near   148       0.0676          0.5714 0.8051  0.0003 source_nr_marginal
   NCI pooled   448       0.1674          0.4286 0.5476  0.2464 source_nr_marginal
   NCI    far   100       0.1600          0.4286 0.5722  0.1931 source_nr_marginal
   NCI    mid   200       0.1450          0.4286 0.4970  0.5165 source_nr_marginal
   NCI   near   148       0.2027          0.4286 0.5907  0.1636 source_nr_marginal
```
