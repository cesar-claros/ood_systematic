# Step 6 — Within-ε label table (variants A, B, C)

**Date:** 2026-05-02
**Source:** `code/nc_csf_predictivity/data/within_eps.py`

## Worked examples — three within-ε variants on the same toy cell

Toy cell: `(paradigm=confidnet, source=cifar10, dropout=False, reward=2.2, regime=near)`. The CIFAR-10 near regime contains **two** eval_datasets: `{cifar100, tinyimagenet}`. Three CSFs (MSR, Energy, NeCo), 5 runs.

Per-(csf, run, eval) raw AUGRC:

```
   csf  run  cifar100  tinyimagenet
Energy    1     107.6         252.3
Energy    2     105.2         253.4
Energy    3     106.4         247.4
Energy    4     106.1         247.1
Energy    5     107.6         249.9
   MSR    1     100.9         256.9
   MSR    2     102.3         262.8
   MSR    3      94.1         256.1
   MSR    4     100.4         259.1
   MSR    5      99.9         257.4
  NeCo    1     129.4         243.0
  NeCo    2     133.7         244.5
  NeCo    3     128.7         243.9
  NeCo    4     131.6         246.1
  NeCo    5     131.2         246.3
```

### Variant C — raw AUGRC mean (original specification)

Per-(csf, run) mean across `{cifar100, tinyimagenet}`. Note tinyimagenet (~250) dominates cifar100 (~100) in this average:

```
   csf  summary_raw
Energy       178.31
   MSR       178.99
  NeCo       187.85
```

Best by raw mean: **Energy** (summary = 178.31). Bootstrap ε on best's 5 per-run means = **1.15** (raw AUGRC units). Set = `['Energy', 'MSR']`.

### Variant A — mean of `augrc_rank` (harmonized first)

Same operation but on `augrc_rank` (within-(source, eval_dataset) percentile rank from step 4). Each eval_dataset now contributes equally because rank ∈ [0, 1] regardless of native scale:

```
   csf  summary_rank
Energy         0.533
   MSR         0.533
  NeCo         0.533
```

Best by rank mean: **Energy** (summary rank = 0.533). Bootstrap ε = **0.060** (rank units). Set = `['Energy', 'MSR', 'NeCo']`.

### Variant B — per-eval within-ε, then aggregate

Per-eval base table:

```
   csf eval_dataset  summary_augrc_eval best_csf_eval  eps_eval  in_set_eval
Energy     cifar100              106.60           MSR      2.38        False
   MSR     cifar100               99.53           MSR      2.38         True
  NeCo     cifar100              130.93           MSR      2.38        False
Energy tinyimagenet              250.02          NeCo      1.10        False
   MSR tinyimagenet              258.46          NeCo      1.10        False
  NeCo tinyimagenet              244.77          NeCo      1.10         True
```

Aggregation to regime level (n_evals = 2):

```
   csf  set_count_per_eval  n_evals_in_regime  in_within_eps_set_majority  in_within_eps_set_unanimous
Energy                   0                  2                       False                        False
   MSR                   1                  2                        True                        False
  NeCo                   1                  2                        True                        False
```

Reading: with N_evals=2, `majority` = ≥1 of 2; `unanimous` = 2 of 2. Variant B exposes per-eval disagreement that A and C smooth over: a CSF can be `majority`-competitive (passes on some evals) without being `unanimous` (passes on all). For larger regimes (CIFAR-10 mid has 4 evals; all has 8) the gap between majority and unanimous becomes more informative.

### How the three variants compare on this toy cell

```
   csf  C_raw  A_rank  B_majority  B_unanimous
   MSR   True    True        True        False
Energy   True    True       False        False
  NeCo  False    True        True        False
```

## Run summary

- Cells processed: 280
- Total (cell × csf) rows: 5,600
- Per-eval base rows (Variant B intermediate): 19,040
- Bootstrap n_boot = 2000, seed = 0

## ε distribution per regime

```
        eps_raw (AUGRC units)  eps_rank (rank units)
regime                                              
all                     1.832                 0.0323
far                     1.821                 0.0346
mid                     2.378                 0.0414
near                    2.183                 0.0376
test                    1.811                 0.0302
```

## Within-ε set-size distribution per regime, per variant

```
        mean_set_size_C_raw  median_set_size_C_raw  mean_set_size_A_rank  median_set_size_A_rank  mean_set_size_B_majority  median_set_size_B_majority  mean_set_size_B_unanimous  median_set_size_B_unanimous
regime                                                                                                                                                                                                        
all                    3.46                    2.0                  3.36                     2.0                      3.86                         2.0                       2.56                          1.0
far                    3.50                    2.0                  3.38                     1.0                      5.05                         4.0                       3.02                          1.0
mid                    3.82                    2.0                  3.84                     2.0                      5.16                         4.5                       1.88                          1.0
near                   4.48                    2.5                  4.59                     3.0                      6.27                         6.0                       3.24                          1.5
test                   4.52                    3.0                  4.52                     3.0                      4.52                         3.0                       4.52                          3.0
```

## Pairwise agreement between variants (Jaccard)

Per-cell Jaccard between the in-set indicators of two variants, averaged across cells (within each regime). 1.0 = perfect agreement, 0.0 = disjoint sets.

```
        C_raw vs A_rank  C_raw vs B_majority  A_rank vs B_majority  B_majority vs B_unanimous
regime                                                                                       
all               0.907                0.730                 0.697                      0.213
far               0.872                0.711                 0.661                      0.484
mid               0.887                0.627                 0.610                      0.294
near              0.829                0.675                 0.701                      0.420
test              0.952                1.000                 0.952                      1.000
```

## Spot check — three variants vs Track 1 top clique

(dropout=False, lowest reward, regime ∈ {near, mid, far, all}) per (paradigm, source). Each row shows the top clique from step 5 alongside the three within-ε variants.

```
 paradigm        source  reward regime                                        top_clique                                                                 C_raw                                                      A_rank                                                                                               B_majority                     B_unanimous
confidnet       cifar10     2.2    all                                       PE, NNGuide                                                      PE, CTM, NNGuide                                            PE, CTM, NNGuide                                                                                             CTM, NNGuide                             NaN
confidnet       cifar10     2.2    far                 PE, CTM, Confidence, PCE, NNGuide                                                                   CTM                                                         CTM                                                                                             CTM, NNGuide                             CTM
confidnet       cifar10     2.2    mid                         NNGuide, Energy, PE, NeCo                               NNGuide, PE, Energy, NeCo, MLS, GE, CTM                     NNGuide, PE, Energy, NeCo, MLS, GE, CTM                                                                          NNGuide, Energy, NeCo, MLS, CTM                             NaN
confidnet       cifar10     2.2   near                               Confidence, PE, PCE                                                   PE, Confidence, PCE                                         PE, Confidence, PCE                                                           PE, Confidence, NNGuide, GE, NeCo, MLS, Energy                             NaN
confidnet      cifar100     2.2    all                                              fDBD                                                                  fDBD                                                        fDBD                                                                                                fDBD, CTM                             NaN
confidnet      cifar100     2.2    far                                              fDBD                                                                  fDBD                                                        fDBD                                                                                           fDBD, REN, GEN                            fDBD
confidnet      cifar100     2.2    mid                                     fDBD, NNGuide                                                             fDBD, ViM                                              fDBD, ViM, CTM                                                                             fDBD, ViM, CTM, NNGuide, REN                             NaN
confidnet      cifar100     2.2   near                      fDBD, GEN, REN, PCE, PE, CTM                                                                  fDBD                                                        fDBD                                                                            fDBD, CTM, ViM, GEN, PCE, MSR                             NaN
confidnet supercifar100     2.2    all                         CTM, KPCA RecError global                                                                   CTM                                                         CTM                                                                                                      CTM                             NaN
confidnet supercifar100     2.2    far          fDBD, KPCA RecError global, CTM, NNGuide                                       fDBD, CTM, KPCA RecError global                                                        fDBD                                                                     fDBD, CTM, KPCA RecError global, ViM                             NaN
confidnet supercifar100     2.2    mid                         CTM, KPCA RecError global                                                                   CTM                                                         CTM                                                                                CTM, KPCA RecError global                             CTM
confidnet supercifar100     2.2   near                                          REN, CTM                                                              CTM, REN                      CTM, REN, GEN, PE, PCE, NeCo, MLS, MSR                                                                      CTM, REN, GEN, PCE, MSR, Confidence                             CTM
confidnet  tinyimagenet     2.2    all                                               CTM                                                                   CTM                                                         CTM                                                                                                      CTM                             CTM
confidnet  tinyimagenet     2.2    far                                               CTM                                                                   CTM                                                         CTM                                                                                                      CTM                             CTM
confidnet  tinyimagenet     2.2    mid                                               CTM                                                             CTM, fDBD                                                         CTM                                                                                       CTM, fDBD, NNGuide                             CTM
confidnet  tinyimagenet     2.2   near                                               CTM                                                                   CTM                                                         CTM                                                                                                      CTM                             CTM
  devries       cifar10     2.2    all                                        Confidence                                                            Confidence                                                  Confidence                                                                                               Confidence                      Confidence
  devries       cifar10     2.2    far                                        Confidence                                      Confidence, KPCA RecError global                            Confidence, KPCA RecError global                                        Confidence, KPCA RecError global, NNGuide, PE, PCE, REN, MSR, GEN                      Confidence
  devries       cifar10     2.2    mid                                        Confidence                                                            Confidence                                                  Confidence                                                                              Confidence, NNGuide, Energy                      Confidence
  devries       cifar10     2.2   near                                        Confidence                                                            Confidence                                                  Confidence                                                                                               Confidence                      Confidence
  devries      cifar100     2.2    all                                              fDBD                                                                  fDBD                                                        fDBD                                                                                                fDBD, CTM                             NaN
  devries      cifar100     2.2    far                                              fDBD                                                                  fDBD                                                        fDBD                                                                                                fDBD, ViM                            fDBD
  devries      cifar100     2.2    mid                                              fDBD                                                                  fDBD                                                        fDBD                                                                                                fDBD, CTM                            fDBD
  devries      cifar100     2.2   near                       fDBD, PCE, NNGuide, PE, MSR                                                                  fDBD                                                        fDBD                                                                                  fDBD, CTM, PCE, PE, MSR                             NaN
  devries supercifar100     2.2    all                         CTM, KPCA RecError global                                  CTM, KPCA RecError global, ViM, fDBD                   CTM, KPCA RecError global, ViM, fDBD, REN                                                            KPCA RecError global, ViM, fDBD, REN, NNGuide                             NaN
  devries supercifar100     2.2    far                   KPCA RecError global, pNML, ViM                                       pNML, ViM, KPCA RecError global                       pNML, ViM, KPCA RecError global, fDBD                                               pNML, ViM, KPCA RecError global, fDBD, NNGuide, Confidence                             NaN
  devries supercifar100     2.2    mid                    CTM, KPCA RecError global, ViM                                        ViM, CTM, KPCA RecError global    ViM, CTM, KPCA RecError global, fDBD, pNML, REN, NNGuide                 ViM, CTM, KPCA RecError global, fDBD, pNML, REN, NNGuide, PE, NeCo, PCE, MLS, Energy, GE                             NaN
  devries supercifar100     2.2   near                                          REN, CTM CTM, REN, NNGuide, KPCA RecError global, PE, PCE, GEN, NeCo, MLS, MSR CTM, REN, NNGuide, PE, PCE, GEN, NeCo, MLS, MSR, Confidence CTM, REN, NNGuide, KPCA RecError global, PE, PCE, GEN, NeCo, MLS, MSR, fDBD, GE, Energy, Confidence, ViM                             REN
  devries  tinyimagenet     2.2    all                                               CTM                                                                   CTM                                                         CTM                                                                                                      CTM                             CTM
  devries  tinyimagenet     2.2    far                         CTM, KPCA RecError global                                       CTM, KPCA RecError global, fDBD                             CTM, KPCA RecError global, fDBD                                                                          CTM, KPCA RecError global, fDBD CTM, KPCA RecError global, fDBD
  devries  tinyimagenet     2.2    mid                                               CTM                                                                   CTM                                                         CTM                                                                                       CTM, fDBD, NNGuide                             CTM
  devries  tinyimagenet     2.2   near                                               CTM                                                                   CTM                                                         CTM                                                                                                      CTM                             CTM
       dg       cifar10     2.2    all                                   CTM, Confidence                                                       CTM, Confidence                                             CTM, Confidence                                                                                               Confidence                             NaN
       dg       cifar10     2.2    far                                              Maha                                                                  Maha                                                        Maha                                                                                                Maha, CTM                             NaN
       dg       cifar10     2.2    mid                                   Confidence, CTM                                                            Confidence                                                  Confidence                                                                                               Confidence                      Confidence
       dg       cifar10     2.2   near                                   CTM, Confidence                                                       CTM, Confidence                                             CTM, Confidence                                                                                          CTM, Confidence                             NaN
       dg      cifar100     6.0    all                                     fDBD, NNGuide                                                                  fDBD                                                        fDBD                                                                                  fDBD, CTM, NNGuide, ViM                             NaN
       dg      cifar100     6.0    far                                              fDBD                                                                  fDBD                                                        fDBD                                                                                           fDBD, CTM, ViM                            fDBD
       dg      cifar100     6.0    mid                                     fDBD, NNGuide                                                         fDBD, NNGuide                                               fDBD, NNGuide                                                                      fDBD, NNGuide, CTM, ViM, Confidence                            fDBD
       dg      cifar100     6.0   near PCE, fDBD, PE, CTM, NNGuide, REN, MSR, Confidence                                                         fDBD, NNGuide                                      fDBD, NNGuide, PE, PCE                                                               fDBD, NNGuide, CTM, PE, PCE, MSR, REN, GEN                             NaN
       dg supercifar100     2.2    all                                         ViM, Maha                                                             ViM, Maha                                                   ViM, Maha                                                                     ViM, Maha, pNML, PCA RecError global                             NaN
       dg supercifar100     2.2    far                                              Maha                                                             Maha, ViM                                                   Maha, ViM                                                                                 Maha, ViM, Residual, CTM                             NaN
       dg supercifar100     2.2    mid                                         Maha, ViM                                             ViM, Maha, Residual, pNML                                   ViM, Maha, Residual, pNML                                                                     ViM, Maha, pNML, PCA RecError global                             ViM
       dg supercifar100     2.2   near             Confidence, KPCA RecError global, CTM                            Confidence, KPCA RecError global, ViM, CTM                                                  Confidence                         Confidence, KPCA RecError global, ViM, pNML, PCA RecError global, Maha, Residual                             NaN
       dg  tinyimagenet    10.0    all                                               CTM                                                                   CTM                                                         CTM                                                                                CTM, KPCA RecError global                             NaN
       dg  tinyimagenet    10.0    far                         KPCA RecError global, CTM                                       KPCA RecError global, CTM, fDBD                                   KPCA RecError global, CTM                                                                          KPCA RecError global, CTM, fDBD KPCA RecError global, CTM, fDBD
       dg  tinyimagenet    10.0    mid                                      CTM, NNGuide                                                                   CTM                                                         CTM                                                                                       CTM, fDBD, NNGuide                             CTM
       dg  tinyimagenet    10.0   near                                               CTM                                             CTM, KPCA RecError global                                   CTM, KPCA RecError global                                                                                CTM, KPCA RecError global                             NaN
```
