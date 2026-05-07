# Step 5 — Track 1 clique recomputation

**Date:** 2026-05-02
**Source:** `code/nc_csf_predictivity/data/cliques_track1.py`

## Worked example — Friedman/Conover/clique on a 3×4 toy table

Three CSFs (A, B, C) evaluated on four blocks. Values shown are the raw AUGRC (lower = better). The cell pipeline negates these to get scores (higher = better) so the existing Friedman/post-hoc code can be reused unchanged.

Raw AUGRC:

```
         A   B   C
block1  10  20  11
block2  12  18  12
block3  11  22  12
block4  13  19  13
```

After negation (`score = -augrc`, higher = better):

```
         A   B   C
block1 -10 -20 -11
block2 -12 -18 -12
block3 -11 -22 -12
block4 -13 -19 -13
```

Rank within each block (`ascending=False` ⇒ rank 1 = highest score = best):

```
          A    B    C
block1  1.0  3.0  2.0
block2  1.5  3.0  1.5
block3  1.0  3.0  2.0
block4  1.5  3.0  1.5
```

Mean rank per CSF (sorted):

```
A    1.25
C    1.75
B    3.00
```

Friedman χ² statistic = **7.429**, p = **0.0244**. With p < 0.05, we reject H₀ that all CSFs are equivalent on these blocks → there is at least one significant pairwise difference, so a Conover post-hoc is run to identify which pairs.

Conover post-hoc p-matrix:

```
        A       B       C
A  1.0000  0.0001  0.0498
B  0.0001  1.0000  0.0009
C  0.0498  0.0009  1.0000
```

Build the 'not-significant' graph: edge between CSF i and j iff p_ij ≥ 0.05. Maximal cliques (Bron-Kerbosch) on this graph are the candidate top cliques.

Edges (not significantly different): none

With no edges, every CSF is its own clique. The top clique is the singleton with the lowest mean rank — here `{A}`.

## Run summary

- Cells processed: 280

Rows in flat parquet per regime:

```
regime
all     1120
far     1120
mid     1120
near    1120
test    1120
```

## Top-clique size distribution per regime

```
        count  mean   std  min  25%  50%  75%   max
regime                                             
all      56.0  1.75  0.92  1.0  1.0  2.0  2.0   5.0
far      56.0  2.66  1.87  1.0  1.0  2.0  4.0  11.0
mid      56.0  1.95  1.17  1.0  1.0  2.0  2.0   7.0
near     56.0  3.39  2.68  1.0  1.0  3.0  5.0  13.0
test     56.0  1.71  0.99  1.0  1.0  1.0  2.0   6.0
```

## Spot check — top cliques for one cell per (paradigm, source)

Showing the top clique for `(dropout=False, lowest reward, regime ∈ {near, mid, far, all})` per (paradigm, source). This lets us eyeball the cliques against the published per-paradigm cliques in `code/ood_eval_outputs/top_cliques_Conv_False_RC_<paradigm>_cliques.json` (which pool over dropout/reward, so they are coarser than ours).

```
 paradigm        source  reward regime                                        top_clique
confidnet       cifar10     2.2    all                                       PE, NNGuide
confidnet       cifar10     2.2    far                 PE, CTM, Confidence, PCE, NNGuide
confidnet       cifar10     2.2    mid                         NNGuide, Energy, PE, NeCo
confidnet       cifar10     2.2   near                               Confidence, PE, PCE
confidnet      cifar100     2.2    all                                              fDBD
confidnet      cifar100     2.2    far                                              fDBD
confidnet      cifar100     2.2    mid                                     fDBD, NNGuide
confidnet      cifar100     2.2   near                      fDBD, GEN, REN, PCE, PE, CTM
confidnet supercifar100     2.2    all                         CTM, KPCA RecError global
confidnet supercifar100     2.2    far          fDBD, KPCA RecError global, CTM, NNGuide
confidnet supercifar100     2.2    mid                         CTM, KPCA RecError global
confidnet supercifar100     2.2   near                                          REN, CTM
confidnet  tinyimagenet     2.2    all                                               CTM
confidnet  tinyimagenet     2.2    far                                               CTM
confidnet  tinyimagenet     2.2    mid                                               CTM
confidnet  tinyimagenet     2.2   near                                               CTM
  devries       cifar10     2.2    all                                        Confidence
  devries       cifar10     2.2    far                                        Confidence
  devries       cifar10     2.2    mid                                        Confidence
  devries       cifar10     2.2   near                                        Confidence
  devries      cifar100     2.2    all                                              fDBD
  devries      cifar100     2.2    far                                              fDBD
  devries      cifar100     2.2    mid                                              fDBD
  devries      cifar100     2.2   near                       fDBD, PCE, NNGuide, PE, MSR
  devries supercifar100     2.2    all                         CTM, KPCA RecError global
  devries supercifar100     2.2    far                   KPCA RecError global, pNML, ViM
  devries supercifar100     2.2    mid                    CTM, KPCA RecError global, ViM
  devries supercifar100     2.2   near                                          REN, CTM
  devries  tinyimagenet     2.2    all                                               CTM
  devries  tinyimagenet     2.2    far                         CTM, KPCA RecError global
  devries  tinyimagenet     2.2    mid                                               CTM
  devries  tinyimagenet     2.2   near                                               CTM
       dg       cifar10     2.2    all                                   CTM, Confidence
       dg       cifar10     2.2    far                                              Maha
       dg       cifar10     2.2    mid                                   Confidence, CTM
       dg       cifar10     2.2   near                                   CTM, Confidence
       dg      cifar100     6.0    all                                     fDBD, NNGuide
       dg      cifar100     6.0    far                                              fDBD
       dg      cifar100     6.0    mid                                     fDBD, NNGuide
       dg      cifar100     6.0   near PCE, fDBD, PE, CTM, NNGuide, REN, Confidence, MSR
       dg supercifar100     2.2    all                                         ViM, Maha
       dg supercifar100     2.2    far                                              Maha
       dg supercifar100     2.2    mid                                         Maha, ViM
       dg supercifar100     2.2   near             Confidence, KPCA RecError global, CTM
       dg  tinyimagenet    10.0    all                                               CTM
       dg  tinyimagenet    10.0    far                         KPCA RecError global, CTM
       dg  tinyimagenet    10.0    mid                                      CTM, NNGuide
       dg  tinyimagenet    10.0   near                                               CTM
```
