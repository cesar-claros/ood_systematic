# Step 8 — ResNet18 sanity-check cliques

**Date:** 2026-05-03
**Source:** `code/nc_csf_predictivity/data/cliques_resnet18.py`

## Worked examples — degenerate vs richer block cells

Because ResNet18 has 1 run per cell, the block dimensions for Friedman are only `(eval_dataset × metric ∈ {augrc, aurc})`. The block count therefore depends on how many eval_datasets the regime contains.

### Degenerate case — `(source=tinyimagenet, regime=far)`

The CIFAR/CLIP grouping for tinyimagenet puts only `svhn` into the `far` regime (cifar10/cifar100/etc. land in `near`; places365/textures in `mid`). So the cell has:

- 1 eval_dataset × 2 metrics = **2 blocks**
- 20 CSFs (entities)

Friedman with 2 blocks and 20 entities is technically computable but has effectively no statistical power — it almost always fails to reject H₀, and the Conover post-hoc returns p ≈ 1 for nearly all pairs. Consequence: the 'not-significantly-different' graph is fully connected and the top clique = all 20 CSFs.

These cells are flagged with `n_blocks ≤ 4` in the parquet so downstream code can drop them or assign zero weight.

### Richer case — `(source=cifar10, regime=all)`

CIFAR-10's `all` regime pools the 8 OOD eval_datasets, giving:

- 8 eval_datasets × 2 metrics = **16 blocks**
- 20 CSFs (entities)

16 blocks is enough for Friedman to detect real differences across CSFs and for Conover to identify pairwise ties. The top clique here should look qualitatively similar to the corresponding VGG13 cliques from step 5 if the cross-architecture transfer hypothesis holds. Disagreement between the ResNet18 sanity clique and the VGG13 clique (at the same paradigm/source/dropout/reward/regime) flags either (a) genuine cross-arch CSF reordering — interesting for the paper, or (b) ResNet18's single-run noise creating a misleading clique. The regret-based ResNet18 evaluation in step 13 is the authoritative test; this clique is corroborative only.

### Hand-trace of the Friedman block construction

For one ResNet18 cell `(confidnet, cifar10, dropout=False, reward=2.2, regime=all)`, the Friedman pivot is shaped 16 × 20 (blocks × CSFs):

```
block                            CSF1  CSF2  ...  CSF20
cifar100|augrc                   ...
cifar100|aurc                    ...
tinyimagenet|augrc               ...
tinyimagenet|aurc                ...
lsun resize|augrc                ...
lsun resize|aurc                 ...
...                              ...
(8 evals × 2 metrics = 16 rows)
```

Each row is a single block: the per-block ranking of CSFs is what Friedman tests for consistency. With only 1 run, augrc and aurc are deterministic functions of the same logits, so the two metric blocks for a given eval are correlated — but they are not identical (AUGRC weights misclassification differently than AURC), so they do add real information.

## Run summary

- Cells processed: 280
- Total (cell × csf) rows: 5,600

### Status counts per regime

```
status  ok
regime    
all     56
far     56
mid     56
near    56
test    56
```

### n_blocks distribution per regime

```
        count   mean   std   min   25%   50%   75%   max
regime                                                  
all      56.0  16.00  0.00  16.0  16.0  16.0  16.0  16.0
far      56.0   3.57  0.83   2.0   4.0   4.0   4.0   4.0
mid      56.0   7.14  1.66   4.0   8.0   8.0   8.0   8.0
near     56.0   5.29  2.48   4.0   4.0   4.0   4.0  10.0
test     56.0   2.00  0.00   2.0   2.0   2.0   2.0   2.0
```

### Top-clique size distribution per regime (where status==ok)

```
        count  mean   std  min  25%  50%    75%   max
regime                                               
all      56.0  3.21  2.11  1.0  2.0  2.0   4.25  10.0
far      56.0  3.98  1.87  1.0  3.0  4.0   5.00  11.0
mid      56.0  3.52  1.99  1.0  2.0  3.0   4.00  10.0
near     56.0  6.80  4.47  1.0  3.0  6.5  10.00  18.0
test     56.0  3.61  1.92  1.0  2.0  3.0   5.00   9.0
```

### Spot check vs VGG13 step-5 cliques on the same cells

(dropout=False, lowest reward) per (paradigm, source) and regime ∈ {near, mid, far, all}. ResNet18 sanity clique vs VGG13 clique from step 5. Substantial agreement supports cross-arch transfer; disagreement is flagged for inspection.

```
 paradigm        source  reward regime                                                                                           resnet18_sanity                                       vgg13_step5
confidnet       cifar10     2.2    all                                                                                 CTM, KPCA RecError global                                       PE, NNGuide
confidnet       cifar10     2.2    far                                                                     CTM, KPCA RecError global, pNML, Maha                 PE, CTM, Confidence, PCE, NNGuide
confidnet       cifar10     2.2    mid                                                                                 CTM, KPCA RecError global                         NNGuide, Energy, PE, NeCo
confidnet       cifar10     2.2   near                                                                      KPCA RecError global, CTM, pNML, ViM                               Confidence, PE, PCE
confidnet      cifar100     2.2    all                                                                                              NNGuide, CTM                                              fDBD
confidnet      cifar100     2.2    far                                                                                        NNGuide, CTM, fDBD                                              fDBD
confidnet      cifar100     2.2    mid                                                                                              NNGuide, CTM                                     fDBD, NNGuide
confidnet      cifar100     2.2   near                             PE, NNGuide, NeCo, PCE, CTM, MLS, MSR, KPCA RecError global, Confidence, fDBD                      fDBD, GEN, REN, PCE, PE, CTM
confidnet supercifar100     2.2    all                                                      ViM, Maha, pNML, CTM, KPCA RecError global, Residual                         CTM, KPCA RecError global
confidnet supercifar100     2.2    far fDBD, KPCA RecError global, ViM, Maha, CTM, NNGuide, pNML, Residual, PCA RecError global, REN, Confidence          fDBD, KPCA RecError global, CTM, NNGuide
confidnet supercifar100     2.2    mid                                                                                                 Maha, ViM                         CTM, KPCA RecError global
confidnet supercifar100     2.2   near   REN, GEN, pNML, PCA RecError global, CTM, KPCA RecError global, PCE, MSR, ViM, Maha, PE, NeCo, Residual                                          REN, CTM
confidnet  tinyimagenet     2.2    all                                                                                                       CTM                                               CTM
confidnet  tinyimagenet     2.2    far                                                                                                 CTM, fDBD                                               CTM
confidnet  tinyimagenet     2.2    mid                                                                                        CTM, fDBD, NNGuide                                               CTM
confidnet  tinyimagenet     2.2   near                                                                                                 CTM, fDBD                                               CTM
  devries       cifar10     2.2    all                                                                          Confidence, KPCA RecError global                                        Confidence
  devries       cifar10     2.2    far                                                         Confidence, ViM, KPCA RecError global, Maha, fDBD                                        Confidence
  devries       cifar10     2.2    mid                                                     Confidence, KPCA RecError global, PCA RecError global                                        Confidence
  devries       cifar10     2.2   near                                                                     Confidence, KPCA RecError global, ViM                                        Confidence
  devries      cifar100     2.2    all                                                                                              NNGuide, CTM                                              fDBD
  devries      cifar100     2.2    far                                                                                  fDBD, CTM, NNGuide, NeCo                                              fDBD
  devries      cifar100     2.2    mid                                                                                        NNGuide, CTM, fDBD                                              fDBD
  devries      cifar100     2.2   near                                                             NeCo, MLS, PCE, PE, CTM, NNGuide, MSR, Energy                       fDBD, PCE, NNGuide, PE, MSR
  devries supercifar100     2.2    all                                                                     ViM, Confidence, CTM, pNML, REN, Maha                         CTM, KPCA RecError global
  devries supercifar100     2.2    far                                                                      Maha, ViM, Residual, fDBD, pNML, CTM                   KPCA RecError global, pNML, ViM
  devries supercifar100     2.2    mid                                                                ViM, pNML, Confidence, Maha, Residual, REN                    CTM, KPCA RecError global, ViM
  devries supercifar100     2.2   near                                                           Confidence, REN, CTM, GEN, KPCA RecError global                                          REN, CTM
  devries  tinyimagenet     2.2    all                                                                                                       CTM                                               CTM
  devries  tinyimagenet     2.2    far                                                                                                 CTM, fDBD                         CTM, KPCA RecError global
  devries  tinyimagenet     2.2    mid                                                                                   CTM, NNGuide, REN, fDBD                                               CTM
  devries  tinyimagenet     2.2   near                                                                                                 CTM, fDBD                                               CTM
       dg       cifar10     2.2    all                                                                                 CTM, KPCA RecError global                                   CTM, Confidence
       dg       cifar10     2.2    far                                                                CTM, Maha, pNML, KPCA RecError global, ViM                                              Maha
       dg       cifar10     2.2    mid                                                               KPCA RecError global, CTM, Confidence, fDBD                                   Confidence, CTM
       dg       cifar10     2.2   near                               KPCA RecError global, CTM, PCA RecError global, ViM, fDBD, Maha, Confidence                                   CTM, Confidence
       dg      cifar100     6.0    all                                                                                        CTM, NNGuide, fDBD                                     fDBD, NNGuide
       dg      cifar100     6.0    far                                                                                  CTM, NNGuide, fDBD, Maha                                              fDBD
       dg      cifar100     6.0    mid                                                                       CTM, fDBD, NNGuide, ViM, Confidence                                     fDBD, NNGuide
       dg      cifar100     6.0   near                         REN, CTM, NNGuide, GEN, Confidence, PE, PCE, fDBD, KPCA RecError global, MLS, MSR PCE, fDBD, PE, CTM, NNGuide, REN, MSR, Confidence
       dg supercifar100     2.2    all                                                      Maha, Residual, ViM, KPCA RecError global, pNML, CTM                                         ViM, Maha
       dg supercifar100     2.2    far                                                                                       Maha, Residual, ViM                                              Maha
       dg supercifar100     2.2    mid                                                                                       Maha, Residual, ViM                                         Maha, ViM
       dg supercifar100     2.2   near                          Confidence, KPCA RecError global, pNML, CTM, PCA RecError global, ViM, REN, Maha             Confidence, KPCA RecError global, CTM
       dg  tinyimagenet    10.0    all                                                                                                       CTM                                               CTM
       dg  tinyimagenet    10.0    far                                                                                                 CTM, fDBD                         KPCA RecError global, CTM
       dg  tinyimagenet    10.0    mid                                                                                              CTM, NNGuide                                      CTM, NNGuide
       dg  tinyimagenet    10.0   near                                                                                                       CTM                                               CTM
```
