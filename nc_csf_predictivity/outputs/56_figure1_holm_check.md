# Figure 1 cliques: unadjusted vs Holm-adjusted Conover p-values

**Source:** `nc_csf_predictivity/evaluation/figure1_holm_check.py`. 22-CSF Figure 1 inputs; Holm over all CSF pairs of a cell; alpha 0.05. Gate GF passed.

## OOD cells (48)

```
       cells  subset  identical  mean_size_unadjusted  mean_size_holm
panel                                                                
VGG13     36      36         10                  2.92            5.22
ViT       12      12          4                  2.50            4.25
```

## Predictor labels (22-CSF per-configuration VGG13 cells, near/mid/far)

```
 cells  subset  identical  mean_size_unadjusted  mean_size_holm  positive_labels_unadjusted  positive_labels_holm
   168     168         46                  2.71            4.55                         456                   765
```

Cells where the unadjusted clique is NOT a subset of the Holm clique: 0

## Changed OOD cells

```
    panel       source regime                                                      unadjusted                                                                                                                    holm
confidnet      cifar10    mid                                                          Energy                                                                                                    Energy, MLS, NNGuide
  devries      cifar10   near                                                          Energy                                                                                                       Energy, MLS, NeCo
  devries      cifar10    mid                                                          Energy                                                                                                       Energy, MLS, NeCo
  devries      cifar10    far                                               Energy, MLS, NeCo                                                                                              Energy, MLS, NNGuide, NeCo
       dg      cifar10   near                                       CTM, Energy, MLS, NNGuide                                                                             CTM, Confidence, Energy, MLS, NNGuide, NeCo
       dg      cifar10    mid                                                 CTM, Confidence                                                                                                     CTM, Confidence, PE
       dg      cifar10    far                                 CTM, Energy, MLS, Maha, NNGuide                                                                    CTM, Energy, KPCA RecError, MLS, Maha, NNGuide, NeCo
confidnet     cifar100   near                              Energy, GE, MLS, NNGuide, NeCo, PE     CTM, Confidence, Energy, GE, GEN, GradNorm, KPCA RecError, MLS, MSR, MahaPP, NCI, NNGuide, NeCo, PCE, PE, REN, fDBD
confidnet     cifar100    mid                                                    Energy, NeCo                                                            Energy, KPCA RecError, MLS, MahaPP, NCI, NNGuide, NeCo, fDBD
  devries     cifar100   near                           CTM, MSR, NCI, NNGuide, PCE, PE, fDBD                                                      CTM, Energy, GEN, MLS, MSR, NCI, NNGuide, NeCo, PCE, PE, REN, fDBD
  devries     cifar100    mid                                                       NCI, fDBD                                                                                                 CTM, NCI, NNGuide, fDBD
       dg     cifar100   near CTM, GEN, GradNorm, MSR, NCI, NNGuide, NeCo, PCE, PE, REN, fDBD                     CTM, Confidence, GE, GEN, GradNorm, KPCA RecError, MLS, MSR, NCI, NNGuide, NeCo, PCE, PE, REN, fDBD
       dg     cifar100    mid                                              NCI, NNGuide, fDBD                                                                      CTM, Confidence, KPCA RecError, NCI, NNGuide, fDBD
confidnet   supercifar   near                                                Energy, GradNorm                                                     CTM, Energy, GE, GEN, GradNorm, MLS, MahaPP, NNGuide, NeCo, PE, REN
confidnet   supercifar    mid                                                             CTM                                                                                CTM, Energy, KPCA RecError, MahaPP, pNML
confidnet   supercifar    far                                        CTM, KPCA RecError, fDBD                                                                        CTM, KPCA RecError, MahaPP, NCI, ViM, fDBD, pNML
  devries   supercifar   near                                       Energy, GE, MLS, NeCo, PE                                                                                CTM, Energy, GE, GradNorm, MLS, NeCo, PE
  devries   supercifar    mid                                                      Energy, GE                                                              CTM, Energy, GE, KPCA RecError, MLS, MahaPP, NeCo, PE, ViM
  devries   supercifar    far                                   CTM, KPCA RecError, ViM, fDBD                                                                                      CTM, KPCA RecError, NCI, ViM, fDBD
       dg   supercifar   near                                  Energy, GE, MLS, NNGuide, NeCo                                                                  CTM, Energy, GE, GradNorm, MLS, NNGuide, NeCo, PE, REN
       dg   supercifar    mid                          Energy, GE, Maha, NeCo, Residual, pNML                                                                   CTM, Energy, GE, MLS, Maha, NeCo, Residual, ViM, pNML
       dg   supercifar    far                                                  Maha, Residual                                                                                               Maha, Residual, ViM, fDBD
confidnet tinyimagenet    mid                                                             CTM                                                                                                               CTM, fDBD
confidnet tinyimagenet    far                                                             CTM                                                                                                               CTM, fDBD
  devries tinyimagenet    mid                                                             CTM                                                                                                            CTM, NNGuide
       dg tinyimagenet    far                                        CTM, KPCA RecError, fDBD                                                                                           CTM, KPCA RecError, NCI, fDBD
      vit      cifar10   near                                      MahaPP, PCA RecError, pNML                                                                                         MahaPP, PCA RecError, ViM, pNML
      vit     cifar100    mid                                                   KPCA RecError                                                                                                      KPCA RecError, NCI
      vit     cifar100    far                                                   Residual, ViM                                                                                                     Maha, Residual, ViM
      vit   supercifar   near                        GE, MLS, MahaPP, NCI, NNGuide, ViM, pNML Energy, GE, KPCA RecError, MLS, MSR, Maha, MahaPP, NCI, NNGuide, NeCo, PCA RecError, PCE, PE, Residual, ViM, fDBD, pNML
      vit   supercifar    mid                                                   KPCA RecError                                                                                            GradNorm, KPCA RecError, NCI
      vit tinyimagenet   near                                                              PE                                                                                                        GE, GradNorm, PE
      vit tinyimagenet    mid                                                  Residual, pNML                                                                                               Maha, Residual, ViM, pNML
      vit tinyimagenet    far                                                    GE, GradNorm                                                                                               Energy, GE, GradNorm, NCI
```

## Three most frequently tied CSFs per panel (OOD cells)

```
    panel                  top3_unadjusted                      top3_holm
confidnet    Energy (6), CTM (5), NeCo (4)   CTM (7), Energy (7), MLS (6)
  devries    CTM (5), Energy (5), fDBD (5)   CTM (8), Energy (6), MLS (6)
       dg CTM (7), NNGuide (5), Energy (4) CTM (10), MLS (5), NNGuide (5)
      vit  pNML (6), Residual (4), ViM (4)     ViM (6), pNML (6), NCI (5)
```

## Same, all 16 columns

```
    panel                      top3_unadjusted                            top3_holm
confidnet        CTM (6), Energy (6), NeCo (4)         CTM (8), Energy (7), MLS (6)
  devries        CTM (6), Energy (5), fDBD (5)         CTM (9), Energy (6), MLS (6)
       dg    CTM (10), NNGuide (5), Energy (4)   CTM (13), fDBD (6), Confidence (5)
      vit pNML (6), GEN (4), KPCA RecError (4) ViM (6), pNML (6), KPCA RecError (5)
```

## Cells entered per CSF (OOD cells)

```
                               unadjusted  holm
backbone regime csf                            
VGG13    far    fDBD                    7     9
                CTM                     6     6
                NCI                     3     6
                KPCA RecError           4     5
                Energy                  3     3
                MLS                     3     3
                NNGuide                 2     3
                NeCo                    2     3
                ViM                     1     3
                Maha                    2     2
                Residual                1     1
                MahaPP                  0     1
                pNML                    0     1
         mid    CTM                     5     9
                Energy                  5     6
                NNGuide                 1     5
                MLS                     0     5
                NeCo                    2     4
                fDBD                    2     4
                KPCA RecError           0     4
                NCI                     2     3
                MahaPP                  0     3
                Confidence              1     2
                GE                      2     2
                pNML                    1     2
                PE                      0     2
                ViM                     0     2
                Maha                    1     1
                Residual                1     1
         near   CTM                     6    10
                MLS                     5     9
                NeCo                    5     9
                Energy                  7     8
                NNGuide                 6     7
                PE                      4     6
                GE                      3     5
                GradNorm                2     5
                REN                     1     5
                GEN                     1     4
                MSR                     2     3
                NCI                     2     3
                PCE                     2     3
                fDBD                    2     3
                Confidence              0     3
                KPCA RecError           0     2
                MahaPP                  0     2
ViT      far    Residual                3     3
                ViM                     2     2
                pNML                    2     2
                GE                      1     1
                GradNorm                1     1
                Energy                  0     1
                Maha                    0     1
                NCI                     0     1
         mid    KPCA RecError           3     3
                NCI                     1     3
                pNML                    2     2
                Residual                1     1
                GradNorm                0     1
                Maha                    0     1
                ViM                     0     1
         near   ViM                     2     3
                GE                      1     2
                Maha                    1     2
                MahaPP                  2     2
                NeCo                    1     2
                PCA RecError            1     2
                PE                      1     2
                pNML                    2     2
                MLS                     1     1
                NCI                     1     1
                NNGuide                 1     1
                Energy                  0     1
                GradNorm                0     1
                KPCA RecError           0     1
                MSR                     0     1
                PCE                     0     1
                Residual                0     1
                fDBD                    0     1
```
