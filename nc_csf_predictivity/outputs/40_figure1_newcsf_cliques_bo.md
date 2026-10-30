# 40 — Figure-1 cliques from paper scores_all_bo

alpha=0.05; unadjusted Conover; blocks = dataset|model|metric|run; legacy roster = published Figure 1 inputs verbatim.

Figure-1 cells: 48 (expected 48).

## MahaPP: enters 2 of 48 cells

- vit / cifar10 / near: joins {PCA RecError, pNML}
- vit / supercifar / near: joins {GE, MLS, NNGuide, ViM, pNML}

## NCI: enters 9 of 48 cells

- confidnet / cifar100 / far: joins {fDBD}
- devries / cifar100 / near: joins {CTM, MSR, NNGuide, PCE, PE, fDBD}
- devries / cifar100 / mid: joins {fDBD}
- devries / cifar100 / far: joins {fDBD}
- dg / cifar100 / near: joins {CTM, GEN, GradNorm, MSR, NNGuide, NeCo, PCE, PE, REN, fDBD}
- dg / cifar100 / mid: joins {NNGuide, fDBD}
- dg / cifar100 / far: joins {fDBD}
- vit / cifar10 / mid: joins {KPCA RecError, pNML}
- vit / supercifar / near: joins {GE, MLS, NNGuide, ViM, pNML}

## Membership shifts among legacy CSFs: 6 cells

- devries / cifar10 / far: -{NNGuide}
- dg / cifar10 / far: +{MLS} 
- confidnet / supercifar / mid: -{Energy}
- dg / supercifar / mid: +{NeCo, Residual} 
- vit / cifar10 / mid: +{pNML} 
- vit / supercifar / near: +{pNML} 

## All cells (legacy = published reproduction; extended alongside)

| Panel | Source | Regime | Legacy top clique | Extended top clique |
|---|---|---|---|---|
| confidnet | cifar10 | near | Energy, MLS, NNGuide, NeCo | Energy, MLS, NNGuide, NeCo |
| confidnet | cifar10 | mid | Energy | Energy |
| confidnet | cifar10 | far | Energy, MLS, NNGuide, NeCo | Energy, MLS, NNGuide, NeCo |
| confidnet | cifar100 | near | Energy, GE, MLS, NNGuide, NeCo, PE | Energy, GE, MLS, NNGuide, NeCo, PE |
| confidnet | cifar100 | mid | Energy, NeCo | Energy, NeCo |
| confidnet | cifar100 | far | fDBD | **NCI**, fDBD |
| confidnet | supercifar | near | Energy, GradNorm | Energy, GradNorm |
| confidnet | supercifar | mid | CTM, Energy | CTM |
| confidnet | supercifar | far | CTM, KPCA RecError, fDBD | CTM, KPCA RecError, fDBD |
| confidnet | tinyimagenet | near | CTM | CTM |
| confidnet | tinyimagenet | mid | CTM | CTM |
| confidnet | tinyimagenet | far | CTM | CTM |
| devries | cifar10 | near | Energy | Energy |
| devries | cifar10 | mid | Energy | Energy |
| devries | cifar10 | far | Energy, MLS, NNGuide, NeCo | Energy, MLS, NeCo |
| devries | cifar100 | near | CTM, MSR, NNGuide, PCE, PE, fDBD | CTM, MSR, **NCI**, NNGuide, PCE, PE, fDBD |
| devries | cifar100 | mid | fDBD | **NCI**, fDBD |
| devries | cifar100 | far | fDBD | **NCI**, fDBD |
| devries | supercifar | near | Energy, GE, MLS, NeCo, PE | Energy, GE, MLS, NeCo, PE |
| devries | supercifar | mid | Energy, GE | Energy, GE |
| devries | supercifar | far | CTM, KPCA RecError, ViM, fDBD | CTM, KPCA RecError, ViM, fDBD |
| devries | tinyimagenet | near | CTM | CTM |
| devries | tinyimagenet | mid | CTM | CTM |
| devries | tinyimagenet | far | CTM, KPCA RecError, fDBD | CTM, KPCA RecError, fDBD |
| dg | cifar10 | near | CTM, Energy, MLS, NNGuide | CTM, Energy, MLS, NNGuide |
| dg | cifar10 | mid | CTM, Confidence | CTM, Confidence |
| dg | cifar10 | far | CTM, Energy, Maha, NNGuide | CTM, Energy, MLS, Maha, NNGuide |
| dg | cifar100 | near | CTM, GEN, GradNorm, MSR, NNGuide, NeCo, PCE, PE, REN, fDBD | CTM, GEN, GradNorm, MSR, **NCI**, NNGuide, NeCo, PCE, PE, REN, fDBD |
| dg | cifar100 | mid | NNGuide, fDBD | **NCI**, NNGuide, fDBD |
| dg | cifar100 | far | fDBD | **NCI**, fDBD |
| dg | supercifar | near | Energy, GE, MLS, NNGuide, NeCo | Energy, GE, MLS, NNGuide, NeCo |
| dg | supercifar | mid | Energy, GE, Maha, pNML | Energy, GE, Maha, NeCo, Residual, pNML |
| dg | supercifar | far | Maha, Residual | Maha, Residual |
| dg | tinyimagenet | near | CTM | CTM |
| dg | tinyimagenet | mid | CTM | CTM |
| dg | tinyimagenet | far | CTM, KPCA RecError, fDBD | CTM, KPCA RecError, fDBD |
| vit | cifar10 | near | PCA RecError, pNML | **MahaPP**, PCA RecError, pNML |
| vit | cifar10 | mid | KPCA RecError | KPCA RecError, **NCI**, pNML |
| vit | cifar10 | far | Residual, ViM, pNML | Residual, ViM, pNML |
| vit | cifar100 | near | Maha, NeCo, ViM | Maha, NeCo, ViM |
| vit | cifar100 | mid | KPCA RecError | KPCA RecError |
| vit | cifar100 | far | Residual, ViM | Residual, ViM |
| vit | supercifar | near | GE, MLS, NNGuide, ViM | GE, MLS, **MahaPP**, **NCI**, NNGuide, ViM, pNML |
| vit | supercifar | mid | KPCA RecError | KPCA RecError |
| vit | supercifar | far | Residual, pNML | Residual, pNML |
| vit | tinyimagenet | near | PE | PE |
| vit | tinyimagenet | mid | Residual, pNML | Residual, pNML |
| vit | tinyimagenet | far | GE, GradNorm | GE, GradNorm |
