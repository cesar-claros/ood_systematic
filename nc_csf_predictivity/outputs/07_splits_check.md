# Step 9 — Train/test splits

**Date:** 2026-05-03
**Source:** `code/nc_csf_predictivity/evaluation/splits.py`

## Worked example — what each split does

Model row counts in the Track 1 universe (376 total):

```
architecture  paradigm  n_models
    ResNet18 confidnet         8
    ResNet18   devries         8
    ResNet18        dg        40
       VGG13 confidnet        40
       VGG13   devries        40
       VGG13        dg       200
         ViT  modelvit        40
```

### `xarch` (one fold)

Fold `vgg13_to_resnet18`: train = 280 model rows (VGG13 CNN), test = 56 (ResNet18 CNN). The predictor sees no ResNet18 rows during training; this is the cross-architecture transfer headline.

### `lopo` (4 folds, pooled VGG13 + ResNet18 + ViT)

```
    fold_label  n_train  n_test
lopo_confidnet      328      48
  lopo_devries      328      48
       lopo_dg      136     240
 lopo_modelvit      336      40
```

The `lopo_modelvit` fold trains on CNN paradigms only and tests on ViT — pure CNN→Transformer paradigm transfer. The other three folds test cross-paradigm generalization with both CNN architectures in the training pool.

### `lodo_vgg13` (4 folds)

```
        fold_label  n_train  n_test
      lodo_cifar10      220      60
     lodo_cifar100      210      70
lodo_supercifar100      190      90
 lodo_tinyimagenet      220      60
```

Each fold holds out one source dataset (with all VGG13 paradigms / configs / runs trained on it) — tests dataset transfer with architecture and paradigm pooled.

### `pxs_vgg13` (12 folds — 3 paradigms × 4 sources)

```
                 fold_label  n_train  n_test
      pxs_confidnet_cifar10      270      10
     pxs_confidnet_cifar100      270      10
pxs_confidnet_supercifar100      270      10
 pxs_confidnet_tinyimagenet      270      10
        pxs_devries_cifar10      270      10
       pxs_devries_cifar100      270      10
  pxs_devries_supercifar100      270      10
   pxs_devries_tinyimagenet      270      10
             pxs_dg_cifar10      240      40
            pxs_dg_cifar100      230      50
       pxs_dg_supercifar100      210      70
        pxs_dg_tinyimagenet      240      40
```

The smallest test fold is whichever (paradigm, source) cell has the fewest model rows — typically `confidnet` or `devries` cells (10 rows = 5 runs × 2 dropouts × 1 reward).

### `single_vgg13` (1 fold, diagnostic)

Fold `single_vgg13_resub`: train = test = 280 VGG13 CNN model rows. This is a resubstitution estimate — the upper bound on what the predictor can learn from VGG13 alone. Used only as a within-population ceiling for the cross-population evaluations.

### `track2_loo` (12 folds, Track 2 cells)

```
                       fold_label  n_train  n_test
      loo_VGG13_confidnet_cifar10       11       1
     loo_VGG13_confidnet_cifar100       11       1
loo_VGG13_confidnet_supercifar100       11       1
 loo_VGG13_confidnet_tinyimagenet       11       1
        loo_VGG13_devries_cifar10       11       1
       loo_VGG13_devries_cifar100       11       1
  loo_VGG13_devries_supercifar100       11       1
   loo_VGG13_devries_tinyimagenet       11       1
             loo_VGG13_dg_cifar10       11       1
            loo_VGG13_dg_cifar100       11       1
       loo_VGG13_dg_supercifar100       11       1
        loo_VGG13_dg_tinyimagenet       11       1
```

Track 2 has 12 VGG13 cells (3 paradigms × 4 sources). With only 8 NC features, LOO is the only viable evaluation — every fold trains on 11 cells and tests on 1. Reported as a sanity check, not a production predictor.

### Ablations

`xarch_vit_in`: 1 fold, train = 320 (VGG13 CNN + ViT), test = 56 (ResNet18 CNN). Same test set as primary xarch but with ViT in the training pool — measures whether ViT's NC diversity helps cross-arch transfer.

`lopo_cnn_only`: 3 folds (CNN paradigms only). Train pool excludes ViT, so this is a cleaner CNN-paradigm transfer without architecture-mediated NC differences confounding the test.

```
    fold_label  n_train  n_test
lopo_confidnet      288      48
  lopo_devries      288      48
       lopo_dg       96     240
```

## Run summary

```
        split  n_folds  n_rows_in_parquet                   leakage
        xarch        1                336                     clean
         lopo        4               1504                     clean
   lodo_vgg13        4               1120                     clean
    pxs_vgg13       12               3360                     clean
 single_vgg13        1                560 expected (resubstitution)
 xarch_vit_in        1                376                     clean
lopo_cnn_only        3               1008                     clean
   track2_loo       12                144                     clean
```

## Leakage check

For every fold of every split (excluding `single_vgg13` which is resubstitution by design), the intersection of `train` and `test` model/cell IDs must be empty. Any non-zero intersection above is a leak that would invalidate downstream evaluation.

