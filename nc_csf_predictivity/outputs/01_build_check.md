# Step 2/3 — Build check

**Date:** 2026-05-02
**Source:** `code/nc_csf_predictivity/data/build_dataset.py`

## Track 1 (`outputs/track1/dataset/long.parquet`)

- Total eval rows: 67,320
- Unique model configurations: 376
- Unique CSFs after §5 filter: 20
- Unique eval_datasets: 10

### Rows per (architecture, regime)

```
regime          far    mid   near  test
architecture                           
ResNet18       2000   4000   2960  1120
VGG13         10000  20000  14800  5600
ViT            1330   2660   2090   760
```

### Model rows per (architecture, paradigm, source)

```
architecture  paradigm   source       
ResNet18      confidnet  cifar10           2
                         cifar100          2
                         supercifar100     2
                         tinyimagenet      2
              devries    cifar10           2
                         cifar100          2
                         supercifar100     2
                         tinyimagenet      2
              dg         cifar10           8
                         cifar100         10
                         supercifar100    14
                         tinyimagenet      8
VGG13         confidnet  cifar10          10
                         cifar100         10
                         supercifar100    10
                         tinyimagenet     10
              devries    cifar10          10
                         cifar100         10
                         supercifar100    10
                         tinyimagenet     10
              dg         cifar10          40
                         cifar100         50
                         supercifar100    70
                         tinyimagenet     40
ViT           modelvit   cifar10          10
                         cifar100         10
                         supercifar100    10
                         tinyimagenet     10
```

### Side breakdown

```
side
head       36864
feature    30456
```

### NaN counts in primary 8 NC features (must be 0)

```
var_collapse          0
equiangular_uc        0
equiangular_wc        0
equinorm_uc           0
equinorm_wc           0
max_equiangular_uc    0
max_equiangular_wc    0
self_duality          0
```

### NaN counts in AUGRC and AURC (must be 0)

```
augrc    0
aurc     0
```

### NC rows with no score match (orphans, dropped from long table)

4 NC model rows without an AUGRC entry. These are excluded from the long table because the predictor cannot be evaluated on cells without an oracle CSF.

```
architecture paradigm   source  run  dropout  reward
    ResNet18       dg cifar100    1    False     2.2
    ResNet18       dg cifar100    1    False     3.0
    ResNet18       dg cifar100    1     True     2.2
    ResNet18       dg cifar100    1     True     3.0
```

**Implication for protocol §13 caveat 2:** the audit noted ResNet18 dg cifar100 as having NC rewards {2.2, 3} that VGG13 lacks. After joining with the actual ResNet18 AUGRC file, those NC rows are orphans (no score data exists for them). The reward grid in the joined long table therefore aligns between VGG13 and ResNet18 for cifar100 dg ({6, 10, 12, 15, 20}), and the original cross-arch concern is moot.

## Track 2 (`outputs/track2/dataset/long.parquet`)

- Total eval rows: 2,844
- Unique (architecture, paradigm, source) cells: 16
- Unique CSFs: 20

### Rows per (architecture, regime)

```
regime        far  mid  near  test
architecture                      
VGG13         420  840   660   240
ViT           133  266   209    76
```

### NaN counts in nc_mean primary features (must be 0)

```
nc_mean_var_collapse          0
nc_mean_equiangular_uc        0
nc_mean_equiangular_wc        0
nc_mean_equinorm_uc           0
nc_mean_equinorm_wc           0
nc_mean_max_equiangular_uc    0
nc_mean_max_equiangular_wc    0
nc_mean_self_duality          0
```

### Caveat

Track 2 contains **VGG13 and ViT only**. ResNet18 has no
non-`_fix-config` aggregated AUGRC file because
`code/retrieve_scores.py` has not been run for ResNet18 with the
per-CSF best-config selection. Track 2 evaluation is therefore
restricted to LOO CV within VGG13 and the ViT ablation; the
cross-architecture transfer evaluation lives only in Track 1.
