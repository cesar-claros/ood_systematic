# Step 7 — Oracle and regret tables

**Date:** 2026-05-02
**Source:** `code/nc_csf_predictivity/data/oracle_regret.py`

## Worked example — oracle, worst, per-side, regret variants

Suppose one row at `(architecture=VGG13, paradigm=confidnet, source=cifar10, run=1, dropout=False, reward=2.2, eval_dataset=tinyimagenet)` contains 6 CSFs (3 head-side + 3 feature-side; toy size for illustration). Their AUGRC values:

```
   csf    side  augrc
   MSR    head    100
Energy    head    120
    PE    head    180
  NeCo feature    110
  fDBD feature    130
   CTM feature    200
```

### Sorted by AUGRC ascending (oracle first):

```
   csf    side  augrc
   MSR    head    100
  NeCo feature    110
Energy    head    120
  fDBD feature    130
    PE    head    180
   CTM feature    200
```

- **oracle_csf** = `MSR` (augrc = 100)
- **worst_csf** = `CTM` (augrc = 200)
- **csf_ranking** = `['MSR', 'NeCo', 'Energy', 'fDBD', 'PE', 'CTM']`
- **augrc_ranking** = `[100, 110, 120, 130, 180, 200]`

### Per-side restrictions (the §10e analysis pool):

- Head-side pool sorted: `['MSR', 'Energy', 'PE']` → `oracle_csf_head` = `MSR` (augrc = 100), `worst_csf_head` = `PE` (augrc = 180)
- Feature-side pool sorted: `['NeCo', 'fDBD', 'CTM']` → `oracle_csf_feature` = `NeCo` (augrc = 110), `worst_csf_feature` = `CTM` (augrc = 200)

### Regret variants computed downstream from this row

Suppose the predictor's score-based ranking on this row is `[NeCo, MSR, fDBD, Energy, PE, CTM]` (so its top-1 pick is NeCo).

- **Top-1 regret** = `augrc(NeCo) − augrc(oracle)` = 110 − 100 = **10**.
- **Normalized top-1 regret** = `10 / (worst − oracle)` = `10 / (200 − 100)` = **0.10**.
- **Top-3 regret** with predicted top-3 = `[NeCo, MSR, fDBD]`: `min(110, 100, 130) − 100` = **0** (the oracle landed in the predicted top-3 even though top-1 was wrong).
- **Set regret** for predicted competitive set `{NeCo, Energy}`: `min(110, 120) − 100` = **10**.

On the head-side restricted pool, the oracle is MSR (100), worst is PE (180). If the predictor's head-side top-1 is Energy, head-side regret = 120 − 100 = 20 (normalized 20/80 = 0.25). On the feature-side pool, oracle is NeCo (110); if predictor's feature-side top-1 is fDBD, feature-side regret = 130 − 110 = 20 (normalized 20/90 ≈ 0.22).

All regret variants are computable from the columns this step writes (`oracle_*`, `worst_*`, per-side variants, and `csf_ranking` / `augrc_ranking`); the predictor's per-row score ordering is the only extra input needed at metric time (step 13).

## Track 1

- Output: `outputs/track1/dataset/oracle.parquet`
- Rows (one per regret-evaluable row): 3,384
- Per-row CSF inventory size — distribution:

```
count    3384.00
mean       19.89
std         0.31
min        19.00
25%        20.00
50%        20.00
75%        20.00
max        20.00
```

### Oracle CSF frequency (top-10) per regime

**near** (n = 998):

```
oracle_csf
CTM                     358
Confidence              156
KPCA RecError global     71
Energy                   61
REN                      60
NNGuide                  54
GradNorm                 42
ViM                      33
PCE                      29
MLS                      29
```

**mid** (n = 1,340):

```
oracle_csf
CTM                     315
Confidence              222
Energy                  185
KPCA RecError global    132
NNGuide                 122
ViM                      79
Maha                     57
pNML                     45
fDBD                     40
GradNorm                 32
```

**far** (n = 670):

```
oracle_csf
CTM                     176
fDBD                     85
KPCA RecError global     64
Confidence               62
Energy                   49
NNGuide                  44
ViM                      39
Residual                 36
pNML                     24
REN                      23
```

**test** (n = 376):

```
oracle_csf
MSR                     104
GEN                      79
CTM                      72
Confidence               43
REN                      25
PCE                      17
fDBD                     15
KPCA RecError global      9
MLS                       7
GE                        2
```

### Oracle CSF frequency by side per regime

**near** — share of rows where oracle is head vs feature:

```
oracle_side
feature    0.567
head       0.433
```

**mid** — share of rows where oracle is head vs feature:

```
oracle_side
feature    0.597
head       0.403
```

**far** — share of rows where oracle is head vs feature:

```
oracle_side
feature    0.712
head       0.288
```

**test** — share of rows where oracle is head vs feature:

```
oracle_side
head       0.739
feature    0.261
```

### Per-row dynamic range (worst − oracle) — distribution

```
count    3384.00
mean       71.58
std        48.85
min         1.47
25%        32.95
50%        59.65
75%        99.86
max       237.74
```

This is the denominator of normalized regret. Cells with very small dynamic range (worst ≈ oracle) make normalized regret unstable — downstream metric code should drop or downweight those rows.

## Track 2

- Output: `outputs/track2/dataset/oracle.parquet`
- Rows: 144

### Oracle CSF frequency (top-10) per regime

**near** (n = 44):

```
oracle_csf
CTM                    18
Energy                  7
GradNorm                4
PCE                     3
NNGuide                 2
Confidence              2
NeCo                    2
MLS                     1
PCA RecError global     1
pNML                    1
```

**mid** (n = 56):

```
oracle_csf
CTM                     12
Energy                  11
KPCA RecError global     9
pNML                     5
NNGuide                  3
ViM                      3
fDBD                     3
GradNorm                 3
Confidence               3
Residual                 2
```

**far** (n = 28):

```
oracle_csf
fDBD                    6
CTM                     5
Residual                4
Maha                    3
Energy                  2
NNGuide                 2
ViM                     2
KPCA RecError global    1
REN                     1
pNML                    1
```

**test** (n = 16):

```
oracle_csf
GEN           7
CTM           4
fDBD          2
Confidence    1
REN           1
MSR           1
```

