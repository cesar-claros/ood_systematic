# Data-unit and fDBD regression check (R6; deterministic)

Source table: `code/nc_csf_predictivity/outputs/track1/dataset/long_harmonized.parquet`; total rows 67320.

## Architectures (row counts)

- VGG13: 50400
- ResNet18: 10080
- ViT: 6840

## VGG-13 observation units

- unique trained checkpoints (paradigm|source|run|reward|dropout): **280**
- checkpoints per source: {'cifar10': 60, 'cifar100': 70, 'supercifar100': 90, 'tinyimagenet': 60}
- OOD shifts per checkpoint: min 8, max 8
- distinct OOD set names pool-wide: 9
- unique checkpoint-shift observations: **2240**
- OOD long-format rows over all 20 CSFs: **44800**

## Crossing-audit unit resolution

The crossing audit's report header `Cells: 280` counts unique checkpoints (bootstrap clusters); its estimator operates on **2240 (checkpoint, OOD set) rows** with both Energy and CTM present. Manuscript wording must state checkpoints and checkpoint-shift observations separately and must not call either number 'checkpoint-shift cells'.

## fDBD sourcewise comparisons (mean AUGRC over OOD rows; lower is better)

| source | fDBD | MLS | CTM | fDBD-MLS | fDBD-CTM | fDBD beats MLS | fDBD beats CTM |
|---|---|---|---|---|---|---|---|
| cifar10 | 167.97 | 164.37 | 170.78 | +3.60 | -2.80 | False | True |
| cifar100 | 233.80 | 224.07 | 221.75 | +9.73 | +12.05 | False | False |
| supercifar100 | 282.58 | 273.73 | 271.38 | +8.85 | +11.20 | False | False |
| tinyimagenet | 228.20 | 230.64 | 224.90 | -2.44 | +3.30 | True | False |

- fDBD beats MLS on: ['tinyimagenet']
- fDBD beats CTM on: ['cifar10']

## R5 support: prevalence columns

Columns suggesting per-population sample sizes or prevalence: NONE. If none, the fixed-prevalence premise of the AUGRC crossing-invariance corollary cannot be verified from this table and must be handled at the protocol level (OOD sets differ in size, so prevalence varies across shifts; empirical crossings are computed directly in AUGRC and do not invoke the corollary).
