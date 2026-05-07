# Step 1 — Schema audit

**Date:** 2026-05-02
**Protocol section:** §4, §5
**Status:** PASS with translation rules and four flagged caveats below.

## Summary

All five data sources (NC metrics, AUGRC, AURC, CLIP groupings, cliques JSONs)
are present and joinable for Tracks 1 and 2. The (NC ↔ score-file) join works
1-to-1 once four naming/format mismatches are normalized. The §5 base-CSF
filter leaves a clean 20-CSF inventory (11 head-side + 9 feature-side + 0
unclassified), matching the partition in `code/stats_eval.py:55–62`. The
head-logit metrics fallback file is present and has the same row schema as
NC (380 rows × 60 cols).

| Check | Result |
|---|---|
| NC metrics file present | ✓ 380 rows, 20 cols |
| NC covers all 3 architectures | ✓ VGG13 (280), ResNet18 (60), ViT (40) |
| NC covers all 4 sources | ✓ cifar10, cifar100, supercifar, tinyimagenet |
| AUGRC `_fix-config` files (all 4 sources × 3 backbones) | ✓ |
| AURC `_fix-config` files (all 4 sources × 3 backbones) | ✓ |
| CLIP groupings present | ✓ all 4 sources |
| Per-paradigm clique JSONs (Track 2 labels) | ✓ confidnet, devries, dg, modelvit + combined Conv |
| Surviving CSF inventory after §5 filter | ✓ 20 CSFs (11 head, 9 feature, 0 unclassified) |
| Head-logit metrics fallback | ✓ `head_logit_metrics/hl_metrics.csv`, 380×60 |

## NC metrics (`neural_collapse_metrics/nc_metrics.csv`)

- **Shape:** 380 rows × 20 columns.
- **Identifier columns:** `dataset, architecture, study, dropout, run, reward, lr`.
- **NC columns:** `var_collapse, cdnv_score, bias_collapse, equiangular_uc,
  equiangular_wc, equinorm_uc, equinorm_wc, max_equiangular_uc,
  max_equiangular_wc, self_duality, w_etf_diff, M_etf_diff, wM_etf_diff` (13
  metrics; the protocol's primary 8 are a strict subset).
- **Architectures:** `ResNet18, VGG13, ViT`.
- **Studies:** `confidnet, devries, dg, vit`.
- **Datasets:** `cifar10, cifar100, supercifar, tinyimagenet`.
- **Dropout:** `{False, True}` (numpy bool).
- **Runs:** CNN paradigms 1-indexed (1–5 for VGG13, 1 for ResNet18); ViT
  0-indexed (0–4).
- **Rewards:** 2.2 / 3.0 / 6.0 / 10.0 / 12.0 / 15.0 / 20.0 (DG sweeps); 2.2
  for confidnet/devries; 0.0 for ViT (placeholder, not used).
- **LR:** 0.0 for CNN paradigms; ViT uses 0.0003 / 0.001 / 0.003 / 0.01
  depending on (source, dropout) — see §"ViT lr-dropout coupling" below.

### Per-cell config grid

| arch | study | source | runs | rewards | dropouts | rows |
|---|---|---|---|---|---|---|
| ResNet18 | confidnet | each of 4 | {1} | {2.2} | {F, T} | 2 |
| ResNet18 | devries   | each of 4 | {1} | {2.2} | {F, T} | 2 |
| ResNet18 | dg | cifar10 | {1} | {2.2, 3, 6, 10} | {F, T} | 8 |
| ResNet18 | dg | cifar100 | {1} | {2.2, 3, 6, 10, 12, 15, 20} | {F, T} | 14 |
| ResNet18 | dg | supercifar | {1} | {2.2, 3, 6, 10, 12, 15, 20} | {F, T} | 14 |
| ResNet18 | dg | tinyimagenet | {1} | {10, 12, 15, 20} | {F, T} | 8 |
| VGG13 | confidnet / devries | each of 4 | {1..5} | {2.2} | {F, T} | 10 |
| VGG13 | dg | cifar10 | {1..5} | {2.2, 3, 6, 10} | {F, T} | 40 |
| VGG13 | dg | cifar100 | {1..5} | **{6, 10, 12, 15, 20}** | {F, T} | 50 |
| VGG13 | dg | supercifar | {1..5} | {2.2, 3, 6, 10, 12, 15, 20} | {F, T} | 70 |
| VGG13 | dg | tinyimagenet | {1..5} | {10, 12, 15, 20} | {F, T} | 40 |
| ViT | vit | each of 4 | {0..4} | {0.0} | {F, T} | 10 |

## Score files

### `_fix-config` AUGRC and AURC coverage

- VGG13/Conv and ViT: present in `code/scores_risk/` for both AUGRC and AURC,
  for all 4 sources, with both `MCD-True` and `MCD-False`. We use only
  `MCD-False` per §5.
- ResNet18/Conv: present in `code/scores_risk_resnet18/` for both AUGRC and
  AURC, for all 4 sources. ViT files are **not** present for ResNet18 (this
  matches the protocol's CNN-only cross-arch test).

### Schema (e.g., `scores_all_AUGRC_MCD-False_Conv_cifar10_fix-config.csv`)

- Columns: `model, drop out, methods, reward, run, test, cifar100,
  tinyimagenet, lsun resize, isun, places365, lsun cropped, textures, svhn`.
- 9 evaluation columns per row (`test` + 8 OOD).
- 83 unique values in `methods` before §5 filter (all CSF variants including
  `global` / `class` / CTMmean).
- Sample size for VGG13/Conv cifar10: 60 (model, drop out, reward, run)
  cells × 83 methods = 4980 rows.
- Sample size for ResNet18/Conv cifar10: 12 cells × 83 methods = 996 rows.
- Sample size for ViT cifar10: 10 cells × 83 methods = 830 rows.

## Naming and format mismatches (apply at join time)

| Field | NC format | Score-file format | Mapping |
|---|---|---|---|
| paradigm | `study='vit'` | `model='modelvit'` | rename `vit → modelvit` for join |
| source dataset | `dataset='supercifar'` | source token = `supercifar100` | rename `supercifar → supercifar100` |
| reward | numeric `2.2, 3.0, …` | string `'rew2.2', 'rew3', …` | `float(s.replace('rew',''))` |
| dropout | bool `True / False` | string `'do0' / 'do1'` | `'do0' ↔ False, 'do1' ↔ True` (direction assumed; see Caveat 1) |

### Smoke-test of the join

For VGG13/confidnet/cifar10: 10 NC rows ↔ 10 unique (model, drop out, reward,
run) score triples. 1-to-1. Identical run set {1, 2, 3, 4, 5} and identical
(reward = 2.2). Direction of dropout mapping is consistent but not yet
verified to be the one written above.

## CSF inventory after §5 filter

83 raw → 81 (CTMmean blacklist) → 20 (global|class regex with PCA/KPCA
exceptions) → 20 (drop MCD- prefix; no MCD- methods survive after the
prior step in the MCD-False files).

### Surviving 20 CSFs

- **Head-side (11):** `Confidence, Energy, GE, GEN, GradNorm, MLS, MSR, PCE,
  PE, pNML, REN`.
- **Feature-side (9):** `CTM, fDBD, KPCA RecError global, Maha, NeCo,
  NNGuide, PCA RecError global, Residual, ViM`.
- **Unclassified:** none.

This matches `code/stats_eval.py:55–62` exactly.

## CLIP groupings (per source)

Read with `header=[0,1]`, drop level 0, rename `Unnamed: 5_level_1 → group`.
Group codes: 0 = test (in-distribution), 1 = near, 2 = mid, 3 = far.

| source | near | mid | far |
|---|---|---|---|
| cifar10 | cifar100, tinyimagenet | isun, lsun resize, lsun cropped, svhn | places365, textures |
| cifar100 | cifar10, tinyimagenet | isun, lsun resize, lsun cropped, svhn | places365, textures |
| supercifar100 | cifar10, tinyimagenet | isun, lsun resize, lsun cropped, svhn | places365, textures |
| tinyimagenet | cifar100, cifar10, isun, lsun resize, lsun cropped | places365, textures | svhn |

The tinyimagenet partition is markedly different — most evaluations land in
`near` (5 datasets), only 2 in `mid`, only `svhn` in `far`. This will affect
per-regime statistical power for tinyimagenet-trained models and should be
visible in any per-regime table.

## Existing cliques JSONs (Track 2 labels)

All under `code/ood_eval_outputs/` with stem `top_cliques_<backbone>_False_RC[_<paradigm>]_cliques.json`:

- Combined Conv: `top_cliques_Conv_False_RC_cliques.json`
- Per CNN paradigm: `top_cliques_Conv_False_RC_{confidnet,devries,dg}_cliques.json`
- ViT: `top_cliques_ViT_False_RC_cliques.json`

Each covers all 4 sources × 5 regimes (test, near, mid, far, all). Track 2
will use the per-paradigm files; the combined Conv file is reserved for the
ablation that pools paradigms.

## Sample sizes

### Track 1 (one row per `(arch, study, source, run, dropout, reward, lr)`)

| Pool | Model rows | × 20 CSFs × 9 eval ds = eval rows |
|---|---|---|
| VGG13 (CNN, train for cross-arch) | 280 | 50 400 |
| ResNet18 (CNN, test for cross-arch) | 60 | 10 800 |
| ViT | 40 | 7 200 |
| **LOPO pool VGG13 + ResNet18 only** | 340 | 61 200 |
| **LOPO pool incl. ViT** | 380 | 68 400 |

### Track 2 (one row per `(arch, study, source)`)

| Pool | Cells |
|---|---|
| VGG13 CNN (train) | 12 (3 paradigms × 4 sources) |
| ResNet18 CNN (test) | 12 |
| ViT | 4 |

LOO CV on 12 train cells is the only viable evaluation for Track 2 with 8 NC
features.

## Caveats and follow-ups for Step 2

1. **Dropout direction mapping (`do0/do1` ↔ `False/True`).** The 1-to-1
   join works regardless of direction since the cell counts match. The
   direction needs explicit verification by reading `set_name_dict` in
   `code/src/scores_retrieval_utils.py` (or by joining a known-distinct
   metric and checking which dropout pairing reproduces published numbers).
   Step 2 will read that file and pin the direction in
   `data/build_dataset.py`.

2. **DG cifar100 reward grid mismatch between VGG13 and ResNet18.** VGG13 DG
   on cifar100 uses rewards {6, 10, 12, 15, 20}; ResNet18 DG on cifar100
   uses {2.2, 3, 6, 10, 12, 15, 20}. The two ResNet18-only rewards (2.2 and
   3) appear in the test set but never in training. The predictor must
   generalize across reward; our hypothesis is precisely that NC summarizes
   what reward-tuning changed about the model, so this is a feature not a
   bug. Document the mismatch in §13 of the protocol; quantify the
   ResNet18 cifar100 OOD-test row count contributed by the unseen rewards
   when Track 1 lands.

3. **ViT lr-dropout coupling.** For ViT the lr is not a free axis: cifar10
   has `(lr=0.0003, do=False)` and `(lr=0.01, do=True)` paired; supercifar
   has `(lr=0.001, do=True)` and `(lr=0.003, do=False)`; cifar100 and
   tinyimagenet have lr=0.01 with both dropouts. So a ViT model
   configuration is keyed by `(lr, dropout, run)`, not `(dropout, run)`.
   The ablation in §9a-ablation that adds ViT to the training set must
   include lr in the cell key so that NC features for the two dropout
   variants are not blended.

4. **Run indexing differs across architectures.** VGG13 and ResNet18 use
   1-indexed runs {1..5} and {1}; ViT uses 0-indexed runs {0..4}. The join
   key uses `run` directly so this is harmless, but any code that infers
   "first run" from the index value should not assume run=0 or run=1
   uniformly. Step 2 will treat `run` as an opaque categorical label.

## Files inspected

- `code/neural_collapse_metrics/nc_metrics.csv`
- `code/scores_risk/scores_all_AUGRC_MCD-False_Conv_cifar10_fix-config.csv`
- `code/scores_risk/scores_all_AUGRC_MCD-False_ViT_cifar10_fix-config.csv`
- `code/scores_risk_resnet18/scores_all_AUGRC_MCD-False_Conv_cifar10_fix-config.csv`
- `code/clip_scores/clip_distances_{cifar10,cifar100,supercifar100,tinyimagenet}.csv`
- `code/ood_eval_outputs/top_cliques_*_cliques.json` (5 files)
- `code/head_logit_metrics/hl_metrics.csv`

Directory listings of `code/scores_risk/` and `code/scores_risk_resnet18/`
confirmed all expected `_fix-config` files for AUGRC and AURC.
