# Step 10 — Regression head

**Date:** 2026-05-03
**Source:** `code/nc_csf_predictivity/models/regression.py`
**Model:** `RidgeCV(alphas=10^[-3..3])`, internal LOOCV α selection.
**Features:** 8 NC (var_collapse, equiangular_uc, equiangular_wc, equinorm_uc, equinorm_wc, max_equiangular_uc, max_equiangular_wc, self_duality) + one-hot of ['csf', 'source', 'regime'].
**Target:** `augrc_rank` (rank is primary per step-4 verdict).
**Scope:** OOD only (regime ∈ {near, mid, far}); test regime excluded.

## Worked example — one prediction row from `xarch`

From the cross-arch test set, model row `ResNet18|confidnet|cifar10|1|0|2.2` evaluated on `cifar100` (regime = `near`). The Ridge predicts a score per CSF (lower = better, since target is rank ∈ [0, 1]):

```
                 csf  predicted_score  true_score  raw_augrc
                 CTM            0.423       0.206    165.209
          Confidence            0.477       0.407    168.587
              Energy            0.483       0.626    171.882
                  GE            0.488       0.633    171.997
                 GEN            0.495       0.490    169.892
            GradNorm            0.696       0.993    239.561
KPCA RecError global            0.500       0.200    165.125
                 MLS            0.462       0.625    171.847
```

Top-1 predicted CSF: **CTM** (predicted_score = 0.423, raw AUGRC = 165.21).
Top-1 true (oracle) CSF: **KPCA RecError global** (true_score = 0.200, raw AUGRC = 165.12).
Top-1 regret on this eval row = 0.08 raw AUGRC (computed downstream in step 13).

## Worked example — feature importance computation

Permutation importance shuffles each input column on the test set and measures the increase in MSE (negative R²). Each NC and categorical feature is permuted `n_repeats=5` times with `random_state=0`. The importance is the mean MSE increase; std is across repeats. Higher = more important.

Toy: with 3 features (NC_1, NC_2, csf) and a Ridge fitted on 100 rows, suppose permuting NC_1 increases MSE from 0.20 to 0.27 (Δ = +0.07) and permuting csf increases it from 0.20 to 0.50 (Δ = +0.30). Then `csf` is much more important than `NC_1` for this model. Categoricals usually dominate when the OneHot encoding has many levels (here csf has 19 of them after dropping the first), so the *relative* ordering of NC features is the more interesting readout.

### Feature importance on `xarch` (ResNet18 test set)

NC features (sorted by importance):

```
           feature  importance_mean  importance_std
max_equiangular_uc           0.0662          0.0008
    equiangular_uc           0.0407          0.0010
       equinorm_uc           0.0258          0.0005
max_equiangular_wc           0.0139          0.0006
    equiangular_wc           0.0130          0.0002
      self_duality           0.0009          0.0000
       equinorm_wc           0.0002          0.0000
      var_collapse           0.0000          0.0000
```

Categorical features:

```
feature  importance_mean  importance_std
    csf           0.0334          0.0007
 source           0.0042          0.0003
 regime          -0.0000          0.0000
```

## Run summary

```
track         split  n_folds  n_test_rows  mean_train_r2
    1         xarch        1         8960       0.461294
    1          lopo        4        59840       0.504937
    1    lodo_vgg13        4        44800       0.483468
    1     pxs_vgg13       12        44800       0.466441
    1  single_vgg13        1        44800       0.461294
    1  xarch_vit_in        1         8960       0.528898
    1 lopo_cnn_only        3        53760       0.430762
    2    track2_loo       12         1920       0.328214
```

`mean_train_r2` is reported for sanity only — it does not measure out-of-fold transfer. Test-fold metrics (top-1 / set / top-k regret, ranking, baselines) are computed in step 13.
