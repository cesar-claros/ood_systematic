# Step 11 — Multi-output binary 'is competitive' head

**Date:** 2026-05-03
**Source:** `code/nc_csf_predictivity/models/multilabel_competitive.py`
**Model:** `MultiOutputClassifier(LogisticRegression(C=1.0, lbfgs))`. Fixed C — internal CV across 20 binary heads × every (split, fold, rule) is prohibitively slow; the per-CSF ablation in step 12 will sweep C per CSF.
**Features:** 8 NC + one-hot of (source, regime). `csf` is the OUTPUT dimension.

## Worked example — predicted competitive set on one xarch row

From the cross-arch test set, model row `ResNet18|confidnet|cifar10|1|0|2.2` at regime = `near`. Each of the 20 binary heads outputs a probability that the corresponding CSF is competitive on this row. Predicted competitive set = `{p ≥ 0.5}`.

```
                 csf  p_competitive  predicted_competitive
          Confidence          0.664                   True
                  PE          0.488                  False
             NNGuide          0.478                  False
                 PCE          0.385                  False
                 CTM          0.259                  False
                NeCo          0.206                  False
                 MLS          0.170                  False
KPCA RecError global          0.140                  False
                 REN          0.095                  False
                 MSR          0.054                  False
```

Predicted competitive set on this row (`label_rule=clique`): `['Confidence']` (size 1).

Set-regret on this row will be `min(raw_augrc) over the predicted set − oracle raw_augrc`, computed in step 13.

## Worked example — per-CSF logistic coefficients (NC features only)

Mean |coefficient| across CSFs, per NC feature (xarch fold, label_rule=clique):

```
feature
equiangular_wc        1.5380
max_equiangular_wc    1.1753
equinorm_wc           0.9012
equinorm_uc           0.8831
self_duality          0.7928
equiangular_uc        0.7720
max_equiangular_uc    0.7664
var_collapse          0.3096
```

Per-NC coefficients for the **CTM** binary head:

```
           feature  coefficient
    equiangular_uc       -1.662
       equinorm_wc       -1.579
    equiangular_wc        0.820
max_equiangular_uc        0.512
max_equiangular_wc       -0.482
      var_collapse       -0.242
      self_duality       -0.169
       equinorm_uc       -0.033
```

Positive coefficient ⇒ higher value of that NC metric increases the predicted probability that this CSF is competitive. Magnitude is on the standardized-feature scale.

## Run summary

```
track         split           label_rule  n_folds  n_preds  n_csfs_avg
    1         xarch               clique        1     4480   20.000000
    1         xarch       within_eps_raw        1     4480   20.000000
    1         xarch      within_eps_rank        1     4480   20.000000
    1         xarch  within_eps_majority        1     4480   20.000000
    1         xarch within_eps_unanimous        1     4032   18.000000
    1          lopo               clique        4    27008   19.000000
    1          lopo       within_eps_raw        4    27200   19.250000
    1          lopo      within_eps_rank        4    27200   19.250000
    1          lopo  within_eps_majority        4    27200   19.250000
    1          lopo within_eps_unanimous        4    22272   16.750000
    1    lodo_vgg13               clique        4    20960   19.000000
    1    lodo_vgg13       within_eps_raw        4    20960   19.000000
    1    lodo_vgg13      within_eps_rank        4    20960   19.000000
    1    lodo_vgg13  within_eps_majority        4    21320   19.250000
    1    lodo_vgg13 within_eps_unanimous        4    18480   16.750000
    1     pxs_vgg13               clique       12    21800   19.750000
    1     pxs_vgg13       within_eps_raw       12    21840   19.833333
    1     pxs_vgg13      within_eps_rank       12    21840   19.833333
    1     pxs_vgg13  within_eps_majority       12    21840   19.833333
    1     pxs_vgg13 within_eps_unanimous       12    19040   17.666667
    1  single_vgg13               clique        1    22400   20.000000
    1  single_vgg13       within_eps_raw        1    22400   20.000000
    1  single_vgg13      within_eps_rank        1    22400   20.000000
    1  single_vgg13  within_eps_majority        1    22400   20.000000
    1  single_vgg13 within_eps_unanimous        1    20160   18.000000
    1 lopo_cnn_only               clique        3    23808   18.666667
    1 lopo_cnn_only       within_eps_raw        3    24000   19.000000
    1 lopo_cnn_only      within_eps_rank        3    24000   19.000000
    1 lopo_cnn_only  within_eps_majority        3    24000   19.000000
    1 lopo_cnn_only within_eps_unanimous        3    19392   16.333333
    2    track2_loo               clique       12      888   18.500000
```

`n_csfs_avg` is the average number of binary heads actually trained per fold (some CSFs may be constant=0 in a given fold's training labels, in which case their head is skipped — e.g., a CSF that is never in any cliquer/within-ε set across the training cells of that fold).

Predictor outputs `predicted_competitive` (bool, threshold 0.5 on `p_competitive`). Set-regret and per-side metrics will be computed in step 13 by joining these predictions with the oracle table from step 7.
