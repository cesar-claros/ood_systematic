# Step 12 — Per-CSF binary ablation

**Date:** 2026-05-03
**Source:** `code/nc_csf_predictivity/models/per_csf_binary.py`
**Model:** `LogisticRegressionCV(Cs=10, cv=5)` per CSF, fitted independently. Track 2 uses cv=3 due to small sample size.

## Worked example — per-CSF chosen C on `xarch/clique`

Each CSF's binary head selected its own C from 10 log-spaced values via internal 5-fold CV. Step 11's multilabel head used a single `C=1.0` for all CSFs.

```
                 csf   chosen_C
             NNGuide   0.000100
                  PE   0.000100
          Confidence   0.000774
                 MLS   0.000774
                NeCo   0.000774
                  GE   0.005995
                 GEN   0.005995
            GradNorm   0.005995
            Residual   0.005995
                 REN   0.005995
                 CTM   0.046416
KPCA RecError global   0.046416
                 PCE   0.046416
              Energy   0.359381
                 ViM   0.359381
 PCA RecError global   0.359381
                pNML   2.782559
                fDBD   2.782559
                 MSR  21.544347
                Maha 166.810054
```

Range: chosen C ∈ [0.0001, 166.81]. CSFs that picked C far from 1.0 are the ones whose 'competitive' decision boundary differs most from the multilabel head's fixed regularization.

## Worked example — predicted competitive set on the same xarch row

Same row as step 10/11 (`ResNet18|confidnet|cifar10|1|0|2.2`, regime=near). Side-by-side: per-CSF binary (this step) vs multilabel binary (step 11).

```
                 csf  p_competitive_per_csf  predicted_competitive_per_csf  p_competitive_multi  predicted_competitive_multi  agree
                 CTM                  0.476                          False                0.259                        False   True
          Confidence                  0.223                          False                0.664                         True  False
             NNGuide                  0.163                          False                0.478                        False   True
                NeCo                  0.162                          False                0.206                        False   True
                  PE                  0.144                          False                0.488                        False   True
                 PCE                  0.142                          False                0.385                        False   True
                 MLS                  0.134                          False                0.170                        False   True
KPCA RecError global                  0.115                          False                0.140                        False   True
                  GE                  0.105                          False                0.028                        False   True
                 REN                  0.095                          False                0.095                        False   True
```

Per-row agreement on `predicted_competitive`: 9/10 CSFs in this top-10 view.

## Run summary

```
track      split           label_rule  n_folds  n_csf_fits  n_preds  median_chosen_C
    1      xarch               clique        1          20     4480         0.026205
    1      xarch       within_eps_raw        1          20     4480         0.046416
    1      xarch      within_eps_rank        1          20     4480         0.046416
    1      xarch  within_eps_majority        1          20     4480         0.046416
    1      xarch within_eps_unanimous        1          18     4032         0.026205
    1       lopo               clique        4          76    27008         0.046416
    1       lopo       within_eps_raw        4          77    27200         0.046416
    1       lopo      within_eps_rank        4          77    27200         0.046416
    1       lopo  within_eps_majority        4          77    27200         0.046416
    1       lopo within_eps_unanimous        4          67    22272         0.046416
    1 lodo_vgg13               clique        4          76    20960         0.046416
    1 lodo_vgg13       within_eps_raw        4          76    20960         0.046416
    1 lodo_vgg13      within_eps_rank        4          76    20960         0.046416
    1 lodo_vgg13  within_eps_majority        4          77    21320         0.046416
    1 lodo_vgg13 within_eps_unanimous        4          67    18480         0.046416
    2 track2_loo               clique       12         119      476         0.046416
```

McNemar comparison of per-CSF vs multilabel predictions on the same test rows lives in step 15.
