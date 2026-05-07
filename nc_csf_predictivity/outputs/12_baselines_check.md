# Step 14 — Baseline comparisons + empty-set imputation

**Date:** 2026-05-04
**Source:** `code/nc_csf_predictivity/evaluation/baselines.py`
**Bootstrap:** n=2000, seed=0.

## Worked example — per-row baselines on one xarch row

Same row as steps 10–13: `ResNet18|confidnet|cifar10|1|0|2.2`, eval=`cifar100`, regime=`near`. Showing all comparators (baselines, raw predictors, and imputed binary predictors):

```
  comparator_kind                     comparator_name    side  regret_raw  regret_norm
         baseline                          Always-CTM     all       0.084        0.001
         baseline                       Always-Energy     all       6.757        0.091
         baseline                          Always-MLS     all       6.722        0.090
         baseline                          Always-MSR     all       5.138        0.069
         baseline                      Always-NNGuide     all       5.412        0.073
         baseline                         Always-fDBD     all       7.772        0.104
         baseline               Oracle-on-train (CTM)     all       0.084        0.001
         baseline                          Random-CSF     all       8.644        0.116
predictor_imputed                   multilabel/clique     all       3.462        0.047
predictor_imputed      multilabel/within_eps_majority     all       3.462        0.047
predictor_imputed          multilabel/within_eps_rank     all       3.462        0.047
predictor_imputed           multilabel/within_eps_raw     all       3.462        0.047
predictor_imputed     multilabel/within_eps_unanimous     all       3.462        0.047
predictor_imputed               per_csf_binary/clique     all      74.437        1.000
predictor_imputed  per_csf_binary/within_eps_majority     all       0.084        0.001
predictor_imputed      per_csf_binary/within_eps_rank     all       0.084        0.001
predictor_imputed       per_csf_binary/within_eps_raw     all       0.084        0.001
predictor_imputed per_csf_binary/within_eps_unanimous     all      74.437        1.000
    predictor_raw                   multilabel/clique     all       3.462        0.047
    predictor_raw      multilabel/within_eps_majority     all       3.462        0.047
    predictor_raw          multilabel/within_eps_rank     all       3.462        0.047
    predictor_raw           multilabel/within_eps_raw     all       3.462        0.047
    predictor_raw     multilabel/within_eps_unanimous     all       3.462        0.047
    predictor_raw               per_csf_binary/clique     all         NaN          NaN
    predictor_raw  per_csf_binary/within_eps_majority     all       0.084        0.001
    predictor_raw      per_csf_binary/within_eps_rank     all       0.084        0.001
    predictor_raw       per_csf_binary/within_eps_raw     all       0.084        0.001
    predictor_raw per_csf_binary/within_eps_unanimous     all         NaN          NaN
    predictor_raw                          regression     all       5.412        0.073
         baseline                          Always-CTM feature       0.084        0.005
         baseline                      Always-NNGuide feature       5.412        0.322
         baseline                         Always-fDBD feature       7.772        0.462
         baseline               Oracle-on-train (CTM) feature       0.084        0.005
         baseline                          Random-CSF feature       5.700        0.339
predictor_imputed                   multilabel/clique feature      16.816        1.000
predictor_imputed      multilabel/within_eps_majority feature      16.816        1.000
predictor_imputed          multilabel/within_eps_rank feature       5.412        0.322
predictor_imputed           multilabel/within_eps_raw feature       5.412        0.322
predictor_imputed     multilabel/within_eps_unanimous feature      16.816        1.000
predictor_imputed               per_csf_binary/clique feature      16.816        1.000
predictor_imputed  per_csf_binary/within_eps_majority feature       0.084        0.005
predictor_imputed      per_csf_binary/within_eps_rank feature       0.084        0.005
predictor_imputed       per_csf_binary/within_eps_raw feature       0.084        0.005
predictor_imputed per_csf_binary/within_eps_unanimous feature      16.816        1.000
    predictor_raw                   multilabel/clique feature         NaN          NaN
    predictor_raw      multilabel/within_eps_majority feature         NaN          NaN
    predictor_raw          multilabel/within_eps_rank feature       5.412        0.322
    predictor_raw           multilabel/within_eps_raw feature       5.412        0.322
    predictor_raw     multilabel/within_eps_unanimous feature         NaN          NaN
    predictor_raw               per_csf_binary/clique feature         NaN          NaN
    predictor_raw  per_csf_binary/within_eps_majority feature       0.084        0.005
    predictor_raw      per_csf_binary/within_eps_rank feature       0.084        0.005
    predictor_raw       per_csf_binary/within_eps_raw feature       0.084        0.005
    predictor_raw per_csf_binary/within_eps_unanimous feature         NaN          NaN
    predictor_raw                          regression feature       5.412        0.322
         baseline                       Always-Energy    head       5.338        0.073
         baseline                          Always-MLS    head       5.303        0.073
         baseline                          Always-MSR    head       3.719        0.051
         baseline               Oracle-on-train (MLS)    head       5.303        0.073
         baseline                          Random-CSF    head       9.634        0.132
predictor_imputed                   multilabel/clique    head       2.043        0.028
predictor_imputed      multilabel/within_eps_majority    head       2.043        0.028
predictor_imputed          multilabel/within_eps_rank    head       2.043        0.028
predictor_imputed           multilabel/within_eps_raw    head       2.043        0.028
predictor_imputed     multilabel/within_eps_unanimous    head       2.043        0.028
predictor_imputed               per_csf_binary/clique    head      73.018        1.000
predictor_imputed  per_csf_binary/within_eps_majority    head       2.043        0.028
predictor_imputed      per_csf_binary/within_eps_rank    head       2.043        0.028
predictor_imputed       per_csf_binary/within_eps_raw    head      73.018        1.000
predictor_imputed per_csf_binary/within_eps_unanimous    head      73.018        1.000
    predictor_raw                   multilabel/clique    head       2.043        0.028
    predictor_raw      multilabel/within_eps_majority    head       2.043        0.028
    predictor_raw          multilabel/within_eps_rank    head       2.043        0.028
    predictor_raw           multilabel/within_eps_raw    head       2.043        0.028
    predictor_raw     multilabel/within_eps_unanimous    head       2.043        0.028
    predictor_raw               per_csf_binary/clique    head         NaN          NaN
    predictor_raw  per_csf_binary/within_eps_majority    head       2.043        0.028
    predictor_raw      per_csf_binary/within_eps_rank    head       2.043        0.028
    predictor_raw       per_csf_binary/within_eps_raw    head         NaN          NaN
    predictor_raw per_csf_binary/within_eps_unanimous    head         NaN          NaN
    predictor_raw                          regression    head       5.303        0.073
```

Reading: `predictor_raw` = the raw step-13 predictor regret (NaN for binary heads when set ∩ side was empty). `predictor_imputed` = empty rows filled with worst-case regret (= worst − oracle for that side). `baseline` rows include Always-X, Random-CSF, and Oracle-on-train.

## Aggregate — `xarch` cross-arch headline

Sorted within each (regime, side) by `regret_raw_mean` ascending (lower = better):

### `regime=near, side=all`

```
  comparator_kind                     comparator_name regime side   n  regret_raw_mean  regret_raw_ci_lo  regret_raw_ci_hi  regret_norm_mean
         baseline                          Always-CTM   near  all 148            2.066             1.474             2.707             0.051
         baseline               Oracle-on-train (CTM)   near  all 148            2.066             1.474             2.707             0.051
         baseline                      Always-NNGuide   near  all 148            4.049             3.454             4.728             0.065
         baseline                          Always-MLS   near  all 148            5.463             4.693             6.313             0.081
         baseline                          Always-MSR   near  all 148            6.065             5.324             6.903             0.087
         baseline                       Always-Energy   near  all 148            6.388             5.477             7.386             0.093
         baseline                         Always-fDBD   near  all 148            7.033             5.947             8.207             0.128
         baseline                          Random-CSF   near  all 148           22.158            19.990            24.354             0.228
predictor_imputed  per_csf_binary/within_eps_majority   near  all 148            1.298             0.876             1.790             0.031
predictor_imputed      per_csf_binary/within_eps_rank   near  all 148            1.455             1.023             1.970             0.033
predictor_imputed       per_csf_binary/within_eps_raw   near  all 148            1.509             1.059             2.030             0.034
predictor_imputed                   multilabel/clique   near  all 148            1.858             1.390             2.379             0.038
predictor_imputed      multilabel/within_eps_majority   near  all 148            1.887             1.217             2.660             0.052
predictor_imputed           multilabel/within_eps_raw   near  all 148            8.354             5.753            11.347             0.197
predictor_imputed               per_csf_binary/clique   near  all 148           11.299             7.294            15.992             0.192
predictor_imputed          multilabel/within_eps_rank   near  all 148           12.488             8.456            17.530             0.268
predictor_imputed     multilabel/within_eps_unanimous   near  all 148           19.696            15.347            24.393             0.401
predictor_imputed per_csf_binary/within_eps_unanimous   near  all 148           24.408            19.775            29.162             0.502
    predictor_raw per_csf_binary/within_eps_unanimous   near  all 148            0.770             0.411             1.201             0.005
    predictor_raw     multilabel/within_eps_unanimous   near  all 148            1.103             0.713             1.554             0.016
    predictor_raw          multilabel/within_eps_rank   near  all 148            1.117             0.768             1.517             0.015
    predictor_raw  per_csf_binary/within_eps_majority   near  all 148            1.298             0.876             1.790             0.031
    predictor_raw           multilabel/within_eps_raw   near  all 148            1.427             0.976             1.995             0.025
    predictor_raw      per_csf_binary/within_eps_rank   near  all 148            1.455             1.023             1.970             0.033
    predictor_raw      multilabel/within_eps_majority   near  all 148            1.482             1.071             1.933             0.039
    predictor_raw       per_csf_binary/within_eps_raw   near  all 148            1.509             1.059             2.030             0.034
    predictor_raw               per_csf_binary/clique   near  all 148            1.715             1.203             2.303             0.035
    predictor_raw                   multilabel/clique   near  all 148            1.858             1.390             2.379             0.038
    predictor_raw                          regression   near  all 148            4.049             3.454             4.728             0.065
```

### `regime=near, side=feature`

```
  comparator_kind                     comparator_name regime    side   n  regret_raw_mean  regret_raw_ci_lo  regret_raw_ci_hi  regret_norm_mean
         baseline                          Always-CTM   near feature 148            1.548             1.038             2.120             0.060
         baseline               Oracle-on-train (CTM)   near feature 148            1.548             1.038             2.120             0.060
         baseline                      Always-NNGuide   near feature 148            3.530             2.935             4.205             0.110
         baseline                         Always-fDBD   near feature 148            6.515             5.466             7.683             0.178
         baseline                          Random-CSF   near feature 148           30.945            27.233            34.721             0.361
predictor_imputed       per_csf_binary/within_eps_raw   near feature 148            1.088             0.695             1.537             0.043
predictor_imputed  per_csf_binary/within_eps_majority   near feature 148            1.312             0.820             1.898             0.055
predictor_imputed      per_csf_binary/within_eps_rank   near feature 148            1.527             1.016             2.098             0.060
predictor_imputed      multilabel/within_eps_majority   near feature 148            4.074             2.276             6.507             0.136
predictor_imputed                   multilabel/clique   near feature 148            8.219             4.889            12.324             0.201
predictor_imputed           multilabel/within_eps_raw   near feature 148           12.610             8.357            17.475             0.283
predictor_imputed          multilabel/within_eps_rank   near feature 148           16.232            11.645            21.777             0.371
predictor_imputed               per_csf_binary/clique   near feature 148           16.922            11.083            23.393             0.277
predictor_imputed     multilabel/within_eps_unanimous   near feature 148           27.044            21.310            33.295             0.595
predictor_imputed per_csf_binary/within_eps_unanimous   near feature 148           27.044            21.310            33.295             0.595
    predictor_raw     multilabel/within_eps_unanimous   near feature 148            0.354             0.017             0.804             0.002
    predictor_raw per_csf_binary/within_eps_unanimous   near feature 148            0.354             0.017             0.804             0.002
    predictor_raw       per_csf_binary/within_eps_raw   near feature 148            1.088             0.695             1.537             0.043
    predictor_raw      multilabel/within_eps_majority   near feature 148            1.250             0.729             1.953             0.046
    predictor_raw  per_csf_binary/within_eps_majority   near feature 148            1.312             0.820             1.898             0.055
    predictor_raw               per_csf_binary/clique   near feature 148            1.315             0.832             1.865             0.045
    predictor_raw           multilabel/within_eps_raw   near feature 148            1.350             0.899             1.867             0.053
    predictor_raw      per_csf_binary/within_eps_rank   near feature 148            1.527             1.016             2.098             0.060
    predictor_raw                   multilabel/clique   near feature 148            1.636             1.115             2.267             0.046
    predictor_raw          multilabel/within_eps_rank   near feature 148            1.715             0.888             2.760             0.050
    predictor_raw                          regression   near feature 148            3.530             2.935             4.205             0.110
```

### `regime=near, side=head`

```
  comparator_kind                     comparator_name regime side   n  regret_raw_mean  regret_raw_ci_lo  regret_raw_ci_hi  regret_norm_mean
         baseline                          Always-MLS   near head 148            3.492             2.878             4.183             0.075
         baseline               Oracle-on-train (MLS)   near head 148            3.492             2.878             4.183             0.075
         baseline                          Always-MSR   near head 148            4.094             3.451             4.816             0.085
         baseline                       Always-Energy   near head 148            4.417             3.722             5.181             0.089
         baseline                          Random-CSF   near head 148           12.574            11.533            13.667             0.194
predictor_imputed           multilabel/within_eps_raw   near head 148           37.159            30.101            44.636             0.605
predictor_imputed                   multilabel/clique   near head 148           40.377            32.905            48.098             0.624
predictor_imputed      multilabel/within_eps_majority   near head 148           40.753            33.381            48.485             0.638
predictor_imputed  per_csf_binary/within_eps_majority   near head 148           43.164            35.561            50.973             0.669
predictor_imputed          multilabel/within_eps_rank   near head 148           45.785            37.848            54.535             0.664
predictor_imputed     multilabel/within_eps_unanimous   near head 148           48.038            40.749            55.834             0.738
predictor_imputed       per_csf_binary/within_eps_raw   near head 148           51.835            44.482            59.122             0.827
predictor_imputed      per_csf_binary/within_eps_rank   near head 148           52.579            44.482            60.978             0.743
predictor_imputed               per_csf_binary/clique   near head 148           59.833            51.839            68.300             0.828
predictor_imputed per_csf_binary/within_eps_unanimous   near head 148           61.273            53.785            69.290             0.906
    predictor_raw per_csf_binary/within_eps_unanimous   near head 148            0.951             0.563             1.386             0.007
    predictor_raw  per_csf_binary/within_eps_majority   near head 148            1.250             0.629             2.016             0.019
    predictor_raw               per_csf_binary/clique   near head 148            1.393             0.735             2.271             0.023
    predictor_raw      multilabel/within_eps_majority   near head 148            1.452             0.867             2.121             0.025
    predictor_raw      per_csf_binary/within_eps_rank   near head 148            1.557             0.776             2.581             0.050
    predictor_raw                   multilabel/clique   near head 148            1.601             0.954             2.367             0.022
    predictor_raw           multilabel/within_eps_raw   near head 148            1.604             1.055             2.227             0.025
    predictor_raw          multilabel/within_eps_rank   near head 148            1.636             1.004             2.356             0.026
    predictor_raw       per_csf_binary/within_eps_raw   near head 148            1.917             0.759             3.337             0.018
    predictor_raw     multilabel/within_eps_unanimous   near head 148            2.080             1.221             3.053             0.031
    predictor_raw                          regression   near head 148            3.492             2.878             4.183             0.075
```

### `regime=mid, side=all`

```
  comparator_kind                     comparator_name regime side   n  regret_raw_mean  regret_raw_ci_lo  regret_raw_ci_hi  regret_norm_mean
         baseline                          Always-CTM    mid  all 200            3.519             2.939             4.161             0.074
         baseline               Oracle-on-train (CTM)    mid  all 200            3.519             2.939             4.161             0.074
         baseline                      Always-NNGuide    mid  all 200            4.792             3.972             5.694             0.094
         baseline                       Always-Energy    mid  all 200            6.580             5.540             7.796             0.117
         baseline                          Always-MLS    mid  all 200            6.816             5.872             7.906             0.124
         baseline                          Always-MSR    mid  all 200            8.686             7.798             9.649             0.154
         baseline                         Always-fDBD    mid  all 200            9.901             8.705            11.232             0.164
         baseline                          Random-CSF    mid  all 200           17.374            16.073            18.697             0.234
predictor_imputed      multilabel/within_eps_majority    mid  all 200            1.839             1.448             2.308             0.036
predictor_imputed      per_csf_binary/within_eps_rank    mid  all 200            2.222             1.757             2.768             0.052
predictor_imputed       per_csf_binary/within_eps_raw    mid  all 200            2.262             1.797             2.797             0.052
predictor_imputed  per_csf_binary/within_eps_majority    mid  all 200            2.596             2.029             3.200             0.056
predictor_imputed           multilabel/within_eps_raw    mid  all 200            5.141             2.958             7.930             0.074
predictor_imputed                   multilabel/clique    mid  all 200            5.861             3.661             8.605             0.087
predictor_imputed          multilabel/within_eps_rank    mid  all 200            6.495             3.961             9.442             0.095
predictor_imputed               per_csf_binary/clique    mid  all 200           14.265            10.556            18.769             0.239
predictor_imputed per_csf_binary/within_eps_unanimous    mid  all 200           53.918            46.436            61.198             0.730
predictor_imputed     multilabel/within_eps_unanimous    mid  all 200           58.618            51.209            66.137             0.765
    predictor_raw     multilabel/within_eps_unanimous    mid  all 200            0.981             0.636             1.371             0.021
    predictor_raw      multilabel/within_eps_majority    mid  all 200            1.839             1.448             2.308             0.036
    predictor_raw      per_csf_binary/within_eps_rank    mid  all 200            2.222             1.757             2.768             0.052
    predictor_raw       per_csf_binary/within_eps_raw    mid  all 200            2.262             1.797             2.797             0.052
    predictor_raw per_csf_binary/within_eps_unanimous    mid  all 200            2.341             1.690             3.049             0.037
    predictor_raw  per_csf_binary/within_eps_majority    mid  all 200            2.596             2.029             3.200             0.056
    predictor_raw           multilabel/within_eps_raw    mid  all 200            2.682             2.176             3.228             0.055
    predictor_raw          multilabel/within_eps_rank    mid  all 200            2.911             2.407             3.457             0.057
    predictor_raw                   multilabel/clique    mid  all 200            3.416             2.863             4.048             0.069
    predictor_raw               per_csf_binary/clique    mid  all 200            3.508             2.896             4.157             0.071
    predictor_raw                          regression    mid  all 200            4.792             3.972             5.694             0.094
```

### `regime=mid, side=feature`

```
  comparator_kind                     comparator_name regime    side   n  regret_raw_mean  regret_raw_ci_lo  regret_raw_ci_hi  regret_norm_mean
         baseline                          Always-CTM    mid feature 200            2.919             2.434             3.457             0.108
         baseline               Oracle-on-train (CTM)    mid feature 200            2.919             2.434             3.457             0.108
         baseline                      Always-NNGuide    mid feature 200            4.191             3.378             5.093             0.179
         baseline                         Always-fDBD    mid feature 200            9.301             8.144            10.604             0.242
         baseline                          Random-CSF    mid feature 200           20.650            18.322            22.990             0.362
predictor_imputed      per_csf_binary/within_eps_rank    mid feature 200            1.821             1.474             2.201             0.063
predictor_imputed       per_csf_binary/within_eps_raw    mid feature 200            1.859             1.502             2.259             0.062
predictor_imputed  per_csf_binary/within_eps_majority    mid feature 200            2.270             1.812             2.764             0.095
predictor_imputed      multilabel/within_eps_majority    mid feature 200            5.770             3.504             8.475             0.127
predictor_imputed          multilabel/within_eps_rank    mid feature 200           10.733             7.823            14.063             0.266
predictor_imputed                   multilabel/clique    mid feature 200           20.588            15.737            25.884             0.393
predictor_imputed           multilabel/within_eps_raw    mid feature 200           25.244            19.641            31.196             0.412
predictor_imputed               per_csf_binary/clique    mid feature 200           26.357            20.452            32.570             0.386
predictor_imputed per_csf_binary/within_eps_unanimous    mid feature 200           40.688            34.480            46.984             0.761
predictor_imputed     multilabel/within_eps_unanimous    mid feature 200           44.108            38.246            50.118             0.880
    predictor_raw     multilabel/within_eps_unanimous    mid feature 200            0.510             0.155             0.955             0.003
    predictor_raw      per_csf_binary/within_eps_rank    mid feature 200            1.821             1.474             2.201             0.063
    predictor_raw       per_csf_binary/within_eps_raw    mid feature 200            1.859             1.502             2.259             0.062
    predictor_raw      multilabel/within_eps_majority    mid feature 200            1.919             1.395             2.556             0.052
    predictor_raw  per_csf_binary/within_eps_majority    mid feature 200            2.270             1.812             2.764             0.095
    predictor_raw per_csf_binary/within_eps_unanimous    mid feature 200            2.280             1.582             3.037             0.079
    predictor_raw           multilabel/within_eps_raw    mid feature 200            2.917             2.304             3.557             0.109
    predictor_raw          multilabel/within_eps_rank    mid feature 200            2.985             2.465             3.581             0.105
    predictor_raw               per_csf_binary/clique    mid feature 200            3.091             2.522             3.699             0.123
    predictor_raw                   multilabel/clique    mid feature 200            3.736             2.934             4.668             0.133
    predictor_raw                          regression    mid feature 200            4.191             3.378             5.093             0.179
```

### `regime=mid, side=head`

```
  comparator_kind                     comparator_name regime side   n  regret_raw_mean  regret_raw_ci_lo  regret_raw_ci_hi  regret_norm_mean
         baseline                       Always-Energy    mid head 200            4.201             3.395             5.101             0.093
         baseline                          Always-MLS    mid head 200            4.438             3.719             5.250             0.103
         baseline               Oracle-on-train (MLS)    mid head 200            4.438             3.719             5.250             0.103
         baseline                          Always-MSR    mid head 200            6.307             5.591             7.056             0.144
         baseline                          Random-CSF    mid head 200           11.823            11.037            12.658             0.217
predictor_imputed      multilabel/within_eps_majority    mid head 200           30.509            26.288            35.044             0.603
predictor_imputed  per_csf_binary/within_eps_majority    mid head 200           30.705            26.452            35.246             0.611
predictor_imputed           multilabel/within_eps_raw    mid head 200           32.349            27.859            37.122             0.623
predictor_imputed                   multilabel/clique    mid head 200           33.401            28.735            38.269             0.643
predictor_imputed          multilabel/within_eps_rank    mid head 200           35.645            30.513            40.847             0.638
predictor_imputed      per_csf_binary/within_eps_rank    mid head 200           41.451            37.228            45.842             0.799
predictor_imputed       per_csf_binary/within_eps_raw    mid head 200           43.424            39.422            47.486             0.860
predictor_imputed               per_csf_binary/clique    mid head 200           46.682            42.284            51.572             0.880
predictor_imputed     multilabel/within_eps_unanimous    mid head 200           60.023            53.226            67.067             0.885
predictor_imputed per_csf_binary/within_eps_unanimous    mid head 200           63.243            56.850            69.840             0.980
    predictor_raw       per_csf_binary/within_eps_raw    mid head 200            0.175             0.002             0.450             0.002
    predictor_raw               per_csf_binary/clique    mid head 200            0.177             0.000             0.487             0.002
    predictor_raw          multilabel/within_eps_rank    mid head 200            0.752             0.334             1.390             0.022
    predictor_raw      multilabel/within_eps_majority    mid head 200            0.932             0.504             1.510             0.031
    predictor_raw per_csf_binary/within_eps_unanimous    mid head 200            0.938             0.000             2.815             0.010
    predictor_raw     multilabel/within_eps_unanimous    mid head 200            1.029             0.551             1.563             0.040
    predictor_raw           multilabel/within_eps_raw    mid head 200            1.079             0.537             1.829             0.033
    predictor_raw  per_csf_binary/within_eps_majority    mid head 200            1.411             0.547             2.445             0.052
    predictor_raw                   multilabel/clique    mid head 200            1.715             0.820             2.805             0.061
    predictor_raw      per_csf_binary/within_eps_rank    mid head 200            2.305             0.765             4.191             0.085
    predictor_raw                          regression    mid head 200            4.438             3.719             5.250             0.103
```

### `regime=far, side=all`

```
  comparator_kind                     comparator_name regime side   n  regret_raw_mean  regret_raw_ci_lo  regret_raw_ci_hi  regret_norm_mean
         baseline                          Always-CTM    far  all 100            2.403             1.777             3.157             0.071
         baseline               Oracle-on-train (CTM)    far  all 100            2.403             1.777             3.157             0.071
         baseline                      Always-NNGuide    far  all 100            4.624             3.765             5.582             0.098
         baseline                          Always-MLS    far  all 100            6.355             5.409             7.379             0.142
         baseline                       Always-Energy    far  all 100            6.675             5.666             7.727             0.145
         baseline                          Always-MSR    far  all 100            7.112             6.213             8.082             0.154
         baseline                         Always-fDBD    far  all 100            8.092             6.369            10.158             0.157
         baseline                          Random-CSF    far  all 100           14.374            12.702            16.027             0.239
predictor_imputed      per_csf_binary/within_eps_rank    far  all 100            1.488             1.023             2.075             0.053
predictor_imputed       per_csf_binary/within_eps_raw    far  all 100            1.497             1.028             2.091             0.053
predictor_imputed                   multilabel/clique    far  all 100            1.765             1.361             2.187             0.040
predictor_imputed  per_csf_binary/within_eps_majority    far  all 100            2.047             1.430             2.806             0.065
predictor_imputed      multilabel/within_eps_majority    far  all 100            6.165             2.801            10.309             0.101
predictor_imputed               per_csf_binary/clique    far  all 100           14.350             7.135            22.690             0.188
predictor_imputed           multilabel/within_eps_raw    far  all 100           31.844            21.607            43.074             0.330
predictor_imputed          multilabel/within_eps_rank    far  all 100           34.274            24.277            45.512             0.412
predictor_imputed     multilabel/within_eps_unanimous    far  all 100           36.422            27.950            45.898             0.591
predictor_imputed per_csf_binary/within_eps_unanimous    far  all 100           45.127            35.910            55.283             0.724
    predictor_raw per_csf_binary/within_eps_unanimous    far  all 100            0.959             0.588             1.358             0.014
    predictor_raw               per_csf_binary/clique    far  all 100            1.219             0.922             1.534             0.033
    predictor_raw     multilabel/within_eps_unanimous    far  all 100            1.305             0.862             1.783             0.026
    predictor_raw           multilabel/within_eps_raw    far  all 100            1.358             1.041             1.689             0.042
    predictor_raw      per_csf_binary/within_eps_rank    far  all 100            1.488             1.023             2.075             0.053
    predictor_raw       per_csf_binary/within_eps_raw    far  all 100            1.497             1.028             2.091             0.053
    predictor_raw          multilabel/within_eps_rank    far  all 100            1.638             1.191             2.138             0.051
    predictor_raw      multilabel/within_eps_majority    far  all 100            1.741             1.290             2.221             0.043
    predictor_raw                   multilabel/clique    far  all 100            1.765             1.361             2.187             0.040
    predictor_raw  per_csf_binary/within_eps_majority    far  all 100            2.047             1.430             2.806             0.065
    predictor_raw                          regression    far  all 100            4.624             3.765             5.582             0.098
```

### `regime=far, side=feature`

```
  comparator_kind                     comparator_name regime    side   n  regret_raw_mean  regret_raw_ci_lo  regret_raw_ci_hi  regret_norm_mean
         baseline                          Always-CTM    far feature 100            2.188             1.589             2.923             0.119
         baseline               Oracle-on-train (CTM)    far feature 100            2.188             1.589             2.923             0.119
         baseline                      Always-NNGuide    far feature 100            4.409             3.559             5.352             0.214
         baseline                         Always-fDBD    far feature 100            7.876             6.191             9.935             0.246
         baseline                          Random-CSF    far feature 100           15.976            13.253            18.790             0.369
predictor_imputed      per_csf_binary/within_eps_rank    far feature 100            1.454             0.994             2.047             0.079
predictor_imputed       per_csf_binary/within_eps_raw    far feature 100            1.462             1.003             2.056             0.079
predictor_imputed  per_csf_binary/within_eps_majority    far feature 100            1.981             1.413             2.692             0.113
predictor_imputed      multilabel/within_eps_majority    far feature 100            5.823             3.495             8.604             0.229
predictor_imputed                   multilabel/clique    far feature 100           12.929             7.927            18.751             0.306
predictor_imputed               per_csf_binary/clique    far feature 100           19.147            12.120            27.254             0.289
predictor_imputed           multilabel/within_eps_raw    far feature 100           20.581            13.769            28.560             0.400
predictor_imputed          multilabel/within_eps_rank    far feature 100           21.750            14.954            29.587             0.455
predictor_imputed     multilabel/within_eps_unanimous    far feature 100           29.336            22.772            37.067             0.747
predictor_imputed per_csf_binary/within_eps_unanimous    far feature 100           29.336            22.772            37.067             0.747
    predictor_raw     multilabel/within_eps_unanimous    far feature 100            0.658             0.355             1.012             0.025
    predictor_raw per_csf_binary/within_eps_unanimous    far feature 100            0.658             0.355             1.012             0.025
    predictor_raw               per_csf_binary/clique    far feature 100            1.031             0.723             1.345             0.065
    predictor_raw      multilabel/within_eps_majority    far feature 100            1.079             0.775             1.419             0.060
    predictor_raw           multilabel/within_eps_raw    far feature 100            1.134             0.820             1.484             0.062
    predictor_raw      per_csf_binary/within_eps_rank    far feature 100            1.454             0.994             2.047             0.079
    predictor_raw       per_csf_binary/within_eps_raw    far feature 100            1.462             1.003             2.056             0.079
    predictor_raw          multilabel/within_eps_rank    far feature 100            1.481             1.025             2.013             0.092
    predictor_raw                   multilabel/clique    far feature 100            1.825             0.898             3.191             0.063
    predictor_raw  per_csf_binary/within_eps_majority    far feature 100            1.981             1.413             2.692             0.113
    predictor_raw                          regression    far feature 100            4.409             3.559             5.352             0.214
```

### `regime=far, side=head`

```
  comparator_kind                     comparator_name regime side   n  regret_raw_mean  regret_raw_ci_lo  regret_raw_ci_hi  regret_norm_mean
         baseline                          Always-MLS    far head 100            3.773             2.944             4.753             0.090
         baseline               Oracle-on-train (MLS)    far head 100            3.773             2.944             4.753             0.090
         baseline                       Always-Energy    far head 100            4.093             3.187             5.080             0.093
         baseline                          Always-MSR    far head 100            4.530             3.660             5.521             0.107
         baseline                          Random-CSF    far head 100           10.304             9.220            11.379             0.204
predictor_imputed                   multilabel/clique    far head 100           26.681            20.896            32.526             0.618
predictor_imputed      multilabel/within_eps_majority    far head 100           28.752            23.306            34.345             0.658
predictor_imputed       per_csf_binary/within_eps_raw    far head 100           36.978            31.133            42.976             0.824
predictor_imputed  per_csf_binary/within_eps_majority    far head 100           37.071            30.788            43.506             0.778
predictor_imputed      per_csf_binary/within_eps_rank    far head 100           39.365            33.311            45.641             0.846
predictor_imputed               per_csf_binary/clique    far head 100           49.694            41.988            57.869             0.920
predictor_imputed     multilabel/within_eps_unanimous    far head 100           50.038            41.964            59.132             0.848
predictor_imputed           multilabel/within_eps_raw    far head 100           54.108            45.287            63.440             0.888
predictor_imputed          multilabel/within_eps_rank    far head 100           56.005            47.275            64.964             0.933
predictor_imputed per_csf_binary/within_eps_unanimous    far head 100           58.087            49.673            66.919             0.980
    predictor_raw               per_csf_binary/clique    far head 100            0.292             0.087             0.513             0.002
    predictor_raw per_csf_binary/within_eps_unanimous    far head 100            0.503             0.000             1.006             0.003
    predictor_raw  per_csf_binary/within_eps_majority    far head 100            1.018             0.298             2.133             0.034
    predictor_raw     multilabel/within_eps_unanimous    far head 100            1.114             0.587             1.700             0.050
    predictor_raw       per_csf_binary/within_eps_raw    far head 100            1.241             0.326             2.634             0.021
    predictor_raw      per_csf_binary/within_eps_rank    far head 100            1.554             0.160             3.843             0.039
    predictor_raw                   multilabel/clique    far head 100            1.878             1.091             2.857             0.045
    predictor_raw          multilabel/within_eps_rank    far head 100            1.878             0.230             4.840             0.049
    predictor_raw      multilabel/within_eps_majority    far head 100            2.093             1.206             3.085             0.050
    predictor_raw           multilabel/within_eps_raw    far head 100            2.300             0.842             4.220             0.068
    predictor_raw                          regression    far head 100            3.773             2.944             4.753             0.090
```

