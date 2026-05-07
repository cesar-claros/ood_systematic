# Step 13 — Regret + ranking metrics

**Date:** 2026-05-03
**Source:** `code/nc_csf_predictivity/evaluation/regret.py`
**Bootstrap:** n=2000, seed=0, percentile 95% CI on the mean.

## Worked example — top-1 / set / per-side regret on one xarch row

Same row as steps 10–12: `ResNet18|confidnet|cifar10|1|0|2.2`, eval=`cifar100`, regime=`near`. Showing all (predictor, label_rule, side) combinations evaluated on this row.

```
     predictor           label_rule    side                o_csf  o_augrc  w_augrc top1_csf  top1_regret_raw  top1_regret_norm  set_size  set_regret_raw
    regression                          all KPCA RecError global  165.125  239.561  NNGuide            5.412             0.073       NaN             NaN
    regression                         head                 pNML  166.544  239.561      MLS            5.303             0.073       NaN             NaN
    regression                      feature KPCA RecError global  165.125  181.941  NNGuide            5.412             0.322       NaN             NaN
    multilabel               clique     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       1.0           3.462
    multilabel               clique    head                 pNML  166.544  239.561      NaN              NaN               NaN       1.0           2.043
    multilabel               clique feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       0.0             NaN
    multilabel       within_eps_raw     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       2.0           3.462
    multilabel       within_eps_raw    head                 pNML  166.544  239.561      NaN              NaN               NaN       1.0           2.043
    multilabel       within_eps_raw feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       1.0           5.412
    multilabel      within_eps_rank     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       2.0           3.462
    multilabel      within_eps_rank    head                 pNML  166.544  239.561      NaN              NaN               NaN       1.0           2.043
    multilabel      within_eps_rank feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       1.0           5.412
    multilabel  within_eps_majority     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       1.0           3.462
    multilabel  within_eps_majority    head                 pNML  166.544  239.561      NaN              NaN               NaN       1.0           2.043
    multilabel  within_eps_majority feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       0.0             NaN
    multilabel within_eps_unanimous     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       1.0           3.462
    multilabel within_eps_unanimous    head                 pNML  166.544  239.561      NaN              NaN               NaN       1.0           2.043
    multilabel within_eps_unanimous feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       0.0             NaN
per_csf_binary               clique     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       0.0             NaN
per_csf_binary               clique    head                 pNML  166.544  239.561      NaN              NaN               NaN       0.0             NaN
per_csf_binary               clique feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       0.0             NaN
per_csf_binary       within_eps_raw     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       1.0           0.084
per_csf_binary       within_eps_raw    head                 pNML  166.544  239.561      NaN              NaN               NaN       0.0             NaN
per_csf_binary       within_eps_raw feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       1.0           0.084
per_csf_binary      within_eps_rank     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       2.0           0.084
per_csf_binary      within_eps_rank    head                 pNML  166.544  239.561      NaN              NaN               NaN       1.0           2.043
per_csf_binary      within_eps_rank feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       1.0           0.084
per_csf_binary  within_eps_majority     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       2.0           0.084
per_csf_binary  within_eps_majority    head                 pNML  166.544  239.561      NaN              NaN               NaN       1.0           2.043
per_csf_binary  within_eps_majority feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       1.0           0.084
per_csf_binary within_eps_unanimous     all KPCA RecError global  165.125  239.561      NaN              NaN               NaN       0.0             NaN
per_csf_binary within_eps_unanimous    head                 pNML  166.544  239.561      NaN              NaN               NaN       0.0             NaN
per_csf_binary within_eps_unanimous feature KPCA RecError global  165.125  181.941      NaN              NaN               NaN       0.0             NaN
```

## Aggregate — `xarch` (Track 1, primary headline)

```
     predictor           label_rule regime    side   n  top1_regret_raw_mean  top1_regret_raw_ci_lo  top1_regret_raw_ci_hi  top1_regret_norm_mean  set_regret_raw_mean  set_regret_raw_ci_lo  set_regret_raw_ci_hi  set_size_mean  spearman_rho_mean  mrr_mean  empty_set_share
    multilabel               clique    far     all 100                   NaN                    NaN                    NaN                    NaN                1.765                 1.361                 2.187          2.140                NaN       NaN            0.000
    multilabel  within_eps_majority    far     all 100                   NaN                    NaN                    NaN                    NaN                1.741                 1.290                 2.221          2.020                NaN       NaN            0.060
    multilabel      within_eps_rank    far     all 100                   NaN                    NaN                    NaN                    NaN                1.638                 1.191                 2.138          0.880                NaN       NaN            0.380
    multilabel       within_eps_raw    far     all 100                   NaN                    NaN                    NaN                    NaN                1.358                 1.041                 1.689          1.240                NaN       NaN            0.300
    multilabel within_eps_unanimous    far     all 100                   NaN                    NaN                    NaN                    NaN                1.305                 0.862                 1.783          0.440                NaN       NaN            0.580
per_csf_binary               clique    far     all 100                   NaN                    NaN                    NaN                    NaN                1.219                 0.922                 1.534          1.320                NaN       NaN            0.160
per_csf_binary  within_eps_majority    far     all 100                   NaN                    NaN                    NaN                    NaN                2.047                 1.430                 2.806          1.770                NaN       NaN            0.000
per_csf_binary      within_eps_rank    far     all 100                   NaN                    NaN                    NaN                    NaN                1.488                 1.023                 2.075          1.680                NaN       NaN            0.000
per_csf_binary       within_eps_raw    far     all 100                   NaN                    NaN                    NaN                    NaN                1.497                 1.028                 2.091          1.520                NaN       NaN            0.000
per_csf_binary within_eps_unanimous    far     all 100                   NaN                    NaN                    NaN                    NaN                0.959                 0.588                 1.358          0.310                NaN       NaN            0.720
    regression                         far     all 100                 4.624                  3.765                  5.582                  0.098                  NaN                   NaN                   NaN            NaN              0.410     0.383              NaN
    multilabel               clique    far feature 100                   NaN                    NaN                    NaN                    NaN                1.825                 0.898                 3.191          1.460                NaN       NaN            0.260
    multilabel  within_eps_majority    far feature 100                   NaN                    NaN                    NaN                    NaN                1.079                 0.775                 1.419          1.350                NaN       NaN            0.180
    multilabel      within_eps_rank    far feature 100                   NaN                    NaN                    NaN                    NaN                1.481                 1.025                 2.013          0.640                NaN       NaN            0.400
    multilabel       within_eps_raw    far feature 100                   NaN                    NaN                    NaN                    NaN                1.134                 0.820                 1.484          1.020                NaN       NaN            0.360
    multilabel within_eps_unanimous    far feature 100                   NaN                    NaN                    NaN                    NaN                0.658                 0.355                 1.012          0.260                NaN       NaN            0.740
per_csf_binary               clique    far feature 100                   NaN                    NaN                    NaN                    NaN                1.031                 0.723                 1.345          1.240                NaN       NaN            0.240
per_csf_binary  within_eps_majority    far feature 100                   NaN                    NaN                    NaN                    NaN                1.981                 1.413                 2.692          1.260                NaN       NaN            0.000
per_csf_binary      within_eps_rank    far feature 100                   NaN                    NaN                    NaN                    NaN                1.454                 0.994                 2.047          1.340                NaN       NaN            0.000
per_csf_binary       within_eps_raw    far feature 100                   NaN                    NaN                    NaN                    NaN                1.462                 1.003                 2.056          1.240                NaN       NaN            0.000
per_csf_binary within_eps_unanimous    far feature 100                   NaN                    NaN                    NaN                    NaN                0.658                 0.355                 1.012          0.270                NaN       NaN            0.740
    regression                         far feature 100                 4.409                  3.559                  5.352                  0.214                  NaN                   NaN                   NaN            NaN              0.546     0.485              NaN
    multilabel               clique    far    head 100                   NaN                    NaN                    NaN                    NaN                1.878                 1.091                 2.857          0.680                NaN       NaN            0.600
    multilabel  within_eps_majority    far    head 100                   NaN                    NaN                    NaN                    NaN                2.093                 1.206                 3.085          0.670                NaN       NaN            0.640
    multilabel      within_eps_rank    far    head 100                   NaN                    NaN                    NaN                    NaN                1.878                 0.230                 4.840          0.240                NaN       NaN            0.930
    multilabel       within_eps_raw    far    head 100                   NaN                    NaN                    NaN                    NaN                2.300                 0.842                 4.220          0.220                NaN       NaN            0.880
    multilabel within_eps_unanimous    far    head 100                   NaN                    NaN                    NaN                    NaN                1.114                 0.587                 1.700          0.180                NaN       NaN            0.840
per_csf_binary               clique    far    head 100                   NaN                    NaN                    NaN                    NaN                0.292                 0.087                 0.513          0.080                NaN       NaN            0.920
per_csf_binary  within_eps_majority    far    head 100                   NaN                    NaN                    NaN                    NaN                1.018                 0.298                 2.133          0.510                NaN       NaN            0.770
per_csf_binary      within_eps_rank    far    head 100                   NaN                    NaN                    NaN                    NaN                1.554                 0.160                 3.843          0.340                NaN       NaN            0.840
per_csf_binary       within_eps_raw    far    head 100                   NaN                    NaN                    NaN                    NaN                1.241                 0.326                 2.634          0.280                NaN       NaN            0.820
per_csf_binary within_eps_unanimous    far    head 100                   NaN                    NaN                    NaN                    NaN                0.503                 0.000                 1.006          0.040                NaN       NaN            0.980
    regression                         far    head 100                 3.773                  2.944                  4.753                  0.090                  NaN                   NaN                   NaN            NaN              0.325     0.233              NaN
    multilabel               clique    mid     all 200                   NaN                    NaN                    NaN                    NaN                3.416                 2.863                 4.048          1.260                NaN       NaN            0.020
    multilabel  within_eps_majority    mid     all 200                   NaN                    NaN                    NaN                    NaN                1.839                 1.448                 2.308          2.200                NaN       NaN            0.000
    multilabel      within_eps_rank    mid     all 200                   NaN                    NaN                    NaN                    NaN                2.911                 2.407                 3.457          1.490                NaN       NaN            0.040
    multilabel       within_eps_raw    mid     all 200                   NaN                    NaN                    NaN                    NaN                2.682                 2.176                 3.228          1.370                NaN       NaN            0.020
    multilabel within_eps_unanimous    mid     all 200                   NaN                    NaN                    NaN                    NaN                0.981                 0.636                 1.371          0.240                NaN       NaN            0.760
per_csf_binary               clique    mid     all 200                   NaN                    NaN                    NaN                    NaN                3.508                 2.896                 4.157          0.860                NaN       NaN            0.180
per_csf_binary  within_eps_majority    mid     all 200                   NaN                    NaN                    NaN                    NaN                2.596                 2.029                 3.200          2.150                NaN       NaN            0.000
per_csf_binary      within_eps_rank    mid     all 200                   NaN                    NaN                    NaN                    NaN                2.222                 1.757                 2.768          1.680                NaN       NaN            0.000
per_csf_binary       within_eps_raw    mid     all 200                   NaN                    NaN                    NaN                    NaN                2.262                 1.797                 2.797          1.580                NaN       NaN            0.000
per_csf_binary within_eps_unanimous    mid     all 200                   NaN                    NaN                    NaN                    NaN                2.341                 1.690                 3.049          0.300                NaN       NaN            0.720
    regression                         mid     all 200                 4.792                  3.972                  5.694                  0.094                  NaN                   NaN                   NaN            NaN              0.454     0.353              NaN
    multilabel               clique    mid feature 200                   NaN                    NaN                    NaN                    NaN                3.736                 2.934                 4.668          0.720                NaN       NaN            0.300
    multilabel  within_eps_majority    mid feature 200                   NaN                    NaN                    NaN                    NaN                1.919                 1.395                 2.556          1.540                NaN       NaN            0.080
    multilabel      within_eps_rank    mid feature 200                   NaN                    NaN                    NaN                    NaN                2.985                 2.465                 3.581          0.920                NaN       NaN            0.180
    multilabel       within_eps_raw    mid feature 200                   NaN                    NaN                    NaN                    NaN                2.917                 2.304                 3.557          0.900                NaN       NaN            0.340
    multilabel within_eps_unanimous    mid feature 200                   NaN                    NaN                    NaN                    NaN                0.510                 0.155                 0.955          0.120                NaN       NaN            0.880
per_csf_binary               clique    mid feature 200                   NaN                    NaN                    NaN                    NaN                3.091                 2.522                 3.699          0.740                NaN       NaN            0.300
per_csf_binary  within_eps_majority    mid feature 200                   NaN                    NaN                    NaN                    NaN                2.270                 1.812                 2.764          1.460                NaN       NaN            0.000
per_csf_binary      within_eps_rank    mid feature 200                   NaN                    NaN                    NaN                    NaN                1.821                 1.474                 2.201          1.260                NaN       NaN            0.000
per_csf_binary       within_eps_raw    mid feature 200                   NaN                    NaN                    NaN                    NaN                1.859                 1.502                 2.259          1.280                NaN       NaN            0.000
per_csf_binary within_eps_unanimous    mid feature 200                   NaN                    NaN                    NaN                    NaN                2.280                 1.582                 3.037          0.260                NaN       NaN            0.740
    regression                         mid feature 200                 4.191                  3.378                  5.093                  0.179                  NaN                   NaN                   NaN            NaN              0.557     0.511              NaN
    multilabel               clique    mid    head 200                   NaN                    NaN                    NaN                    NaN                1.715                 0.820                 2.805          0.540                NaN       NaN            0.620
    multilabel  within_eps_majority    mid    head 200                   NaN                    NaN                    NaN                    NaN                0.932                 0.504                 1.510          0.660                NaN       NaN            0.590
    multilabel      within_eps_rank    mid    head 200                   NaN                    NaN                    NaN                    NaN                0.752                 0.334                 1.390          0.570                NaN       NaN            0.630
    multilabel       within_eps_raw    mid    head 200                   NaN                    NaN                    NaN                    NaN                1.079                 0.537                 1.829          0.470                NaN       NaN            0.610
    multilabel within_eps_unanimous    mid    head 200                   NaN                    NaN                    NaN                    NaN                1.029                 0.551                 1.563          0.120                NaN       NaN            0.880
per_csf_binary               clique    mid    head 200                   NaN                    NaN                    NaN                    NaN                0.177                 0.000                 0.487          0.120                NaN       NaN            0.880
per_csf_binary  within_eps_majority    mid    head 200                   NaN                    NaN                    NaN                    NaN                1.411                 0.547                 2.445          0.690                NaN       NaN            0.590
per_csf_binary      within_eps_rank    mid    head 200                   NaN                    NaN                    NaN                    NaN                2.305                 0.765                 4.191          0.420                NaN       NaN            0.780
per_csf_binary       within_eps_raw    mid    head 200                   NaN                    NaN                    NaN                    NaN                0.175                 0.002                 0.450          0.300                NaN       NaN            0.860
per_csf_binary within_eps_unanimous    mid    head 200                   NaN                    NaN                    NaN                    NaN                0.938                 0.000                 2.815          0.040                NaN       NaN            0.980
    regression                         mid    head 200                 4.438                  3.719                  5.250                  0.103                  NaN                   NaN                   NaN            NaN              0.374     0.222              NaN
    multilabel               clique   near     all 148                   NaN                    NaN                    NaN                    NaN                1.858                 1.390                 2.379          1.520                NaN       NaN            0.000
    multilabel  within_eps_majority   near     all 148                   NaN                    NaN                    NaN                    NaN                1.482                 1.071                 1.933          2.068                NaN       NaN            0.014
    multilabel      within_eps_rank   near     all 148                   NaN                    NaN                    NaN                    NaN                1.117                 0.768                 1.517          1.345                NaN       NaN            0.257
    multilabel       within_eps_raw   near     all 148                   NaN                    NaN                    NaN                    NaN                1.427                 0.976                 1.995          1.453                NaN       NaN            0.176
    multilabel within_eps_unanimous   near     all 148                   NaN                    NaN                    NaN                    NaN                1.103                 0.713                 1.554          0.716                NaN       NaN            0.392
per_csf_binary               clique   near     all 148                   NaN                    NaN                    NaN                    NaN                1.715                 1.203                 2.303          1.000                NaN       NaN            0.162
per_csf_binary  within_eps_majority   near     all 148                   NaN                    NaN                    NaN                    NaN                1.298                 0.876                 1.790          1.953                NaN       NaN            0.000
per_csf_binary      within_eps_rank   near     all 148                   NaN                    NaN                    NaN                    NaN                1.455                 1.023                 1.970          1.514                NaN       NaN            0.000
per_csf_binary       within_eps_raw   near     all 148                   NaN                    NaN                    NaN                    NaN                1.509                 1.059                 2.030          1.459                NaN       NaN            0.000
per_csf_binary within_eps_unanimous   near     all 148                   NaN                    NaN                    NaN                    NaN                0.770                 0.411                 1.201          0.514                NaN       NaN            0.500
    regression                        near     all 148                 4.049                  3.454                  4.728                  0.065                  NaN                   NaN                   NaN            NaN              0.631     0.368              NaN
    multilabel               clique   near feature 148                   NaN                    NaN                    NaN                    NaN                1.636                 1.115                 2.267          0.838                NaN       NaN            0.162
    multilabel  within_eps_majority   near feature 148                   NaN                    NaN                    NaN                    NaN                1.250                 0.729                 1.953          1.216                NaN       NaN            0.095
    multilabel      within_eps_rank   near feature 148                   NaN                    NaN                    NaN                    NaN                1.715                 0.888                 2.760          0.676                NaN       NaN            0.338
    multilabel       within_eps_raw   near feature 148                   NaN                    NaN                    NaN                    NaN                1.350                 0.899                 1.867          0.784                NaN       NaN            0.243
    multilabel within_eps_unanimous   near feature 148                   NaN                    NaN                    NaN                    NaN                0.354                 0.017                 0.804          0.405                NaN       NaN            0.595
per_csf_binary               clique   near feature 148                   NaN                    NaN                    NaN                    NaN                1.315                 0.832                 1.865          0.757                NaN       NaN            0.243
per_csf_binary  within_eps_majority   near feature 148                   NaN                    NaN                    NaN                    NaN                1.312                 0.820                 1.898          1.216                NaN       NaN            0.000
per_csf_binary      within_eps_rank   near feature 148                   NaN                    NaN                    NaN                    NaN                1.527                 1.016                 2.098          1.054                NaN       NaN            0.000
per_csf_binary       within_eps_raw   near feature 148                   NaN                    NaN                    NaN                    NaN                1.088                 0.695                 1.537          1.149                NaN       NaN            0.000
per_csf_binary within_eps_unanimous   near feature 148                   NaN                    NaN                    NaN                    NaN                0.354                 0.017                 0.804          0.405                NaN       NaN            0.595
    regression                        near feature 148                 3.530                  2.935                  4.205                  0.110                  NaN                   NaN                   NaN            NaN              0.753     0.526              NaN
    multilabel               clique   near    head 148                   NaN                    NaN                    NaN                    NaN                1.601                 0.954                 2.367          0.682                NaN       NaN            0.615
    multilabel  within_eps_majority   near    head 148                   NaN                    NaN                    NaN                    NaN                1.452                 0.867                 2.121          0.851                NaN       NaN            0.628
    multilabel      within_eps_rank   near    head 148                   NaN                    NaN                    NaN                    NaN                1.636                 1.004                 2.356          0.669                NaN       NaN            0.655
    multilabel       within_eps_raw   near    head 148                   NaN                    NaN                    NaN                    NaN                1.604                 1.055                 2.227          0.669                NaN       NaN            0.595
    multilabel within_eps_unanimous   near    head 148                   NaN                    NaN                    NaN                    NaN                2.080                 1.221                 3.053          0.311                NaN       NaN            0.730
per_csf_binary               clique   near    head 148                   NaN                    NaN                    NaN                    NaN                1.393                 0.735                 2.271          0.243                NaN       NaN            0.824
per_csf_binary  within_eps_majority   near    head 148                   NaN                    NaN                    NaN                    NaN                1.250                 0.629                 2.016          0.736                NaN       NaN            0.662
per_csf_binary      within_eps_rank   near    head 148                   NaN                    NaN                    NaN                    NaN                1.557                 0.776                 2.581          0.459                NaN       NaN            0.730
per_csf_binary       within_eps_raw   near    head 148                   NaN                    NaN                    NaN                    NaN                1.917                 0.759                 3.337          0.311                NaN       NaN            0.824
per_csf_binary within_eps_unanimous   near    head 148                   NaN                    NaN                    NaN                    NaN                0.951                 0.563                 1.386          0.108                NaN       NaN            0.905
    regression                        near    head 148                 3.492                  2.878                  4.183                  0.075                  NaN                   NaN                   NaN            NaN              0.462     0.290              NaN
```

Reading: `top1_regret_raw_mean` is the average top-1 raw AUGRC regret on the cross-arch test set (lower = better; a value of 0 means the predictor always picked the oracle). 95% CIs come from the 2000-resample bootstrap. `set_regret_raw_mean` is the set-regret for binary heads (min raw_augrc within predicted set − oracle). `set_size_mean` is paired so we can read precision-vs-recall trade-offs. `empty_set_share` flags rows where the predicted competitive set was empty (binary head outputs all p < 0.5 on the side restriction).

## Aggregate — `lopo` (4 folds pooled)

```
     predictor           label_rule regime    side    n  top1_regret_raw_mean  set_regret_raw_mean  set_size_mean  spearman_rho_mean
    multilabel               clique    far     all  670                   NaN                2.684          3.442                NaN
    multilabel  within_eps_majority    far     all  670                   NaN                1.111          4.903                NaN
    multilabel      within_eps_rank    far     all  670                   NaN                2.193          3.472                NaN
    multilabel       within_eps_raw    far     all  670                   NaN                2.541          3.640                NaN
    multilabel within_eps_unanimous    far     all  670                   NaN                3.802          1.745                NaN
per_csf_binary               clique    far     all  670                   NaN                3.022          2.716                NaN
per_csf_binary  within_eps_majority    far     all  670                   NaN                1.693          4.446                NaN
per_csf_binary      within_eps_rank    far     all  670                   NaN                3.036          3.510                NaN
per_csf_binary       within_eps_raw    far     all  670                   NaN                3.030          3.764                NaN
per_csf_binary within_eps_unanimous    far     all  670                   NaN                4.185          1.851                NaN
    regression                         far     all  670                 6.871                  NaN            NaN              0.346
    multilabel               clique    far feature  670                   NaN                5.183          1.590                NaN
    multilabel  within_eps_majority    far feature  670                   NaN                1.492          2.619                NaN
    multilabel      within_eps_rank    far feature  670                   NaN                3.141          1.555                NaN
    multilabel       within_eps_raw    far feature  670                   NaN                3.252          1.804                NaN
    multilabel within_eps_unanimous    far feature  670                   NaN                4.475          0.673                NaN
per_csf_binary               clique    far feature  670                   NaN                5.684          1.191                NaN
per_csf_binary  within_eps_majority    far feature  670                   NaN                2.735          2.360                NaN
per_csf_binary      within_eps_rank    far feature  670                   NaN                3.859          1.710                NaN
per_csf_binary       within_eps_raw    far feature  670                   NaN                4.206          1.916                NaN
per_csf_binary within_eps_unanimous    far feature  670                   NaN                5.721          0.672                NaN
    regression                         far feature  670                 6.489                  NaN            NaN              0.433
    multilabel               clique    far    head  670                   NaN                3.150          1.852                NaN
    multilabel  within_eps_majority    far    head  670                   NaN                3.461          2.284                NaN
    multilabel      within_eps_rank    far    head  670                   NaN                3.636          1.916                NaN
    multilabel       within_eps_raw    far    head  670                   NaN                3.796          1.836                NaN
    multilabel within_eps_unanimous    far    head  670                   NaN                4.616          1.072                NaN
per_csf_binary               clique    far    head  670                   NaN                3.479          1.525                NaN
per_csf_binary  within_eps_majority    far    head  670                   NaN                2.111          2.087                NaN
per_csf_binary      within_eps_rank    far    head  670                   NaN                4.249          1.800                NaN
per_csf_binary       within_eps_raw    far    head  670                   NaN                3.864          1.848                NaN
per_csf_binary within_eps_unanimous    far    head  670                   NaN                4.031          1.179                NaN
    regression                         far    head  670                 5.707                  NaN            NaN              0.262
    multilabel               clique    mid     all 1340                   NaN                2.734          2.879                NaN
    multilabel  within_eps_majority    mid     all 1340                   NaN                1.422          5.448                NaN
    multilabel      within_eps_rank    mid     all 1340                   NaN                2.410          3.958                NaN
    multilabel       within_eps_raw    mid     all 1340                   NaN                2.386          3.933                NaN
    multilabel within_eps_unanimous    mid     all 1340                   NaN                4.700          1.363                NaN
per_csf_binary               clique    mid     all 1340                   NaN                3.496          2.457                NaN
per_csf_binary  within_eps_majority    mid     all 1340                   NaN                1.745          4.960                NaN
per_csf_binary      within_eps_rank    mid     all 1340                   NaN                2.820          3.724                NaN
per_csf_binary       within_eps_raw    mid     all 1340                   NaN                2.466          3.875                NaN
per_csf_binary within_eps_unanimous    mid     all 1340                   NaN                4.773          1.372                NaN
    regression                         mid     all 1340                 7.291                  NaN            NaN              0.421
    multilabel               clique    mid feature 1340                   NaN                5.330          1.363                NaN
    multilabel  within_eps_majority    mid feature 1340                   NaN                1.574          2.890                NaN
    multilabel      within_eps_rank    mid feature 1340                   NaN                3.418          1.770                NaN
    multilabel       within_eps_raw    mid feature 1340                   NaN                3.373          1.796                NaN
    multilabel within_eps_unanimous    mid feature 1340                   NaN                4.729          0.507                NaN
per_csf_binary               clique    mid feature 1340                   NaN                7.193          1.094                NaN
per_csf_binary  within_eps_majority    mid feature 1340                   NaN                2.472          2.503                NaN
per_csf_binary      within_eps_rank    mid feature 1340                   NaN                3.779          1.813                NaN
per_csf_binary       within_eps_raw    mid feature 1340                   NaN                3.618          1.869                NaN
per_csf_binary within_eps_unanimous    mid feature 1340                   NaN                7.362          0.549                NaN
    regression                         mid feature 1340                 6.479                  NaN            NaN              0.540
    multilabel               clique    mid    head 1340                   NaN                2.772          1.516                NaN
    multilabel  within_eps_majority    mid    head 1340                   NaN                1.801          2.558                NaN
    multilabel      within_eps_rank    mid    head 1340                   NaN                1.828          2.188                NaN
    multilabel       within_eps_raw    mid    head 1340                   NaN                1.797          2.137                NaN
    multilabel within_eps_unanimous    mid    head 1340                   NaN                3.035          0.855                NaN
per_csf_binary               clique    mid    head 1340                   NaN                2.789          1.363                NaN
per_csf_binary  within_eps_majority    mid    head 1340                   NaN                2.610          2.457                NaN
per_csf_binary      within_eps_rank    mid    head 1340                   NaN                1.859          1.910                NaN
per_csf_binary       within_eps_raw    mid    head 1340                   NaN                1.796          2.006                NaN
per_csf_binary within_eps_unanimous    mid    head 1340                   NaN                3.247          0.822                NaN
    regression                         mid    head 1340                 7.302                  NaN            NaN              0.295
    multilabel               clique   near     all  998                   NaN                1.758          3.498                NaN
    multilabel  within_eps_majority   near     all  998                   NaN                0.761          5.491                NaN
    multilabel      within_eps_rank   near     all  998                   NaN                1.259          3.692                NaN
    multilabel       within_eps_raw   near     all  998                   NaN                1.697          3.764                NaN
    multilabel within_eps_unanimous   near     all  998                   NaN                2.055          1.880                NaN
per_csf_binary               clique   near     all  998                   NaN                2.080          2.693                NaN
per_csf_binary  within_eps_majority   near     all  998                   NaN                1.003          4.563                NaN
per_csf_binary      within_eps_rank   near     all  998                   NaN                4.245          3.732                NaN
per_csf_binary       within_eps_raw   near     all  998                   NaN                2.156          3.867                NaN
per_csf_binary within_eps_unanimous   near     all  998                   NaN                2.470          1.598                NaN
    regression                        near     all  998                 4.818                  NaN            NaN              0.531
    multilabel               clique   near feature  998                   NaN                3.317          1.308                NaN
    multilabel  within_eps_majority   near feature  998                   NaN                0.762          2.475                NaN
    multilabel      within_eps_rank   near feature  998                   NaN                2.655          1.324                NaN
    multilabel       within_eps_raw   near feature  998                   NaN                1.999          1.687                NaN
    multilabel within_eps_unanimous   near feature  998                   NaN                2.375          0.692                NaN
per_csf_binary               clique   near feature  998                   NaN                4.720          0.946                NaN
per_csf_binary  within_eps_majority   near feature  998                   NaN                1.705          2.253                NaN
per_csf_binary      within_eps_rank   near feature  998                   NaN                6.740          1.502                NaN
per_csf_binary       within_eps_raw   near feature  998                   NaN                3.235          1.728                NaN
per_csf_binary within_eps_unanimous   near feature  998                   NaN                3.356          0.469                NaN
    regression                        near feature  998                 4.079                  NaN            NaN              0.684
    multilabel               clique   near    head  998                   NaN                2.375          2.190                NaN
    multilabel  within_eps_majority   near    head  998                   NaN                1.557          3.016                NaN
    multilabel      within_eps_rank   near    head  998                   NaN                1.771          2.369                NaN
    multilabel       within_eps_raw   near    head  998                   NaN                1.923          2.076                NaN
    multilabel within_eps_unanimous   near    head  998                   NaN                2.571          1.187                NaN
per_csf_binary               clique   near    head  998                   NaN                2.461          1.747                NaN
per_csf_binary  within_eps_majority   near    head  998                   NaN                2.065          2.311                NaN
per_csf_binary      within_eps_rank   near    head  998                   NaN                2.214          2.230                NaN
per_csf_binary       within_eps_raw   near    head  998                   NaN                2.497          2.138                NaN
per_csf_binary within_eps_unanimous   near    head  998                   NaN                2.435          1.129                NaN
    regression                        near    head  998                 4.897                  NaN            NaN              0.356
```

