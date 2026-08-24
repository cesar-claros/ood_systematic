# B-axis dose-search report (geometry only)

Reference: baseline var_collapse seed SD 0.00147; A1++ target var_collapse 0.1115; support threshold (LOO) 1.70.

## varreg

| lam | seeds | d_acc (pp) | d_var_collapse (SDs) | var_collapse | match to A1++ | support dist | gates | verdict |
|---|---|---|---|---|---|---|---|---|
| 0.003 | 2 | +4.20 | +0.5129 (349) | 0.6307 | 0.5191 | 54.33 | FAIL GB1,GB2,GB3 | rejected |
| 0.01 | 2 | +0.20 | -0.0045 (3) | 0.1133 | 0.0017 | 1.48 | FAIL GB2,GB3 | rejected |
| 0.03 | 2 | -1.05 | -0.0117 (8) | 0.1060 | 0.0055 | 2.53 | FAIL GB3 | rejected |
| 0.1 | 2 | -1.25 | -0.0290 (20) | 0.0887 | 0.0228 | 5.65 | FAIL GB3 | rejected |
| 0.3 | 2 | -0.85 | -0.0493 (34) | 0.0685 | 0.0431 | 8.50 | FAIL GB3 | rejected |
| 1.0 | 2 | -1.55 | -0.0693 (47) | 0.0485 | 0.0630 | 13.01 | FAIL GB3 | rejected |

Recommended varreg dose: NONE (no dose qualifies)

## ctrreg

| lam | seeds | d_acc (pp) | d_var_collapse (SDs) | var_collapse | match to A1++ | support dist | gates | verdict |
|---|---|---|---|---|---|---|---|---|
| 0.00003 | 2 | -1.75 | -0.0014 (1) | 0.1163 | 0.0048 | 0.75 | FAIL GB2 | rejected |
| 0.0001 | 2 | +0.00 | -0.0057 (4) | 0.1121 | 0.0005 | 1.81 | FAIL GB2,GB3 | rejected |
| 0.0003 | 2 | -1.45 | -0.0142 (10) | 0.1035 | 0.0080 | 3.43 | FAIL GB3 | rejected |
| 0.003 | 2 | -0.80 | -0.0520 (35) | 0.0658 | 0.0458 | 9.03 | FAIL GB3 | rejected |
| 0.01 | 2 | -2.35 | -0.0684 (47) | 0.0494 | 0.0622 | 11.69 | FAIL GB3 | rejected |
| 0.03 | 2 | -1.55 | -0.0768 (52) | 0.0409 | 0.0706 | 16.34 | FAIL GB3 | rejected |
| 0.1 | 2 | -1.10 | -0.0616 (42) | 0.0562 | 0.0554 | 22.66 | FAIL GB3 | rejected |

Recommended ctrreg dose: NONE (no dose qualifies)

**Overall geometry-matched pick: NONE** (closest qualifying var_collapse to the A1++ level; protocol section 6).

## GB3 sensitivity to span expansion (closure audit R3; frozen gate = 25%)

| mechanism | lam | GB3@0.1 | GB3@0.25 | GB3@0.5 |
|---|---|---|---|---|
| varreg | 0.003 | fail | fail | fail |
| varreg | 0.01 | fail | fail | pass |
| varreg | 0.03 | fail | fail | fail |
| varreg | 0.1 | fail | fail | fail |
| varreg | 0.3 | fail | fail | fail |
| varreg | 1.0 | fail | fail | fail |
| ctrreg | 0.00003 | pass | pass | pass |
| ctrreg | 0.0001 | fail | fail | fail |
| ctrreg | 0.0003 | fail | fail | fail |
| ctrreg | 0.003 | fail | fail | fail |
| ctrreg | 0.01 | fail | fail | fail |
| ctrreg | 0.03 | fail | fail | fail |
| ctrreg | 0.1 | fail | fail | fail |

## Per-seed trajectories (closure audit R3)

| mechanism | lam | run | var_collapse | eig_max_over_mean | logit_scale | d_acc (pp) | support dist |
|---|---|---|---|---|---|---|---|
| varreg | 0.003 | 1 | 1.1465 | 23.92 | 8.202 | +9.00 | 107.82 |
| varreg | 0.003 | 2 | 0.1148 | 20.91 | 9.814 | -0.60 | 0.84 |
| varreg | 0.01 | 1 | 0.1134 | 20.19 | 9.803 | +0.30 | 1.39 |
| varreg | 0.01 | 2 | 0.1131 | 20.03 | 9.826 | +0.10 | 1.57 |
| varreg | 0.03 | 1 | 0.1080 | 18.96 | 9.767 | -1.50 | 2.50 |
| varreg | 0.03 | 2 | 0.1041 | 19.11 | 9.735 | -0.60 | 2.57 |
| varreg | 0.1 | 1 | 0.0895 | 16.36 | 9.588 | -1.50 | 5.52 |
| varreg | 0.1 | 2 | 0.0880 | 16.01 | 9.622 | -1.00 | 5.77 |
| varreg | 0.3 | 1 | 0.0690 | 14.63 | 9.310 | -1.30 | 8.64 |
| varreg | 0.3 | 2 | 0.0680 | 15.28 | 9.306 | -0.40 | 8.36 |
| varreg | 1.0 | 1 | 0.0487 | 16.65 | 8.561 | -2.60 | 13.14 |
| varreg | 1.0 | 2 | 0.0483 | 17.47 | 8.571 | -0.50 | 12.89 |
| ctrreg | 0.00003 | 1 | 0.1170 | 21.00 | 9.803 | -2.60 | 0.73 |
| ctrreg | 0.00003 | 2 | 0.1156 | 20.86 | 9.755 | -0.90 | 0.77 |
| ctrreg | 0.0001 | 1 | 0.1130 | 19.72 | 9.786 | -0.40 | 1.75 |
| ctrreg | 0.0001 | 2 | 0.1112 | 19.63 | 9.779 | +0.40 | 1.87 |
| ctrreg | 0.0003 | 1 | 0.1038 | 17.86 | 9.720 | -1.40 | 3.54 |
| ctrreg | 0.0003 | 2 | 0.1033 | 18.13 | 9.746 | -1.50 | 3.33 |
| ctrreg | 0.003 | 1 | 0.0656 | 14.21 | 9.276 | -0.50 | 9.22 |
| ctrreg | 0.003 | 2 | 0.0659 | 14.53 | 9.322 | -1.10 | 8.85 |
| ctrreg | 0.01 | 1 | 0.0492 | 16.41 | 8.794 | -3.00 | 11.70 |
| ctrreg | 0.01 | 2 | 0.0495 | 16.49 | 8.788 | -1.70 | 11.69 |
| ctrreg | 0.03 | 1 | 0.0408 | 18.54 | 8.136 | -1.50 | 16.16 |
| ctrreg | 0.03 | 2 | 0.0411 | 18.34 | 8.088 | -1.60 | 16.52 |
| ctrreg | 0.1 | 1 | 0.0560 | 20.20 | 7.213 | -1.60 | 22.64 |
| ctrreg | 0.1 | 2 | 0.0563 | 22.32 | 7.203 | -0.60 | 22.68 |
