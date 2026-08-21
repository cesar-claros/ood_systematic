# B-axis dose-search report (geometry only)

Reference: baseline var_collapse seed SD 0.00147; A1++ target var_collapse 0.1115; support threshold (LOO) 1.70.

## varreg

| lam | seeds | d_acc (pp) | d_var_collapse (SDs) | var_collapse | match to A1++ | support dist | gates | verdict |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 2 | -1.25 | -0.0290 (20) | 0.0887 | 0.0228 | 5.65 | FAIL GB3 | rejected |
| 0.3 | 2 | -0.85 | -0.0493 (34) | 0.0685 | 0.0431 | 8.50 | FAIL GB3 | rejected |
| 1.0 | 2 | -1.55 | -0.0693 (47) | 0.0485 | 0.0630 | 13.01 | FAIL GB3 | rejected |

Recommended varreg dose: NONE (no dose qualifies)

## ctrreg

| lam | seeds | d_acc (pp) | d_var_collapse (SDs) | var_collapse | match to A1++ | support dist | gates | verdict |
|---|---|---|---|---|---|---|---|---|
| 0.003 | 2 | -0.80 | -0.0520 (35) | 0.0658 | 0.0458 | 9.03 | FAIL GB3 | rejected |
| 0.01 | 2 | -2.35 | -0.0684 (47) | 0.0494 | 0.0622 | 11.69 | FAIL GB3 | rejected |
| 0.03 | 2 | -1.55 | -0.0768 (52) | 0.0409 | 0.0706 | 16.34 | FAIL GB3 | rejected |
| 0.1 | 2 | -1.10 | -0.0616 (42) | 0.0562 | 0.0554 | 22.66 | FAIL GB3 | rejected |

Recommended ctrreg dose: NONE (no dose qualifies)

**Overall geometry-matched pick: NONE** (closest qualifying var_collapse to the A1++ level; protocol section 6).
