# Crossing reconciliation (R1; frozen equalities per claim contract 2026-08-24)

Checkpoints: 280; (checkpoint, OOD set) rows: 2240; fine grid 301 points over [-1.413, 1.557]; B = 2000; seeds A=11, B=0; identical severity, grid, clusters, tie definition.

| estimator | first up-crossing | all crossings | sign changes | band q95 | zero-set span | zero-set segments | bounded right | g(d_min) | g(d_max) | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| A_per_score_isotonic | -1.089 | [-1.089] | 1 | 2.471 | [-1.116, -1.057] | [[-1.116, -1.057]] | True | -6.17 | 2.47 | 1.4 |
| B_direct_gap_isotonic | -1.08 | [-1.08] | 1 | 4.924 | [-1.156, 1.557] | [[-1.156, 1.557]] | False | -14.16 | 3.65 | 0.8 |

Point-estimate agreement: |d_A - d_B| = 0.009

Historical note: the manuscript's narrow interval (-1.097, CI [-1.116, -1.057]) came from `documentation/x6_spectral_scripts/x4_real_fit.py` at B=300 on a z-unit severity axis; it is superseded by the table above and may only be cited as a shape-constrained, conditional interval next to the simultaneous zero set.
