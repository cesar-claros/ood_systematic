# E1 Forensic Reanalysis (post hoc, design data only)

Scale: L = 1 - AUROC_f. Nothing here amends the registered Pilot 1 / Pilot 2 verdicts (evaluation doc section 5.3).

## A. A1-only E1 (5.1)

| group | seed effects | mean [95% CI] | t p (one-sided, sensitivity) | sign test p | cells agree |
|---|---|---|---|---|---|
| A1- | +0.0329, +0.0013, -0.0341, +0.0258 | +0.0064 [-0.0417, +0.0546] | 0.3493 | 0.3125 | 4/5 |
| A1+ | +0.0727, +0.0340, -0.0281, +0.0255 | +0.0260 [-0.0400, +0.0921] | 0.1493 | 0.3125 | 3/4 |
| A1++ | +0.0306, +0.0145, +0.0109, +0.0104 | +0.0166 [+0.0014, +0.0317] | 0.0199 | 0.0625 | 1/2 |
| A2 | +0.1397, +0.0971, +0.1519, +0.1276 | +0.1290 [+0.0916, +0.1665] | 0.0008 | 0.0625 | 8/8 |
| A1_pooled | +0.0470, +0.0156, -0.0237, +0.0229 | +0.0154 [-0.0313, +0.0621] | 0.1853 | 0.3125 | 8/11 |

A2 is listed for reference only and is never pooled with A1. The sign-test floor at four seeds is 0.0625.

## B. MLS / Mahalanobis decomposition (5.2)

| arm | mean |dL_MLS| | mean |dL_Maha| | Maha share | reading |
|---|---|---|---|---|
| A1- | 0.0151 | 0.0236 | 0.61 | feature/covariance channel dominates |
| A1+ | 0.0162 | 0.0254 | 0.61 | feature/covariance channel dominates |
| A1++ | 0.0170 | 0.0180 | 0.51 | feature/covariance channel dominates |
| A2 | 0.0178 | 0.1282 | 0.88 | feature/covariance channel dominates |

## C. Paired-response transport (5.3, POST HOC)

| arm | sign agreement (raw) | MAE raw | MAE no-change |
|---|---|---|---|
| A1- | 0.750 | 0.0226 | 0.0275 |
| A1+ | 0.594 | 0.0278 | 0.0290 |
| A1++ | 0.469 | 0.0225 | 0.0221 |
| A2 | 1.000 | 0.2064 | 0.1290 |

- A1-fitted -> A2 (alpha +0.0091, beta +1.586): calibrated_plugin 0.3918; raw_plugin 0.2064; no_change 0.1290; response_cell_mean 0.1419; delta_nuisance 0.2726; calibrated sign agreement 1.000
- Within-A1 leave-one-dose-out (on-support preview):
  - hold out A1-: MAE 0.0293 vs no-change 0.0275, sign 0.531
  - hold out A1+: MAE 0.0253 vs no-change 0.0290, sign 0.688
  - hold out A1++: MAE 0.0213 vs no-change 0.0221, sign 0.688

## D. Amplitude attribution (5.4, artifact-computable parts)

| group | component | mean obs dL | mean pred dL | MAE pred |
|---|---|---|---|---|
| A1 | MLS | +0.0026 | -0.0005 | 0.0148 |
| A1 | Maha | -0.0103 | -0.0029 | 0.0173 |
| A2 | MLS | -0.0009 | +0.0001 | 0.0163 |
| A2 | Maha | +0.1282 | +0.3343 | 0.2073 |
- A1: Mahalanobis share of prediction error 0.54
- A2: Mahalanobis share of prediction error 0.93
