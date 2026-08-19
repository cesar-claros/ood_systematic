# Pilot 1 Manipulation Report (blinded stage)

Runs measured: 20 (final checkpoints). No OOD or detector information enters this report.

- **M1 manipulation**: strength PASS (max standardized move 1036.79 baseline SDs); monotone FAIL (Spearman(dose, self_duality) = +0.000, gate <= -0.8) => **FAIL**
- **M1' (amended, manifest addendum A; A1 dial only)**: **PASS** (Spearman -0.970, gate <= -0.8)
- **A2 relabeled (measured geometry)**: the fixed-ETF arm's measured self-duality lands ABOVE baseline, refuting the alignment-endpoint premise; A2 is a distinct joint mechanism, its E1 direction is generated from measured geometry via the frozen plug-in (addendum A), and it does not enter the dose-ordering gate.
- **M2 accuracy**: **PASS** (median val-acc drop vs baseline, pp: A1- -0.70, A1+ -0.75, A1++ -0.70, A2 +1.30)

## Self-duality and accuracy by arm

| arm | self_duality mean (sd) | std move vs baseline | exits benchmark span [0.03, 0.13] | val acc median |
|---|---|---|---|---|
| A1- | 0.0495 (0.0009) | 97.74 | no | 0.7560 |
| baseline | 0.0142 (0.0004) | - | - | 0.7475 |
| A1+ | 0.0041 (0.0001) | 28.12 | yes | 0.7565 |
| A1++ | 0.0027 (0.0002) | 31.95 | yes | 0.7560 |
| A2 | 0.3887 (0.0010) | 1036.79 | yes | 0.7360 |

## Selectivity (reported, not gated; ratio > 0.25 flags joint-intervention relabeling)

- A1-: equinorm_wc (0.76x), max_equiangular_wc (0.61x), equiangular_wc (0.60x)
- A1+: max_equiangular_wc (2.34x), equiangular_wc (2.17x), equinorm_wc (1.65x)
- A1++: max_equiangular_wc (2.94x), equiangular_wc (2.49x), equinorm_wc (1.42x), max_equiangular_uc (0.51x)
- A2: max_equiangular_wc (0.58x), var_collapse (0.55x), equiangular_wc (0.52x)

## Per-run record (final checkpoints)

| experiment | self_dual | var_col | eqn_uc | eqa_wc | head_resid | logit_scale | train acc | val acc |
|---|---|---|---|---|---|---|---|---|
| etfreg_bbvgg13_do0_run1_lam-0.1 | 0.0504 | 0.1209 | 0.0465 | 0.0621 | 0.0209 | 9.70 | 0.9926 | 0.7570 |
| etfreg_bbvgg13_do0_run2_lam-0.1 | 0.0492 | 0.1216 | 0.0383 | 0.0623 | 0.0210 | 9.66 | 0.9923 | 0.7490 |
| etfreg_bbvgg13_do0_run3_lam-0.1 | 0.0501 | 0.1219 | 0.0443 | 0.0621 | 0.0203 | 9.69 | 0.9925 | 0.7630 |
| etfreg_bbvgg13_do0_run4_lam-0.1 | 0.0484 | 0.1216 | 0.0410 | 0.0624 | 0.0204 | 9.70 | 0.9925 | 0.7550 |
| etfreg_bbvgg13_do0_run1_lam0.0 | 0.0147 | 0.1190 | 0.0316 | 0.0698 | 0.0057 | 9.79 | 0.9923 | 0.7430 |
| etfreg_bbvgg13_do0_run2_lam0.0 | 0.0139 | 0.1166 | 0.0272 | 0.0700 | 0.0056 | 9.80 | 0.9920 | 0.7470 |
| etfreg_bbvgg13_do0_run3_lam0.0 | 0.0143 | 0.1198 | 0.0318 | 0.0698 | 0.0054 | 9.79 | 0.9924 | 0.7580 |
| etfreg_bbvgg13_do0_run4_lam0.0 | 0.0140 | 0.1195 | 0.0301 | 0.0699 | 0.0054 | 9.80 | 0.9921 | 0.7480 |
| etfreg_bbvgg13_do0_run1_lam0.3 | 0.0040 | 0.1124 | 0.0247 | 0.0780 | 0.0016 | 9.82 | 0.9924 | 0.7530 |
| etfreg_bbvgg13_do0_run2_lam0.3 | 0.0042 | 0.1115 | 0.0225 | 0.0779 | 0.0017 | 9.83 | 0.9920 | 0.7550 |
| etfreg_bbvgg13_do0_run3_lam0.3 | 0.0039 | 0.1140 | 0.0228 | 0.0776 | 0.0016 | 9.80 | 0.9919 | 0.7580 |
| etfreg_bbvgg13_do0_run4_lam0.3 | 0.0041 | 0.1142 | 0.0226 | 0.0780 | 0.0016 | 9.84 | 0.9915 | 0.7590 |
| etfreg_bbvgg13_do0_run1_lam1.0 | 0.0027 | 0.1123 | 0.0246 | 0.0803 | 0.0008 | 9.76 | 0.9918 | 0.7520 |
| etfreg_bbvgg13_do0_run2_lam1.0 | 0.0028 | 0.1104 | 0.0251 | 0.0801 | 0.0009 | 9.76 | 0.9925 | 0.7600 |
| etfreg_bbvgg13_do0_run3_lam1.0 | 0.0025 | 0.1123 | 0.0225 | 0.0806 | 0.0008 | 9.78 | 0.9913 | 0.7610 |
| etfreg_bbvgg13_do0_run4_lam1.0 | 0.0027 | 0.1112 | 0.0249 | 0.0801 | 0.0008 | 9.75 | 0.9917 | 0.7420 |
| etfhard_bbvgg13_do0_run1_lamhard | 0.3874 | 0.9501 | 0.0490 | 0.0000 | 0.1893 | 13.42 | 0.9956 | 0.7340 |
| etfhard_bbvgg13_do0_run2_lamhard | 0.3889 | 0.9535 | 0.0499 | 0.0000 | 0.1908 | 13.42 | 0.9948 | 0.7380 |
| etfhard_bbvgg13_do0_run3_lamhard | 0.3898 | 0.9548 | 0.0524 | 0.0000 | 0.1915 | 13.49 | 0.9947 | 0.7440 |
| etfhard_bbvgg13_do0_run4_lamhard | 0.3888 | 0.9467 | 0.0509 | 0.0000 | 0.1894 | 13.53 | 0.9945 | 0.7340 |
