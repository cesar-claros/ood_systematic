# Hero-figure pair-specific boundary audit (R4)

Frozen rules: m_min = 0.2 (gap units per gamma-a unit), displacement bound tol/m_min = 0.05; boundary level = -tol = -0.01; grid 41x26; MC SE target 0.0025 per score; seed base 777.

## Panel A: Energy vs CTM over (gamma a, s)

- boundary points: 38; well-conditioned (slope >= 0.2): 16; display-sharp after all rules: 15
- max |displacement|: all points 0.009; well-conditioned only 0.004; unresolved brackets 2
- sign agreement on resolvable subgrid pixels: 1.0 (44 resolvable)
- |analytic - MC| gap error on subgrid: p95 0.0024, max 0.0047

| y | gamma_a | slope | well-cond | MC bracket | displacement | sign | sharp |
|---|---|---|---|---|---|---|---|
| 8.0 | 0.853 | 0.108 | False | [-0.0103, -0.015] | None | True | False |
| 8.0 | 1.474 | 0.078 | False | [-0.0115, -0.008] | -0.003 | True | False |
| 8.94 | 0.829 | 0.114 | False | [-0.0071, -0.0128] | -0.004 | True | False |
| 8.94 | 1.468 | 0.088 | False | [-0.0115, -0.0068] | -0.002 | True | False |
| 10.0 | 0.811 | 0.148 | False | [-0.0087, -0.0158] | -0.002 | True | False |
| 10.0 | 1.463 | 0.095 | False | [-0.0111, -0.0065] | -0.001 | True | False |
| 11.18 | 0.797 | 0.119 | False | [-0.0054, -0.0113] | -0.008 | True | False |
| 11.18 | 1.459 | 0.101 | False | [-0.0109, -0.0059] | 0.0 | True | False |
| 12.5 | 0.782 | 0.144 | False | [-0.0063, -0.0133] | -0.005 | True | False |
| 12.5 | 1.46 | 0.109 | False | [-0.0108, -0.0056] | -0.003 | True | False |
| 13.97 | 0.771 | 0.172 | False | [-0.0067, -0.0153] | -0.002 | True | False |
| 13.97 | 1.467 | 0.123 | False | [-0.012, -0.0058] | -0.001 | True | False |
| 15.62 | 0.763 | 0.202 | True | [-0.0073, -0.0179] | 0.0 | True | True |
| 15.62 | 1.481 | 0.148 | False | [-0.0141, -0.0072] | -0.002 | True | False |
| 17.46 | 0.756 | 0.233 | True | [-0.0088, -0.0206] | -0.001 | True | True |
| 17.46 | 1.503 | 0.107 | False | [-0.0101, -0.0049] | -0.003 | True | False |
| 19.52 | 0.752 | 0.263 | True | [-0.0103, -0.0232] | None | True | False |
| 19.52 | 1.545 | 0.167 | False | [-0.0172, -0.0083] | -0.005 | True | False |
| 21.83 | 0.745 | 0.133 | False | [-0.0044, -0.0122] | -0.009 | True | False |
| 21.83 | 1.611 | 0.102 | False | [-0.0109, -0.0058] | -0.003 | True | False |
| 24.4 | 0.737 | 0.148 | False | [-0.0041, -0.0124] | -0.002 | True | False |
| 24.4 | 1.715 | 0.093 | False | [-0.0109, -0.0063] | -0.005 | True | False |
| 27.28 | 0.731 | 0.165 | False | [-0.0048, -0.0127] | 0.002 | True | False |
| 27.28 | 1.887 | 0.087 | False | [-0.0129, -0.0087] | -0.003 | True | False |
| 30.5 | 0.727 | 0.184 | False | [-0.0047, -0.0158] | -0.003 | True | False |
| 34.1 | 0.724 | 0.204 | True | [-0.0054, -0.0152] | -0.001 | True | True |
| 38.12 | 0.724 | 0.221 | True | [-0.004, -0.0166] | 0.0 | True | True |
| 42.62 | 0.724 | 0.232 | True | [-0.0055, -0.0138] | 0.003 | True | True |
| 47.65 | 0.725 | 0.24 | True | [-0.0043, -0.0154] | 0.001 | True | True |
| 53.27 | 0.726 | 0.249 | True | [-0.003, -0.0179] | -0.002 | True | True |
| 59.55 | 0.726 | 0.26 | True | [-0.003, -0.0165] | -0.001 | True | True |
| 66.58 | 0.727 | 0.271 | True | [-0.0031, -0.0145] | 0.004 | True | True |
| 74.43 | 0.727 | 0.283 | True | [-0.0019, -0.0172] | -0.0 | True | True |
| 83.22 | 0.727 | 0.296 | True | [-0.0021, -0.0181] | -0.002 | True | True |
| 93.03 | 0.727 | 0.307 | True | [-0.0017, -0.0173] | -0.0 | True | True |
| 104.01 | 0.727 | 0.319 | True | [-0.0014, -0.0183] | -0.001 | True | True |
| 116.28 | 0.727 | 0.329 | True | [-0.0011, -0.0198] | -0.003 | True | True |
| 130.0 | 0.727 | 0.34 | True | [-0.0008, -0.0157] | 0.004 | True | True |

## Panel B: MLS vs Maha over (gamma a, theta)

- boundary points: 26; well-conditioned (slope >= 0.2): 20; display-sharp after all rules: 20
- max |displacement|: all points 0.008; well-conditioned only 0.008; unresolved brackets 1
- sign agreement on resolvable subgrid pixels: 1.0 (58 resolvable)
- |analytic - MC| gap error on subgrid: p95 0.0024, max 0.0039

| y | gamma_a | slope | well-cond | MC bracket | displacement | sign | sharp |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.726 | 0.217 | True | [-0.0036, -0.0173] | -0.002 | True | True |
| 2.4 | 0.726 | 0.217 | True | [-0.0047, -0.0172] | -0.005 | True | True |
| 4.8 | 0.726 | 0.217 | True | [-0.0054, -0.0157] | -0.003 | True | True |
| 7.2 | 0.725 | 0.218 | True | [-0.0062, -0.0172] | -0.008 | True | True |
| 9.6 | 0.725 | 0.219 | True | [-0.0055, -0.0144] | 0.0 | True | True |
| 12.0 | 0.725 | 0.22 | True | [-0.0051, -0.0178] | -0.005 | True | True |
| 14.4 | 0.724 | 0.221 | True | [-0.0054, -0.0162] | -0.003 | True | True |
| 16.8 | 0.723 | 0.223 | True | [-0.0042, -0.0182] | -0.002 | True | True |
| 19.2 | 0.722 | 0.226 | True | [-0.0058, -0.0147] | 0.001 | True | True |
| 21.6 | 0.721 | 0.229 | True | [-0.0051, -0.0173] | -0.001 | True | True |
| 24.0 | 0.72 | 0.232 | True | [-0.0066, -0.0184] | -0.006 | True | True |
| 26.4 | 0.719 | 0.236 | True | [-0.0053, -0.0176] | 0.0 | True | True |
| 28.8 | 0.717 | 0.241 | True | [-0.0061, -0.0167] | 0.001 | True | True |
| 31.2 | 0.715 | 0.247 | True | [-0.0071, -0.0186] | -0.003 | True | True |
| 33.6 | 0.713 | 0.253 | True | [-0.0068, -0.0181] | 0.001 | True | True |
| 36.0 | 0.711 | 0.261 | True | [-0.0072, -0.0182] | 0.001 | True | True |
| 38.4 | 0.709 | 0.27 | True | [-0.0084, -0.0229] | -0.003 | True | True |
| 40.8 | 0.706 | 0.282 | True | [-0.009, -0.0219] | -0.002 | True | True |
| 43.2 | 0.703 | 0.295 | True | [-0.0093, -0.0248] | -0.001 | True | True |
| 45.6 | 0.699 | 0.133 | False | [-0.0046, -0.01] | None | True | False |
| 48.0 | 0.691 | 0.145 | False | [-0.0044, -0.0108] | 0.003 | True | False |
| 50.4 | 0.682 | 0.159 | False | [-0.0042, -0.0133] | -0.0 | True | False |
| 52.8 | 0.673 | 0.176 | False | [-0.0061, -0.014] | 0.001 | True | False |
| 55.2 | 0.664 | 0.198 | False | [-0.0067, -0.0184] | 0.0 | True | False |
| 57.6 | 0.654 | 0.224 | True | [-0.0094, -0.0196] | -0.001 | True | True |
| 60.0 | 0.639 | 0.128 | False | [-0.0051, -0.0123] | -0.005 | True | False |
