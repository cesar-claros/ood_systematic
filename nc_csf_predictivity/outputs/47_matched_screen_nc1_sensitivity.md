# Matched screen: NC1 correctness sensitivity and M1 rescoring (descriptive, 2026-09-15)

columns with corrected/provenance: ['var_collapse_corrected', 'nc1_convention']

| Source | Median legacy NC1 | Median corrected NC1 | Median ratio |
|---|---:|---:|---:|
| cifar10 | 0.00944 | 0.09489 | 9.97 |
| cifar100 | 0.00848 | 0.86397 | 100.21 |
| supercifar100 | 0.01819 | 0.34332 | 19.06 |
| tinyimagenet | 0.00514 | 2.60829 | 259.01 |

## Pooled regret (AUGRC x1e3)

| Policy | Legacy, 20 | Corrected, 20 | Legacy, no Conf | Corrected, no Conf |
|---|---:|---:|---:|---:|
| FIXED | 7.370 | 7.370 | 7.046 | 7.046 |
| M1 | 6.297 | 6.297 | 5.818 | 5.818 |
| M1+G | 10.040 | 13.403 | 9.692 | 13.080 |
| M2 | 8.428 | 8.428 | 8.875 | 8.875 |
| M2+G | 7.301 | 8.828 | 7.888 | 8.504 |
| G | 8.698 | 11.915 | 8.374 | 11.636 |

Corrected NC1, 20 detectors: M1+G on tinyimagenet 30.423; on the grouped pair 8.252; M1 on tinyimagenet 1.440

## M1 versus fixed by source (legacy run; M1 does not use NC1)

| Source | Fixed | M1 | Improvement |
|---|---:|---:|---:|
| cifar10 | 9.310 | 9.310 | 0.000 |
| cifar100 | 4.159 | 5.292 | -1.133 |
| supercifar100 | 10.977 | 8.308 | 2.669 |
| tinyimagenet | 3.764 | 1.440 | 2.324 |

| Weighting | Fixed | M1 | Improvement |
|---|---:|---:|---:|
| rows | 7.370 | 6.297 | 1.073 |
| equal named sources | 7.052 | 6.087 | 0.965 |
| equal image-source groups | 7.023 | 5.913 | 1.110 |
| equal source/paradigm | 7.522 | 6.284 | 1.238 |
- M1 selections on cifar10: CTM 60
- M1 selections on cifar100+supercifar100: CTM 75, ViM 36, Confidence 20, NeCo 19, NNGuide 10
- M1 selections on tinyimagenet: CTM 45, NNGuide 5, MLS 5, GE 5
