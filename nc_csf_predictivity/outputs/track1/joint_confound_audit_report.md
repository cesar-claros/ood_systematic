# Post-hoc source-, training-paradigm-, and training-hyperparameter-confounding sensitivity

Audit #7 (2026-08-26) sections 3.4-3.8. POST HOC: specified after Stage-2 outcomes were inspected; the frozen specification is in joint_confound_audit.py's header and was not tuned afterward. The frozen Stage-2 results remain the evidence of record.

## Section 3.4: support and positivity

| source/paradigm | ckpt | mat | frac+ | 1-class | sev | do | var_collapse 5/50/95 |
|---|---|---|---|---|---|---|---|
| cifar10/confidnet | 10 | 29 | 0.0 | Y | 8 | 0.5 | [0.0096, 0.0124, 0.0163] |
| cifar10/devries | 10 | 36 | 0.0 | Y | 8 | 0.5 | [0.008, 0.0122, 0.0177] |
| cifar10/dg | 40 | 105 | 0.181 |  | 8 | 0.5 | [0.0061, 0.0092, 0.0155] |
| cifar100/confidnet | 10 | 23 | 1.0 | Y | 8 | 0.5 | [0.0075, 0.0086, 0.0097] |
| cifar100/devries | 10 | 26 | 1.0 | Y | 8 | 0.5 | [0.0064, 0.0081, 0.0109] |
| cifar100/dg | 50 | 41 | 0.951 |  | 8 | 0.5 | [0.0067, 0.0085, 0.053] |
| supercifar100/confidnet | 10 | 32 | 0.656 |  | 8 | 0.5 | [0.011, 0.0128, 0.0165] |
| supercifar100/devries | 10 | 29 | 0.655 |  | 8 | 0.5 | [0.0082, 0.0139, 0.0255] |
| supercifar100/dg | 70 | 253 | 0.553 |  | 8 | 0.5 | [0.008, 0.0257, 1.3089] |
| tinyimagenet/confidnet | 10 | 39 | 1.0 | Y | 8 | 0.5 | [0.002, 0.0061, 0.0105] |
| tinyimagenet/devries | 10 | 26 | 1.0 | Y | 8 | 0.5 | [0.0023, 0.0062, 0.0104] |
| tinyimagenet/dg | 40 | 79 | 1.0 | Y | 8 | 0.5 | [0.0019, 0.0051, 0.0097] |

DG reward strata (source/reward: ckpt, material, frac+, median var_collapse):
- cifar10/rew2.2: 10 ckpt, 41 mat, frac+ 0.463, vc 0.0077
- cifar10/rew3: 10 ckpt, 17 mat, frac+ 0.0, vc 0.0098
- cifar10/rew6: 10 ckpt, 23 mat, frac+ 0.0, vc 0.0122
- cifar10/rew10: 10 ckpt, 24 mat, frac+ 0.0, vc 0.0119
- cifar100/rew6: 10 ckpt, 11 mat, frac+ 1.0, vc 0.0098
- cifar100/rew10: 10 ckpt, 3 mat, frac+ 1.0, vc 0.0221
- cifar100/rew12: 10 ckpt, 11 mat, frac+ 1.0, vc 0.0227
- cifar100/rew15: 10 ckpt, 8 mat, frac+ 0.875, vc 0.0085
- cifar100/rew20: 10 ckpt, 8 mat, frac+ 0.875, vc 0.008
- supercifar100/rew2.2: 10 ckpt, 30 mat, frac+ 0.9, vc 0.3333
- supercifar100/rew3: 10 ckpt, 53 mat, frac+ 0.585, vc 0.0181
- supercifar100/rew6: 10 ckpt, 26 mat, frac+ 0.692, vc 0.0169
- supercifar100/rew10: 10 ckpt, 30 mat, frac+ 0.667, vc 0.0176
- supercifar100/rew12: 10 ckpt, 38 mat, frac+ 0.184, vc 0.1231
- supercifar100/rew15: 10 ckpt, 42 mat, frac+ 0.333, vc 0.1258
- supercifar100/rew20: 10 ckpt, 34 mat, frac+ 0.676, vc 0.0124
- tinyimagenet/rew10: 10 ckpt, 7 mat, frac+ 1.0, vc 0.0051
- tinyimagenet/rew12: 10 ckpt, 15 mat, frac+ 1.0, vc 0.0052
- tinyimagenet/rew15: 10 ckpt, 20 mat, frac+ 1.0, vc 0.0052
- tinyimagenet/rew20: 10 ckpt, 37 mat, frac+ 1.0, vc 0.0057

Paradigm contrasts within source (per G feature: standardized mean difference / range overlap):
- cifar10:confidnet-vs-devries: var_collapse: 0.05/0.69, self_duality: 0.68/0.5, equinorm_uc: 0.2/0.81, max_equiangular_wc: 0.25/0.78
- cifar10:confidnet-vs-dg: var_collapse: 0.74/0.61, self_duality: 0.45/0.7, equinorm_uc: 0.16/0.78, max_equiangular_wc: 0.76/0.55
- cifar10:devries-vs-dg: var_collapse: 0.55/0.66, self_duality: -0.16/0.55, equinorm_uc: -0.05/0.73, max_equiangular_wc: 0.58/0.52
- cifar100:confidnet-vs-devries: var_collapse: 0.15/0.5, self_duality: 1.27/0.25, equinorm_uc: -0.01/0.66, max_equiangular_wc: -0.12/0.88
- cifar100:confidnet-vs-dg: var_collapse: -0.73/0.04, self_duality: 0.39/0.15, equinorm_uc: -0.49/0.08, max_equiangular_wc: -0.35/0.2
- cifar100:devries-vs-dg: var_collapse: -0.74/0.07, self_duality: -0.42/0.24, equinorm_uc: -0.48/0.13, max_equiangular_wc: -0.29/0.19
- supercifar100:confidnet-vs-devries: var_collapse: -0.47/0.34, self_duality: 0.39/0.67, equinorm_uc: -0.05/0.87, max_equiangular_wc: -0.13/0.88
- supercifar100:confidnet-vs-dg: var_collapse: -0.52/0.0, self_duality: -0.7/0.16, equinorm_uc: -0.92/0.15, max_equiangular_wc: -0.59/0.2
- supercifar100:devries-vs-dg: var_collapse: -0.51/0.01, self_duality: -0.84/0.11, equinorm_uc: -0.89/0.17, max_equiangular_wc: -0.51/0.21
- tinyimagenet:confidnet-vs-devries: var_collapse: -0.02/0.94, self_duality: 0.08/0.93, equinorm_uc: -0.62/0.6, max_equiangular_wc: -0.25/0.78
- tinyimagenet:confidnet-vs-dg: var_collapse: 0.21/0.94, self_duality: 0.36/0.73, equinorm_uc: -1.3/0.18, max_equiangular_wc: 0.21/0.84
- tinyimagenet:devries-vs-dg: var_collapse: 0.24/0.92, self_duality: 0.29/0.78, equinorm_uc: -0.94/0.3, max_equiangular_wc: 0.46/0.76

## Sections 3.5-3.6: joint models (out-of-fold, material cells)

n material 718, frac positive 0.6, material cells on out-of-support checkpoints 0.323

| model | bal macro | bal macro 2-class | bal row | sign macro | sign row |
|---|---|---|---|---|---|
| M0 | 0.859 | 0.669 | 0.797 | 0.919 | 0.829 |
| M0plus | 0.947 | 0.873 | 0.919 | 0.972 | 0.919 |
| M1 | 0.954 | 0.89 | 0.921 | 0.975 | 0.929 |
| MH | 0.946 | 0.869 | 0.911 | 0.966 | 0.919 |
| MHG | 0.952 | 0.885 | 0.915 | 0.972 | 0.923 |
| sev_pooled | 0.719 | 0.664 | 0.608 | 0.742 | 0.684 |
| sev_source | 0.875 | 0.714 | 0.801 | 0.917 | 0.83 |

| comparison | bal macro (CI95) | bal row (CI95) | sign row (CI95) |
|---|---|---|---|
| M0plus-M0 | +0.089 [0.057, 0.107] | +0.121 [0.078, 0.166] | +0.091 [0.048, 0.136] |
| M1-M0plus | +0.007 [-0.001, 0.017] | +0.002 [-0.019, 0.023] | +0.010 [-0.009, 0.027] |
| M1-M0 | +0.096 [0.065, 0.113] | +0.124 [0.084, 0.167] | +0.100 [0.064, 0.141] |
| MH-M0plus | -0.002 [-0.016, 0.012] | -0.008 [-0.036, 0.019] | +0.000 [-0.024, 0.023] |
| MHG-MH | +0.007 [0.0, 0.015] | +0.003 [-0.004, 0.012] | +0.004 [-0.003, 0.011] |
| M1-MH | +0.009 [-0.005, 0.025] | +0.010 [-0.009, 0.03] | +0.010 [-0.006, 0.028] |
| M0plus-sev_source | +0.073 [0.03, 0.107] | +0.118 [0.071, 0.162] | +0.089 [0.048, 0.134] |

Per source (balanced accuracy):

| group | n mat | frac+ | M0+ | M1 | MH | MHG |
|---|---|---|---|---|---|---|
| cifar10 | 170 | 0.112 | 0.921 | 1.0 | 0.921 | 0.947 |
| cifar100 | 90 | 0.978 | 0.744 | 0.744 | 0.744 | 0.744 |
| supercifar100 | 314 | 0.573 | 0.831 | 0.832 | 0.818 | 0.824 |
| tinyimagenet | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Per paradigm (balanced accuracy):

| group | n mat | frac+ | M0+ | M1 | MH | MHG |
|---|---|---|---|---|---|---|
| confidnet | 123 | 0.675 | 0.975 | 0.975 | 1.0 | 1.0 |
| devries | 117 | 0.607 | 1.0 | 1.0 | 0.972 | 0.986 |
| dg | 478 | 0.579 | 0.884 | 0.89 | 0.877 | 0.879 |

Influence (M1 - M0+ after dropping each group; macro / row):
- drop_cifar100: +0.009 / +0.007
- drop_cifar10: +0.001 / -0.020
- drop_supercifar100: +0.009 / +0.006
- drop_tinyimagenet: +0.009 / +0.012
- drop_confidnet: +0.011 / +0.004
- drop_devries: +0.011 / +0.003
- drop_dg: +0.000 / +0.000

## Section 3.7: paradigm-stratified crossings

### confidnet: NOT-RETAINED
- ckpt 40, vc range [0.002, 0.017], severities 32, material 123, frac+ 0.675
- strong: crossing -1.142, tie [-1.205, -1.077], ckpt 13
- middle: crossing <=min(censored), tie [-1.413, -1.097], ckpt 13
- weak: crossing 0.117, tie [-0.928, 1.557], ckpt 14
- pooled first up-crossing -1.09, tie [-1.295, 1.557]

### devries: RETAINED
- ckpt 40, vc range [0.0023, 0.0258], severities 32, material 117, frac+ 0.607
- strong: crossing -1.178, tie [-1.275, -1.126], ckpt 13
- middle: crossing -1.186, tie [-1.196, 0.121], ckpt 13
- weak: crossing None, tie [-1.077, 1.557], ckpt 14
- pooled first up-crossing -1.07, tie [-1.285, 1.557]

### dg: RETAINED
- ckpt 200, vc range [0.0013, 2.3971], severities 32, material 478, frac+ 0.579
- strong: crossing -1.212, tie [-1.413, -1.196], ckpt 66
- middle: crossing -1.184, tie [-1.314, 1.557], ckpt 67
- weak: crossing None, tie [0.111, 1.557], ckpt 67
- pooled first up-crossing -1.08, tie [-1.156, 1.557]

Equal-paradigm-weighted pooled curve: first up-crossing -1.081, tie region [-1.215, 1.557] (band q95 6.414).

Saturation by paradigm (frozen theory cache):

| paradigm | cells | mat | sign acc | zero margin | both>0.99 | gamma*a 5/50/95 |
|---|---|---|---|---|---|---|
| confidnet | 320 | 123 | 0.041 | 0.966 | 1.0 | [0.13, 0.267, 0.467] |
| devries | 320 | 117 | 0.077 | 0.938 | 0.997 | [0.14, 0.277, 0.483] |
| dg | 1600 | 478 | 0.119 | 0.744 | 0.838 | [0.129, 0.291, 0.579] |

## Section 3.8: leave-one-paradigm-out transport

| held-out paradigm | n mat | frac+ | sev bal | geo bal | G-S bal | sev sign | geo sign | out-of-support |
|---|---|---|---|---|---|---|---|---|
| confidnet | 123 | 0.675 | 0.625 | 0.815 | +0.190 | 0.756 | 0.837 | 0.225 |
| devries | 117 | 0.607 | 0.609 | 0.663 | +0.054 | 0.692 | 0.735 | 0.275 |
| dg | 478 | 0.579 | 0.604 | 0.787 | +0.184 | 0.663 | 0.803 | 0.785 |

Declared transport rule: SUCCEEDS (three results reported individually; never a paradigm population).
