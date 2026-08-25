# Stage-2 scientific closure (audit #6 E1-E4; frozen rules in stage2_closure.py docstring)

## E1: leave-one-OOD-set-out

| held-out set | n mat | frac + | theory | severity | geometry | flexible | G-S |
|---|---|---|---|---|---|---|---|
| cifar10 | 58 | 0.38 | 0.000 | 0.983 | 0.931 | 1.000 | -0.052 |
| cifar100 | 52 | 0.42 | 0.000 | 0.577 | 0.904 | 0.596 | +0.327 |
| isun | 78 | 0.72 | 0.128 | 0.718 | 0.821 | 0.808 | +0.103 |
| lsun cropped | 79 | 0.62 | 0.076 | 0.620 | 0.873 | 0.848 | +0.253 |
| lsun resize | 90 | 0.66 | 0.189 | 0.656 | 0.833 | 0.800 | +0.178 |
| places365 | 52 | 0.27 | 0.000 | 0.269 | 0.288 | 0.288 | +0.019 |
| svhn | 101 | 0.81 | 0.248 | 0.812 | 0.703 | 0.762 | -0.109 |
| textures | 128 | 0.69 | 0.008 | 0.688 | 0.695 | 0.703 | +0.008 |
| tinyimagenet | 80 | 0.49 | 0.150 | 0.487 | 0.800 | 0.762 | +0.313 |

Pooled LOO-OOD: theory 0.099, severity 0.660, geometry 0.763, G-S +0.103.
Pooled G-S after dropping each set: cifar10: +0.117, cifar100: +0.086, isun: +0.103, lsun cropped: +0.085, lsun resize: +0.092, places365: +0.110, svhn: +0.138, textures: +0.124, tinyimagenet: +0.077

## E2: Gate 3 (held-out strata ordering)

| held-out source | strong | middle | weak | outcome |
|---|---|---|---|---|
| cifar10 | <=range-min | None | None | RETAINED |
| cifar100 | -0.794 | -0.646 | -0.604 | RETAINED |
| supercifar100 | None | <=range-min | -0.152 | REVERSED |
| tinyimagenet | <=range-min | <=range-min | None | RETAINED |

**Gate 3 verdict: INCONCLUSIVE** (outcomes: ['RETAINED', 'RETAINED', 'REVERSED', 'RETAINED'])

## E3: geometry-vs-severity uncertainty

### ckpt5: n material 718, frac positive 0.6
- sign-acc G-S: +0.091, CI95 [0.051, 0.13]
- balanced-acc G-S: +0.129, CI95 [0.088, 0.174]

| source | n mat | frac + | geometry | severity | geo bal | sev bal |
|---|---|---|---|---|---|---|
| cifar10 | 170 | 0.112 | 0.429 | 0.259 | 0.679 | 0.468 |
| cifar100 | 90 | 0.978 | 0.911 | 0.989 | 0.71 | 0.75 |
| supercifar100 | 314 | 0.573 | 0.818 | 0.682 | 0.812 | 0.628 |
| tinyimagenet | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

G-S after dropping each source: cifar100: +0.115, cifar10: +0.066, supercifar100: +0.054, tinyimagenet: +0.113

### loso: n material 718, frac positive 0.6
- sign-acc G-S: +0.096, CI95 [0.042, 0.151]
- balanced-acc G-S: +0.118, CI95 [0.059, 0.179]

| source | n mat | frac + | geometry | severity | geo bal | sev bal |
|---|---|---|---|---|---|---|
| cifar10 | 170 | 0.112 | 0.182 | 0.259 | 0.54 | 0.468 |
| cifar100 | 90 | 0.978 | 0.878 | 0.989 | 0.693 | 0.75 |
| supercifar100 | 314 | 0.573 | 0.803 | 0.682 | 0.813 | 0.628 |
| tinyimagenet | 144 | 1.0 | 0.993 | 0.618 | 0.993 | 0.618 |

G-S after dropping each source: cifar100: +0.126, cifar10: +0.150, supercifar100: +0.077, tinyimagenet: +0.026

## E4: coordinate-support diagnostic

- cells with theory evaluation: 2240
- gamma*a quantiles (5/50/95) by source: {'cifar10': [0.122, 0.213, 0.418], 'cifar100': [0.131, 0.318, 0.499], 'supercifar100': [0.16, 0.33, 0.659], 'tinyimagenet': [0.09, 0.221, 0.46]}
- rho quantiles by source: {'cifar10': [0.716, 1.438, 2.188], 'cifar100': [0.863, 1.037, 1.377], 'supercifar100': [0.754, 1.216, 1.669], 'tinyimagenet': [0.771, 1.01, 1.243]}
- dictionary SNR quantiles by source: {'cifar10': [21.912, 29.295, 36.4], 'cifar100': [43.079, 107.518, 122.812], 'supercifar100': [3.795, 30.668, 46.02], 'tinyimagenet': [138.378, 224.269, 322.313]}
- analytic winner-margin quantiles (50/90/95/99): [0.0, 0.00976, 0.05308, 0.13106]
- fraction with margin exactly zero: 0.803
- fraction with both AUROCs > 0.99: 0.883
- fraction on the CTM-material side of the displayed boundary (gap <= -0.01): 0.087
- median |observed gap| on material cells: 15.4 (AUGRC x 1000)
- Spearman(|analytic margin|, |observed gap|) on nonzero-margin material cells: 0.087
