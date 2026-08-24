# ViT and foundation-regime crossing replication (P1; frozen spec in vit_poola_crossing.py)

Existing tables only; predeclared pairs; band bootstrap B = 50; gap = AUGRC_A - AUGRC_B (positive = second score better).

## Fine-tuned ViT (40 cells)

### MLS vs KPCA RecError global  ->  **smeared**

| estimator | sign changes | first up-crossing | tie region | g(d_min) | g(d_max) |
|---|---|---|---|---|---|
| pava | 1 | -0.57 | [-0.76, 1.557] | -11.6 | +6.7 |
| loclin | 3 | -0.686 | None | -0.4 | +4.3 |
| spline | 4 | -0.711 | None | +3.8 | +4.9 |
| piecewise | 10 | -0.57 | None | +1.3 | +0.5 |

- strata crossings: strong: -0.795; middle: -0.931; weak: -0.516
- leave-one-source-out crossings: cifar10: -0.57; cifar100: -0.625; supercifar100: -0.41; tinyimagenet: -0.57
- leave-one-OOD-set-out: 9/9 refits keep a crossing, range [-0.67, -0.54]

### MLS vs Residual  ->  **sharp**

| estimator | sign changes | first up-crossing | tie region | g(d_min) | g(d_max) |
|---|---|---|---|---|---|
| pava | 1 | 1.358 | [-0.037, 1.418] | -42.3 | +25.6 |
| loclin | 2 | 1.41 | None | +6.0 | +25.2 |
| spline | 4 | 0.616 | None | +21.8 | +27.4 |
| piecewise | 14 | -0.536 | None | +0.5 | +9.1 |

- strata crossings: strong: -0.533; middle: 0.367; weak: 1.364
- leave-one-source-out crossings: cifar10: 1.318; cifar100: 1.361; supercifar100: 1.355; tinyimagenet: 0.912
- leave-one-OOD-set-out: 8/9 refits keep a crossing, range [0.60, 1.37]

### Energy vs CTM  ->  **smeared**

| estimator | sign changes | first up-crossing | tie region | g(d_min) | g(d_max) |
|---|---|---|---|---|---|
| pava | 1 | 1.454 | [-1.413, 1.537] | -2.9 | +3.7 |
| loclin | 1 | 1.406 | None | -2.4 | +2.0 |
| spline | 3 | -1.085 | None | -2.5 | +1.8 |
| piecewise | 15 | -1.267 | None | -2.9 | +3.7 |

- strata crossings: strong: -0.412; middle: None; weak: 1.44
- leave-one-source-out crossings: cifar10: 1.456; cifar100: None; supercifar100: 1.403; tinyimagenet: 1.444
- leave-one-OOD-set-out: 8/9 refits keep a crossing, range [1.44, 1.46]

## Frozen probes, Pool A (40 cells)

### MLS vs Maha  ->  **smeared**

| estimator | sign changes | first up-crossing | tie region | g(d_min) | g(d_max) |
|---|---|---|---|---|---|
| pava | 1 | 0.496 | [-1.413, 1.329] | -24.2 | +84.4 |
| loclin | 1 | 0.496 | None | -20.9 | +78.0 |
| spline | 1 | 0.421 | None | -20.2 | +75.8 |
| piecewise | 11 | -1.048 | None | -19.3 | +66.7 |

- strata crossings: strong: 1.328; middle: 0.486; weak: 0.108; probe_dinov2_vitb14: 1.284; probe_clip_vitb16: 0.078; low_residue: 1.284; high_residue: 0.078
- leave-one-source-out crossings: cifar10: 0.499; cifar100: 0.357; supercifar100: 0.498; tinyimagenet: 0.549
- leave-one-OOD-set-out: 9/9 refits keep a crossing, range [0.12, 0.74]
- probe_training_size_sensitivity: NOT AVAILABLE: no probe-size column in the harmonized tables
