# Crossing robustness audit (P0; frozen spec in crossing_robustness_audit.py)

Cells: 280; band bootstrap B = 2000; gap = AUGRC_Energy - AUGRC_CTM (raw AUGRC units); positive gap = CTM better.

## A. Estimators (pooled and by var-collapse tertile)

| estimator | stratum | sign changes | first up-crossing | tie region | g(d_min) | g(d_max) | bracketed |
|---|---|---|---|---|---|---|---|
| pava | pooled | 1 | -1.08 | [-1.156, 1.557] | -14.2 | +3.7 | True |
| pava | strong | 1 | -1.204 | [-1.413, -1.196] | -1.9 | +8.1 | True |
| pava | middle | 1 | -1.189 | [-1.295, 1.557] | -6.3 | +5.3 | True |
| pava | weak | 0 | None | [0.101, 1.557] | -33.2 | -1.4 | True |
| loclin | pooled | 2 | -1.083 | [-1.186, 1.557] | -14.2 | -0.5 | True |
| loclin | strong | 0 | None | [-1.413, 1.557] | +1.1 | +1.0 | True |
| loclin | middle | 1 | -1.173 | [-1.384, 1.557] | -6.7 | +0.1 | True |
| loclin | weak | 2 | 0.106 | [-0.701, 1.557] | -32.4 | -1.6 | True |
| spline | pooled | 2 | -1.072 | [-1.215, 1.557] | -16.2 | -1.1 | True |
| spline | strong | 1 | -1.286 | [-1.413, 1.557] | -6.2 | +0.1 | True |
| spline | middle | 2 | -1.187 | [-1.394, 1.557] | -6.4 | -1.2 | True |
| spline | weak | 3 | 0.063 | [-1.007, 1.557] | -32.3 | +2.3 | True |
| piecewise | pooled | 16 | -1.113 | [-1.225, 1.557] | -14.2 | -0.4 | True |
| piecewise | strong | 3 | -1.197 | [-1.413, 1.557] | +4.2 | -0.2 | True |
| piecewise | middle | 16 | -1.189 | [-1.394, 1.557] | -6.3 | -0.0 | True |
| piecewise | weak | 6 | -0.613 | [-1.225, 1.557] | -33.2 | -1.4 | True |

## B. Leave-one-OOD-dataset-out (pava)

| held-out OOD | pooled | strong | middle | weak | ordering retained |
|---|---|---|---|---|---|
| cifar10 | -1.214 | <=min(censored) | -1.213 | None | True |
| cifar100 | -0.668 | -1.195 | -0.699 | None | True |
| isun | -1.08 | -1.2 | -1.192 | None | True |
| lsun cropped | -1.05 | -1.257 | -1.211 | None | True |
| lsun resize | -1.071 | -1.14 | -1.123 | None | True |
| places365 | -1.0 | -1.084 | -1.079 | 0.345 | True |
| svhn | -0.933 | <=min(censored) | -1.283 | None | True |
| textures | -0.95 | -1.059 | -1.032 | None | True |
| tinyimagenet | -1.097 | -1.279 | -1.251 | None | True |

## C. Leave-one-ID-source-out (pava)

| held-out source | pooled | tie interval | strong | middle | weak | ordering retained |
|---|---|---|---|---|---|---|
| cifar10 | -1.096 | [-1.112, -1.075] | -1.197 | -1.188 | -0.156 | True |
| cifar100 | -1.069 | [-1.138, 1.435] | <=min(censored) | -1.134 | None | True |
| supercifar100 | -1.113 | [-1.225, 1.557] | -1.196 | -1.062 | None | True |
| tinyimagenet | -0.402 | [-1.077, 1.557] | -1.189 | -0.401 | None | True |

## D. Severity-definition sensitivity (pava)

| severity variant | pooled | strong | middle | weak | ordering retained |
|---|---|---|---|---|---|
| full_composite | -1.08 | -1.204 | -1.189 | None | True |
| kid_only | -1.424 | -2.003 | -1.984 | None | True |
| fd_only | -1.425 | -1.691 | -1.682 | 1.27 | True |
| img_centroid_only | -0.739 | <=min(censored) | <=min(censored) | None | True |
| inverse_text_align_only | -1.668 | -2.033 | -2.019 | 1.218 | True |
| without_kid | -0.812 | <=min(censored) | -0.859 | None | True |
| without_fd | -0.965 | -1.041 | -1.019 | None | True |
| without_text_align | -1.077 | -1.48 | -1.474 | None | True |
| without_img_centroid | -1.099 | -1.358 | -1.338 | None | True |

CLIP model variants: NOT AVAILABLE: severity table contains a single CLIP model.

## E. Uncertainty targets

- Conditional checkpoint uncertainty: first up-crossing -1.08, tie region [-1.156, 1.557] (cluster bootstrap over checkpoints, conditional on the fixed OOD suite).
- Shift sensitivity (LOO ranges): pooled crossing spans [-1.214, -0.402].
- Two-way bootstrap: NOT REPORTED: ~8 OOD sets; would require a calibration simulation (section 2.7).

## Decision (section 2.8)

- ordering across estimators/strata: True; unconstrained fits single-crossing: False
- LOO-OOD retained 9/9; LOO-source retained 4/4; severity variants retained 9/9
- **Verdict: PASS**
