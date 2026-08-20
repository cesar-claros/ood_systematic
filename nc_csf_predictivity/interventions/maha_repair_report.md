# Maha amplitude-operator repair: Pilot 1 validation (design data)

## 1. Operator comparison (paired responses, L = 1 - AUROC_f)

| group | mean obs | mean old | mean min | MAE old | MAE min | repair improves |
|---|---|---|---|---|---|---|
| A1 | -0.0103 | -0.0026 | -0.0001 | 0.0175 | 0.0220 | False |
| A2 | +0.1282 | +0.3329 | +0.5147 | 0.2053 | 0.3865 | False |

- extraction sanity: corr(rank-AUROC from features, pipeline AUROC_f) = 0.999

## 2. Mechanism diagnostics (audit 5.4)

| arm | MC id switch | empirical id switch | OOD score-mean rel gap |
|---|---|---|---|
| baseline | 0.000 | 0.247 | 0.140 |
| A1- | 0.000 | 0.244 | 0.143 |
| A1+ | 0.000 | 0.243 | 0.141 |
| A1++ | 0.000 | 0.244 | 0.160 |
| A2 | 0.000 | 0.260 | 0.036 |

## 3. Bounded calibration (input d_min; fit on A1, evaluated on A2)

No-change MAE: A1 0.0223, A2 0.1282.

| form | params | A1 LOO-dose CV MAE | admissible | A2 MAE | A2 sign |
|---|---|---|---|---|---|
| linear | -0.0087, +18.5973 | 0.0188 | True | 9.4346 | 1.000 |
| slope | +19.7378 | 0.0192 | True | 10.0303 | 1.000 |
| cap | +25.1000 | 0.0220 | True | 0.3864 | 1.000 |
| slope_cap | +0.0424, +1562.6564 | 0.0165 | True | 0.0858 | 1.000 |

**Selected form (fresh-confirmation candidate): slope_cap**
