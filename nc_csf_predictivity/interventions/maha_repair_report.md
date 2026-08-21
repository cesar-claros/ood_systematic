# Maha amplitude-operator repair: Pilot 1 validation (design data)

## 1. Operator comparison (paired responses, L = 1 - AUROC_f)

| group | mean obs | mean d_old / MAE | mean d_min / MAE | mean d_old_val / MAE | mean d_min_val / MAE | best |
|---|---|---|---|---|---|---|
| A1 | -0.0103 | -0.0030 / 0.0170 | -0.0000 / 0.0220 | -0.0062 / 0.0149 | -0.0017 / 0.0153 | d_old_val |
| A2 | +0.1282 | +0.3299 / 0.2038 | +0.5111 / 0.3829 | +0.0698 / 0.0605 | +0.1231 / 0.0719 | d_old_val |

- extraction sanity: corr(rank-AUROC from features, pipeline AUROC_f) = 0.999

## 2. Mechanism diagnostics (audit 5.4)

| arm | MC id switch (train-fit) | MC id switch (val-fit) | empirical val switch | empirical test switch | OOD score-mean rel gap |
|---|---|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.257 | 0.246 | 0.137 |
| A1- | 0.000 | 0.000 | 0.248 | 0.244 | 0.144 |
| A1+ | 0.000 | 0.000 | 0.246 | 0.244 | 0.140 |
| A1++ | 0.000 | 0.000 | 0.246 | 0.244 | 0.159 |
| A2 | 0.000 | 0.000 | 0.272 | 0.258 | 0.036 |

## 3. Bounded calibration (all operator inputs; fit on A1, evaluated on A2)

No-change MAE: A1 0.0223, A2 0.1282.

| input | form | params | A1 LOO-dose CV MAE | admissible | A2 MAE | A2 sign |
|---|---|---|---|---|---|---|
| d_old | linear | -0.0048, +1.8356 | 0.0158 | True | 0.4726 | 1.000 |
| d_old | slope | +1.9260 | 0.0154 | True | 0.5072 | 1.000 |
| d_old | cap | +94.0306 | 0.0170 | True | 0.2038 | 1.000 |
| d_old | slope_cap | -0.0626, +3.3640 | 0.0148 | True | 0.0664 | 1.000 |
| d_min | linear | -0.0092, +20.8572 | 0.0180 | True | 10.5234 | 1.000 |
| d_min | slope | +21.6519 | 0.0194 | True | 10.9389 | 1.000 |
| d_min | cap | +15.1000 | 0.0220 | True | 0.3827 | 1.000 |
| d_min | slope_cap | -0.0453, +896.1057 | 0.0156 | True | 0.0829 | 1.000 |
| d_old_val | linear | -0.0056, +0.7598 | 0.0162 | True | 0.0808 | 0.906 |
| d_old_val | slope | +0.7921 | 0.0148 | True | 0.0733 | 1.000 |
| d_old_val | cap | +0.0833 | 0.0142 | True | 0.0806 | 1.000 |
| d_old_val | slope_cap | +0.0922, +0.9315 | 0.0147 | True | 0.0800 | 1.000 |
| d_min_val | linear | -0.0084, +1.0794 | 0.0157 | True | 0.0778 | 0.906 |
| d_min_val | slope | +1.1092 | 0.0160 | True | 0.0800 | 1.000 |
| d_min_val | cap | +0.1163 | 0.0152 | True | 0.0562 | 1.000 |
| d_min_val | slope_cap | +0.0520, +3.0280 | 0.0151 | True | 0.0813 | 1.000 |

**Selected (fresh-confirmation candidate): d_min_val + cap (A2 MAE 0.0562)**
