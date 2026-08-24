# Maha amplitude-operator repair: Pilot 1 validation (design data)

## 1. Operator comparison (paired responses, L = 1 - AUROC_f)

| group | mean obs | mean d_old / MAE | mean d_min / MAE | mean d_old_val / MAE | mean d_min_val / MAE | mean d_old_valc / MAE | mean d_min_valc / MAE | best |
|---|---|---|---|---|---|---|---|---|
| A1 | -0.0103 | -0.0025 / 0.0170 | -0.0001 / 0.0220 | -0.0065 / 0.0143 | -0.0035 / 0.0145 | -0.0015 / 0.0187 | -0.0000 / 0.0222 | d_old_val |
| A2 | +0.1282 | +0.3315 / 0.2046 | +0.5125 / 0.3843 | +0.0685 / 0.0605 | +0.1206 / 0.0701 | +0.0390 / 0.0892 | +0.0461 / 0.0821 | d_old_val |

- extraction sanity: corr(rank-AUROC from features, pipeline AUROC_f) = 0.999

## 2. Mechanism diagnostics (audit 5.4)

| arm | MC id switch (train-fit) | MC id switch (val-fit) | empirical val switch | empirical test switch | OOD score-mean rel gap |
|---|---|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.257 | 0.246 | 0.134 |
| A1- | 0.000 | 0.000 | 0.251 | 0.245 | 0.142 |
| A1+ | 0.000 | 0.000 | 0.247 | 0.244 | 0.141 |
| A1++ | 0.000 | 0.000 | 0.245 | 0.244 | 0.157 |
| A2 | 0.000 | 0.000 | 0.270 | 0.258 | 0.036 |

Correct-filtered validation population (R2):
- baseline: MC id switch (valc-fit) 0.000, empirical correct-filtered val switch 0.013
- A1-: MC id switch (valc-fit) 0.000, empirical correct-filtered val switch 0.020
- A1+: MC id switch (valc-fit) 0.000, empirical correct-filtered val switch 0.016
- A1++: MC id switch (valc-fit) 0.000, empirical correct-filtered val switch 0.012
- A2: MC id switch (valc-fit) 0.000, empirical correct-filtered val switch 0.040

## 3. Bounded calibration (all operator inputs; fit on A1, evaluated on A2)

No-change MAE: A1 0.0223, A2 0.1282.

| input | form | params | A1 LOO-dose CV MAE | admissible | A2 MAE | A2 sign |
|---|---|---|---|---|---|---|
| d_old | linear | -0.0054, +1.9378 | 0.0152 | True | 0.5089 | 1.000 |
| d_old | slope | +2.0311 | 0.0153 | True | 0.5452 | 1.000 |
| d_old | cap | +101.8501 | 0.0170 | True | 0.2046 | 1.000 |
| d_old | slope_cap | +0.0675, +3.5916 | 0.0143 | True | 0.0622 | 1.000 |
| d_min | linear | -0.0090, +15.6700 | 0.0190 | True | 7.8937 | 1.000 |
| d_min | slope | +16.5456 | 0.0195 | True | 8.3514 | 1.000 |
| d_min | cap | +15.1000 | 0.0220 | True | 0.3840 | 1.000 |
| d_min | slope_cap | +0.0448, +779.3662 | 0.0160 | True | 0.0834 | 1.000 |
| d_old_val | linear | -0.0053, +0.7650 | 0.0146 | True | 0.0811 | 0.969 |
| d_old_val | slope | +0.7963 | 0.0141 | True | 0.0737 | 1.000 |
| d_old_val | cap | +0.0845 | 0.0132 | True | 0.0808 | 1.000 |
| d_old_val | slope_cap | +0.0797, +1.0701 | 0.0135 | True | 0.0807 | 1.000 |
| d_min_val | linear | -0.0068, +0.9837 | 0.0147 | True | 0.0690 | 0.938 |
| d_min_val | slope | +1.0240 | 0.0152 | True | 0.0719 | 1.000 |
| d_min_val | cap | +0.0992 | 0.0145 | True | 0.0614 | 1.000 |
| d_min_val | slope_cap | -0.0566, +2.2480 | 0.0137 | True | 0.0793 | 1.000 |
| d_old_valc | linear | -0.0067, +2.3531 | 0.0170 | True | 0.0689 | 0.781 |
| d_old_valc | slope | +2.4762 | 0.0169 | True | 0.0694 | 1.000 |
| d_old_valc | cap | +85.2929 | 0.0187 | True | 0.0892 | 1.000 |
| d_old_valc | slope_cap | -0.0628, +5.0355 | 0.0166 | True | 0.0823 | 1.000 |
| d_min_valc | linear | -0.0103, +56.2546 | 0.0187 | True | 2.4689 | 1.000 |
| d_min_valc | slope | +56.2797 | 0.0203 | True | 2.4770 | 1.000 |
| d_min_valc | cap | +15.1000 | 0.0222 | True | 0.0821 | 1.000 |
| d_min_valc | slope_cap | +0.0426, +5317.7046 | 0.0163 | True | 0.0856 | 1.000 |

**Selected (fresh-confirmation candidate): d_min_val + cap (A2 MAE 0.0614)**
