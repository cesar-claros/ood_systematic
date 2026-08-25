# Crossing-estimator coverage under the realistic design (R2)

Clusters: 280 (real source split); supports: 8 per source (real severities); reps 200; B_boot 500; seed 2026; noise calibrated from the real table (cluster sd Energy/CTM = 18.7/20.5, corr 0.93; residual sd = 37.4/37.3, corr 0.98).

## S1_single (true zeros: [-1.083])

| estimator | declared crossing | zero-set covers all true | window covers first | median |first err| | median window width | mean sign changes | bounded zero set | left-censor call | right-censor call |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.245 | 0.920 | 0.240 | 0.100 | 2.970 | 0.2 | 0.000 | 0.755 | 0.000 |
| B | 1.000 | 1.000 | 1.000 | 0.020 | 0.921 | 1.0 | 1.000 | 0.000 | 0.000 |

## S2_none (true zeros: none in range)

| estimator | declared crossing | zero-set covers all true | window covers first | median |first err| | median window width | mean sign changes | bounded zero set | left-censor call | right-censor call |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.005 | - | - | - | - | 0.0 | 0.000 | 0.000 | 0.995 |
| B | 0.320 | - | - | - | - | 0.3 | 0.005 | 0.000 | 0.680 |

## S4_left_censored (true zeros: none in range)

| estimator | declared crossing | zero-set covers all true | window covers first | median |first err| | median window width | mean sign changes | bounded zero set | left-censor call | right-censor call |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.000 | - | - | - | - | 0.0 | 0.000 | 1.000 | 0.000 |
| B | 1.000 | - | - | - | - | 1.0 | 0.995 | 0.000 | 0.000 |

## S5_right_censored (true zeros: none in range)

| estimator | declared crossing | zero-set covers all true | window covers first | median |first err| | median window width | mean sign changes | bounded zero set | left-censor call | right-censor call |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.045 | - | - | - | - | 0.1 | 0.000 | 0.000 | 0.955 |
| B | 0.715 | - | - | - | - | 0.7 | 0.010 | 0.000 | 0.285 |

## S3_multiple (true zeros: [-1.083, 1.349])

| estimator | declared crossing | zero-set covers all true | window covers first | median |first err| | median window width | mean sign changes | bounded zero set | left-censor call | right-censor call |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.220 | 0.890 | 0.220 | 0.140 | 2.970 | 0.2 | 0.000 | 0.780 | 0.000 |
| B | 1.000 | 0.050 | 1.000 | 0.015 | 0.940 | 1.0 | 0.955 | 0.000 | 0.000 |

## S6_per_source_realistic (true zeros: [-1.079])

| estimator | declared crossing | zero-set covers all true | window covers first | median |first err| | median window width | mean sign changes | bounded zero set | left-censor call | right-censor call |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.990 | 1.000 | 0.990 | 0.010 | 2.703 | 1.0 | 0.040 | 0.010 | 0.000 |
| B | 1.000 | 1.000 | 1.000 | 0.006 | 0.871 | 1.0 | 0.975 | 0.000 | 0.000 |
