# A2 nullspace geometry (evaluation doc 4.2)

| model | lam | rank(W) | eta_perp | per-class max | self-duality full | self-duality projected |
|---|---|---|---|---|---|---|
| run1 | -0.1 | 100 | 0.0193 | 0.0308 | 0.0506 | 0.0315 |
| run2 | -0.1 | 100 | 0.0196 | 0.0314 | 0.0491 | 0.0296 |
| run3 | -0.1 | 100 | 0.0189 | 0.0312 | 0.0502 | 0.0315 |
| run4 | -0.1 | 100 | 0.0191 | 0.0316 | 0.0487 | 0.0298 |
| run1 | 0.0 | 100 | 0.0055 | 0.0119 | 0.0148 | 0.0093 |
| run2 | 0.0 | 100 | 0.0054 | 0.0109 | 0.0138 | 0.0085 |
| run3 | 0.0 | 100 | 0.0052 | 0.0094 | 0.0144 | 0.0092 |
| run4 | 0.0 | 100 | 0.0052 | 0.0107 | 0.0141 | 0.0089 |
| run1 | 0.3 | 100 | 0.0016 | 0.0040 | 0.0041 | 0.0025 |
| run2 | 0.3 | 100 | 0.0017 | 0.0032 | 0.0043 | 0.0026 |
| run3 | 0.3 | 100 | 0.0016 | 0.0038 | 0.0040 | 0.0024 |
| run4 | 0.3 | 100 | 0.0016 | 0.0032 | 0.0040 | 0.0024 |
| run1 | 1.0 | 100 | 0.0009 | 0.0026 | 0.0028 | 0.0019 |
| run2 | 1.0 | 100 | 0.0009 | 0.0023 | 0.0029 | 0.0020 |
| run3 | 1.0 | 100 | 0.0008 | 0.0015 | 0.0024 | 0.0016 |
| run4 | 1.0 | 100 | 0.0008 | 0.0016 | 0.0027 | 0.0019 |
| run1 | hard | 100 | 0.1870 | 0.2580 | 0.3874 | 0.2115 |
| run2 | hard | 100 | 0.1897 | 0.2640 | 0.3882 | 0.2095 |
| run3 | hard | 100 | 0.1887 | 0.2547 | 0.3895 | 0.2120 |
| run4 | hard | 100 | 0.1870 | 0.2558 | 0.3879 | 0.2122 |

Reading: if A2's projected self-duality is small while its full-space value is large, the A2 refutation is a nullspace-leakage result (features escape through ker W, which cross-entropy cannot constrain), and fixed-classifier training needs explicit span or class-mean control.
