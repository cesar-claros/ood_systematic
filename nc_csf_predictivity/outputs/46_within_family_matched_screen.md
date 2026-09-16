# Grouped within-family decomposition and matched development screen (descriptive, 2026-09-15)


## Grouped decomposition, 20 detectors (20 detectors)

| Held-out group | Rows | Training rule | Regret | Source-fixed gain | Beyond source-fixed | Beyond source-by-paradigm | Ceiling over training rule |
|---|---:|---|---:|---:|---:|---:|---:|
| cifar10 | 480 | CTM | 9.310 | 7.518 | 1.231 | 0.912 | 8.748 |
| cifar100+supercifar100 | 1280 | NNGuide | 7.994 | 1.963 | 3.042 | 2.976 | 5.006 |
| tinyimagenet | 480 | NNGuide | 3.764 | 3.266 | 0.057 | 0.057 | 3.323 |

## Grouped decomposition, without Confidence (19 detectors)

| Held-out group | Rows | Training rule | Regret | Source-fixed gain | Beyond source-fixed | Beyond source-by-paradigm | Ceiling over training rule |
|---|---:|---|---:|---:|---:|---:|---:|
| cifar10 | 480 | CTM | 8.831 | 7.143 | 1.045 | 1.045 | 8.187 |
| cifar100+supercifar100 | 1280 | NNGuide | 7.653 | 1.963 | 2.934 | 2.895 | 4.898 |
| tinyimagenet | 480 | NNGuide | 3.643 | 3.266 | 0.057 | 0.057 | 3.323 |

## Matched development screen, 20 detectors (20 detectors); alpha by inner leave-one-training-group-out

| Held-out group | Rows | FIXED (training) | M1 (alpha) | M1+G (alpha) | M2 (alpha) | M2+G (alpha) | G (alpha) | Hindsight fixed | Checkpoint oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | 480 | 9.310 | 9.310 (1000) | 10.235 (1000) | 17.837 (10) | 6.505 (1000) | 9.310 (1000) | 1.792 | 0.562 |
| cifar100+supercifar100 | 1280 | 7.994 | 6.989 (0.1) | 6.667 (10) | 7.292 (1) | 6.979 (10) | 10.318 (1000) | 6.031 | 2.988 |
| tinyimagenet | 480 | 3.764 | 1.440 (1) | 18.840 (10) | 2.049 (0.1) | 8.955 (0.1) | 3.764 (1000) | 0.498 | 0.441 |
| pooled | 2240 | 7.370 | 6.297 | 10.040 | 8.428 | 7.301 | 8.698 | 3.937 | 1.922 |

## Matched development screen, without Confidence (19 detectors); alpha by inner leave-one-training-group-out

| Held-out group | Rows | FIXED (training) | M1 (alpha) | M1+G (alpha) | M2 (alpha) | M2+G (alpha) | G (alpha) | Hindsight fixed | Checkpoint oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | 480 | 8.831 | 8.831 (1000) | 9.642 (1000) | 17.358 (10) | 6.027 (1000) | 8.831 (1000) | 1.689 | 0.644 |
| cifar100+supercifar100 | 1280 | 7.653 | 6.395 (0.1) | 6.326 (10) | 6.872 (1) | 6.638 (10) | 9.977 (1000) | 5.689 | 2.755 |
| tinyimagenet | 480 | 3.643 | 1.267 (1) | 18.719 (10) | 5.731 (1) | 13.081 (0.1) | 3.643 (1000) | 0.377 | 0.320 |
| pooled | 2240 | 7.046 | 5.818 | 9.692 | 8.875 | 7.888 | 8.374 | 3.694 | 1.781 |
