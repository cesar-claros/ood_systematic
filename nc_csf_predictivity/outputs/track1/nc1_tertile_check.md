# NC1 impact check 1: tertile composition and ordering under the corrected panel

Frozen machinery, seed 1071, B=2000; no manuscript change; feeds the next audit round.

```
{
 "within_source_spearman_old_vs_new_panel": {
  "cifar10": 0.994,
  "cifar100": 0.9047,
  "supercifar100": 0.9993,
  "tinyimagenet": 0.4807
 },
 "tertile_membership_retained": "84/280",
 "membership_overlap_counts": {
  "strong->strong": 24,
  "strong->middle": 46,
  "strong->weak": 23,
  "middle->strong": 46,
  "middle->middle": 18,
  "middle->weak": 29,
  "weak->strong": 23,
  "weak->middle": 29,
  "weak->weak": 42
 },
 "source_composition_old": {
  "strong": {
   "cifar10": "14",
   "cifar100": "31",
   "supercifar100": "9",
   "tinyimagenet": "39"
  },
  "middle": {
   "cifar10": "23",
   "cifar100": "23",
   "supercifar100": "26",
   "tinyimagenet": "21"
  },
  "weak": {
   "cifar10": "23",
   "cifar100": "16",
   "supercifar100": "55"
  }
 },
 "source_composition_new": {
  "strong": {
   "cifar10": "60",
   "cifar100": "1",
   "supercifar100": "32"
  },
  "middle": {
   "cifar100": "41",
   "supercifar100": "32",
   "tinyimagenet": "20"
  },
  "weak": {
   "cifar100": "28",
   "supercifar100": "26",
   "tinyimagenet": "40"
  }
 },
 "strata_old_panel_reference": {
  "pooled": {
   "first_up_crossing": -1.08,
   "tie_region": [
    -1.156,
    1.557
   ],
   "n_sign_changes": 1
  },
  "strong": {
   "first_up_crossing": -1.204,
   "tie_region": [
    -1.413,
    -1.196
   ],
   "n_sign_changes": 1
  },
  "middle": {
   "first_up_crossing": -1.189,
   "tie_region": [
    -1.334,
    1.557
   ],
   "n_sign_changes": 1
  },
  "weak": {
   "first_up_crossing": null,
   "tie_region": [
    0.101,
    1.557
   ],
   "n_sign_changes": 0
  },
  "ordering_retained": true
 },
 "strata_new_panel": {
  "pooled": {
   "first_up_crossing": -1.08,
   "tie_region": [
    -1.156,
    1.557
   ],
   "n_sign_changes": 1
  },
  "strong": {
   "first_up_crossing": -0.4,
   "tie_region": [
    -1.265,
    1.408
   ],
   "n_sign_changes": 1
  },
  "middle": {
   "first_up_crossing": -1.074,
   "tie_region": [
    -1.097,
    -0.305
   ],
   "n_sign_changes": 1
  },
  "weak": {
   "first_up_crossing": -1.071,
   "tie_region": [
    -1.106,
    1.557
   ],
   "n_sign_changes": 1
  },
  "ordering_retained": false
 }
}
```
