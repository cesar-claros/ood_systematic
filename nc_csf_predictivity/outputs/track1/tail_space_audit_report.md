# Phase-0 stable tail-space audit (saturation plan sections 4-5)

Post-outcome numerical-correctness and sensitivity audit; mathematically equivalent stable representation of the frozen formulas; frozen primaries unchanged.

```
{
 "continuity": {
  "max_abs_dev_E": 0.0,
  "max_abs_dev_C": 0.0
 },
 "n_cells": 2240,
 "n_material": 718,
 "frac_direct_zero": 0.8031,
 "frac_direct_zero_with_nonzero_tail": 1.0,
 "sign_agreement_direct_nonzero_vs_tail": 1.0,
 "material_sign_acc_tail": 0.4401,
 "material_sign_acc_ci95": [
  0.391,
  0.493
 ],
 "material_balanced_acc_tail": 0.4568,
 "frozen_material_sign_acc_direct": 0.099,
 "tail_minus_severity": {
  "point": -0.2437,
  "ci95": [
   -0.339,
   -0.146
  ]
 },
 "severity_sign_acc": 0.6838,
 "trainfold_mean_sign_acc": 0.6003,
 "spearman_abs_mtail_vs_abs_gap_material": -0.2279,
 "spearman_abs_mtail_vs_abs_gap_all": -0.0968,
 "per_source": {
  "cifar10": {
   "sign_acc": 0.594,
   "n_material": 170
  },
  "cifar100": {
   "sign_acc": 0.533,
   "n_material": 90
  },
  "supercifar100": {
   "sign_acc": 0.404,
   "n_material": 314
  },
  "tinyimagenet": {
   "sign_acc": 0.278,
   "n_material": 144
  }
 },
 "per_paradigm": {
  "confidnet": {
   "sign_acc": 0.545,
   "n_material": 123
  },
  "devries": {
   "sign_acc": 0.359,
   "n_material": 117
  },
  "dg": {
   "sign_acc": 0.433,
   "n_material": 478
  }
 },
 "leave_one_source_sign_acc": {
  "cifar10": 0.392,
  "cifar100": 0.427,
  "supercifar100": 0.468,
  "tinyimagenet": 0.481
 },
 "leave_one_oodset_sign_acc": {
  "cifar10": 0.448,
  "cifar100": 0.452,
  "isun": 0.442,
  "lsun cropped": 0.444,
  "lsun resize": 0.419,
  "places365": 0.432,
  "svhn": 0.392,
  "textures": 0.495,
  "tinyimagenet": 0.437
 },
 "float32_sign_flips": 0,
 "fast_path_sign_disagreements": 0,
 "stability": {
  "mean": 1.0,
  "frac_fully_stable": 1.0
 },
 "selective": {
  "0.5": {
   "coverage": 1.0,
   "sign_acc": 0.4401
  },
  "0.8": {
   "coverage": 1.0,
   "sign_acc": 0.4401
  },
  "0.95": {
   "coverage": 1.0,
   "sign_acc": 0.4401
  },
  "1.0": {
   "coverage": 1.0,
   "sign_acc": 0.4401
  }
 },
 "high_precision": {
  "n_checked": 20,
  "n_sign_agree": 20,
  "max_rel_diff": 4.362121144195956e-14
 },
 "scientific_uncertainty": "extraction-derived per-cell coordinate uncertainty unavailable without re-extraction; recorded per plan 5.4"
}
```
