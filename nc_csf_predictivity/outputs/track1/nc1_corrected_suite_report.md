# Audit-11 R11.1/R11.3: corrected cache + held-out suite

Post-outcome correctness audit; margins unchanged from the corrected-dictionary cache; nothing fitted.

```
{
 "E4_diagnostics_corrected": {
  "frac_direct_zero_equiv": 0.0728,
  "abs_margin_quantiles_50_90_95_99": [
   1.36,
   7.5626,
   29.9727,
   43.1333
  ],
  "frac_both_pred_auroc_above_099": 0.7174,
  "frac_ctm_material_side_displayed": 0.171,
  "frac_energy_material_side_displayed": 0.0567,
  "spearman_absM_absgap_all": 0.1568,
  "spearman_absM_absgap_material": -0.0395
 },
 "G11A": {
  "sign_acc_here": 0.7312,
  "sign_acc_ref": 0.7312,
  "balanced_here": 0.6934,
  "balanced_ref": 0.6934,
  "frac_resolvable_here": 0.9272,
  "frac_resolvable_ref": 0.9272,
  "PASS": true
 },
 "ckpt5": {
  "n_material": 718,
  "arms": {
   "theory": {
    "sign": 0.7312,
    "balanced": 0.6934
   },
   "severity": {
    "sign": 0.6838,
    "balanced": 0.608
   },
   "geometry": {
    "sign": 0.7646,
    "balanced": 0.7236
   },
   "flexible": {
    "sign": 0.7646,
    "balanced": 0.7254
   },
   "mean": {
    "sign": 0.6003,
    "balanced": 0.5
   },
   "src_id": {
    "sign": 0.7841,
    "balanced": 0.741
   }
  },
  "theory_minus_severity": {
   "point": 0.0474,
   "ci95": [
    -0.0095,
    0.1028
   ]
  },
  "per_half": {
   "dev": {
    "sign": 0.7479,
    "n": 353
   },
   "val": {
    "sign": 0.7151,
    "n": 365
   }
  },
  "leave_one_oodset_sign": {
   "cifar10": 0.762,
   "cifar100": 0.755,
   "isun": 0.727,
   "lsun cropped": 0.714,
   "lsun resize": 0.723,
   "places365": 0.722,
   "svhn": 0.718,
   "textures": 0.736,
   "tinyimagenet": 0.723
  }
 },
 "loso": {
  "n_material": 718,
  "arms": {
   "theory": {
    "sign": 0.7312,
    "balanced": 0.6934
   },
   "severity": {
    "sign": 0.6072,
    "balanced": 0.5442
   },
   "geometry": {
    "sign": 0.6978,
    "balanced": 0.6348
   },
   "flexible": {
    "sign": 0.6365,
    "balanced": 0.5761
   },
   "mean": {
    "sign": 0.6003,
    "balanced": 0.5
   }
  },
  "theory_minus_severity": {
   "point": 0.124,
   "ci95": [
    0.0649,
    0.1786
   ]
  },
  "per_half": {
   "dev": {
    "sign": 0.7479,
    "n": 353
   },
   "val": {
    "sign": 0.7151,
    "n": 365
   }
  },
  "per_source": {
   "cifar10": 0.647,
   "cifar100": 0.778,
   "supercifar100": 0.685,
   "tinyimagenet": 0.903
  },
  "leave_one_oodset_sign": {
   "cifar10": 0.762,
   "cifar100": 0.755,
   "isun": 0.727,
   "lsun cropped": 0.714,
   "lsun resize": 0.723,
   "places365": 0.722,
   "svhn": 0.718,
   "textures": 0.736,
   "tinyimagenet": 0.723
  }
 }
}
```
