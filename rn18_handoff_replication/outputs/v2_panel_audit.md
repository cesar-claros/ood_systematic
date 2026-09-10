# Version-2 panel: extraction-chain audit

```
{
 "label": "EXTRACTION-CHAIN AUDIT (post-outcome, descriptive)",
 "consistency": {
  "rn18": {
   "n_common": 96,
   "only_v1": [],
   "only_v2": [],
   "by_group": {
    "outcomes": {
     "n_records": 96,
     "n_fields_total": 12672,
     "n_fields_equal": 12672,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew20",
      "fashionmnist_new/a"
     ]
    },
    "coords_p10": {
     "n_records": 96,
     "n_fields_total": 45580,
     "n_fields_equal": 45580,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew20",
      "fashionmnist_new/R"
     ]
    },
    "gaussian": {
     "n_records": 96,
     "n_fields_total": 17080,
     "n_fields_equal": 17080,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew20",
      "fashionmnist_new/component_ids/0"
     ]
    },
    "papyan": {
     "n_records": 96,
     "n_fields_total": 768,
     "n_fields_equal": 768,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew20",
      "equiangular_uc"
     ]
    },
    "geometry": {
     "n_records": 96,
     "n_fields_total": 1056,
     "n_fields_equal": 1056,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew20",
      "class_mean_radius"
     ]
    },
    "iid_test": {
     "n_records": 96,
     "n_fields_total": 8608,
     "n_fields_equal": 8608,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew20",
      "a"
     ]
    },
    "record": {
     "n_records": 96,
     "n_fields_total": 192,
     "n_fields_equal": 192,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew20",
      "dim"
     ]
    }
   }
  },
  "vgg": {
   "n_common": 20,
   "only_v1": [],
   "only_v2": [],
   "by_group": {
    "outcomes": {
     "n_records": 20,
     "n_fields_total": 2640,
     "n_fields_equal": 2640,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
      "fashionmnist_new/a"
     ]
    },
    "coords_p10": {
     "n_records": 20,
     "n_fields_total": 9900,
     "n_fields_equal": 9900,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
      "fashionmnist_new/R"
     ]
    },
    "gaussian": {
     "n_records": 20,
     "n_fields_total": 3720,
     "n_fields_equal": 3720,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
      "fashionmnist_new/component_ids/0"
     ]
    },
    "papyan": {
     "n_records": 20,
     "n_fields_total": 160,
     "n_fields_equal": 160,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
      "equiangular_uc"
     ]
    },
    "geometry": {
     "n_records": 20,
     "n_fields_total": 220,
     "n_fields_equal": 220,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
      "class_mean_radius"
     ]
    },
    "iid_test": {
     "n_records": 20,
     "n_fields_total": 1865,
     "n_fields_equal": 1865,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
      "a"
     ]
    },
    "record": {
     "n_records": 20,
     "n_fields_total": 40,
     "n_fields_equal": 40,
     "max_abs_diff": 0.0,
     "worst": [
      0.0,
      "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
      "dim"
     ]
    }
   }
  }
 },
 "unrounded_sensitivity": {
  "SEL_reference_contrast": {
   "unrounded": 0.0001449596874999997,
   "rounded_v2_records": 0.00014437499999999867,
   "corrected_readout": 0.00014437499999999867,
   "ci_unrounded_mult1": [
    -0.0014082094663001156,
    0.001698128841300115
   ],
   "ci_corrected_readout_mult1": [
    -0.0014102695882384688,
    0.0016990195882384661
   ]
  },
  "SEL_mean_regret_P00": {
   "unrounded": 0.008987842841880344,
   "rounded_v2_records": 0.00898875,
   "corrected_readout": 0.00898875
  },
  "SEL_P00_choice_counts_unrounded": {
   "CTM": 160,
   "Energy": 0,
   "tie": 0
  },
  "SEL_material_unrounded": {
   "n_material": 17,
   "n_all": 160,
   "sign_accuracy_material_dG": 1.0,
   "sign_accuracy_all_nonzero_dG": 0.86875
  },
  "SEL_all_comparators_D_unrounded": {
   "always_energy": 0.0065164884127938034,
   "always_ctm": 0.0,
   "vgg_kid_isotonic": 0.002837980878071581,
   "vgg_fd_isotonic": 0.002837980878071581,
   "vgg_source_shift_mean": 0.002779025322516027,
   "vgg_geometry_severity_ridge": -0.0003741358246527795,
   "vgg_matched_scalar_ridge": 0.0001449596874999997,
   "vgg_no_target_batch_ridge": 0.005737586641960472,
   "vgg_source_majority": 0.005737586641960472
  },
  "LEVEL_delta": {
   "unrounded": 0.011144496547958072,
   "rounded_v2_records": 0.011144496547958066,
   "corrected_readout": 0.011144496547958066,
   "ci_unrounded_mult1": [
    0.007609731790751684,
    0.014679261305164459
   ],
   "ci_corrected_readout_mult1": [
    0.007609731790751679,
    0.014679261305164453
   ]
  },
  "HO_unrounded_verdicts": {
   "dK": {
    "cifar10": "HO-UNINFORMATIVE",
    "cifar100": "HO-RETAINED",
    "supercifar100": "HO-UNINFORMATIVE",
    "tinyimagenet": "HO-UNINFORMATIVE"
   },
   "dF": {
    "cifar10": "HO-UNINFORMATIVE",
    "cifar100": "HO-RETAINED",
    "supercifar100": "HO-UNINFORMATIVE",
    "tinyimagenet": "HO-UNINFORMATIVE"
   }
  },
  "HO_corrected_readout_verdicts": {
   "dK": {
    "cifar10": "HO-UNINFORMATIVE",
    "cifar100": "HO-RETAINED",
    "supercifar100": "HO-UNINFORMATIVE",
    "tinyimagenet": "HO-UNINFORMATIVE"
   },
   "dF": {
    "cifar10": "HO-UNINFORMATIVE",
    "cifar100": "HO-RETAINED",
    "supercifar100": "HO-UNINFORMATIVE",
    "tinyimagenet": "HO-UNINFORMATIVE"
   }
  },
  "note": "all three SEL/LEVEL columns are descriptive at multiplier 1; the reader of record used the rounded version-1 fields"
 },
 "per_example": {
  "rn18": {
   "n_sets": 384,
   "max_identity_residual": 1.6653345369377348e-16,
   "pi_raw_range": [
    0.494,
    0.7570588235294118
   ],
   "pi_balanced_range": [
    0.5221111111111111,
    0.762
   ],
   "failure_auroc_gap_balanced_CTM_minus_Energy": {
    "mean": 0.014982253878591941,
    "min": -0.24873196254913577,
    "max": 0.1504408031422333
   },
   "score_tie_fraction_energy_ctm": {
    "mean": 2.7089783281732116e-05,
    "max": 0.00011764705882355564
   },
   "n_ceiling_aurocs_energy_ctm": 0,
   "id_error_rate_range": [
    0.044222222222222225,
    0.5234615384615384
   ],
   "test_over_train_sigma_iso": {
    "min": 1.083814050686667,
    "max": 2.62713822473332,
    "mean": 1.7263016937804256
   }
  },
  "vgg": {
   "n_sets": 80,
   "max_identity_residual": 1.1102230246251565e-16,
   "pi_raw_range": [
    0.49629411764705883,
    0.7147058823529412
   ],
   "pi_balanced_range": [
    0.5242777777777777,
    0.7215625
   ],
   "failure_auroc_gap_balanced_CTM_minus_Energy": {
    "mean": 0.028109379350905718,
    "min": -0.015258682137087587,
    "max": 0.10843259259259252
   },
   "score_tie_fraction_energy_ctm": {
    "mean": 2.7089783281732116e-05,
    "max": 0.00011764705882355564
   },
   "n_ceiling_aurocs_energy_ctm": 0,
   "id_error_rate_range": [
    0.04855555555555555,
    0.4403846153846154
   ],
   "test_over_train_sigma_iso": {
    "min": 1.0549885918284478,
    "max": 1.4022350066808549,
    "mean": 1.2824568109509142
   }
  }
 },
 "inventory": {
  "rn18": {
   "n_json": 96,
   "n_npz": 96,
   "n_failed": 0,
   "extractor_revisions": [
    "68b3af46611b118f1f26878c05fadd3ae4034c5803dc7da6e178b19e82876719"
   ],
   "id_methods": {
    "contiguous offset (label-sequence match)": "74",
    "file-path mapping": "22"
   }
  },
  "vgg": {
   "n_json": 20,
   "n_npz": 20,
   "n_failed": 0,
   "extractor_revisions": [
    "68b3af46611b118f1f26878c05fadd3ae4034c5803dc7da6e178b19e82876719"
   ],
   "id_methods": {
    "contiguous offset (label-sequence match)": "15",
    "file-path mapping": "5"
   }
  }
 }
}
```
