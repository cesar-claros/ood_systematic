# rn18_report

```
{
 "denominators": {
  "n_records": 96,
  "n_cells": 384,
  "per_source_checkpoints": {
   "cifar10": 22,
   "cifar100": 24,
   "supercifar100": 28,
   "tinyimagenet": 22
  },
  "sets_per_source": {
   "cifar10": [
    "fashionmnist_new",
    "kmnist_new",
    "mnist_new",
    "stl10_new"
   ],
   "cifar100": [
    "fashionmnist_new",
    "kmnist_new",
    "mnist_new",
    "stl10_new"
   ],
   "supercifar100": [
    "fashionmnist_new",
    "kmnist_new",
    "mnist_new",
    "stl10_new"
   ],
   "tinyimagenet": [
    "fashionmnist_new",
    "kmnist_new",
    "mnist_new",
    "stl10_new"
   ]
  },
  "ce_families": [
   "do0_run1",
   "do0_run2",
   "do0_run3",
   "do0_run4",
   "do0_run5",
   "do1_run1",
   "do1_run2",
   "do1_run3",
   "do1_run4",
   "do1_run5"
  ],
  "vgg_checkpoints_primary_view": 20,
  "vgg_checkpoints_aug_view": 20,
  "multipliers": {
   "SEL_Nf10": 1.1,
   "SEL_Nf5": 1.1,
   "LEVEL_Nf10": 1.1,
   "LEVEL_Nf5": 1.1
  }
 },
 "HO": {
  "dK": {
   "cifar10": {
    "n_checkpoints": 22,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 7,
     "middle": 7,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0033999999999999994,
      "g_at_max_d": 0.0050214285714285725,
      "band_q95": 0.002091642857142856,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0009892857142857145,
      "g_at_max_d": 0.0067800000000000004,
      "band_q95": 0.0032288520893403267,
      "tie_region": [
       -1.57,
       0.813
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0010362499999999998,
      "g_at_max_d": 0.011903750000000001,
      "band_q95": 0.0023453124999999997,
      "tie_region": [
       -1.57,
       -0.776
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     22,
     22
    ],
    "composition": {
     "strong": {
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 4,
      "dg/do1": 1
     },
     "middle": {
      "ce/do0": 2,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 3
     },
     "weak": {
      "ce/do0": 3,
      "ce/do1": 5
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "verdict": "UNINFORMATIVE (< 9 DG checkpoints)"
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": true
    }
   },
   "cifar100": {
    "n_checkpoints": 24,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 8,
     "middle": 8,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0033768749999999997,
      "g_at_max_d": 0.005915,
      "band_q95": 0.003268125,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.00162375,
      "g_at_max_d": 0.007115,
      "band_q95": 0.003537343806981895,
      "tie_region": [
       -1.271,
       0.064
      ]
     },
     "weak": {
      "n_sign_changes": 1,
      "all_crossings": [
       -0.331
      ],
      "first_up_crossing": -0.331,
      "bracketed_by_observed": true,
      "g_at_min_d": -0.00444625,
      "g_at_max_d": 0.0076237499999999995,
      "band_q95": 0.007247625,
      "tie_region": [
       -1.271,
       1.002
      ]
     }
    },
    "informative": true,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     24,
     24
    ],
    "composition": {
     "strong": {
      "ce/do0": 1,
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 5
     },
     "middle": {
      "ce/do0": 4,
      "ce/do1": 4
     },
     "weak": {
      "ce/do1": 1,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 5
     }
    },
    "verdict": "HO-RETAINED",
    "within_dg": {
     "retained": true,
     "informative": true,
     "sizes": {
      "strong": 3,
      "middle": 3,
      "weak": 4
     }
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": true
    }
   },
   "supercifar100": {
    "n_checkpoints": 28,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 9,
     "middle": 9,
     "weak": 10
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": true,
      "g_at_min_d": 0.002418611111111112,
      "g_at_max_d": 0.002418611111111112,
      "band_q95": 0.0021747777777777785,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.00258,
      "g_at_max_d": 0.006938888888888889,
      "band_q95": 0.003126916666666667,
      "tie_region": [
       -1.273,
       -0.487
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0024514999999999997,
      "g_at_max_d": 0.012273999999999998,
      "band_q95": 0.005215449999999999,
      "tie_region": [
       -1.273,
       -0.163
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     28,
     28
    ],
    "composition": {
     "strong": {
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 7
     },
     "middle": {
      "ce/do0": 4,
      "dg/do1": 5
     },
     "weak": {
      "ce/do0": 1,
      "ce/do1": 5,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 2
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "retained": true,
     "informative": false,
     "sizes": {
      "strong": 4,
      "middle": 5,
      "weak": 5
     }
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": false
    }
   },
   "tinyimagenet": {
    "n_checkpoints": 22,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 7,
     "middle": 7,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.003829642857142857,
      "g_at_max_d": 0.003829642857142857,
      "band_q95": 0.0026179761904761914,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0016114285714285714,
      "g_at_max_d": 0.007874285714285714,
      "band_q95": 0.0049351666666666676,
      "tie_region": [
       -1.611,
       -0.348
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": true,
      "g_at_min_d": -0.006830937499999999,
      "g_at_max_d": -0.006830937499999999,
      "band_q95": 0.0095135625,
      "tie_region": [
       -1.611,
       0.882
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     22,
     22
    ],
    "composition": {
     "strong": {
      "ce/do0": 5,
      "confidnet/do0": 1,
      "dg/do0": 1
     },
     "middle": {
      "ce/do1": 5,
      "devries/do0": 1,
      "dg/do0": 1
     },
     "weak": {
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do0": 2,
      "dg/do1": 4
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "verdict": "UNINFORMATIVE (< 9 DG checkpoints)"
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": false
    }
   }
  },
  "dF": {
   "cifar10": {
    "n_checkpoints": 22,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 7,
     "middle": 7,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0033999999999999994,
      "g_at_max_d": 0.0050214285714285725,
      "band_q95": 0.002091642857142856,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0009892857142857145,
      "g_at_max_d": 0.0067800000000000004,
      "band_q95": 0.0032280094692150817,
      "tie_region": [
       -1.684,
       0.597
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0010362499999999998,
      "g_at_max_d": 0.011903750000000001,
      "band_q95": 0.0023453124999999997,
      "tie_region": [
       -1.684,
       -0.586
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     22,
     22
    ],
    "composition": {
     "strong": {
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 4,
      "dg/do1": 1
     },
     "middle": {
      "ce/do0": 2,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 3
     },
     "weak": {
      "ce/do0": 3,
      "ce/do1": 5
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "verdict": "UNINFORMATIVE (< 9 DG checkpoints)"
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": true
    }
   },
   "cifar100": {
    "n_checkpoints": 24,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 8,
     "middle": 8,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0033768749999999997,
      "g_at_max_d": 0.005915,
      "band_q95": 0.003268125,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.00162375,
      "g_at_max_d": 0.007115,
      "band_q95": 0.003546769802021209,
      "tie_region": [
       -1.541,
       0.157
      ]
     },
     "weak": {
      "n_sign_changes": 1,
      "all_crossings": [
       0.018
      ],
      "first_up_crossing": 0.018,
      "bracketed_by_observed": true,
      "g_at_min_d": -0.00444625,
      "g_at_max_d": 0.0076237499999999995,
      "band_q95": 0.007247625,
      "tie_region": [
       -1.541,
       1.056
      ]
     }
    },
    "informative": true,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     24,
     24
    ],
    "composition": {
     "strong": {
      "ce/do0": 1,
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 5
     },
     "middle": {
      "ce/do0": 4,
      "ce/do1": 4
     },
     "weak": {
      "ce/do1": 1,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 5
     }
    },
    "verdict": "HO-RETAINED",
    "within_dg": {
     "retained": true,
     "informative": true,
     "sizes": {
      "strong": 3,
      "middle": 3,
      "weak": 4
     }
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": true
    }
   },
   "supercifar100": {
    "n_checkpoints": 28,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 9,
     "middle": 9,
     "weak": 10
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": true,
      "g_at_min_d": 0.002418611111111112,
      "g_at_max_d": 0.002418611111111112,
      "band_q95": 0.0021747777777777785,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.00258,
      "g_at_max_d": 0.006938888888888889,
      "band_q95": 0.003126916666666667,
      "tie_region": [
       -1.531,
       -0.041
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0024514999999999997,
      "g_at_max_d": 0.012273999999999998,
      "band_q95": 0.005215449999999999,
      "tie_region": [
       -1.531,
       0.069
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     28,
     28
    ],
    "composition": {
     "strong": {
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 7
     },
     "middle": {
      "ce/do0": 4,
      "dg/do1": 5
     },
     "weak": {
      "ce/do0": 1,
      "ce/do1": 5,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 2
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "retained": true,
     "informative": false,
     "sizes": {
      "strong": 4,
      "middle": 5,
      "weak": 5
     }
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": false
    }
   },
   "tinyimagenet": {
    "n_checkpoints": 22,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 7,
     "middle": 7,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.003829642857142857,
      "g_at_max_d": 0.003829642857142857,
      "band_q95": 0.002718130952380952,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0016114285714285714,
      "g_at_max_d": 0.006416190476190476,
      "band_q95": 0.0038056869782507725,
      "tie_region": [
       -1.64,
       -0.863
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": true,
      "g_at_min_d": -0.006830937499999999,
      "g_at_max_d": -0.006830937499999999,
      "band_q95": 0.008032916666666664,
      "tie_region": [
       -1.64,
       0.98
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     22,
     22
    ],
    "composition": {
     "strong": {
      "ce/do0": 5,
      "confidnet/do0": 1,
      "dg/do0": 1
     },
     "middle": {
      "ce/do1": 5,
      "devries/do0": 1,
      "dg/do0": 1
     },
     "weak": {
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do0": 2,
      "dg/do1": 4
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "verdict": "UNINFORMATIVE (< 9 DG checkpoints)"
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": false
    }
   }
  },
  "global": {
   "retained_sources_dK": [
    "cifar100"
   ],
   "retained_sources_dF": [
    "cifar100"
   ],
   "global_wording_available": false,
   "attribution": "compatible with geometry organizing the handoff and equally with the training objective doing so; the design does not separate them"
  }
 },
 "SEL_ce": {
  "N_f": 10,
  "n_cells": 160,
  "score_domain_failures_excluded": 0,
  "mean_regret_P00": 0.00898875,
  "comparators": {
   "always_energy": {
    "mean_regret": 0.015504999999999996,
    "D_b": 0.006516249999999996,
    "ci": [
     -0.00885,
     0.02188
    ],
    "degenerate": null
   },
   "always_ctm": {
    "mean_regret": 0.00898875,
    "D_b": 0.0,
    "ci": null,
    "degenerate": "INFERENCE_DEGENERATE"
   },
   "vgg_kid_isotonic": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "ci": [
     -0.0029,
     0.00858
    ],
    "degenerate": null
   },
   "vgg_fd_isotonic": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "ci": [
     -0.0029,
     0.00858
    ],
    "degenerate": null
   },
   "vgg_source_shift_mean": {
    "mean_regret": 0.011768749999999998,
    "D_b": 0.0027799999999999978,
    "ci": [
     -0.00336,
     0.00892
    ],
    "degenerate": null
   },
   "vgg_geometry_severity_ridge": {
    "mean_regret": 0.008616875000000001,
    "D_b": -0.00037187499999999894,
    "ci": [
     -0.00233,
     0.00158
    ],
    "degenerate": null
   },
   "vgg_matched_scalar_ridge": {
    "mean_regret": 0.009133124999999999,
    "D_b": 0.00014437499999999867,
    "ci": [
     -0.00157,
     0.00185
    ],
    "degenerate": null
   },
   "vgg_no_target_batch_ridge": {
    "mean_regret": 0.014729375,
    "D_b": 0.005740624999999999,
    "ci": [
     -0.00052,
     0.012
    ],
    "degenerate": null
   },
   "vgg_source_majority": {
    "mean_regret": 0.014729375,
    "D_b": 0.005740624999999999,
    "ci": [
     -0.00052,
     0.012
    ],
    "degenerate": null
   }
  },
  "verdict": "PRACTICALLY EQUIVALENT TO THE REFERENCE",
  "material_sign_accuracy_dG": 1.0
 },
 "SEL_ce_do0_sensitivity_Nf5": {
  "N_f": 5,
  "n_cells": 80,
  "score_domain_failures_excluded": 0,
  "mean_regret_P00": 0.010761250000000002,
  "comparators": {
   "always_energy": {
    "mean_regret": 0.010666249999999999,
    "D_b": -9.500000000000307e-05,
    "ci": [
     -0.02846,
     0.02827
    ],
    "degenerate": null
   },
   "always_ctm": {
    "mean_regret": 0.010761250000000002,
    "D_b": 0.0,
    "ci": null,
    "degenerate": "INFERENCE_DEGENERATE"
   },
   "vgg_kid_isotonic": {
    "mean_regret": 0.012115,
    "D_b": 0.001353749999999999,
    "ci": [
     -0.00693,
     0.00964
    ],
    "degenerate": null
   },
   "vgg_fd_isotonic": {
    "mean_regret": 0.012115,
    "D_b": 0.001353749999999999,
    "ci": [
     -0.00693,
     0.00964
    ],
    "degenerate": null
   },
   "vgg_source_shift_mean": {
    "mean_regret": 0.011656250000000002,
    "D_b": 0.000895,
    "ci": [
     -0.00756,
     0.00935
    ],
    "degenerate": null
   },
   "vgg_geometry_severity_ridge": {
    "mean_regret": 0.011380000000000005,
    "D_b": 0.000618750000000003,
    "ci": [
     -0.00306,
     0.00429
    ],
    "degenerate": null
   },
   "vgg_matched_scalar_ridge": {
    "mean_regret": 0.011145,
    "D_b": 0.0003837499999999987,
    "ci": [
     -0.00529,
     0.00606
    ],
    "degenerate": null
   },
   "vgg_no_target_batch_ridge": {
    "mean_regret": 0.013905000000000004,
    "D_b": 0.0031437500000000024,
    "ci": [
     -0.00554,
     0.01183
    ],
    "degenerate": null
   },
   "vgg_source_majority": {
    "mean_regret": 0.013905000000000004,
    "D_b": 0.0031437500000000024,
    "ci": [
     -0.00554,
     0.01183
    ],
    "degenerate": null
   }
  },
  "verdict": "UNRESOLVED",
  "material_sign_accuracy_dG": 1.0
 },
 "SEL_ce_augview_comparators_sensitivity": {
  "N_f": 10,
  "n_cells": 160,
  "score_domain_failures_excluded": 0,
  "mean_regret_P00": 0.00898875,
  "comparators": {
   "always_energy": {
    "mean_regret": 0.015504999999999996,
    "D_b": 0.006516249999999996,
    "ci": [
     -0.00885,
     0.02188
    ],
    "degenerate": null
   },
   "always_ctm": {
    "mean_regret": 0.00898875,
    "D_b": 0.0,
    "ci": null,
    "degenerate": "INFERENCE_DEGENERATE"
   },
   "vgg_kid_isotonic": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "ci": [
     -0.0029,
     0.00858
    ],
    "degenerate": null
   },
   "vgg_fd_isotonic": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "ci": [
     -0.0029,
     0.00858
    ],
    "degenerate": null
   },
   "vgg_source_shift_mean": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "ci": [
     -0.0029,
     0.00858
    ],
    "degenerate": null
   },
   "vgg_geometry_severity_ridge": {
    "mean_regret": 0.0087425,
    "D_b": -0.00024624999999999994,
    "ci": [
     -0.00258,
     0.00209
    ],
    "degenerate": null
   },
   "vgg_matched_scalar_ridge": {
    "mean_regret": 0.0073693749999999975,
    "D_b": -0.0016193750000000028,
    "ci": [
     -0.00829,
     0.00505
    ],
    "degenerate": null
   },
   "vgg_no_target_batch_ridge": {
    "mean_regret": 0.014729375,
    "D_b": 0.005740624999999999,
    "ci": [
     -0.00052,
     0.012
    ],
    "degenerate": null
   },
   "vgg_source_majority": {
    "mean_regret": 0.014729375,
    "D_b": 0.005740624999999999,
    "ci": [
     -0.00052,
     0.012
    ],
    "degenerate": null
   }
  },
  "verdict": "UNRESOLVED",
  "material_sign_accuracy_dG": 1.0
 },
 "SEL_paradigm_pool_descriptive": {
  "mean_regret_P00": 0.012820982142857141
 },
 "LEVEL_ce": {
  "N_f": 10,
  "delta": 0.011144496547958066,
  "ci": [
   0.00726,
   0.01503
  ],
  "verdict": "resolved improvement",
  "equivalent_within_0.01": false,
  "at_least_one_point": false
 },
 "LEVEL_ce_do0_sensitivity_Nf5": {
  "N_f": 5,
  "delta": 0.0074288872221739185,
  "ci": [
   0.00677,
   0.00808
  ],
  "verdict": "resolved improvement",
  "equivalent_within_0.01": true,
  "at_least_one_point": false
 },
 "bridge_A_G": {
  "primary_view": -0.00125575,
  "aug_view": -0.001478
 },
 "ORG_descriptive": {
  "ce_do0": {
   "A_G": 0.0018057499999999996,
   "ci95_descriptive": [
    -0.0002,
    0.00381
   ],
   "N_f": 5,
   "equivalence_0.003": false,
   "resolvable": true
  },
  "full_panel_descriptive": {
   "A_G": 0.0032976609209680874,
   "caveat": "geometry and training objective move together on this panel"
  }
 },
 "E4": {
  "full": {
   "spearman_absM_absdG_all": -0.1427845038756117,
   "spearman_absM_absdG_material": -0.252716107797133
  },
  "ce": {
   "spearman_absM_absdG_all": -0.11171034446243945,
   "spearman_absM_absdG_material": -0.047823430202802525
  }
 },
 "comparator_fits": {
  "primary_view": {
   "vgg_geometry_severity_ridge": {
    "lambda": 10.0,
    "cv_losses": {
     "0.0001": 0.001229238799443499,
     "0.001": 0.0012292347510209564,
     "0.01": 0.0012291943078959686,
     "0.1": 0.0012287939425876765,
     "1.0": 0.001225157205964494,
     "10.0": 0.0012066635433142072
    }
   },
   "vgg_matched_scalar_ridge": {
    "lambda": 0.1,
    "cv_losses": {
     "0.0001": 0.0005528502577014783,
     "0.001": 0.0005499529320381832,
     "0.01": 0.0005472454632546581,
     "0.1": 0.0005459801704182712,
     "1.0": 0.0005547223148980342,
     "10.0": 0.0008073791713627316
    }
   },
   "vgg_no_target_batch_ridge": {
    "lambda": 10.0,
    "cv_losses": {
     "0.0001": 0.0013592237549551226,
     "0.001": 0.0013524770767664946,
     "0.01": 0.0013377710427991954,
     "0.1": 0.0013214045607284008,
     "1.0": 0.001307312481862802,
     "10.0": 0.0013029024805854376
    }
   },
   "source_majority": {
    "cifar10": -1.0,
    "cifar100": 1.0,
    "supercifar100": -1.0,
    "tinyimagenet": 1.0
   },
   "folds": "leave-one-VGG-checkpoint-out (seed reuse across sources not audited)"
  },
  "aug_view": {
   "vgg_geometry_severity_ridge": {
    "lambda": 10.0,
    "cv_losses": {
     "0.0001": 0.001230544602407013,
     "0.001": 0.001230541718425619,
     "0.01": 0.0012305129070332606,
     "0.1": 0.001230227605741492,
     "1.0": 0.001227629249961061,
     "10.0": 0.0012143770180766999
    }
   },
   "vgg_matched_scalar_ridge": {
    "lambda": 0.0001,
    "cv_losses": {
     "0.0001": 0.0006202676805690365,
     "0.001": 0.0006210132569474194,
     "0.01": 0.0006238369262680349,
     "0.1": 0.0006302314737357201,
     "1.0": 0.0006635855569136134,
     "10.0": 0.0009490327833837317
    }
   },
   "vgg_no_target_batch_ridge": {
    "lambda": 10.0,
    "cv_losses": {
     "0.0001": 0.001429376755753756,
     "0.001": 0.0014141538274670407,
     "0.01": 0.0013910020809534632,
     "0.1": 0.0013843656088782558,
     "1.0": 0.0013833599157311235,
     "10.0": 0.0013822624997759164
    }
   },
   "source_majority": {
    "cifar10": -1.0,
    "cifar100": 1.0,
    "supercifar100": -1.0,
    "tinyimagenet": 1.0
   },
   "folds": "leave-one-VGG-checkpoint-out (seed reuse across sources not audited)"
  }
 }
}
```
