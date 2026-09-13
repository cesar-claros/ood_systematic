# Target-aligned audit

```
{
 "label": "TARGET-ALIGNED AUDIT (post-outcome, descriptive)",
 "materiality": 0.01,
 "identity_check_rn18_max_abs_resid": 1.1362438767648086e-16,
 "rn18": {
  "n_cells": 384,
  "id_error_range": [
   0.044222222222222274,
   0.5234615384615384
  ],
  "pi_raw_range": [
   0.494,
   0.7570588235294118
  ],
  "disagreement_all": {
   "dA_vs_dF": {
    "n": 384,
    "opposite": 87,
    "opposite_frac": 0.2265625,
    "opposite_among_dA_material": [
     53,
     278
    ],
    "opposite_among_dF_material": [
     34,
     278
    ]
   },
   "dA_vs_dG": {
    "n": 384,
    "opposite": 87,
    "opposite_frac": 0.2265625,
    "opposite_among_dA_material": [
     53,
     278
    ],
    "opposite_among_dG_material": [
     4,
     60
    ]
   },
   "dF_vs_dG": {
    "n": 384,
    "opposite": 0,
    "opposite_frac": 0.0,
    "opposite_among_dF_material": [
     0,
     278
    ],
    "opposite_among_dG_material": [
     0,
     60
    ]
   }
  },
  "disagreement_ce": {
   "dA_vs_dF": {
    "n": 160,
    "opposite": 44,
    "opposite_frac": 0.275,
    "opposite_among_dA_material": [
     31,
     115
    ],
    "opposite_among_dF_material": [
     18,
     110
    ]
   },
   "dA_vs_dG": {
    "n": 160,
    "opposite": 44,
    "opposite_frac": 0.275,
    "opposite_among_dA_material": [
     31,
     115
    ],
    "opposite_among_dG_material": [
     0,
     17
    ]
   },
   "dF_vs_dG": {
    "n": 160,
    "opposite": 0,
    "opposite_frac": 0.0,
    "opposite_among_dF_material": [
     0,
     110
    ],
    "opposite_among_dG_material": [
     0,
     17
    ]
   }
  },
  "winner_accuracy_all": {
   "dA": {
    "frozen_arm_all_nonzero": 0.5703125,
    "n_all_nonzero": 384,
    "frozen_arm_material": 0.5719424460431655,
    "n_material": 278,
    "always_ctm_all": 0.609375,
    "always_ctm_material": 0.5971223021582733,
    "frozen_arm_regret": 0.01121956253241297,
    "always_ctm_regret": 0.016281747065443598,
    "always_energy_regret": 0.018699129463113206,
    "p00_tie_fraction": 0.0
   },
   "dF": {
    "frozen_arm_all_nonzero": 0.734375,
    "n_all_nonzero": 384,
    "frozen_arm_material": 0.7877697841726619,
    "n_material": 278,
    "always_ctm_all": 0.7890625,
    "always_ctm_material": 0.8669064748201439,
    "frozen_arm_regret": 0.005400377918270448,
    "always_ctm_regret": 0.005900883090882595,
    "always_energy_regret": 0.02082225436134285,
    "p00_tie_fraction": 0.0
   },
   "dG": {
    "frozen_arm_all_nonzero": 0.734375,
    "n_all_nonzero": 384,
    "frozen_arm_material": 0.85,
    "n_material": 60,
    "always_ctm_all": 0.7890625,
    "always_ctm_material": 0.8,
    "frozen_arm_regret": 0.0011675192210281484,
    "always_ctm_regret": 0.0012179729832399774,
    "always_energy_regret": 0.004495308905149036,
    "p00_tie_fraction": 0.0
   }
  },
  "winner_accuracy_ce": {
   "dA": {
    "frozen_arm_all_nonzero": 0.59375,
    "n_all_nonzero": 160,
    "frozen_arm_material": 0.5826086956521739,
    "n_material": 115,
    "always_ctm_all": 0.59375,
    "always_ctm_material": 0.5826086956521739,
    "frozen_arm_regret": 0.008987842841880344,
    "always_ctm_regret": 0.008987842841880344,
    "always_energy_regret": 0.015504331254674148,
    "p00_tie_fraction": 0.0
   },
   "dF": {
    "frozen_arm_all_nonzero": 0.86875,
    "n_all_nonzero": 160,
    "frozen_arm_material": 0.990909090909091,
    "n_material": 110,
    "always_ctm_all": 0.86875,
    "always_ctm_material": 0.990909090909091,
    "frozen_arm_regret": 0.000652345812014729,
    "always_ctm_regret": 0.000652345812014729,
    "always_energy_regret": 0.020740648320504332,
    "p00_tie_fraction": 0.0
   },
   "dG": {
    "frozen_arm_all_nonzero": 0.86875,
    "n_all_nonzero": 160,
    "frozen_arm_material": 1.0,
    "n_material": 17,
    "always_ctm_all": 0.86875,
    "always_ctm_material": 1.0,
    "frozen_arm_regret": 0.00013525219506920483,
    "always_ctm_regret": 0.00013525219506920483,
    "always_energy_regret": 0.004442011130926285,
    "p00_tie_fraction": 0.0
   }
  },
  "ho_per_target": {
   "dG": {
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
   "dA": {
    "dK": {
     "cifar10": "HO-RETAINED",
     "cifar100": "HO-NOT-RETAINED",
     "supercifar100": "HO-NOT-RETAINED",
     "tinyimagenet": "HO-NOT-RETAINED"
    },
    "dF": {
     "cifar10": "HO-RETAINED",
     "cifar100": "HO-NOT-RETAINED",
     "supercifar100": "HO-NOT-RETAINED",
     "tinyimagenet": "HO-NOT-RETAINED"
    }
   },
   "dF": {
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
   "dGbal": {
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
   "dFbal": {
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
   }
  }
 },
 "vgg_new_shifts": {
  "n_cells": 1120,
  "id_error_available": true,
  "disagreement": {
   "dA_vs_dF": {
    "n": 1120,
    "opposite": 185,
    "opposite_frac": 0.16517857142857142,
    "opposite_among_dA_material": [
     107,
     904
    ],
    "opposite_among_dF_material": [
     89,
     873
    ]
   },
   "dA_vs_dG": {
    "n": 1120,
    "opposite": 185,
    "opposite_frac": 0.16517857142857142,
    "opposite_among_dA_material": [
     107,
     904
    ],
    "opposite_among_dG_material": [
     36,
     346
    ]
   },
   "dF_vs_dG": {
    "n": 1120,
    "opposite": 0,
    "opposite_frac": 0.0,
    "opposite_among_dF_material": [
     0,
     873
    ],
    "opposite_among_dG_material": [
     0,
     346
    ]
   }
  },
  "winner_accuracy": {
   "dA": {
    "frozen_arm_all_nonzero": 0.5133928571428571,
    "n_all_nonzero": 1120,
    "frozen_arm_material": 0.5176991150442478,
    "n_material": 904,
    "always_ctm_all": 0.48839285714285713,
    "always_ctm_material": 0.49004424778761063,
    "frozen_arm_regret": 0.022320178571428574,
    "always_ctm_regret": 0.023968035714285716,
    "always_energy_regret": 0.032236875,
    "p00_tie_fraction": 0.0
   },
   "dF": {
    "frozen_arm_all_nonzero": 0.55,
    "n_all_nonzero": 1120,
    "frozen_arm_material": 0.54524627720504,
    "n_material": 873,
    "always_ctm_all": 0.5321428571428571,
    "always_ctm_material": 0.5429553264604811,
    "frozen_arm_regret": 0.022262700407663632,
    "always_ctm_regret": 0.023190169950805795,
    "always_energy_regret": 0.02063032769157852,
    "p00_tie_fraction": 0.0
   },
   "dG": {
    "frozen_arm_all_nonzero": 0.55,
    "n_all_nonzero": 1120,
    "frozen_arm_material": 0.5780346820809249,
    "n_material": 346,
    "always_ctm_all": 0.5321428571428571,
    "always_ctm_material": 0.5606936416184971,
    "frozen_arm_regret": 0.004037160714285714,
    "always_ctm_regret": 0.004167428571428571,
    "always_energy_regret": 0.004395776785714285,
    "p00_tie_fraction": 0.0
   }
  },
  "e2_per_target": {
   "dG": {
    "dK": {
     "cifar10": [
      "HO-RETAINED",
      true
     ],
     "cifar100": [
      "HO-RETAINED",
      true
     ],
     "supercifar100": [
      "HO-RETAINED",
      true
     ],
     "tinyimagenet": [
      "HO-RETAINED",
      true
     ]
    },
    "dF": {
     "cifar10": [
      "HO-RETAINED",
      true
     ],
     "cifar100": [
      "HO-RETAINED",
      true
     ],
     "supercifar100": [
      "HO-RETAINED",
      true
     ],
     "tinyimagenet": [
      "HO-UNINFORMATIVE",
      false
     ]
    }
   },
   "dA": {
    "dK": {
     "cifar10": [
      "HO-NOT-RETAINED",
      true
     ],
     "cifar100": [
      "HO-NOT-RETAINED",
      true
     ],
     "supercifar100": [
      "HO-NOT-RETAINED",
      true
     ],
     "tinyimagenet": [
      "HO-RETAINED",
      true
     ]
    },
    "dF": {
     "cifar10": [
      "HO-NOT-RETAINED",
      true
     ],
     "cifar100": [
      "HO-NOT-RETAINED",
      true
     ],
     "supercifar100": [
      "HO-NOT-RETAINED",
      true
     ],
     "tinyimagenet": [
      "HO-RETAINED",
      true
     ]
    }
   },
   "dF": {
    "dK": {
     "cifar10": [
      "HO-RETAINED",
      true
     ],
     "cifar100": [
      "HO-UNINFORMATIVE",
      false
     ],
     "supercifar100": [
      "HO-RETAINED",
      true
     ],
     "tinyimagenet": [
      "HO-RETAINED",
      true
     ]
    },
    "dF": {
     "cifar10": [
      "HO-RETAINED",
      true
     ],
     "cifar100": [
      "HO-UNINFORMATIVE",
      false
     ],
     "supercifar100": [
      "HO-RETAINED",
      true
     ],
     "tinyimagenet": [
      "HO-UNINFORMATIVE",
      false
     ]
    }
   }
  },
  "e1_registered_target_dG_reproduced": {
   "ckpt5": {
    "theory_acc": 0.5780346820809249,
    "severity_acc": 0.7167630057803468,
    "n_material": 346,
    "diff_point": -0.1387283236994219,
    "diff_ci95_refit_bootstrap": [
     -0.21854101033548887,
     -0.07849117038945198
    ],
    "B": 500
   },
   "loso": {
    "theory_acc": 0.5780346820809249,
    "severity_acc": 0.5924855491329479,
    "n_material": 346,
    "diff_point": -0.014450867052023031,
    "diff_ci95_refit_bootstrap": [
     -0.1818621305371932,
     0.04498853623449015
    ],
    "B": 500
   }
  },
  "e1_on_dA": {
   "ckpt5": {
    "theory_acc": 0.5176991150442478,
    "severity_acc": 0.6969026548672567,
    "n_material": 904,
    "diff_point": -0.17920353982300885,
    "diff_ci95_refit_bootstrap": [
     -0.22957235451599267,
     -0.12170985686815862
    ],
    "B": 500
   },
   "loso": {
    "theory_acc": 0.5176991150442478,
    "severity_acc": 0.6482300884955752,
    "n_material": 904,
    "diff_point": -0.13053097345132736,
    "diff_ci95_refit_bootstrap": [
     -0.1995500953952457,
     -0.033827692315272946
    ],
    "B": 500
   }
  },
  "e1_on_dF": {
   "ckpt5": {
    "theory_acc": 0.54524627720504,
    "severity_acc": 0.6460481099656358,
    "n_material": 873,
    "diff_point": -0.10080183276059573,
    "diff_ci95_refit_bootstrap": [
     -0.16262184636011973,
     -0.050215611890777515
    ],
    "B": 500
   },
   "loso": {
    "theory_acc": 0.54524627720504,
    "severity_acc": 0.56815578465063,
    "n_material": 873,
    "diff_point": -0.02290950744558995,
    "diff_ci95_refit_bootstrap": [
     -0.11617030540763404,
     0.02258925664895223
    ],
    "B": 500
   }
  }
 }
}
```
