# Audit-11 R11.5/R11.6: E2, paradigm crossings, joint audit under the corrected panel

```
{
 "frozen_reference": {
  "joint_primary_M1_minus_M0plus": "+0.007 [-0.001, 0.017]",
  "e2": "retained cifar10/cifar100/tinyimagenet; reversed supercifar100; verdict INCONCLUSIVE",
  "paradigm_crossings": "RETAINED dg + devries; NOT-RETAINED confidnet (left-censored middle)"
 },
 "e2_corrected": {
  "per_source": {
   "cifar10": {
    "tertile_bounds_from_train": [
     0.48133631158733997,
     2.266577117162825
    ],
    "strata": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": -14.15741833079225,
      "g_at_max_d": -5.433948609063059,
      "band_q95": 4.958158493266296,
      "tie_region": null,
      "n_ckpt": 60
     },
     "middle": {
      "n_ckpt": 0,
      "note": "too few checkpoints"
     },
     "weak": {
      "n_ckpt": 0,
      "note": "too few checkpoints"
     }
    },
    "crossing_values": {
     "strong": null,
     "middle": null,
     "weak": null
    },
    "outcome": "INCONCLUSIVE"
   },
   "cifar100": {
    "tertile_bounds_from_train": [
     0.16332256660577887,
     0.5674245934133577
    ],
    "strata": {
     "strong": {
      "n_ckpt": 1,
      "note": "too few checkpoints"
     },
     "middle": {
      "n_ckpt": 0,
      "note": "too few checkpoints"
     },
     "weak": {
      "n_sign_changes": 1,
      "all_crossings": [
       -0.664
      ],
      "first_up_crossing": -0.664,
      "bracketed_by_observed": true,
      "g_at_min_d": -4.097881023101608,
      "g_at_max_d": 5.246479125590219,
      "band_q95": 1.3033985519024793,
      "tie_region": [
       -0.696,
       -0.632
      ],
      "n_ckpt": 69
     }
    },
    "crossing_values": {
     "strong": null,
     "middle": null,
     "weak": -0.664
    },
    "outcome": "REVERSED"
   },
   "supercifar100": {
    "tertile_bounds_from_train": [
     0.3694687027536223,
     0.9717123606212885
    ],
    "strata": {
     "strong": {
      "n_sign_changes": 1,
      "all_crossings": [
       -1.08
      ],
      "first_up_crossing": -1.08,
      "bracketed_by_observed": true,
      "g_at_min_d": -1.1181990778133692,
      "g_at_max_d": 12.848871123423585,
      "band_q95": 2.986861520780066,
      "tie_region": [
       -1.175,
       -0.835
      ],
      "n_ckpt": 46
     },
     "middle": {
      "n_sign_changes": 1,
      "all_crossings": [
       -0.003
      ],
      "first_up_crossing": -0.003,
      "bracketed_by_observed": true,
      "g_at_min_d": -11.596972915405047,
      "g_at_max_d": 5.890755568870771,
      "band_q95": 3.574774140463048,
      "tie_region": [
       -0.113,
       0.104
      ],
      "n_ckpt": 18
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": true,
      "g_at_min_d": -15.423041509643488,
      "g_at_max_d": -3.3224208969435054,
      "band_q95": 6.200409461625014,
      "tie_region": [
       -0.635,
       1.435
      ],
      "n_ckpt": 26
     }
    },
    "crossing_values": {
     "strong": -1.08,
     "middle": -0.003,
     "weak": null
    },
    "outcome": "RETAINED"
   },
   "tinyimagenet": {
    "tertile_bounds_from_train": [
     0.16858164699433786,
     0.8011719130961892
    ],
    "strata": {
     "strong": {
      "n_ckpt": 0,
      "note": "too few checkpoints"
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 10.557764385077682,
      "g_at_max_d": 14.042398467870267,
      "band_q95": 1.6145482441497438,
      "tie_region": null,
      "n_ckpt": 20
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 5.222819438004288,
      "g_at_max_d": 5.524514009877119,
      "band_q95": 1.2256592093909442,
      "tie_region": null,
      "n_ckpt": 40
     }
    },
    "crossing_values": {
     "strong": null,
     "middle": "<=range-min",
     "weak": "<=range-min"
    },
    "outcome": "INCONCLUSIVE"
   }
  },
  "verdict": "INCONCLUSIVE",
  "outcomes": [
   "INCONCLUSIVE",
   "REVERSED",
   "RETAINED",
   "INCONCLUSIVE"
  ]
 },
 "paradigm_crossings_corrected": {
  "confidnet": {
   "n_ckpt": 40,
   "vc_range": [
    0.0952,
    2.3763
   ],
   "n_severities": 32,
   "n_material": 123,
   "frac_positive": 0.675,
   "pooled": {
    "n_sign_changes": 1,
    "all_crossings": [
     -1.09
    ],
    "first_up_crossing": -1.09,
    "bracketed_by_observed": true,
    "g_at_min_d": -13.200487760483,
    "g_at_max_d": 6.067109081074113,
    "band_q95": 9.486135224986153,
    "tie_region": [
     -1.295,
     1.557
    ]
   },
   "strata": {
    "strong": {
     "first_up_crossing": 1.406,
     "display": 1.406,
     "tie_region": [
      -1.354,
      1.557
     ],
     "n_ckpt": 13
    },
    "middle": {
     "first_up_crossing": -1.077,
     "display": -1.077,
     "tie_region": [
      -1.136,
      -0.285
     ],
     "n_ckpt": 13
    },
    "weak": {
     "first_up_crossing": -1.115,
     "display": -1.115,
     "tie_region": [
      -1.176,
      -0.701
     ],
     "n_ckpt": 14
    }
   },
   "ordering_retained": false,
   "verdict": "NOT-RETAINED"
  },
  "devries": {
   "n_ckpt": 40,
   "vc_range": [
    0.0795,
    2.9038
   ],
   "n_severities": 32,
   "n_material": 117,
   "frac_positive": 0.607,
   "pooled": {
    "n_sign_changes": 1,
    "all_crossings": [
     -1.07
    ],
    "first_up_crossing": -1.07,
    "bracketed_by_observed": true,
    "g_at_min_d": -29.181653196514013,
    "g_at_max_d": 3.1201637288307134,
    "band_q95": 15.085828680268593,
    "tie_region": [
     -1.285,
     1.557
    ]
   },
   "strata": {
    "strong": {
     "first_up_crossing": 1.408,
     "display": 1.408,
     "tie_region": [
      -1.334,
      1.557
     ],
     "n_ckpt": 13
    },
    "middle": {
     "first_up_crossing": -0.25,
     "display": -0.25,
     "tie_region": [
      -1.344,
      1.557
     ],
     "n_ckpt": 13
    },
    "weak": {
     "first_up_crossing": -1.125,
     "display": -1.125,
     "tie_region": [
      -1.176,
      -1.077
     ],
     "n_ckpt": 14
    }
   },
   "ordering_retained": false,
   "verdict": "NOT-RETAINED"
  },
  "dg": {
   "n_ckpt": 200,
   "vc_range": [
    0.0593,
    250.5125
   ],
   "n_severities": 32,
   "n_material": 478,
   "frac_positive": 0.579,
   "pooled": {
    "n_sign_changes": 1,
    "all_crossings": [
     -1.08
    ],
    "first_up_crossing": -1.08,
    "bracketed_by_observed": true,
    "g_at_min_d": -10.640592256939124,
    "g_at_max_d": 3.4085566064229678,
    "band_q95": 4.880666813639675,
    "tie_region": [
     -1.156,
     1.557
    ]
   },
   "strata": {
    "strong": {
     "first_up_crossing": -1.232,
     "display": -1.232,
     "tie_region": [
      -1.324,
      1.398
     ],
     "n_ckpt": 66
    },
    "middle": {
     "first_up_crossing": -1.062,
     "display": -1.062,
     "tie_region": [
      -1.097,
      -0.285
     ],
     "n_ckpt": 67
    },
    "weak": {
     "first_up_crossing": -1.057,
     "display": -1.057,
     "tie_region": [
      -1.106,
      1.557
     ],
     "n_ckpt": 67
    }
   },
   "ordering_retained": true,
   "verdict": "RETAINED"
  },
  "equal_paradigm_pooled": {
   "first_up_crossing": -1.081,
   "band_q95": 6.414,
   "tie_region": [
    -1.215,
    1.557
   ]
  }
 },
 "joint_corrected": {
  "n_material": 718,
  "frac_positive": 0.6,
  "frac_material_out_of_support": 0.313,
  "arms": {
   "M0": {
    "bal_macro": 0.859,
    "bal_macro_two_class": 0.669,
    "bal_row": 0.797,
    "sign_macro": 0.919,
    "sign_row": 0.829
   },
   "M0plus": {
    "bal_macro": 0.947,
    "bal_macro_two_class": 0.873,
    "bal_row": 0.919,
    "sign_macro": 0.972,
    "sign_row": 0.919
   },
   "M1": {
    "bal_macro": 0.954,
    "bal_macro_two_class": 0.89,
    "bal_row": 0.926,
    "sign_macro": 0.977,
    "sign_row": 0.933
   },
   "MH": {
    "bal_macro": 0.946,
    "bal_macro_two_class": 0.869,
    "bal_row": 0.911,
    "sign_macro": 0.966,
    "sign_row": 0.919
   },
   "MHG": {
    "bal_macro": 0.954,
    "bal_macro_two_class": 0.889,
    "bal_row": 0.927,
    "sign_macro": 0.976,
    "sign_row": 0.933
   },
   "sev_pooled": {
    "bal_macro": 0.719,
    "bal_macro_two_class": 0.664,
    "bal_row": 0.608,
    "sign_macro": 0.742,
    "sign_row": 0.684
   },
   "sev_source": {
    "bal_macro": 0.875,
    "bal_macro_two_class": 0.714,
    "bal_row": 0.801,
    "sign_macro": 0.917,
    "sign_row": 0.83
   }
  },
  "differences": {
   "M0plus-M0": {
    "bal_macro": {
     "point": 0.089,
     "ci95": [
      0.057,
      0.107
     ]
    },
    "bal_row": {
     "point": 0.121,
     "ci95": [
      0.078,
      0.166
     ]
    },
    "sign_row": {
     "point": 0.091,
     "ci95": [
      0.048,
      0.136
     ]
    }
   },
   "M1-M0plus": {
    "bal_macro": {
     "point": 0.007,
     "ci95": [
      0.001,
      0.017
     ]
    },
    "bal_row": {
     "point": 0.008,
     "ci95": [
      -0.007,
      0.024
     ]
    },
    "sign_row": {
     "point": 0.014,
     "ci95": [
      0.0,
      0.028
     ]
    }
   },
   "M1-M0": {
    "bal_macro": {
     "point": 0.096,
     "ci95": [
      0.065,
      0.113
     ]
    },
    "bal_row": {
     "point": 0.129,
     "ci95": [
      0.088,
      0.171
     ]
    },
    "sign_row": {
     "point": 0.104,
     "ci95": [
      0.068,
      0.145
     ]
    }
   },
   "MH-M0plus": {
    "bal_macro": {
     "point": -0.002,
     "ci95": [
      -0.016,
      0.012
     ]
    },
    "bal_row": {
     "point": -0.008,
     "ci95": [
      -0.036,
      0.019
     ]
    },
    "sign_row": {
     "point": 0.0,
     "ci95": [
      -0.024,
      0.023
     ]
    }
   },
   "MHG-MH": {
    "bal_macro": {
     "point": 0.008,
     "ci95": [
      0.002,
      0.016
     ]
    },
    "bal_row": {
     "point": 0.016,
     "ci95": [
      0.003,
      0.032
     ]
    },
    "sign_row": {
     "point": 0.014,
     "ci95": [
      0.003,
      0.027
     ]
    }
   },
   "M1-MH": {
    "bal_macro": {
     "point": 0.009,
     "ci95": [
      -0.005,
      0.025
     ]
    },
    "bal_row": {
     "point": 0.015,
     "ci95": [
      -0.003,
      0.037
     ]
    },
    "sign_row": {
     "point": 0.014,
     "ci95": [
      -0.001,
      0.033
     ]
    }
   },
   "M0plus-sev_source": {
    "bal_macro": {
     "point": 0.073,
     "ci95": [
      0.03,
      0.107
     ]
    },
    "bal_row": {
     "point": 0.118,
     "ci95": [
      0.071,
      0.162
     ]
    },
    "sign_row": {
     "point": 0.089,
     "ci95": [
      0.048,
      0.134
     ]
    }
   }
  },
  "per_source": {
   "cifar10": {
    "n_material": 170,
    "frac_positive": 0.112,
    "M0plus_bal": 0.921,
    "M1_bal": 0.974,
    "MH_bal": 0.921,
    "MHG_bal": 0.921
   },
   "cifar100": {
    "n_material": 90,
    "frac_positive": 0.978,
    "M0plus_bal": 0.744,
    "M1_bal": 0.75,
    "MH_bal": 0.744,
    "MHG_bal": 0.75
   },
   "supercifar100": {
    "n_material": 314,
    "frac_positive": 0.573,
    "M0plus_bal": 0.831,
    "M1_bal": 0.844,
    "MH_bal": 0.818,
    "MHG_bal": 0.851
   },
   "tinyimagenet": {
    "n_material": 144,
    "frac_positive": 1.0,
    "M0plus_bal": 1.0,
    "M1_bal": 1.0,
    "MH_bal": 1.0,
    "MHG_bal": 1.0
   }
  },
  "per_paradigm": {
   "confidnet": {
    "n_material": 123,
    "frac_positive": 0.675,
    "M0plus_bal": 0.975,
    "M1_bal": 0.975,
    "MH_bal": 1.0,
    "MHG_bal": 1.0
   },
   "devries": {
    "n_material": 117,
    "frac_positive": 0.607,
    "M0plus_bal": 1.0,
    "M1_bal": 1.0,
    "MH_bal": 0.972,
    "MHG_bal": 0.986
   },
   "dg": {
    "n_material": 478,
    "frac_positive": 0.579,
    "M0plus_bal": 0.884,
    "M1_bal": 0.897,
    "MH_bal": 0.877,
    "MHG_bal": 0.897
   }
  },
  "influence": {
   "drop_cifar100": {
    "M1-M0plus_bal_macro": 0.008,
    "M1-M0plus_bal_row": 0.011
   },
   "drop_cifar10": {
    "M1-M0plus_bal_macro": 0.004,
    "M1-M0plus_bal_row": -0.008
   },
   "drop_supercifar100": {
    "M1-M0plus_bal_macro": 0.007,
    "M1-M0plus_bal_row": 0.006
   },
   "drop_tinyimagenet": {
    "M1-M0plus_bal_macro": 0.009,
    "M1-M0plus_bal_row": 0.017
   },
   "drop_confidnet": {
    "M1-M0plus_bal_macro": 0.011,
    "M1-M0plus_bal_row": 0.01
   },
   "drop_devries": {
    "M1-M0plus_bal_macro": 0.011,
    "M1-M0plus_bal_row": 0.009
   },
   "drop_dg": {
    "M1-M0plus_bal_macro": 0.0,
    "M1-M0plus_bal_row": 0.0
   }
  }
 }
}
```
