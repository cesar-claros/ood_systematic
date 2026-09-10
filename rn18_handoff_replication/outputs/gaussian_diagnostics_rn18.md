# Gaussian/Taylor diagnostics (fourshift_rn18; PRELIMINARY float32)

```
{
 "n_cells": 384,
 "n_psd_rejected": 13,
 "psd_rejected": [
  [
   "cifar100_paper_sweep__dg_bbresnet18_do1_run1_rew12",
   "mnist_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -1.036880333086158e-08"
   }
  ],
  [
   "cifar100_paper_sweep__dg_bbresnet18_do1_run1_rew6",
   "mnist_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -1.2928922978281216e-08"
   }
  ],
  [
   "cifar10_paper_sweep__confidnet_bbresnet18_do0_run1_rew2.2",
   "stl10_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -2.5883997778087247e-09"
   }
  ],
  [
   "cifar10_paper_sweep__devries_bbresnet18_do0_run1_rew2.2",
   "mnist_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -1.159427353104806e-09"
   }
  ],
  [
   "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew10",
   "mnist_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -1.0469542944879117e-09",
    "cov_res": "covariance not PSD beyond tolerance: min eig -3.4809373461365085e-10"
   }
  ],
  [
   "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew10",
   "fashionmnist_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -2.5888113676110176e-09"
   }
  ],
  [
   "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew10",
   "stl10_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -5.783388742363721e-09"
   }
  ],
  [
   "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew3",
   "mnist_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -1.670300755993601e-09"
   }
  ],
  [
   "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew3",
   "kmnist_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -9.447920713372925e-10"
   }
  ],
  [
   "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew3",
   "stl10_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -3.835937138797203e-09"
   }
  ],
  [
   "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew6",
   "mnist_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -1.733975814962497e-09"
   }
  ],
  [
   "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew6",
   "stl10_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -2.2724789754011463e-09"
   }
  ],
  [
   "cifar10_paper_sweep__dg_bbresnet18_do1_run1_rew3",
   "mnist_new",
   {
    "cov_glob": "covariance not PSD beyond tolerance: min eig -9.984556254329372e-10"
   }
  ]
 ],
 "by_source": {
  "cifar10": {
   "n": 77,
   "n_material": 43,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.07320606738347975,
     "approximation_A1_vs_G1": 0.0018042354167343668,
     "mismatch_G0_vs_observed": 0.13758617484538585,
     "mismatch_G1_vs_observed": 0.10401215398280651,
     "mapping_dictionary_vs_observed": 0.17010752588350256,
     "dictionary_vs_G1": 0.06609537190069606
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.049603508898129145,
     "approximation_A1_vs_G1": 0.026529624139630163,
     "mismatch_G0_vs_observed": 0.15743285663344644,
     "mismatch_G1_vs_observed": 0.14329133155686516,
     "mapping_dictionary_vs_observed": 0.15879478787406526,
     "dictionary_vs_G1": 0.015503456317200109
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": -1.1693463697059314e-05,
    "CTM": 0.0004425514053988749
   },
   "origin_ratio_mean": 0.777884751325791,
   "gap_sign_agreement_material": {
    "dictionary": 0.7209302325581395,
    "G0_MC": 0.8372093023255814,
    "G1_MC": 0.8837209302325582,
    "A1_TAYLOR": 0.7441860465116279
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.5844155844155844,
    "G0_MC": 0.7272727272727273,
    "G1_MC": 0.7532467532467533,
    "A1_TAYLOR": 0.7272727272727273
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 1.9023944805194807e-06,
    "ood_G1": 0.11649897914895126
   },
   "G1_MC_resolved_fraction": 1.0
  },
  "cifar100": {
   "n": 94,
   "n_material": 64,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.015489759634921104,
     "approximation_A1_vs_G1": 0.0013980767187789134,
     "mismatch_G0_vs_observed": 0.20767806819804166,
     "mismatch_G1_vs_observed": 0.20575229202838655,
     "mapping_dictionary_vs_observed": 0.23472432721127753,
     "dictionary_vs_G1": 0.031520500178154
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.03546974162023892,
     "approximation_A1_vs_G1": 0.010037187326497344,
     "mismatch_G0_vs_observed": 0.2141053479871851,
     "mismatch_G1_vs_observed": 0.1985131446556842,
     "mapping_dictionary_vs_observed": 0.2265755501591513,
     "dictionary_vs_G1": 0.03740647088470438
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.004271491093838476,
    "CTM": 0.003335617894821984
   },
   "origin_ratio_mean": 0.6530711507550103,
   "gap_sign_agreement_material": {
    "dictionary": 0.40625,
    "G0_MC": 0.640625,
    "G1_MC": 0.625,
    "A1_TAYLOR": 0.578125
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.30851063829787234,
    "G0_MC": 0.6063829787234043,
    "G1_MC": 0.6063829787234043,
    "A1_TAYLOR": 0.5
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 0.010642349567819148,
    "ood_G1": 0.15062192521455
   },
   "G1_MC_resolved_fraction": 0.8723404255319149
  },
  "supercifar100": {
   "n": 112,
   "n_material": 83,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.028278262778826568,
     "approximation_A1_vs_G1": 0.0010070223864199399,
     "mismatch_G0_vs_observed": 0.36022721156265053,
     "mismatch_G1_vs_observed": 0.3571444683021733,
     "mapping_dictionary_vs_observed": 0.4129044628320185,
     "dictionary_vs_G1": 0.05603140169407948
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.06095847273130223,
     "approximation_A1_vs_G1": 0.0196603122691556,
     "mismatch_G0_vs_observed": 0.3880212740178619,
     "mismatch_G1_vs_observed": 0.3733517414229257,
     "mapping_dictionary_vs_observed": 0.4160187055959171,
     "dictionary_vs_G1": 0.04704261984404661
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.010330725461244588,
    "CTM": 0.004995524164821416
   },
   "origin_ratio_mean": 0.5640312326227938,
   "gap_sign_agreement_material": {
    "dictionary": 0.3614457831325301,
    "G0_MC": 0.5180722891566265,
    "G1_MC": 0.5180722891566265,
    "A1_TAYLOR": 0.7469879518072289
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.36607142857142855,
    "G0_MC": 0.5714285714285714,
    "G1_MC": 0.5714285714285714,
    "A1_TAYLOR": 0.7232142857142857
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 0.0001998557183975564,
    "ood_G1": 0.15699374059296106
   },
   "G1_MC_resolved_fraction": 0.9910714285714286
  },
  "tinyimagenet": {
   "n": 88,
   "n_material": 76,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.0032092960164881194,
     "approximation_A1_vs_G1": 0.0008597841915115538,
     "mismatch_G0_vs_observed": 0.22488907485306261,
     "mismatch_G1_vs_observed": 0.2246182746960358,
     "mapping_dictionary_vs_observed": 0.23328968396473357,
     "dictionary_vs_G1": 0.010631620758479645
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.026301901059854086,
     "approximation_A1_vs_G1": 0.024845080994168288,
     "mismatch_G0_vs_observed": 0.24231838551407509,
     "mismatch_G1_vs_observed": 0.23291802651990545,
     "mapping_dictionary_vs_observed": 0.251372251347427,
     "dictionary_vs_G1": 0.023060053814750837
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.00019958639009432208,
    "CTM": 0.018310100382024595
   },
   "origin_ratio_mean": 0.5219400251625788,
   "gap_sign_agreement_material": {
    "dictionary": 0.3026315789473684,
    "G0_MC": 0.5394736842105263,
    "G1_MC": 0.6578947368421053,
    "A1_TAYLOR": 0.6710526315789473
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.29545454545454547,
    "G0_MC": 0.5454545454545454,
    "G1_MC": 0.6363636363636364,
    "A1_TAYLOR": 0.625
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 0.003980019309303977,
    "ood_G1": 0.21743462108159192
   },
   "G1_MC_resolved_fraction": 0.875
  }
 }
}
```
