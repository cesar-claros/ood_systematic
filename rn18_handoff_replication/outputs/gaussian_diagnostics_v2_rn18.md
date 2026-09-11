# Gaussian/Taylor diagnostics (fourshift_v2_rn18; float64)

```
{
 "n_cells": 384,
 "n_psd_rejected": 0,
 "psd_rejected": [],
 "by_source": {
  "cifar10": {
   "n": 88,
   "n_material": 53,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.07000732416526333,
     "approximation_A1_vs_G1": 0.0016333745425057,
     "mismatch_G0_vs_observed": 0.1413542435824871,
     "mismatch_G1_vs_observed": 0.10839928970065983,
     "mapping_dictionary_vs_observed": 0.17195202062797924,
     "dictionary_vs_G1": 0.0635527309273194
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.04898101514175452,
     "approximation_A1_vs_G1": 0.026252886487302726,
     "mismatch_G0_vs_observed": 0.15844942624108357,
     "mismatch_G1_vs_observed": 0.14598785576007584,
     "mapping_dictionary_vs_observed": 0.1596681666625344,
     "dictionary_vs_G1": 0.013680310902458543
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": -0.0003084832633083519,
    "CTM": 0.00037295892834663215
   },
   "origin_ratio_mean": 0.7540853688528785,
   "gap_sign_agreement_material": {
    "dictionary": 0.6792452830188679,
    "G0_MC": 0.8679245283018868,
    "G1_MC": 0.9056603773584906,
    "A1_TAYLOR": 0.7169811320754716
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.5795454545454546,
    "G0_MC": 0.7613636363636364,
    "G1_MC": 0.7840909090909091,
    "A1_TAYLOR": 0.7045454545454546
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 1.6645951704545458e-06,
    "ood_G1": 0.11403113911506245
   },
   "G1_MC_resolved_fraction": 1.0
  },
  "cifar100": {
   "n": 96,
   "n_material": 66,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.016186891963362083,
     "approximation_A1_vs_G1": 0.0014037505788770208,
     "mismatch_G0_vs_observed": 0.20624419404168923,
     "mismatch_G1_vs_observed": 0.2037800322905183,
     "mapping_dictionary_vs_observed": 0.23263861651848994,
     "dictionary_vs_G1": 0.031734549650605764
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.036304184319809134,
     "approximation_A1_vs_G1": 0.010285940300480158,
     "mismatch_G0_vs_observed": 0.21196911904489002,
     "mismatch_G1_vs_observed": 0.19722957032471897,
     "mapping_dictionary_vs_observed": 0.2246905865632983,
     "dictionary_vs_G1": 0.04229749438611036
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.0049593323220809395,
    "CTM": 0.003958694698909923
   },
   "origin_ratio_mean": 0.6655790719387363,
   "gap_sign_agreement_material": {
    "dictionary": 0.42424242424242425,
    "G0_MC": 0.6515151515151515,
    "G1_MC": 0.6060606060606061,
    "A1_TAYLOR": 0.5606060606060606
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.3229166666666667,
    "G0_MC": 0.6145833333333334,
    "G1_MC": 0.59375,
    "A1_TAYLOR": 0.4895833333333333
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 0.013390909830729165,
    "ood_G1": 0.15246977776541873
   },
   "G1_MC_resolved_fraction": 0.875
  },
  "supercifar100": {
   "n": 112,
   "n_material": 83,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.028278263026129253,
     "approximation_A1_vs_G1": 0.0010070224921571046,
     "mismatch_G0_vs_observed": 0.36022721174891503,
     "mismatch_G1_vs_observed": 0.3571444683553917,
     "mapping_dictionary_vs_observed": 0.4129044628320185,
     "dictionary_vs_G1": 0.05603140164086106
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.06095847234660036,
     "approximation_A1_vs_G1": 0.019660312440728436,
     "mismatch_G0_vs_observed": 0.3880212736985514,
     "mismatch_G1_vs_observed": 0.3733517411302243,
     "mapping_dictionary_vs_observed": 0.4160187055959171,
     "dictionary_vs_G1": 0.047042620243184854
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.010330725381416942,
    "CTM": 0.004995524164821417
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
     "approximation_A0_vs_G0": 0.003209296052619684,
     "approximation_A1_vs_G1": 0.0008597840873473223,
     "mismatch_G0_vs_observed": 0.22488907481919637,
     "mismatch_G1_vs_observed": 0.2246182747976346,
     "mapping_dictionary_vs_observed": 0.23328968396473357,
     "dictionary_vs_G1": 0.01063162065688082
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.0263019005990966,
     "approximation_A1_vs_G1": 0.0248450812461495,
     "mismatch_G0_vs_observed": 0.242318385006081,
     "mismatch_G1_vs_observed": 0.23291802645217288,
     "mapping_dictionary_vs_observed": 0.251372251347427,
     "dictionary_vs_G1": 0.023060053611553187
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.00019958632236177272,
    "CTM": 0.018310100144960664
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
    "ood_G1": 0.21743470515205507
   },
   "G1_MC_resolved_fraction": 0.875
  }
 }
}
```
