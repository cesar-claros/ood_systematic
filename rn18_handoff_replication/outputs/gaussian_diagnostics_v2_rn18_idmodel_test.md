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
     "approximation_A0_vs_G0": 0.1071251057261632,
     "approximation_A1_vs_G1": 0.00496230606693184,
     "mismatch_G0_vs_observed": 0.09929876726107163,
     "mismatch_G1_vs_observed": 0.02995046108283779,
     "mapping_dictionary_vs_observed": 0.17195202062797924,
     "dictionary_vs_G1": 0.15558274370422523
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.04407408727115787,
     "approximation_A1_vs_G1": 0.018193789728639914,
     "mismatch_G0_vs_observed": 0.13750795598246834,
     "mismatch_G1_vs_observed": 0.03796663937080991,
     "mapping_dictionary_vs_observed": 0.1596681666625344,
     "dictionary_vs_G1": 0.14258820436507313
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.005103115805170754,
    "CTM": 0.00920437374575571
   },
   "origin_ratio_mean": 0.7540853688528785,
   "gap_sign_agreement_material": {
    "dictionary": 0.6792452830188679,
    "G0_MC": 0.8679245283018868,
    "G1_MC": 0.8679245283018868,
    "A1_TAYLOR": 0.8113207547169812
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.5795454545454546,
    "G0_MC": 0.7613636363636364,
    "G1_MC": 0.7727272727272727,
    "A1_TAYLOR": 0.7272727272727273
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 0.0007712624289772727,
    "ood_G1": 0.11403113911506245
   },
   "G1_MC_resolved_fraction": 0.9545454545454546
  },
  "cifar100": {
   "n": 96,
   "n_material": 66,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.04388877035938308,
     "approximation_A1_vs_G1": 0.010091722610311013,
     "mismatch_G0_vs_observed": 0.08309802547643581,
     "mismatch_G1_vs_observed": 0.06432061016112565,
     "mapping_dictionary_vs_observed": 0.23263861651848994,
     "dictionary_vs_G1": 0.23955289808443428
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.04474700928354367,
     "approximation_A1_vs_G1": 0.07350872441324995,
     "mismatch_G0_vs_observed": 0.1263662143436571,
     "mismatch_G1_vs_observed": 0.1585071004378299,
     "mapping_dictionary_vs_observed": 0.2246905865632983,
     "dictionary_vs_G1": 0.3691855933657986
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.00437134858220815,
    "CTM": 0.01824329271912575
   },
   "origin_ratio_mean": 0.6655790719387363,
   "gap_sign_agreement_material": {
    "dictionary": 0.42424242424242425,
    "G0_MC": 0.5,
    "G1_MC": 0.3939393939393939,
    "A1_TAYLOR": 0.3181818181818182
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.3229166666666667,
    "G0_MC": 0.5104166666666666,
    "G1_MC": 0.3854166666666667,
    "A1_TAYLOR": 0.3333333333333333
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 0.021632944742838542,
    "ood_G1": 0.15246977776541873
   },
   "G1_MC_resolved_fraction": 0.9895833333333334
  },
  "supercifar100": {
   "n": 112,
   "n_material": 83,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.07222932006270429,
     "approximation_A1_vs_G1": 0.01605573034331778,
     "mismatch_G0_vs_observed": 0.09250832081990583,
     "mismatch_G1_vs_observed": 0.19840765129987684,
     "mapping_dictionary_vs_observed": 0.4129044628320185,
     "dictionary_vs_G1": 0.6113121141318955
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.15145351307681384,
     "approximation_A1_vs_G1": 0.0455934482145813,
     "mismatch_G0_vs_observed": 0.19102197915358202,
     "mismatch_G1_vs_observed": 0.3684710232019424,
     "mapping_dictionary_vs_observed": 0.4160187055959171,
     "dictionary_vs_G1": 0.7822943670754449
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.013878239478383746,
    "CTM": 0.018565926381519864
   },
   "origin_ratio_mean": 0.5640312326227938,
   "gap_sign_agreement_material": {
    "dictionary": 0.3614457831325301,
    "G0_MC": 0.5783132530120482,
    "G1_MC": 0.4578313253012048,
    "A1_TAYLOR": 0.4578313253012048
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.36607142857142855,
    "G0_MC": 0.6160714285714286,
    "G1_MC": 0.4107142857142857,
    "A1_TAYLOR": 0.4107142857142857
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 0.24596617275610905,
    "ood_G1": 0.15699374059296106
   },
   "G1_MC_resolved_fraction": 1.0
  },
  "tinyimagenet": {
   "n": 88,
   "n_material": 76,
   "mean_abs_error": {
    "Energy": {
     "approximation_A0_vs_G0": 0.02486795559071021,
     "approximation_A1_vs_G1": 0.01281054420869044,
     "mismatch_G0_vs_observed": 0.08905225958607414,
     "mismatch_G1_vs_observed": 0.08259716661735014,
     "mapping_dictionary_vs_observed": 0.23328968396473357,
     "dictionary_vs_G1": 0.2678671476999673
    },
    "CTM": {
     "approximation_A0_vs_G0": 0.06354740633878654,
     "approximation_A1_vs_G1": 0.19393638823382495,
     "mismatch_G0_vs_observed": 0.13846202099241994,
     "mismatch_G1_vs_observed": 0.21381089100133294,
     "mapping_dictionary_vs_observed": 0.251372251347427,
     "dictionary_vs_G1": 0.46518314234875985
    }
   },
   "mean_origin_effect_G1_centered_minus_uncentered": {
    "Energy": 0.017886649478565556,
    "CTM": 0.07830033295533874
   },
   "origin_ratio_mean": 0.5219400251625788,
   "gap_sign_agreement_material": {
    "dictionary": 0.3026315789473684,
    "G0_MC": 0.6578947368421053,
    "G1_MC": 0.631578947368421,
    "A1_TAYLOR": 0.6052631578947368
   },
   "gap_sign_agreement_all": {
    "dictionary": 0.29545454545454547,
    "G0_MC": 0.6136363636363636,
    "G1_MC": 0.6022727272727273,
    "A1_TAYLOR": 0.5795454545454546
   },
   "A1_CTM_undefined": 0,
   "A0_CTM_undefined": 0,
   "branch_switch_mean": {
    "id": 0.02288665771484375,
    "ood_G1": 0.21743470515205507
   },
   "G1_MC_resolved_fraction": 0.9886363636363636
  }
 }
}
```
