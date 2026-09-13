# Boundary audit (literal model)

```
{
 "summary": {
  "n_points": 154,
  "mls_boundary": {
   "n": 112,
   "max_abs_err_MLS": 0.006777356944478563,
   "max_abs_err_Energy": 0.006757862796734071,
   "max_abs_err_CTM": 0.004959762925543121,
   "max_abs_err_Maha": 0.005147209623846516,
   "frac_within_0.01_MLS": 1.0,
   "frac_within_0.01_Energy": 1.0,
   "frac_within_0.01_CTM": 1.0,
   "frac_within_0.01_Maha": 1.0,
   "max_gap_err": 0.007851823734930097,
   "median_gap_err_over_abs_gap": 0.005457906672638598,
   "sign_agreement": 1.0,
   "id_switch_max": 0.0021,
   "ood_switch_max": 0.0341,
   "id_switch_vs_union_bound_max_ratio": 2.112902485190394
  },
  "energy_ctm_boundary": {
   "n": 42,
   "max_abs_err_MLS": 0.0016893237391200078,
   "max_abs_err_Energy": 0.005750893128397605,
   "max_abs_err_CTM": 0.0008937699692425083,
   "max_abs_err_Maha": 0.12118417330929743,
   "frac_within_0.01_MLS": 1.0,
   "frac_within_0.01_Energy": 1.0,
   "frac_within_0.01_CTM": 1.0,
   "frac_within_0.01_Maha": 0.5476190476190477,
   "max_gap_err": 0.005078827773314587,
   "median_gap_err_over_abs_gap": 0.8563620622226236,
   "sign_agreement": 0.5714285714285714,
   "id_switch_max": 0.0021,
   "ood_switch_max": 0.8579,
   "id_switch_vs_union_bound_max_ratio": 2.112902485190394
  },
  "mls_boundary_displacement_max_abs": 0.002662562475799879,
  "energy_ctm_rel_displacement_max_abs": 0.09880239520958094
 },
 "displacements": [
  {
   "C": 10,
   "D": 128,
   "s": 6,
   "a": 0.4,
   "theta_deg": 0.0,
   "mls_boundary_ga_mc": 0.9973374375242001,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.8598998775161195,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 10,
   "D": 128,
   "s": 6,
   "a": 0.4,
   "theta_deg": 20.0,
   "mls_boundary_ga_mc": 0.9988147552240639,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.7214535448267296,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 10,
   "D": 128,
   "s": 6,
   "a": 0.8,
   "theta_deg": 0.0,
   "mls_boundary_ga_mc": 0.9998347021121738,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.7148224603289411,
   "energy_ctm_gamma_mc": 0.6949259188588908,
   "energy_ctm_rel_displacement": -0.027834242170978442
  },
  {
   "C": 10,
   "D": 128,
   "s": 6,
   "a": 0.8,
   "theta_deg": 20.0,
   "mls_boundary_ga_mc": 1.0016667638208703,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.47261264606284187,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 10,
   "D": 128,
   "s": 16,
   "a": 0.4,
   "theta_deg": 0.0,
   "mls_boundary_ga_mc": 0.9991590247105456,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.6438347595210153,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 10,
   "D": 128,
   "s": 16,
   "a": 0.4,
   "theta_deg": 20.0,
   "mls_boundary_ga_mc": 0.9997063463212859,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.6438347595210153,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 10,
   "D": 128,
   "s": 16,
   "a": 0.8,
   "theta_deg": 0.0,
   "mls_boundary_ga_mc": 1.0004522646520457,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.7499572787265791,
   "energy_ctm_gamma_mc": 0.6784660241189988,
   "energy_ctm_rel_displacement": -0.09532710280373816
  },
  {
   "C": 10,
   "D": 128,
   "s": 16,
   "a": 0.8,
   "theta_deg": 20.0,
   "mls_boundary_ga_mc": 0.9997031026758224,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.7037018730489468,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 100,
   "D": 512,
   "s": 6,
   "a": 0.4,
   "theta_deg": 0.0,
   "mls_boundary_ga_mc": 1.0003444204872474,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.6448698786585417,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 100,
   "D": 512,
   "s": 6,
   "a": 0.4,
   "theta_deg": 20.0,
   "mls_boundary_ga_mc": 0.9982821107139365,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.8516218337387103,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 100,
   "D": 512,
   "s": 6,
   "a": 0.8,
   "theta_deg": 0.0,
   "mls_boundary_ga_mc": 0.9975505689611933,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.31025898339471064,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 100,
   "D": 512,
   "s": 6,
   "a": 0.8,
   "theta_deg": 20.0,
   "mls_boundary_ga_mc": 1.000246681762183,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.39224259295496317,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  },
  {
   "C": 100,
   "D": 512,
   "s": 16,
   "a": 0.4,
   "theta_deg": 0.0,
   "mls_boundary_ga_mc": 1.0000176570647898,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": null
  },
  {
   "C": 100,
   "D": 512,
   "s": 16,
   "a": 0.4,
   "theta_deg": 20.0,
   "mls_boundary_ga_mc": 0.9991820073044563,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": null
  },
  {
   "C": 100,
   "D": 512,
   "s": 16,
   "a": 0.8,
   "theta_deg": 0.0,
   "mls_boundary_ga_mc": 1.0003466443134519,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.7659706265548213,
   "energy_ctm_gamma_mc": 0.6902908939910215,
   "energy_ctm_rel_displacement": -0.09880239520958094
  },
  {
   "C": 100,
   "D": 512,
   "s": 16,
   "a": 0.8,
   "theta_deg": 20.0,
   "mls_boundary_ga_mc": 1.000782244558397,
   "mls_boundary_ga_predicted": 1.0,
   "energy_ctm_gamma_analytic": 0.3266689242466116,
   "energy_ctm_gamma_mc": null,
   "energy_ctm_rel_displacement": null
  }
 ]
}
```
