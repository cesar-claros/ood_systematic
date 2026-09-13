# Temperature audit

```
{
 "label": "TEMPERATURE AUDIT (post-outcome, diagnostic; cross-fitted T on the ID test set; benchmark validation temperatures not available locally)",
 "facts": {
  "campaign_energy_temperature": 1.0,
  "benchmark_pipeline": "fitted T by validation NLL; Energy = T logsumexp(z/T)",
  "fitter_positivity": "the benchmark fitter optimizes T directly without a positivity constraint (implementation note, no invalid value observed here)"
 },
 "by_source": {
  "cifar10": {
   "n_checkpoints": 22,
   "T_crossfit_range": [
    1.0897889218562684,
    1.5003524721253243
   ],
   "T_fold_disagreement_max": 0.05068124002260799,
   "nll_before_after": [
    0.2531666155722585,
    0.24433584858382085
   ],
   "energy_auroc_T1_mean": 0.8277540781881313,
   "energy_auroc_Tcrossfit_mean": 0.8279787748421717,
   "mean_abs_change": 0.00028695574494949763,
   "winner_flips": 0,
   "n_cells": 88,
   "ctm_wins_T1": 67,
   "ctm_wins_Tcrossfit": 67
  },
  "cifar100": {
   "n_checkpoints": 24,
   "T_crossfit_range": [
    0.8034154459417729,
    1.6698351601502979
   ],
   "T_fold_disagreement_max": 0.018455859005857334,
   "nll_before_after": [
    1.269582665523787,
    1.2362772502402744
   ],
   "energy_auroc_T1_mean": 0.7393674872685185,
   "energy_auroc_Tcrossfit_mean": 0.7396089886863426,
   "mean_abs_change": 0.0009974814525462925,
   "winner_flips": 0,
   "n_cells": 96,
   "ctm_wins_T1": 64,
   "ctm_wins_Tcrossfit": 64
  },
  "supercifar100": {
   "n_checkpoints": 28,
   "T_crossfit_range": [
    1.5471053172997236,
    2.124867439928036
   ],
   "T_fold_disagreement_max": 0.049189388979100634,
   "nll_before_after": [
    1.947503562166554,
    1.6224975851752819
   ],
   "energy_auroc_T1_mean": 0.5741668210207761,
   "energy_auroc_Tcrossfit_mean": 0.577537767599588,
   "mean_abs_change": 0.005655394724416207,
   "winner_flips": 2,
   "n_cells": 112,
   "ctm_wins_T1": 66,
   "ctm_wins_Tcrossfit": 66
  },
  "tinyimagenet": {
   "n_checkpoints": 22,
   "T_crossfit_range": [
    0.7952398160291427,
    1.117263407507084
   ],
   "T_fold_disagreement_max": 0.019580828655695548,
   "nll_before_after": [
    1.8714651341231239,
    1.8357160958814653
   ],
   "energy_auroc_T1_mean": 0.7622459714015151,
   "energy_auroc_Tcrossfit_mean": 0.762195671829542,
   "mean_abs_change": 0.002510539763628703,
   "winner_flips": 1,
   "n_cells": 88,
   "ctm_wins_T1": 37,
   "ctm_wins_Tcrossfit": 38
  }
 }
}
```
