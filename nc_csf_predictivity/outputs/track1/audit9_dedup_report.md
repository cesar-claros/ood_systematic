# Audit-9 duplicate-exclusion sensitivity (gates A and B)

Baseline vs dedup re-extraction; analytic margins are train-fit and unchanged by construction.

## BREEDS (Gate A)
```
{
 "n_excluded_per_ckpt": 4,
 "max_changes": {
  "id_error": 0.0001,
  "gap_balanced": 0.0001,
  "auroc_Energy": 0.0001,
  "auroc_CTM": 0.0004,
  "auroc_MSR": 0.0001,
  "auroc_MLS": 0.0001,
  "auroc_Maha": 0.0005,
  "auroc_fDBD": 0.0002
 },
 "winner_sign_flips": 0,
 "materiality_changes": 0,
 "n_material_dedup": 17,
 "frac_positive_material_dedup": 0.0,
 "spearman_dedup": 0.848,
 "boot_ci95": [
  0.676,
  0.932
 ],
 "perm_p": 0.0001,
 "leave_one_reward_rho": {
  "drop_rew2.2": 0.846,
  "drop_rew3": 0.832,
  "drop_rew6": 0.836,
  "drop_rew10": 0.85,
  "drop_rew15": 0.855
 },
 "GATE_A": "PASS"
}
```

## ImageNet-200 (Gate B)
```
{
 "per_run": [
  {
   "run": "imagenet200_resnet18_224x224_base_e90_lr0.1_default__s0",
   "n_excluded": 22,
   "trend_rho_ga_vs_gap": 0.5
  },
  {
   "run": "imagenet200_resnet18_224x224_base_e90_lr0.1_default__s1",
   "n_excluded": 22,
   "trend_rho_ga_vs_gap": 0.8
  },
  {
   "run": "imagenet200_resnet18_224x224_base_e90_lr0.1_default__s2",
   "n_excluded": 22,
   "trend_rho_ga_vs_gap": 0.7
  }
 ],
 "max_gap_change": 0.00032,
 "sign_pattern_unchanged": true,
 "ctm_advantage_trend_retained": true,
 "GATE_B": "PASS"
}
```
