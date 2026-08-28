# Audit-8 checks R1/R2/R4/R5/R7 (post-hoc sensitivities; frozen primaries unchanged)

## R1 two-class-only joint-audit interval
- two-class cells: ['cifar10/dg', 'cifar100/dg', 'supercifar100/confidnet', 'supercifar100/devries', 'supercifar100/dg']
- M1 - M0+ two-class macro: {'point': 0.017, 'ci95': [-0.0, 0.038]}

## R2 categorical DG-reward sensitivity

| model | bal macro | bal 2-class macro | bal row |
|---|---|---|---|
| M0 | 0.859 | 0.669 | 0.797 |
| M0plus | 0.947 | 0.873 | 0.919 |
| M1 | 0.954 | 0.89 | 0.921 |
| M0pluscat | 0.95 | 0.88 | 0.929 |
| M1cat | 0.95 | 0.88 | 0.934 |
- M0pluscat-M0: bal_macro +0.091 [0.057, 0.11], bal_row +0.132 [0.086, 0.178]
- M1cat-M0pluscat: bal_macro -0.000 [-0.004, 0.002], bal_two_class_macro -0.000 [-0.01, 0.006], bal_row +0.005 [-0.003, 0.013]
- leave-one-reward influence (M1cat-M0pluscat macro): {'drop_rew2.2': -0.001, 'drop_rew3': 0.0, 'drop_rew6': 0.001, 'drop_rew10': -0.0, 'drop_rew12': -0.0, 'drop_rew15': -0.0, 'drop_rew20': -0.0}

## R4 BREEDS rank uncertainty
- Spearman 0.851 (n=28), boot CI95 [0.653, 0.932], permutation p 0.0001
- DG-only: {'n': 20, 'rho': 0.853, 'boot_ci95': [0.651, 0.948]}
- leave-one-reward rho: {'drop_rew2.2': 0.85, 'drop_rew3': 0.836, 'drop_rew6': 0.839, 'drop_rew10': 0.851, 'drop_rew15': 0.858}
- per paradigm: {'confidnet': {'n': 4, 'rho': 0.4, 'underpowered': True}, 'devries': {'n': 4, 'rho': 0.2, 'underpowered': True}, 'dg': {'n': 20, 'rho': 0.853, 'underpowered': False}}

## R5 numerical stability
- {
 "float32": {
  "breeds_sign_flips": 0,
  "breeds_rank_spearman_base_vs_alt": 1.0,
  "max_rel_margin_change_breeds": 0.0
 },
 "rel1e-6": {
  "breeds_sign_flips": 0,
  "breeds_rank_spearman_base_vs_alt": 1.0,
  "max_rel_margin_change_breeds": 0.0
 },
 "magnitude_vs_error": {
  "breeds_min_margin": 7.423791082650233e-06,
  "numerical_error_scale": 1e-12,
  "imagenet200_median_margin": 2.434007995155696e-09,
  "verdict": "BREEDS margins >> numerical error; ImageNet-200 margins remain classified degenerate"
 }
}

## R7 matched correlation decomposition
- pool within-shift/across-ckpt: {'n_pairs_usable': 17, 'n_pairs_total': 32, 'median': 0.076, 'iqr': [-0.321, 0.283], 'values': [0.8, -0.732, 0.473, -0.374, 0.345, -0.562, -0.345, 0.094, 0.38, 0.283, 0.003, -0.17, 0.076, 0.128, -0.321, 0.042, 0.195]}
- pool within-ckpt/across-shift: {'n_ckpt_usable': 46, 'n_ckpt_total': 280, 'median': 0.048}
- svhn within-shift values: [0.686, 0.75, -0.016, 0.621, -0.001, 0.675, 0.66, 0.185]
- breeds within-shift value: [0.851]
- imagenet200 per-ckpt: {'imagenet200_resnet18_224x224_base_e90_lr0.1_default__s0': 0.5, 'imagenet200_resnet18_224x224_base_e90_lr0.1_default__s1': 0.8, 'imagenet200_resnet18_224x224_base_e90_lr0.1_default__s2': 0.7}
