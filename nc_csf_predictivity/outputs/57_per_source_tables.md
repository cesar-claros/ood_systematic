# Per-source tables of the three transfer targets

**Source:** `nc_csf_predictivity/evaluation/per_source_tables.py`. Joint side, mean imputed set-regret (AUGRC x1000), 22-CSF NC-only heads trained on VGG13; best fixed CSF with hindsight per cell (ResNet18: Always-X candidates of Sec 4.4; ViT and SSL: full 21-CSF roster). Gates GR, GV, GS passed.

## resnet18

Beats the best fixed CSF: 9 of 12 per-source cells, 3 of 3 pooled. Beats the random set: 12 of 12, 3 of 3.

```
regime        source  rows  nc_only best_fixed  best_fixed_regret  random_set  set_size  beats_best_fixed  beats_random
  near       cifar10    24     0.33        NCI               2.58        0.80      8.92              True          True
  near      cifar100    28     0.66        CTM               1.54        0.73      8.57              True          True
  near supercifar100    36     1.42        CTM               4.19        1.49      8.17              True          True
  near  tinyimagenet    60     1.00        CTM               0.55        2.04      7.50             False          True
  near           all   148     0.93        CTM               2.08        1.46      8.09              True          True
   mid       cifar10    48     0.24    NNGuide               1.59        0.60      8.92              True          True
   mid      cifar100    56     0.68    NNGuide               1.14        1.15      8.57              True          True
   mid supercifar100    72     1.57        CTM               6.64        2.03      8.17              True          True
   mid  tinyimagenet    24     0.56        CTM               0.56        1.52      7.50             False          True
   mid           all   200     0.88        CTM               3.56        1.38      8.38              True          True
   far       cifar10    24     0.24        CTM               2.82        0.98      8.92              True          True
   far      cifar100    28     0.66        CTM               1.46        1.05      8.57              True          True
   far supercifar100    36     0.44        CTM               4.02        1.91      8.17              True          True
   far  tinyimagenet    12     0.88        CTM               0.32        1.77      7.50             False          True
   far           all   100     0.50        CTM               2.57        1.43      8.38              True          True
```

## vit

Beats the best fixed CSF: 4 of 12 per-source cells, 2 of 3 pooled. Beats the random set: 5 of 12, 0 of 3.

```
regime        source  rows  nc_only           best_fixed  best_fixed_regret  random_set  set_size  beats_best_fixed  beats_random
  near       cifar10    20     0.57                  ViM               0.34        0.57      7.40             False         False
  near      cifar100    20     0.49                  ViM               0.73        0.53     10.40              True          True
  near supercifar100    20     5.70                  ViM               9.35        2.67      8.40              True         False
  near  tinyimagenet    50     0.48             GradNorm               1.51        1.35      7.20              True          True
  near           all   110     1.44                   GE               5.85        1.30      8.04              True         False
   mid       cifar10    40     0.45 KPCA RecError global               0.40        0.34      7.40             False         False
   mid      cifar100    40     0.63 KPCA RecError global               2.75        1.11     10.40              True          True
   mid supercifar100    40    10.35 KPCA RecError global               3.88        9.03      8.40             False         False
   mid  tinyimagenet    20     5.39                  ViM               0.71        1.60      7.20             False         False
   mid           all   140     4.04 KPCA RecError global               3.67        3.22      8.51             False         False
   far       cifar10    20     0.12             Residual               0.02        0.15      7.40             False          True
   far      cifar100    20     1.41                  ViM               0.66        0.83     10.40             False         False
   far supercifar100    20    10.42             Residual               0.18        2.17      8.40             False         False
   far  tinyimagenet    10     0.12             GradNorm               0.10        0.92      7.20             False          True
   far           all    70     3.43                  ViM               5.90        1.03      8.51              True         False
```

## ssl

Beats the best fixed CSF: 3 of 12 per-source cells, 2 of 3 pooled. Beats the random set: 5 of 12, 1 of 3.

```
regime        source  rows  nc_only best_fixed  best_fixed_regret  random_set  set_size  beats_best_fixed  beats_random
  near       cifar10    20     1.12        ViM               1.41        0.95      6.50              True         False
  near      cifar100    20     2.47        ViM               1.75        2.29      7.50             False         False
  near supercifar100    20    16.18        ViM               5.20        2.40      8.40             False         False
  near  tinyimagenet    50     1.59        NCI               4.25        1.79      7.50              True          True
  near           all   110     4.32        ViM               6.03        1.84      7.48              True         False
   mid       cifar10    40     0.21        CTM               0.37        0.40      6.50              True          True
   mid      cifar100    40     2.80        NCI               0.44        3.06      7.50             False          True
   mid supercifar100    40     1.97        NCI               1.77        5.15      8.40             False          True
   mid  tinyimagenet    20     1.41        ViM               0.57        1.19      7.50             False         False
   mid           all   140     1.63        NCI               2.07        2.63      7.47              True          True
   far       cifar10    20     1.00       Maha               0.07        0.36      6.50             False         False
   far      cifar100    20     0.70     MahaPP               0.18        1.85      7.50             False          True
   far supercifar100    20    26.25     MahaPP               0.06        2.08      8.40             False         False
   far  tinyimagenet    10     0.64        NCI               0.50        0.38      7.50             False         False
   far           all    70     8.08     MahaPP               0.85        1.28      7.47             False         False
```

