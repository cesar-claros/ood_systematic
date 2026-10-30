# RC vs ROC top-clique agreement

RC = AUGRC+AURC blocks; ROC = AUROC_f+FPR@95TPR blocks; both through the identical Friedman-Conover clique pipeline (filter-methods, alpha=0.05).

## Aggregate by regime

|                                  |   cells |   mean_jaccard |   rc_winner_in_roc |   roc_winner_in_rc |   mean_size_rc |   mean_size_roc |   featshare_rc |   featshare_roc |
|:---------------------------------|--------:|---------------:|-------------------:|-------------------:|---------------:|----------------:|---------------:|----------------:|
| ('ROC (AUROC_f+FPR@95)', 'near') |      16 |          0.484 |              0.562 |              0.625 |          3.5   |           3.438 |          0.49  |           0.428 |
| ('ROC (AUROC_f+FPR@95)', 'mid')  |      16 |          0.529 |              0.875 |              0.688 |          1.562 |           2.875 |          0.641 |           0.456 |
| ('ROC (AUROC_f+FPR@95)', 'far')  |      16 |          0.53  |              0.812 |              0.75  |          2.5   |           2.875 |          0.807 |           0.539 |
| ('FPR@95 only', 'near')          |      16 |          0.454 |              0.75  |              0.5   |          3.5   |           5.062 |          0.49  |           0.517 |
| ('FPR@95 only', 'mid')           |      16 |          0.394 |              0.875 |              0.688 |          1.562 |           3.938 |          0.641 |           0.42  |
| ('FPR@95 only', 'far')           |      16 |          0.466 |              0.875 |              0.688 |          2.5   |           3.375 |          0.807 |           0.613 |

## By panel x regime

|                                                    |   mean_jaccard |   rc_winner_in_roc |   featshare_rc |   featshare_roc |
|:---------------------------------------------------|---------------:|-------------------:|---------------:|----------------:|
| ('ROC (AUROC_f+FPR@95)', 'Conv/confidnet', 'near') |          0.708 |               0.75 |          0.458 |           0.525 |
| ('ROC (AUROC_f+FPR@95)', 'Conv/confidnet', 'mid')  |          0.725 |               1    |          0.5   |           0.525 |
| ('ROC (AUROC_f+FPR@95)', 'Conv/confidnet', 'far')  |          0.625 |               1    |          0.875 |           0.625 |
| ('ROC (AUROC_f+FPR@95)', 'Conv/devries', 'near')   |          0.681 |               0.75 |          0.425 |           0.362 |
| ('ROC (AUROC_f+FPR@95)', 'Conv/devries', 'mid')    |          0.625 |               0.75 |          0.5   |           0.25  |
| ('ROC (AUROC_f+FPR@95)', 'Conv/devries', 'far')    |          0.604 |               0.75 |          0.875 |           0.667 |
| ('ROC (AUROC_f+FPR@95)', 'Conv/dg', 'near')        |          0.138 |               0    |          0.575 |           0.3   |
| ('ROC (AUROC_f+FPR@95)', 'Conv/dg', 'mid')         |          0.258 |               0.75 |          0.688 |           0.246 |
| ('ROC (AUROC_f+FPR@95)', 'Conv/dg', 'far')         |          0.142 |               0.5  |          0.938 |           0.383 |
| ('ROC (AUROC_f+FPR@95)', 'ViT/modelvit', 'near')   |          0.41  |               0.75 |          0.5   |           0.524 |
| ('ROC (AUROC_f+FPR@95)', 'ViT/modelvit', 'mid')    |          0.508 |               1    |          0.875 |           0.804 |
| ('ROC (AUROC_f+FPR@95)', 'ViT/modelvit', 'far')    |          0.75  |               1    |          0.542 |           0.479 |
| ('FPR@95 only', 'Conv/confidnet', 'near')          |          0.684 |               1    |          0.458 |           0.504 |
| ('FPR@95 only', 'Conv/confidnet', 'mid')           |          0.558 |               1    |          0.5   |           0.475 |
| ('FPR@95 only', 'Conv/confidnet', 'far')           |          0.544 |               1    |          0.875 |           0.65  |
| ('FPR@95 only', 'Conv/devries', 'near')            |          0.558 |               1    |          0.425 |           0.585 |
| ('FPR@95 only', 'Conv/devries', 'mid')             |          0.5   |               0.75 |          0.5   |           0.333 |
| ('FPR@95 only', 'Conv/devries', 'far')             |          0.479 |               0.75 |          0.875 |           0.5   |
| ('FPR@95 only', 'Conv/dg', 'near')                 |          0.241 |               0.25 |          0.575 |           0.479 |
| ('FPR@95 only', 'Conv/dg', 'mid')                  |          0.225 |               1    |          0.688 |           0.349 |
| ('FPR@95 only', 'Conv/dg', 'far')                  |          0.194 |               0.75 |          0.938 |           0.719 |
| ('FPR@95 only', 'ViT/modelvit', 'near')            |          0.332 |               0.75 |          0.5   |           0.497 |
| ('FPR@95 only', 'ViT/modelvit', 'mid')             |          0.292 |               0.75 |          0.875 |           0.521 |
| ('FPR@95 only', 'ViT/modelvit', 'far')             |          0.646 |               1    |          0.542 |           0.583 |

## Per-cell detail

| comparison           | panel          | source        | regime   |   jaccard | rc_winner_in_roc   | roc_winner_in_rc   |   size_rc |   size_roc |   featshare_rc |   featshare_roc |
|:---------------------|:---------------|:--------------|:---------|----------:|:-------------------|:-------------------|----------:|-----------:|---------------:|----------------:|
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar10       | near     |     1     | True               | True               |         4 |          4 |          0.5   |           0.5   |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar10       | mid      |     1     | True               | True               |         1 |          1 |          0     |           0     |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar10       | far      |     1     | True               | True               |         4 |          4 |          0.5   |           0.5   |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | supercifar100 | near     |     0     | False              | False              |         2 |          5 |          0     |           0.4   |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | supercifar100 | mid      |     0.4   | True               | True               |         2 |          5 |          0.5   |           0.6   |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | supercifar100 | far      |     0.5   | True               | True               |         3 |          6 |          1     |           0.5   |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar100      | near     |     0.833 | True               | True               |         6 |          5 |          0.333 |           0.2   |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar100      | mid      |     1     | True               | True               |         2 |          2 |          0.5   |           0.5   |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar100      | far      |     0.5   | True               | True               |         1 |          2 |          1     |           0.5   |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | tinyimagenet  | near     |     1     | True               | True               |         1 |          1 |          1     |           1     |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | tinyimagenet  | mid      |     0.5   | True               | True               |         1 |          2 |          1     |           1     |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | tinyimagenet  | far      |     0.5   | True               | True               |         1 |          2 |          1     |           1     |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar10       | near     |     1     | True               | True               |         1 |          1 |          0     |           0     |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar10       | mid      |     1     | True               | True               |         1 |          1 |          0     |           0     |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar10       | far      |     0.75  | True               | True               |         4 |          3 |          0.5   |           0.667 |
| ROC (AUROC_f+FPR@95) | Conv/devries   | supercifar100 | near     |     0.625 | True               | False              |         5 |          8 |          0.2   |           0.25  |
| ROC (AUROC_f+FPR@95) | Conv/devries   | supercifar100 | mid      |     1     | True               | True               |         2 |          2 |          0     |           0     |
| ROC (AUROC_f+FPR@95) | Conv/devries   | supercifar100 | far      |     0     | False              | False              |         4 |          2 |          1     |           0     |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar100      | near     |     0.1   | False              | False              |         6 |          5 |          0.5   |           0.2   |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar100      | mid      |     0     | False              | False              |         1 |          1 |          1     |           0     |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar100      | far      |     1     | True               | True               |         1 |          1 |          1     |           1     |
| ROC (AUROC_f+FPR@95) | Conv/devries   | tinyimagenet  | near     |     1     | True               | True               |         1 |          1 |          1     |           1     |
| ROC (AUROC_f+FPR@95) | Conv/devries   | tinyimagenet  | mid      |     0.5   | True               | True               |         1 |          2 |          1     |           1     |
| ROC (AUROC_f+FPR@95) | Conv/devries   | tinyimagenet  | far      |     0.667 | True               | True               |         3 |          2 |          1     |           1     |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar10       | near     |     0.2   | False              | False              |         4 |          2 |          0.5   |           0     |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar10       | mid      |     0.667 | True               | True               |         2 |          3 |          0.5   |           0.333 |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar10       | far      |     0.4   | True               | True               |         4 |          3 |          0.75  |           0.667 |
| ROC (AUROC_f+FPR@95) | Conv/dg        | supercifar100 | near     |     0.25  | False              | True               |         5 |          5 |          0.4   |           0.2   |
| ROC (AUROC_f+FPR@95) | Conv/dg        | supercifar100 | mid      |     0.2   | True               | False              |         4 |          8 |          0.25  |           0.25  |
| ROC (AUROC_f+FPR@95) | Conv/dg        | supercifar100 | far      |     0.167 | True               | False              |         2 |          5 |          1     |           0.2   |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar100      | near     |     0.1   | False              | True               |        10 |          1 |          0.4   |           1     |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar100      | mid      |     0.167 | True               | False              |         2 |          5 |          1     |           0.4   |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar100      | far      |     0     | False              | False              |         1 |          3 |          1     |           0.667 |
| ROC (AUROC_f+FPR@95) | Conv/dg        | tinyimagenet  | near     |     0     | False              | False              |         1 |          1 |          1     |           0     |
| ROC (AUROC_f+FPR@95) | Conv/dg        | tinyimagenet  | mid      |     0     | False              | False              |         1 |          1 |          1     |           0     |
| ROC (AUROC_f+FPR@95) | Conv/dg        | tinyimagenet  | far      |     0     | False              | False              |         3 |          3 |          1     |           0     |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar10       | near     |     0.667 | True               | True               |         2 |          3 |          0.5   |           0.667 |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar10       | mid      |     1     | True               | True               |         1 |          1 |          1     |           1     |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar10       | far      |     1     | True               | True               |         3 |          3 |          0.667 |           0.667 |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | supercifar100 | near     |     0.571 | True               | True               |         4 |          7 |          0.5   |           0.429 |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | supercifar100 | mid      |     0.333 | True               | True               |         1 |          3 |          1     |           0.667 |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | supercifar100 | far      |     1     | True               | True               |         2 |          2 |          0.5   |           0.5   |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar100      | near     |     0.4   | True               | True               |         3 |          4 |          1     |           1     |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar100      | mid      |     0.2   | True               | False              |         1 |          5 |          1     |           0.8   |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar100      | far      |     0.5   | True               | True               |         2 |          4 |          1     |           0.75  |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | tinyimagenet  | near     |     0     | False              | False              |         1 |          2 |          0     |           0     |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | tinyimagenet  | mid      |     0.5   | True               | True               |         2 |          4 |          0.5   |           0.75  |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | tinyimagenet  | far      |     0.5   | True               | True               |         2 |          1 |          0     |           0     |
| FPR@95 only          | Conv/confidnet | cifar10       | near     |     1     | True               | True               |         4 |          4 |          0.5   |           0.5   |
| FPR@95 only          | Conv/confidnet | cifar10       | mid      |     1     | True               | True               |         1 |          1 |          0     |           0     |
| FPR@95 only          | Conv/confidnet | cifar10       | far      |     0.8   | True               | True               |         4 |          5 |          0.5   |           0.6   |
| FPR@95 only          | Conv/confidnet | supercifar100 | near     |     0.111 | True               | False              |         2 |          8 |          0     |           0.375 |
| FPR@95 only          | Conv/confidnet | supercifar100 | mid      |     0.333 | True               | True               |         2 |          6 |          0.5   |           0.5   |
| FPR@95 only          | Conv/confidnet | supercifar100 | far      |     0.375 | True               | False              |         3 |          8 |          1     |           0.5   |
| FPR@95 only          | Conv/confidnet | cifar100      | near     |     0.625 | True               | True               |         6 |          7 |          0.333 |           0.143 |
| FPR@95 only          | Conv/confidnet | cifar100      | mid      |     0.4   | True               | True               |         2 |          5 |          0.5   |           0.4   |
| FPR@95 only          | Conv/confidnet | cifar100      | far      |     0.5   | True               | False              |         1 |          2 |          1     |           0.5   |
| FPR@95 only          | Conv/confidnet | tinyimagenet  | near     |     1     | True               | True               |         1 |          1 |          1     |           1     |
| FPR@95 only          | Conv/confidnet | tinyimagenet  | mid      |     0.5   | True               | True               |         1 |          2 |          1     |           1     |
| FPR@95 only          | Conv/confidnet | tinyimagenet  | far      |     0.5   | True               | True               |         1 |          2 |          1     |           1     |
| FPR@95 only          | Conv/devries   | cifar10       | near     |     0.333 | True               | True               |         1 |          3 |          0     |           0.667 |
| FPR@95 only          | Conv/devries   | cifar10       | mid      |     1     | True               | True               |         1 |          1 |          0     |           0     |
| FPR@95 only          | Conv/devries   | cifar10       | far      |     0.25  | True               | True               |         4 |          1 |          0.5   |           0     |
| FPR@95 only          | Conv/devries   | supercifar100 | near     |     0.5   | True               | False              |         5 |         10 |          0.2   |           0.3   |
| FPR@95 only          | Conv/devries   | supercifar100 | mid      |     0.667 | True               | True               |         2 |          3 |          0     |           0     |
| FPR@95 only          | Conv/devries   | supercifar100 | far      |     0     | False              | False              |         4 |          3 |          1     |           0     |
| FPR@95 only          | Conv/devries   | cifar100      | near     |     0.4   | True               | False              |         6 |          8 |          0.5   |           0.375 |
| FPR@95 only          | Conv/devries   | cifar100      | mid      |     0     | False              | False              |         1 |          3 |          1     |           0.333 |
| FPR@95 only          | Conv/devries   | cifar100      | far      |     1     | True               | True               |         1 |          1 |          1     |           1     |
| FPR@95 only          | Conv/devries   | tinyimagenet  | near     |     1     | True               | True               |         1 |          1 |          1     |           1     |
| FPR@95 only          | Conv/devries   | tinyimagenet  | mid      |     0.333 | True               | True               |         1 |          3 |          1     |           1     |
| FPR@95 only          | Conv/devries   | tinyimagenet  | far      |     0.667 | True               | True               |         3 |          2 |          1     |           1     |
| FPR@95 only          | Conv/dg        | cifar10       | near     |     0.143 | False              | False              |         4 |          4 |          0.5   |           0.25  |
| FPR@95 only          | Conv/dg        | cifar10       | mid      |     0.25  | True               | True               |         2 |          3 |          0.5   |           0.333 |
| FPR@95 only          | Conv/dg        | cifar10       | far      |     0.5   | True               | True               |         4 |          2 |          0.75  |           1     |
| FPR@95 only          | Conv/dg        | supercifar100 | near     |     0.222 | False              | False              |         5 |          6 |          0.4   |           0.167 |
| FPR@95 only          | Conv/dg        | supercifar100 | mid      |     0.25  | True               | False              |         4 |         11 |          0.25  |           0.364 |
| FPR@95 only          | Conv/dg        | supercifar100 | far      |     0.111 | True               | False              |         2 |          8 |          1     |           0.625 |
| FPR@95 only          | Conv/dg        | cifar100      | near     |     0.1   | False              | True               |        10 |          1 |          0.4   |           1     |
| FPR@95 only          | Conv/dg        | cifar100      | mid      |     0.2   | True               | False              |         2 |          4 |          1     |           0.5   |
| FPR@95 only          | Conv/dg        | cifar100      | far      |     0     | False              | False              |         1 |          2 |          1     |           1     |
| FPR@95 only          | Conv/dg        | tinyimagenet  | near     |     0.5   | True               | True               |         1 |          2 |          1     |           0.5   |
| FPR@95 only          | Conv/dg        | tinyimagenet  | mid      |     0.2   | True               | True               |         1 |          5 |          1     |           0.2   |
| FPR@95 only          | Conv/dg        | tinyimagenet  | far      |     0.167 | True               | True               |         3 |          4 |          1     |           0.25  |
| FPR@95 only          | ViT/modelvit   | cifar10       | near     |     0.667 | True               | True               |         2 |          3 |          0.5   |           0.667 |
| FPR@95 only          | ViT/modelvit   | cifar10       | mid      |     0.5   | True               | True               |         1 |          2 |          1     |           0.5   |
| FPR@95 only          | ViT/modelvit   | cifar10       | far      |     1     | True               | True               |         3 |          3 |          0.667 |           0.667 |
| FPR@95 only          | ViT/modelvit   | supercifar100 | near     |     0.286 | True               | False              |         4 |         14 |          0.5   |           0.571 |
| FPR@95 only          | ViT/modelvit   | supercifar100 | mid      |     0.167 | True               | False              |         1 |          6 |          1     |           0.333 |
| FPR@95 only          | ViT/modelvit   | supercifar100 | far      |     1     | True               | True               |         2 |          2 |          0.5   |           0.5   |
| FPR@95 only          | ViT/modelvit   | cifar100      | near     |     0.375 | True               | False              |         3 |          8 |          1     |           0.75  |
| FPR@95 only          | ViT/modelvit   | cifar100      | mid      |     0     | False              | False              |         1 |          4 |          1     |           0.5   |
| FPR@95 only          | ViT/modelvit   | cifar100      | far      |     0.333 | True               | True               |         2 |          6 |          1     |           0.833 |
| FPR@95 only          | ViT/modelvit   | tinyimagenet  | near     |     0     | False              | False              |         1 |          1 |          0     |           0     |
| FPR@95 only          | ViT/modelvit   | tinyimagenet  | mid      |     0.5   | True               | True               |         2 |          4 |          0.5   |           0.75  |
| FPR@95 only          | ViT/modelvit   | tinyimagenet  | far      |     0.25  | True               | True               |         2 |          3 |          0     |           0.333 |

## Cliques (RC vs ROC)

| comparison           | panel          | source        | regime   | rc_clique                                                  | roc_clique                                                                                           |
|:---------------------|:---------------|:--------------|:---------|:-----------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar10       | near     | Energy, NeCo, NNGuide, MLS                                 | Energy, NeCo, NNGuide, MLS                                                                           |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar10       | mid      | Energy                                                     | Energy                                                                                               |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar10       | far      | NNGuide, NeCo, Energy, MLS                                 | NNGuide, NeCo, Energy, MLS                                                                           |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | supercifar100 | near     | Energy, GradNorm                                           | NNGuide, NeCo, MLS, PE, REN                                                                          |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | supercifar100 | mid      | CTM, Energy                                                | Energy, CTM, GE, KPCA RecError global, NeCo                                                          |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | supercifar100 | far      | fDBD, KPCA RecError global, CTM                            | fDBD, Energy, Confidence, GE, CTM, KPCA RecError global                                              |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar100      | near     | NeCo, MLS, Energy, GE, NNGuide, PE                         | NeCo, MLS, PE, GE, Energy                                                                            |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar100      | mid      | Energy, NeCo                                               | Energy, NeCo                                                                                         |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | cifar100      | far      | fDBD                                                       | fDBD, Confidence                                                                                     |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | tinyimagenet  | near     | CTM                                                        | CTM                                                                                                  |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | tinyimagenet  | mid      | CTM                                                        | CTM, fDBD                                                                                            |
| ROC (AUROC_f+FPR@95) | Conv/confidnet | tinyimagenet  | far      | CTM                                                        | CTM, fDBD                                                                                            |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar10       | near     | Energy                                                     | Energy                                                                                               |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar10       | mid      | Energy                                                     | Energy                                                                                               |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar10       | far      | Energy, NeCo, MLS, NNGuide                                 | Energy, NeCo, NNGuide                                                                                |
| ROC (AUROC_f+FPR@95) | Conv/devries   | supercifar100 | near     | MLS, NeCo, Energy, GE, PE                                  | NNGuide, NeCo, MLS, PE, GE, Energy, PCE, GEN                                                         |
| ROC (AUROC_f+FPR@95) | Conv/devries   | supercifar100 | mid      | Energy, GE                                                 | Energy, GE                                                                                           |
| ROC (AUROC_f+FPR@95) | Conv/devries   | supercifar100 | far      | KPCA RecError global, ViM, fDBD, CTM                       | Energy, GE                                                                                           |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar100      | near     | fDBD, PCE, NNGuide, PE, MSR, CTM                           | NeCo, GE, MLS, PE, Energy                                                                            |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar100      | mid      | fDBD                                                       | Energy                                                                                               |
| ROC (AUROC_f+FPR@95) | Conv/devries   | cifar100      | far      | fDBD                                                       | fDBD                                                                                                 |
| ROC (AUROC_f+FPR@95) | Conv/devries   | tinyimagenet  | near     | CTM                                                        | CTM                                                                                                  |
| ROC (AUROC_f+FPR@95) | Conv/devries   | tinyimagenet  | mid      | CTM                                                        | CTM, NNGuide                                                                                         |
| ROC (AUROC_f+FPR@95) | Conv/devries   | tinyimagenet  | far      | CTM, KPCA RecError global, fDBD                            | CTM, fDBD                                                                                            |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar10       | near     | Energy, NNGuide, CTM, MLS                                  | Confidence, MLS                                                                                      |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar10       | mid      | Confidence, CTM                                            | Confidence, CTM, MLS                                                                                 |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar10       | far      | CTM, Energy, NNGuide, Maha                                 | CTM, Confidence, Maha                                                                                |
| ROC (AUROC_f+FPR@95) | Conv/dg        | supercifar100 | near     | NNGuide, Energy, NeCo, MLS, GE                             | Energy, CTM, GEN, REN, MLS                                                                           |
| ROC (AUROC_f+FPR@95) | Conv/dg        | supercifar100 | mid      | Energy, GE, Maha, pNML                                     | MLS, MSR, Confidence, CTM, REN, GEN, Maha, Energy                                                    |
| ROC (AUROC_f+FPR@95) | Conv/dg        | supercifar100 | far      | Maha, Residual                                             | MSR, MLS, Maha, Confidence, PCE                                                                      |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar100      | near     | fDBD, PE, PCE, NeCo, GradNorm, MSR, REN, NNGuide, GEN, CTM | CTM                                                                                                  |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar100      | mid      | NNGuide, fDBD                                              | CTM, MSR, NNGuide, MLS, Confidence                                                                   |
| ROC (AUROC_f+FPR@95) | Conv/dg        | cifar100      | far      | fDBD                                                       | CTM, NNGuide, Confidence                                                                             |
| ROC (AUROC_f+FPR@95) | Conv/dg        | tinyimagenet  | near     | CTM                                                        | REN                                                                                                  |
| ROC (AUROC_f+FPR@95) | Conv/dg        | tinyimagenet  | mid      | CTM                                                        | MLS                                                                                                  |
| ROC (AUROC_f+FPR@95) | Conv/dg        | tinyimagenet  | far      | CTM, fDBD, KPCA RecError global                            | GEN, MSR, REN                                                                                        |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar10       | near     | pNML, PCA RecError global                                  | pNML, PCA RecError global, ViM                                                                       |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar10       | mid      | KPCA RecError global                                       | KPCA RecError global                                                                                 |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar10       | far      | pNML, Residual, ViM                                        | pNML, Residual, ViM                                                                                  |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | supercifar100 | near     | NNGuide, ViM, GE, MLS                                      | NNGuide, GE, MLS, PE, ViM, fDBD, PCE                                                                 |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | supercifar100 | mid      | KPCA RecError global                                       | KPCA RecError global, NNGuide, MSR                                                                   |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | supercifar100 | far      | Residual, pNML                                             | Residual, pNML                                                                                       |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar100      | near     | ViM, Maha, NeCo                                            | ViM, fDBD, NeCo, CTM                                                                                 |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar100      | mid      | KPCA RecError global                                       | CTM, KPCA RecError global, GradNorm, fDBD, ViM                                                       |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | cifar100      | far      | Residual, ViM                                              | ViM, Residual, pNML, PCA RecError global                                                             |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | tinyimagenet  | near     | PE                                                         | GradNorm, PCE                                                                                        |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | tinyimagenet  | mid      | Residual, pNML                                             | Residual, pNML, Maha, ViM                                                                            |
| ROC (AUROC_f+FPR@95) | ViT/modelvit   | tinyimagenet  | far      | GradNorm, GE                                               | GradNorm                                                                                             |
| FPR@95 only          | Conv/confidnet | cifar10       | near     | Energy, NeCo, NNGuide, MLS                                 | Energy, NeCo, MLS, NNGuide                                                                           |
| FPR@95 only          | Conv/confidnet | cifar10       | mid      | Energy                                                     | Energy                                                                                               |
| FPR@95 only          | Conv/confidnet | cifar10       | far      | NNGuide, NeCo, Energy, MLS                                 | Energy, NNGuide, NeCo, CTM, MLS                                                                      |
| FPR@95 only          | Conv/confidnet | supercifar100 | near     | Energy, GradNorm                                           | NNGuide, NeCo, PCE, KPCA RecError global, MLS, PE, REN, Energy                                       |
| FPR@95 only          | Conv/confidnet | supercifar100 | mid      | CTM, Energy                                                | CTM, Energy, KPCA RecError global, GE, fDBD, PE                                                      |
| FPR@95 only          | Conv/confidnet | supercifar100 | far      | fDBD, KPCA RecError global, CTM                            | Energy, GE, fDBD, CTM, KPCA RecError global, PE, Confidence, NeCo                                    |
| FPR@95 only          | Conv/confidnet | cifar100      | near     | NeCo, MLS, Energy, GE, NNGuide, PE                         | NeCo, MLS, PE, Confidence, PCE, GE, Energy                                                           |
| FPR@95 only          | Conv/confidnet | cifar100      | mid      | Energy, NeCo                                               | Energy, NeCo, GE, MLS, KPCA RecError global                                                          |
| FPR@95 only          | Conv/confidnet | cifar100      | far      | fDBD                                                       | Confidence, fDBD                                                                                     |
| FPR@95 only          | Conv/confidnet | tinyimagenet  | near     | CTM                                                        | CTM                                                                                                  |
| FPR@95 only          | Conv/confidnet | tinyimagenet  | mid      | CTM                                                        | CTM, fDBD                                                                                            |
| FPR@95 only          | Conv/confidnet | tinyimagenet  | far      | CTM                                                        | CTM, fDBD                                                                                            |
| FPR@95 only          | Conv/devries   | cifar10       | near     | Energy                                                     | Energy, NNGuide, NeCo                                                                                |
| FPR@95 only          | Conv/devries   | cifar10       | mid      | Energy                                                     | Energy                                                                                               |
| FPR@95 only          | Conv/devries   | cifar10       | far      | Energy, NeCo, MLS, NNGuide                                 | Energy                                                                                               |
| FPR@95 only          | Conv/devries   | supercifar100 | near     | MLS, NeCo, Energy, GE, PE                                  | NNGuide, PCE, PE, MSR, GEN, NeCo, GE, MLS, Energy, fDBD                                              |
| FPR@95 only          | Conv/devries   | supercifar100 | mid      | Energy, GE                                                 | Energy, GE, PE                                                                                       |
| FPR@95 only          | Conv/devries   | supercifar100 | far      | KPCA RecError global, ViM, fDBD, CTM                       | Energy, GE, PE                                                                                       |
| FPR@95 only          | Conv/devries   | cifar100      | near     | fDBD, PCE, NNGuide, PE, MSR, CTM                           | NeCo, GE, MLS, PE, PCE, fDBD, Energy, CTM                                                            |
| FPR@95 only          | Conv/devries   | cifar100      | mid      | fDBD                                                       | Energy, GE, KPCA RecError global                                                                     |
| FPR@95 only          | Conv/devries   | cifar100      | far      | fDBD                                                       | fDBD                                                                                                 |
| FPR@95 only          | Conv/devries   | tinyimagenet  | near     | CTM                                                        | CTM                                                                                                  |
| FPR@95 only          | Conv/devries   | tinyimagenet  | mid      | CTM                                                        | CTM, NNGuide, fDBD                                                                                   |
| FPR@95 only          | Conv/devries   | tinyimagenet  | far      | CTM, KPCA RecError global, fDBD                            | CTM, fDBD                                                                                            |
| FPR@95 only          | Conv/dg        | cifar10       | near     | Energy, NNGuide, CTM, MLS                                  | Confidence, MLS, NeCo, GE                                                                            |
| FPR@95 only          | Conv/dg        | cifar10       | mid      | Confidence, CTM                                            | Confidence, MLS, NeCo                                                                                |
| FPR@95 only          | Conv/dg        | cifar10       | far      | CTM, Energy, NNGuide, Maha                                 | CTM, Maha                                                                                            |
| FPR@95 only          | Conv/dg        | supercifar100 | near     | NNGuide, Energy, NeCo, MLS, GE                             | CTM, GEN, Energy, REN, Confidence, MLS                                                               |
| FPR@95 only          | Conv/dg        | supercifar100 | mid      | Energy, GE, Maha, pNML                                     | CTM, MLS, MSR, Maha, Confidence, REN, GEN, pNML, Energy, Residual, NeCo                              |
| FPR@95 only          | Conv/dg        | supercifar100 | far      | Maha, Residual                                             | MLS, MSR, CTM, fDBD, Maha, NeCo, PCE, NNGuide                                                        |
| FPR@95 only          | Conv/dg        | cifar100      | near     | fDBD, PE, PCE, NeCo, GradNorm, MSR, REN, NNGuide, GEN, CTM | CTM                                                                                                  |
| FPR@95 only          | Conv/dg        | cifar100      | mid      | NNGuide, fDBD                                              | CTM, NNGuide, MSR, MLS                                                                               |
| FPR@95 only          | Conv/dg        | cifar100      | far      | fDBD                                                       | CTM, NNGuide                                                                                         |
| FPR@95 only          | Conv/dg        | tinyimagenet  | near     | CTM                                                        | CTM, REN                                                                                             |
| FPR@95 only          | Conv/dg        | tinyimagenet  | mid      | CTM                                                        | CTM, MLS, PCE, MSR, GEN                                                                              |
| FPR@95 only          | Conv/dg        | tinyimagenet  | far      | CTM, fDBD, KPCA RecError global                            | CTM, GEN, MSR, PCE                                                                                   |
| FPR@95 only          | ViT/modelvit   | cifar10       | near     | pNML, PCA RecError global                                  | PCA RecError global, pNML, ViM                                                                       |
| FPR@95 only          | ViT/modelvit   | cifar10       | mid      | KPCA RecError global                                       | KPCA RecError global, pNML                                                                           |
| FPR@95 only          | ViT/modelvit   | cifar10       | far      | pNML, Residual, ViM                                        | pNML, Residual, ViM                                                                                  |
| FPR@95 only          | ViT/modelvit   | supercifar100 | near     | NNGuide, ViM, GE, MLS                                      | fDBD, NNGuide, GE, ViM, GradNorm, PE, PCE, MLS, Maha, CTM, Residual, PCA RecError global, pNML, NeCo |
| FPR@95 only          | ViT/modelvit   | supercifar100 | mid      | KPCA RecError global                                       | NNGuide, KPCA RecError global, REN, GEN, MSR, GradNorm                                               |
| FPR@95 only          | ViT/modelvit   | supercifar100 | far      | Residual, pNML                                             | Residual, pNML                                                                                       |
| FPR@95 only          | ViT/modelvit   | cifar100      | near     | ViM, Maha, NeCo                                            | CTM, ViM, fDBD, NNGuide, GE, GradNorm, NeCo, Maha                                                    |
| FPR@95 only          | ViT/modelvit   | cifar100      | mid      | KPCA RecError global                                       | CTM, GradNorm, PE, fDBD                                                                              |
| FPR@95 only          | ViT/modelvit   | cifar100      | far      | Residual, ViM                                              | ViM, Residual, Maha, PCA RecError global, pNML, CTM                                                  |
| FPR@95 only          | ViT/modelvit   | tinyimagenet  | near     | PE                                                         | GradNorm                                                                                             |
| FPR@95 only          | ViT/modelvit   | tinyimagenet  | mid      | Residual, pNML                                             | Residual, pNML, Maha, ViM                                                                            |
| FPR@95 only          | ViT/modelvit   | tinyimagenet  | far      | GradNorm, GE                                               | GradNorm, PCE, KPCA RecError global                                                                  |
