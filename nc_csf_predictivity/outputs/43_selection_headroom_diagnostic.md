# Selection headroom and executable-policy diagnostic (descriptive, 2026-09-14)

AUGRC scale by architecture (mean over OOD rows): {'ResNet18': 234.5, 'VGG13': 239.5, 'ViT': 186.5} 

## Headroom: per-regime training-chosen fixed detector vs per-row oracle

                  pool regime  n_rows  n_det  oracle_level train_fixed  regret_train_fixed      hindsight_fixed  regret_hindsight  regret_random  rel_headroom_pct
       ResNet18<-VGG13   near     148     20         213.1     NNGuide                4.05                  CTM              2.07          22.16              1.90
       ResNet18<-VGG13    mid     200     20         226.1         CTM                3.52                  CTM              3.52          17.37              1.56
       ResNet18<-VGG13    far     100     20         200.9         CTM                2.40                  CTM              2.40          14.37              1.20
            ViT<-VGG13   near     110     19         167.7     NNGuide                7.72                   GE              5.85          19.03              4.61
            ViT<-VGG13    mid     140     19         184.8         CTM               17.86 KPCA RecError global              3.66          16.47              9.66
            ViT<-VGG13    far      70     19         141.5         CTM               11.42                  ViM              5.90          15.06              8.07
      VGG lodo:cifar10   near     120     20         158.3         CTM               13.01           Confidence              1.59           9.13              8.22
      VGG lodo:cifar10    mid     240     20         181.0         CTM                6.48           Confidence              1.21           9.07              3.58
      VGG lodo:cifar10    far     120     20         125.5         CTM               11.27              NNGuide              3.00           8.20              8.98
     VGG lodo:cifar100   near     140     20         214.3     NNGuide                3.34              NNGuide              3.34          16.51              1.56
     VGG lodo:cifar100    mid     280     20         238.3         CTM                3.44                  CTM              3.44          16.48              1.44
     VGG lodo:cifar100    far     140     20         182.3         CTM                3.27                  CTM              3.27          16.01              1.79
VGG lodo:supercifar100   near     180     20         264.8     NNGuide                6.21                  REN              5.99          14.12              2.35
VGG lodo:supercifar100    mid     360     20         276.8     NNGuide               13.00                  CTM              8.27          17.62              4.70
VGG lodo:supercifar100    far     180     20         234.9     NNGuide               11.69                  CTM              4.89          12.56              4.98
 VGG lodo:tinyimagenet   near     300     20         212.9     NNGuide                3.79                  CTM              0.39          33.81              1.78
 VGG lodo:tinyimagenet    mid     120     20         192.0         CTM                0.38                  CTM              0.38          25.91              0.20
 VGG lodo:tinyimagenet    far      60     20         346.8         CTM                1.30                  CTM              1.30          27.52              0.38 

One-detector fixed rule chosen on all VGG13 OOD rows, applied to ResNet18: CTM; regret near/mid/far = 2.07 / 3.52 / 2.4
One-detector fixed rule chosen on all VGG13 OOD rows, applied to ViT: CTM; regret near/mid/far = 8.13 / 17.86 / 11.42

## Probe pool headroom (harmonized 21-detector table)

                      pool regime  n_rows  n_det  oracle_level train_fixed  regret_train_fixed hindsight_fixed  regret_hindsight  regret_random  rel_headroom_pct
  probes:probe_clip_vitb16   near      55     21         166.1     NNGuide               21.39             ViM              6.37          36.13             12.88
  probes:probe_clip_vitb16    mid      70     21         177.9         CTM                8.49             ViM              0.83          27.95              4.77
  probes:probe_clip_vitb16    far      35     21         140.7         CTM               19.26             ViM              0.26          55.74             13.69
probes:probe_dinov2_vitb14   near      55     21         157.4     NNGuide                6.65            NeCo              4.50          19.65              4.22
probes:probe_dinov2_vitb14    mid      70     21         176.9         CTM               11.87             NCI              0.79          20.90              6.71
probes:probe_dinov2_vitb14    far      35     21         134.6         CTM                7.47          MahaPP              0.53          18.89              5.55
               probes:both   near     110     21         161.8     NNGuide               14.02             ViM              6.03          27.89              8.67
               probes:both    mid     140     21         177.4         CTM               10.18             NCI              2.07          24.43              5.74
               probes:both    far      70     21         137.6         CTM               13.37          MahaPP              0.85          37.32              9.71 

## Executable policies from the stored heads (fixed = one-detector rule chosen on VGG13; abstention falls back to it)

### ResNet18, NC only, regime-free (ablations/calib_cliques_regime_free/track1/xarch/none_nr_marginal/preds.parquet, fold=None, models=56)

regime   n fixed  regret_fixed  top1_raw  top1_prior_corrected  best_member  random_member  set_size  empty  policy@100%  policy@80%  policy@60%  policy@40%  policy@20%
  near 148   CTM          2.07      6.28                  4.94         1.23          12.05       7.5    0.0         6.28        5.07        4.08        3.17        2.43
   mid 200   CTM          3.52      9.11                  5.60         1.16          12.33       7.3    0.0         9.11        5.59        5.48        4.46        3.77
   far 100   CTM          2.40      5.05                  3.63         0.80           9.41       7.3    0.0         5.05        3.67        3.56        2.68        2.54 

### ResNet18, NC+source, regime-free (ablations/calib_cliques_regime_free/track1/xarch/source_nr_marginal/preds.parquet, fold=None, models=56)

regime   n fixed  regret_fixed  top1_raw  top1_prior_corrected  best_member  random_member  set_size  empty  policy@100%  policy@80%  policy@60%  policy@40%  policy@20%
  near 148   CTM          2.07      6.03                  2.97         1.06           6.67       5.6    0.0         6.03        5.89        4.50        3.29        2.32
   mid 200   CTM          3.52      9.17                  3.80         0.96           8.63       5.8    0.0         9.17        8.89        5.47        4.61        4.21
   far 100   CTM          2.40      5.19                  2.84         0.64           6.37       5.8    0.0         5.19        5.33        3.76        3.32        3.10 

### ResNet18, paper arm with regime input (ablations/calib_cliques/track1/xarch/source/preds.parquet, fold=None, models=56)

regime   n fixed  regret_fixed  top1_raw  top1_prior_corrected  best_member  random_member  set_size  empty  policy@100%  policy@80%  policy@60%  policy@40%  policy@20%
  near 148   CTM          2.07      4.78                  3.17         1.02           6.66       6.5    0.0         4.78        3.61        3.45        3.10        2.31
   mid 200   CTM          3.52      5.73                  3.12         1.18           6.17       4.2    0.0         5.73        5.03        3.24        2.75        3.12
   far 100   CTM          2.40      4.61                  2.68         0.39           6.86       6.1    0.0         4.61        4.79        4.29        3.01        2.29 

### ViT, NC only, regime-free (ablations/calib_cliques_regime_free/track1/lopo/none_nr_marginal/preds.parquet, fold=lopo_modelvit, models=40)

regime   n fixed  regret_fixed  top1_raw  top1_prior_corrected  best_member  random_member  set_size  empty  policy@100%  policy@80%  policy@60%  policy@40%  policy@20%
  near 110   CTM          8.13      7.05                  7.13         1.35          14.34       8.5    0.0         7.05        7.04        7.51        7.93        8.38
   mid 140   CTM         17.86     15.03                 14.31         2.24          14.34       7.9    0.0        15.03       15.01       16.07       16.81       18.42
   far  70   CTM         11.42     10.80                 10.94         3.94          13.42       7.9    0.0        10.80       10.99       11.66       11.90       12.31 

## Exploratory per-detector ridge loss regressor on the 8 NC features (alpha = 1, trained on VGG13), top-1 regret

- ResNet18, train_only standardization: {'near': 4.83, 'mid': 5.89, 'far': 6.36}
- ResNet18, target_pool standardization: {'near': 20.87, 'mid': 13.95, 'far': 20.07}
- ViT, train_only standardization: {'near': 61.39, 'mid': 25.48, 'far': 18.47}
- ViT, target_pool standardization: {'near': 9.93, 'mid': 17.43, 'far': 10.16}
