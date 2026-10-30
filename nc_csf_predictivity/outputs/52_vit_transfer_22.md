# CNN -> ViT transfer of the NC-only predictor (22-CSF heads)

**Source:** `nc_csf_predictivity/evaluation/vit_transfer_22.py`. VGG13-trained regime-free NC-only heads applied without re-training to the 40 fine-tuned ViT models; NC z-scored within the ViT pool; mean imputed set-regret (AUGRC x1000) over the 21-CSF ViT roster; best fixed CSF chosen with hindsight per cell; tests one-sided (NC only better), Holm over the four main cells; per-source cells exploratory (raw p).

Gates GH and GV passed.

## Main cells

```
  cell  rows  nc_only  set_size           best_fixed  best_fixed_regret  always_ctm  random_set  contains_oracle  random_expected  p_wil_best_holm  p_mean_best_holm  p_wil_random_holm  p_mean_random_holm
pooled   320   3.0127    8.3500 KPCA RecError global             7.5109     13.1164      2.0822           0.3312           0.3976           0.0000            0.0000             0.7887                 1.0
  near   110   1.4441    8.0364                   GE             5.8535      8.1386      1.2984           0.4455           0.3827           0.0000            0.0000             0.0072                 1.0
   mid   140   4.0355    8.5143 KPCA RecError global             3.6697     17.8770      3.2224           0.2786           0.4054           0.5044            0.6366             0.9765                 1.0
   far    70   3.4320    8.5143                  ViM             5.8997     11.4175      1.0334           0.2571           0.4054           0.0742            0.2456             0.5966                 1.0
```

## Per source (exploratory, raw p)

```
              cell  rows  nc_only  set_size           best_fixed  best_fixed_regret  always_ctm  random_set  contains_oracle  random_expected  p_wil_best  p_mean_best  p_wil_random  p_mean_random
      cifar10 near    20   0.5688       7.4                  ViM             0.3365      8.5008      0.5650            0.250           0.3524      0.9621       0.9301        0.4347         0.5158
       cifar10 mid    40   0.4503       7.4 KPCA RecError global             0.4021      6.1370      0.3366            0.075           0.3524      0.3926       0.5904        0.8841         0.9122
       cifar10 far    20   0.1176       7.4             Residual             0.0237      2.3996      0.1489            0.100           0.3524      0.9958       1.0000        0.0047         0.1066
     cifar100 near    20   0.4853      10.4                  ViM             0.7315      4.1924      0.5318            0.650           0.4952      0.0881       0.0147        0.1559         0.3939
      cifar100 mid    40   0.6307      10.4 KPCA RecError global             2.7537      5.4034      1.1092            0.750           0.4952      0.0018       0.0310        0.0018         0.1048
      cifar100 far    20   1.4131      10.4                  ViM             0.6627      5.9869      0.8324            0.250           0.4952      0.8575       0.7512        0.9819         0.9941
supercifar100 near    20   5.6986       8.4                  ViM             9.3535     19.6024      2.6749            0.200           0.4000      0.0379       0.0446        0.7738         0.8968
 supercifar100 mid    40  10.3489       8.4 KPCA RecError global             3.8848     47.3521      9.0339            0.150           0.4000      0.9999       0.9869        0.9441         0.9208
 supercifar100 far    20  10.4201       8.4             Residual             0.1850     28.9575      2.1745            0.150           0.4000      0.9988       0.9963        0.7271         0.8804
 tinyimagenet near    50   0.4760       7.2             GradNorm             1.5120      4.9867      1.3479            0.540           0.3429      0.0015       0.0009        0.0001         0.0058
  tinyimagenet mid    20   5.3889       7.2                  ViM             0.7070      7.3542      1.5978            0.000           0.3429      1.0000       1.0000        1.0000         1.0000
  tinyimagenet far    10   0.1227       7.2             GradNorm             0.0988      5.2347      0.9219            0.800           0.3429      0.7500       0.7501        0.0020         0.0020
```
