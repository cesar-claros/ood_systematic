# Mantel results on the paper's CSF roster

**Source:** `nc_csf_predictivity/evaluation/mantel_all_csfs.py` (decision rule in the docstring, fixed before the run). Gates GP, GT, GN passed. Rosters: reported = 22 methods including CTMmean and CTMmeanOC (and Confidence on ViT); 20 = paper roster without Maha++ and NCI (19 on ViT); 22 = paper roster (21 on ViT).

## Decision rule

- reported: {'M1': True, 'M2': True, 'M3_mid_best': 'confidnet:mid, devries:far, dg:far, vit:mid'}
- 20: {'M1': True, 'M2': True, 'M3_mid_best': 'confidnet:mid, devries:far, dg:far, vit:mid'}
- 22: {'M1': True, 'M2': True, 'M3_mid_best': 'confidnet:mid, devries:far, dg:far, vit:mid'}

## Global Mantel r, rank distance

```
roster     reported      20      22
study                              
confidnet    0.6436  0.6839  0.6304
devries      0.6792  0.6227  0.6231
dg           0.3093  0.3085  0.3692
vit          0.5897  0.6985  0.7041
```

## Global Mantel p

```
roster     reported      20      22
study                              
confidnet    0.0001  0.0001  0.0001
devries      0.0001  0.0001  0.0001
dg           0.0034  0.0029  0.0012
vit          0.0001  0.0001  0.0001
```

## Global Mantel r, Jaccard

```
roster                             reported      20      22
study     test                                             
confidnet full_NC_vs_jaccard_top1    0.5542  0.5526  0.5535
          full_NC_vs_jaccard_top3    0.6692  0.5864  0.4696
devries   full_NC_vs_jaccard_top1    0.3598  0.3755  0.2723
          full_NC_vs_jaccard_top3    0.5154  0.5465  0.5053
dg        full_NC_vs_jaccard_top1    0.4704  0.4493  0.4421
          full_NC_vs_jaccard_top3    0.3437  0.2386  0.3931
vit       full_NC_vs_jaccard_top1    0.3748  0.3991  0.3991
          full_NC_vs_jaccard_top3    0.4910  0.5036  0.5619
```

## Per regime r, rank distance

```
roster               reported      20      22
study     ood_group                          
confidnet far          0.3158  0.2751  0.3044
          mid          0.7270  0.7153  0.6668
          near         0.5763  0.5891  0.5286
devries   far          0.7702  0.7215  0.7039
          mid          0.6755  0.6399  0.6239
          near         0.6579  0.6124  0.6175
dg        far          0.4401  0.3982  0.4967
          mid          0.3613  0.3613  0.4057
          near         0.3196  0.3175  0.3623
vit       far          0.1796  0.1566  0.1844
          mid          0.7834  0.8316  0.8428
          near         0.4002  0.4256  0.4317
```

## Per metric r, rank distance

```
roster                        reported      20      22
nc_metric          study                              
equiangular_uc     confidnet    0.6453  0.6752  0.6527
                   devries      0.7116  0.6664  0.6596
                   dg           0.2015  0.1795  0.1475
                   vit          0.5460  0.6392  0.5753
equiangular_wc     confidnet    0.7252  0.7688  0.7419
                   devries      0.5600  0.4941  0.4855
                   dg           0.6763  0.7108  0.8786
                   vit          0.4414  0.5343  0.5646
equinorm_uc        confidnet    0.5113  0.5603  0.5096
                   devries      0.4577  0.4837  0.4508
                   dg           0.2482  0.1101  0.1187
                   vit          0.4792  0.5217  0.4823
equinorm_wc        confidnet    0.2526  0.3251  0.3288
                   devries      0.1704  0.1966  0.2136
                   dg           0.0882  0.0592  0.0651
                   vit          0.3391  0.3185  0.3952
max_equiangular_uc confidnet    0.7225  0.7683  0.7094
                   devries      0.5507  0.4825  0.5060
                   dg           0.1612  0.1296  0.1342
                   vit          0.5007  0.5885  0.5063
max_equiangular_wc confidnet    0.7972  0.8204  0.7710
                   devries      0.7527  0.6824  0.6849
                   dg           0.4268  0.5641  0.7354
                   vit          0.7017  0.6811  0.7697
self_duality       confidnet    0.3366  0.4049  0.4108
                   devries      0.2585  0.2845  0.2884
                   dg           0.1691  0.1799  0.1750
                   vit          0.5811  0.6711  0.7194
var_collapse       confidnet    0.3820  0.3458  0.3378
                   devries      0.4525  0.4676  0.4901
                   dg           0.4416  0.4859  0.4639
                   vit          0.5129  0.4804  0.4587
```

## Partial Mantel r

```
roster                               reported      20      22
study     control                                            
confidnet (none, partial-code path)    0.6436  0.6839  0.6304
          (none, replication)          0.6436  0.6839  0.6304
          dataset                      0.3262  0.4081  0.3116
          iid_mean                     0.4586  0.5302  0.4553
          iid_msr                      0.5806  0.6288  0.5572
          iid_msr+dataset              0.4691  0.5122  0.3960
devries   (none, partial-code path)    0.6792  0.6227  0.6231
          (none, replication)          0.6792  0.6227  0.6231
          dataset                      0.4741  0.3647  0.3690
          iid_mean                     0.6199  0.5809  0.5745
          iid_msr                      0.6393  0.5739  0.5605
          iid_msr+dataset              0.6270  0.4979  0.5051
dg        (none, partial-code path)    0.3093  0.3085  0.3692
          (none, replication)          0.3093  0.3085  0.3692
          dataset                     -0.1749 -0.1872 -0.0757
          iid_mean                     0.2395  0.2341  0.3414
          iid_msr                      0.2156  0.2061  0.3222
          iid_msr+dataset              0.1495  0.1415  0.3618
vit       (none, partial-code path)    0.5897  0.6985  0.7041
          (none, replication)          0.5897  0.6985  0.7041
          dataset                      0.1886  0.4043  0.4138
          iid_mean                     0.4249  0.5793  0.5950
          iid_msr                      0.4289  0.5775  0.5921
          iid_msr+dataset              0.1788  0.3986  0.4084
          iid_msr+lr                   0.1789  0.3439  0.2777
          iid_msr+lr+dataset          -0.0422  0.1807  0.0877
```

## Partial Mantel p

```
roster                               reported      20      22
study     control                                            
confidnet (none, partial-code path)    0.0001  0.0001  0.0001
          (none, replication)          0.0001  0.0001  0.0001
          dataset                      0.0004  0.0001  0.0004
          iid_mean                     0.0001  0.0001  0.0001
          iid_msr                      0.0001  0.0001  0.0001
          iid_msr+dataset              0.0001  0.0001  0.0001
devries   (none, partial-code path)    0.0001  0.0001  0.0001
          (none, replication)          0.0001  0.0001  0.0001
          dataset                      0.0001  0.0002  0.0002
          iid_mean                     0.0001  0.0001  0.0001
          iid_msr                      0.0001  0.0001  0.0001
          iid_msr+dataset              0.0001  0.0001  0.0001
dg        (none, partial-code path)    0.0031  0.0027  0.0010
          (none, replication)          0.0034  0.0029  0.0012
          dataset                      0.9806  0.9903  0.7985
          iid_mean                     0.0089  0.0093  0.0017
          iid_msr                      0.0102  0.0123  0.0008
          iid_msr+dataset              0.0560  0.0597  0.0010
vit       (none, partial-code path)    0.0001  0.0001  0.0001
          (none, replication)          0.0001  0.0001  0.0001
          dataset                      0.0109  0.0001  0.0002
          iid_mean                     0.0001  0.0001  0.0001
          iid_msr                      0.0001  0.0001  0.0001
          iid_msr+dataset              0.0153  0.0001  0.0002
          iid_msr+lr                   0.0204  0.0002  0.0012
          iid_msr+lr+dataset           0.7239  0.0137  0.1429
```

## NC1 variants on the 22 roster: r

```
variant                         C-log   C-raw   K-log   L-log   L-raw
study     control                                                    
confidnet (none, replication)  0.6019  0.5954  0.6312  0.6334  0.6304
          dataset              0.2583  0.2461  0.3132  0.3173  0.3116
          iid_msr              0.5357  0.5288  0.5568  0.5583  0.5572
devries   (none, replication)  0.7068  0.7006  0.6098  0.5924  0.6231
          dataset              0.5143  0.5035  0.3459  0.3157  0.3690
          iid_msr              0.6367  0.6416  0.5752  0.5607  0.5605
dg        (none, replication)  0.3233  0.3477  0.3417  0.3399  0.3692
          dataset             -0.1556 -0.1131 -0.1236 -0.1268 -0.0757
          iid_msr              0.2192  0.2525  0.2942  0.2930  0.3222
vit       (none, replication)  0.7557  0.7318  0.6965  0.6969  0.7041
          dataset              0.5150  0.4669  0.4020  0.4021  0.4138
          iid_msr              0.6491  0.6216  0.5786  0.5800  0.5921
```

## NC1 variants on the 22 roster: p

```
variant                         C-log   C-raw   K-log   L-log   L-raw
study     control                                                    
confidnet (none, replication)  0.0001  0.0001  0.0001  0.0001  0.0001
          dataset              0.0023  0.0034  0.0004  0.0003  0.0004
          iid_msr              0.0001  0.0001  0.0001  0.0001  0.0001
devries   (none, replication)  0.0001  0.0001  0.0001  0.0001  0.0001
          dataset              0.0001  0.0001  0.0003  0.0006  0.0002
          iid_msr              0.0001  0.0001  0.0001  0.0001  0.0001
dg        (none, replication)  0.0034  0.0022  0.0013  0.0014  0.0012
          dataset              0.9399  0.8619  0.9219  0.9259  0.7985
          iid_msr              0.0123  0.0069  0.0014  0.0016  0.0008
vit       (none, replication)  0.0001  0.0001  0.0001  0.0001  0.0001
          dataset              0.0001  0.0001  0.0002  0.0002  0.0002
          iid_msr              0.0001  0.0001  0.0001  0.0001  0.0001
```

