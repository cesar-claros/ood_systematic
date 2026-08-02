# Pool A pilot (X8): frozen DINOv2/CLIP probes

**Source:** `x8_pool_a/pool_a_analysis.py`

## Probe accuracy and descriptors (mean per encoder x source)

```
                                     acc  var_collapse  self_duality  rho_res
paradigm            source                                                   
probe_clip_vitb16   cifar10        0.951         0.038         1.109    0.813
                    cifar100       0.803         0.081         1.112    0.769
                    supercifar100  0.691         0.112         1.285    0.800
                    tinyimagenet   0.750         0.097         1.064    0.708
probe_dinov2_vitb14 cifar10        0.982         0.017         0.607    0.583
                    cifar100       0.897         0.014         0.834    0.413
                    supercifar100  0.719         0.027         0.989    0.526
                    tinyimagenet   0.870         0.011         0.937    0.320
```

## Top cliques per (encoder, source, regime)

```
           paradigm        source regime                                  csf
  probe_clip_vitb16       cifar10    far                             Residual
  probe_clip_vitb16       cifar10    mid                             CTM, NCI
  probe_clip_vitb16       cifar10   near                       CTM, ViM, fDBD
  probe_clip_vitb16      cifar100    far                             Residual
  probe_clip_vitb16      cifar100    mid                                  NCI
  probe_clip_vitb16      cifar100   near                       NCI, ViM, fDBD
  probe_clip_vitb16 supercifar100    far                             Residual
  probe_clip_vitb16 supercifar100    mid                             NCI, ViM
  probe_clip_vitb16 supercifar100   near                  CTM, NCI, ViM, fDBD
  probe_clip_vitb16  tinyimagenet    far                                  NCI
  probe_clip_vitb16  tinyimagenet    mid                        Residual, ViM
  probe_clip_vitb16  tinyimagenet   near Energy, GEN, MLS, NCI, NNGuide, fDBD
probe_dinov2_vitb14       cifar10    far                               MahaPP
probe_dinov2_vitb14       cifar10    mid                     Energy, GEN, MLS
probe_dinov2_vitb14       cifar10   near          CTM, GEN, MahaPP, ViM, fDBD
probe_dinov2_vitb14      cifar100    far                               MahaPP
probe_dinov2_vitb14      cifar100    mid                          Energy, NCI
probe_dinov2_vitb14      cifar100   near                          MahaPP, ViM
probe_dinov2_vitb14 supercifar100    far                               MahaPP
probe_dinov2_vitb14 supercifar100    mid                        GradNorm, NCI
probe_dinov2_vitb14 supercifar100   near           MahaPP, NNGuide, ViM, fDBD
probe_dinov2_vitb14  tinyimagenet    far                               Energy
probe_dinov2_vitb14  tinyimagenet    mid                          MahaPP, ViM
probe_dinov2_vitb14  tinyimagenet   near   Energy, GE, GEN, MLS, NCI, PE, REN
```

## New CSFs on the SSL pool (mean AUGRC per encoder x regime)

```
paradigm probe_clip_vitb16                 probe_dinov2_vitb14                
regime                 far     mid    near                 far     mid    near
csf                                                                           
CTM                 159.94  186.39  178.37              142.08  188.75  165.21
Energy              230.09  197.89  194.02              142.10  181.66  163.02
Maha                141.54  199.47  209.92              140.39  223.45  179.86
MahaPP              141.85  204.02  213.82              135.14  202.11  169.66
NCI                 159.93  181.29  177.75              143.13  176.82  164.50
NeCo                148.54  255.27  250.79              138.09  206.13  179.14
Residual            141.27  201.16  208.47              145.10  229.69  193.54
ViM                 141.11  179.12  172.47              137.07  191.42  163.37
```

## H1: Mantel, classical vs extended descriptors

```
                       vector  n_models      r      p
             classical NC (8)        40 0.7209 0.0001
extended (8 + X8 descriptors)        40 0.7557 0.0001
```

## H2/H4: benchmark-trained predictors on the probe pool
(train pools: VGG-13 CNNs, fine-tuned ViTs, and both; the ViT pool is the weak-collapse regime closest to frozen probes)

```
train_pool    side regime  predictor_regret  empty_pct best_baseline  baseline_regret
     VGG13     all    far              1.44        0.0          fDBD            11.21
     VGG13     all    mid              3.22        0.0          fDBD             7.11
     VGG13     all   near              5.99        0.0          fDBD             7.39
     VGG13    head    far             11.64        7.1           MSR             4.30
     VGG13    head    mid             26.60       14.3           MLS             2.90
     VGG13    head   near              0.49        0.0           MLS             2.85
     VGG13 feature    far              1.56       11.4          fDBD            11.18
     VGG13 feature    mid             32.13       25.7          fDBD             5.18
     VGG13 feature   near             10.23        5.5          fDBD             6.80
       ViT     all    far             18.43        0.0          fDBD            11.21
       ViT     all    mid             78.46       32.9          fDBD             7.11
       ViT     all   near             21.31        0.0          fDBD             7.39
       ViT    head    far            109.36       38.6           MSR             4.30
       ViT    head    mid            105.64       45.7           MLS             2.90
       ViT    head   near             56.70       22.7           MLS             2.85
       ViT feature    far              3.19       14.3          fDBD            11.18
       ViT feature    mid             43.63       67.1          fDBD             5.18
       ViT feature   near             40.74       35.5          fDBD             6.80
 VGG13+ViT     all    far              0.57        0.0          fDBD            11.21
 VGG13+ViT     all    mid              5.08        0.0          fDBD             7.11
 VGG13+ViT     all   near              5.31        0.0          fDBD             7.39
 VGG13+ViT    head    far             11.72        0.0           MSR             4.30
 VGG13+ViT    head    mid             50.29       25.7           MLS             2.90
 VGG13+ViT    head   near              0.49        0.0           MLS             2.85
 VGG13+ViT feature    far              0.71        0.0          fDBD            11.18
 VGG13+ViT feature    mid             20.12        5.7          fDBD             5.18
 VGG13+ViT feature   near              9.04        0.0          fDBD             6.80
```
