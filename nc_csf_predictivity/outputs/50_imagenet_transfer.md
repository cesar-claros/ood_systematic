# ImageNet-scale transfer of the NC-only predictor (Phase 3)

**Source:** `x9_imagenet/phase3_transfer.py`. VGG13-trained regime-free NC-only heads (22-CSF labels) applied without re-training to 47 timm ImageNet-1k models; NC z-scored per family; mean imputed set-regret (AUGRC x1000) over the 21-CSF ImageNet roster; best fixed CSF chosen with hindsight per cell; tests one-sided (NC only better), Holm over the five main cells; family cells are exploratory (raw p).

Gates: GH heads reproduce the stored ResNet18 predictions; GD 47 x 21 x 8 complete; GR k = 2 grouping without singleton (near ['ninco', 'ssb hard']). G1 (August): ResNet-50 accuracy 76.15 vs 76.18; MSR and Energy AUROC within tolerance of OpenOOD v1.5.

## 21 CSFs: main cells

```
           cell  rows  models  nc_only  set_size best_fixed  best_fixed_regret  always_ctm  random_set  p_wil_best_holm  p_mean_best_holm  p_wil_random_holm  p_mean_random_holm
pooled (8 sets)   376      47   1.0513    7.7872    NNGuide             4.6370      7.8192      1.2547              0.0               0.0             0.0001              0.5836
      CLIP near    94      47   0.9996    7.7872         PE             2.8041     14.9271      1.2624              0.0               0.0             0.0017              0.5836
       CLIP far   282      47   1.0686    7.7872       fDBD             3.3390      5.4500      1.2521              0.0               0.0             0.0021              0.5836
   OpenOOD near    94      47   0.9996    7.7872         PE             2.8041     14.9271      1.2624              0.0               0.0             0.0017              0.5836
    OpenOOD far   141      47   0.9226    7.7872       fDBD             2.9925      5.1834      1.3096              0.0               0.0             0.0021              0.2055
```

## 21 CSFs: per family (exploratory)

```
              cell  rows  nc_only best_fixed  best_fixed_regret  always_ctm  random_set  p_wil_best  p_mean_best  p_wil_random  p_mean_random
    breadth pooled    88   1.4776        REN             4.3425     10.5022      1.3663      0.0000       0.0004        0.0255         0.5987
 breadth CLIP near    22   1.6903        REN             1.0199     16.7993      1.0546      0.4377       0.7432        0.1308         0.7508
  breadth CLIP far    66   1.4066        NCI             3.4893      8.4032      1.4702      0.0006       0.0716        0.0487         0.4196
   convnext pooled    48   1.4438     MahaPP             0.9073      5.9876      0.8908      0.9906       0.7945        0.9561         0.7941
convnext CLIP near    12   1.6349     MahaPP             1.0805     13.9098      1.4425      0.8438       0.8727        0.6614         0.6043
 convnext CLIP far    36   1.3801     MahaPP             0.8496      3.3468      0.7069      0.9798       0.7945        0.9678         0.8257
   resnet50 pooled   128   0.8283       fDBD             4.9299      8.1021      1.4227      0.0000       0.0000        0.0001         0.0540
resnet50 CLIP near    32   0.2958        REN             1.0659     15.6095      1.0266      0.0043       0.0056        0.0003         0.0035
 resnet50 CLIP far    96   1.0058        NCI             2.3080      5.5997      1.5548      0.0001       0.0192        0.0063         0.1037
       swin pooled    48   1.0715     MahaPP             2.1586      6.7110      1.2642      0.2002       0.0310        0.1093         0.3565
    swin CLIP near    12   0.9791        NCI             5.1463     15.5307      1.8715      0.0105       0.0157        0.0461         0.0766
     swin CLIP far    36   1.1023     MahaPP             1.0722      3.7710      1.0618      0.5194       0.5909        0.3576         0.5435
    vit_b16 pooled    64   0.6019     MahaPP             1.1211      5.7694      1.0307      0.0297       0.0860        0.0002         0.0737
 vit_b16 CLIP near    16   0.9965     MahaPP             1.9942     11.2981      1.4276      0.1206       0.0930        0.0583         0.1658
  vit_b16 CLIP far    48   0.4704     MahaPP             0.8300      3.9265      0.8984      0.0657       0.1496        0.0009         0.0659
```

## 20 CSFs, no KPCA: main cells

```
           cell  rows  models  nc_only  set_size best_fixed  best_fixed_regret  always_ctm  random_set  p_wil_best_holm  p_mean_best_holm  p_wil_random_holm  p_mean_random_holm
pooled (8 sets)   376      47   1.0513    7.4043    NNGuide             4.6260      7.8083      1.2548              0.0               0.0             0.0001              0.6076
      CLIP near    94      47   0.9996    7.4043         PE             2.8041     14.9271      1.2747              0.0               0.0             0.0019              0.6076
       CLIP far   282      47   1.0686    7.4043       fDBD             3.3244      5.4354      1.2481              0.0               0.0             0.0021              0.6076
   OpenOOD near    94      47   0.9996    7.4043         PE             2.8041     14.9271      1.2747              0.0               0.0             0.0019              0.6076
    OpenOOD far   141      47   0.9226    7.4043       fDBD             2.9925      5.1834      1.3142              0.0               0.0             0.0021              0.2079
```

## Oracle CSF frequency (rows, 21 CSFs)

```
MahaPP                  0.335
NCI                     0.165
ViM                     0.109
REN                     0.096
NNGuide                 0.064
Maha                    0.056
GE                      0.045
fDBD                    0.027
PE                      0.027
MLS                     0.027
PCA RecError global     0.021
Energy                  0.013
CTM                     0.005
NeCo                    0.005
KPCA RecError global    0.003
PCE                     0.003
```

## Shortlist inclusion rate per CSF (47 models)

```
csf
MahaPP                  0.49
PE                      0.47
MSR                     0.47
CTM                     0.45
PCE                     0.45
fDBD                    0.43
REN                     0.43
NeCo                    0.43
NNGuide                 0.43
pNML                    0.40
KPCA RecError global    0.38
Energy                  0.36
GEN                     0.36
MLS                     0.34
ViM                     0.34
NCI                     0.34
Maha                    0.32
GE                      0.32
GradNorm                0.30
Residual                0.30
```

## NC metrics across pools (median [min, max], scale-free metrics)

```
             pool  models         self_duality   max_equiangular_wc       equiangular_wc          equinorm_wc
            VGG13     280 0.085 [0.011, 0.678] 0.072 [0.022, 0.548] 0.101 [0.030, 0.570] 0.032 [0.009, 1.117]
         ResNet18      56 0.019 [0.008, 0.108] 0.056 [0.022, 0.199] 0.074 [0.030, 0.341] 0.018 [0.005, 0.574]
              ViT      40 1.070 [0.336, 1.297] 0.050 [0.029, 0.112] 0.037 [0.033, 0.082] 0.020 [0.010, 0.048]
       SSL probes      40 1.038 [0.551, 1.287] 0.115 [0.056, 0.189] 0.152 [0.071, 0.210] 0.131 [0.071, 0.219]
 ImageNet breadth      11 0.745 [0.288, 1.214] 0.045 [0.022, 0.360] 0.057 [0.031, 0.072] 0.064 [0.036, 0.080]
ImageNet convnext       6 0.511 [0.389, 0.617] 0.026 [0.022, 0.031] 0.037 [0.032, 0.045] 0.054 [0.038, 0.071]
ImageNet resnet50      16 0.890 [0.401, 1.201] 0.026 [0.017, 0.279] 0.031 [0.025, 0.064] 0.059 [0.030, 0.119]
    ImageNet swin       6 0.523 [0.391, 0.645] 0.027 [0.020, 0.031] 0.042 [0.032, 0.048] 0.057 [0.044, 0.074]
 ImageNet vit_b16       8 0.599 [0.138, 0.648] 0.046 [0.024, 0.350] 0.041 [0.030, 0.067] 0.047 [0.022, 0.087]
```
