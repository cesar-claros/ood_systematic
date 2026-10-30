# Leave-one-ViT-source-out test of the coverage explanation

**Source:** `nc_csf_predictivity/evaluation/vit_loso.py` (decision rule in the docstring, fixed before the run). Mean imputed set-regret (AUGRC x1000) over the 40 ViTs, 4 folds; tests one-sided, model-clustered sign-flip (mean) and Wilcoxon.

Gates GH, GL, GV, GK passed.

## Cells

```
arm                     cell  rows  regret  set_size  random_set  oracle_in_set  random_expected
  A                   pooled   320  3.0127    8.3500      2.0822         0.3312           0.3976
  B                   pooled   320 12.0274    4.3250     11.1782         0.2969           0.2060
  D                   pooled   320 41.4545    0.5000     42.0678         0.0156           0.0238
 D*                   pooled   320 10.7071    4.3250     11.1782         0.3719           0.2060
  C                   pooled   320  5.6726    8.6250      2.2172         0.2906           0.4107
 B'                   pooled   320 13.1780    5.2000     12.6802         0.3094           0.2476
  A                     near   110  1.4441    8.0364      1.2984         0.4455           0.3827
  A                      mid   140  4.0355    8.5143      3.2224         0.2786           0.4054
  A                      far    70  3.4320    8.5143      1.0334         0.2571           0.4054
  B                     near   110  4.8020    4.3727      5.3211         0.1636           0.2082
  B                      mid   140 21.0549    4.3000     16.5524         0.3286           0.2048
  B                      far    70  5.3268    4.3000      9.6338         0.4429           0.2048
  D                     near   110 34.8583    0.6364     35.6958         0.0000           0.0303
  D                      mid   140 45.5517    0.4286     43.8168         0.0357           0.0204
  D                      far    70 43.6257    0.4286     48.5831         0.0000           0.0204
 D*                     near   110  5.5247    4.3727      5.3211         0.2273           0.2082
 D*                      mid   140 17.2033    4.3000     16.5524         0.4500           0.2048
 D*                      far    70  5.8587    4.3000      9.6338         0.4429           0.2048
  C                     near   110  3.3452    8.5364      1.2982         0.4000           0.4065
  C                      mid   140  6.5428    8.6714      3.4871         0.2929           0.4129
  C                      far    70  7.5897    8.6714      1.1219         0.1143           0.4129
 B'                     near   110  5.5654    5.8273      5.4229         0.1818           0.2775
 B'                      mid   140 21.9243    4.8714     19.3184         0.3500           0.2320
 B'                      far    70  7.6479    4.8714     10.8081         0.4286           0.2320
  A      cifar100 (held out)    80  0.7899   10.4000      0.8956         0.6000           0.4952
  A       cifar10 (held out)    80  0.3967    7.4000      0.3468         0.1250           0.3524
  A supercifar100 (held out)    80  9.2041    8.4000      5.7293         0.1625           0.4000
  A  tinyimagenet (held out)    80  1.6600    7.2000      1.3571         0.4375           0.3429
  B      cifar100 (held out)    80  0.7366    6.1000      1.8192         0.4500           0.2905
  B       cifar10 (held out)    80  0.1336    6.0000      0.5511         0.4500           0.2857
  B supercifar100 (held out)    80 44.7660    0.7000     40.2585         0.1750           0.0333
  B  tinyimagenet (held out)    80  2.4736    4.5000      2.0840         0.1125           0.2143
  D      cifar100 (held out)    80 28.4140    0.0000     28.4140         0.0000           0.0000
  D       cifar10 (held out)    80 86.4098    0.0000     86.4098         0.0000           0.0000
  D supercifar100 (held out)    80 26.6525    1.0000     28.3997         0.0000           0.0476
  D  tinyimagenet (held out)    80 24.3419    1.0000     25.0477         0.0625           0.0476
 D*      cifar100 (held out)    80  0.5244    6.1000      1.8192         0.7125           0.2905
 D*       cifar10 (held out)    80  0.1027    6.0000      0.5511         0.5125           0.2857
 D* supercifar100 (held out)    80 38.8472    0.7000     40.2585         0.0000           0.0333
 D*  tinyimagenet (held out)    80  3.3543    4.5000      2.0840         0.2625           0.2143
  C      cifar100 (held out)    80  1.2954   12.0000      0.6961         0.3375           0.5714
  C       cifar10 (held out)    80  0.5673    6.9000      0.3693         0.2250           0.3286
  C supercifar100 (held out)    80 19.1926    7.3000      6.6862         0.0875           0.3476
  C  tinyimagenet (held out)    80  1.6353    8.3000      1.1174         0.5125           0.3952
 B'      cifar100 (held out)    80  0.8764    4.9000      2.1798         0.4500           0.2333
 B'       cifar10 (held out)    80  0.1125    7.9000      0.2971         0.5000           0.3762
 B' supercifar100 (held out)    80 50.1297    0.5000     46.9890         0.1250           0.0238
 B'  tinyimagenet (held out)    80  1.5932    7.5000      1.2548         0.1625           0.3571
```

## Primary tests (Holm over the two pooled comparisons)

```
  comparison   cell     diff  p_wilcoxon  p_mean  p_mean_holm
       B < D pooled -29.4271      0.0000  0.0001       0.0002
       B < D   near -30.0563      0.0000  0.0000          NaN
       B < D    mid -24.4968      0.0000  0.0080          NaN
       B < D    far -38.2989      0.0000  0.0001          NaN
B < random_B pooled   0.8492      0.0000  0.9695       0.9695
B < random_B   near  -0.5191      0.0012  0.0017          NaN
B < random_B    mid   4.5024      0.0065  0.9937          NaN
B < random_B    far  -4.3070      0.0000  0.0000          NaN
```

## Secondary tests (raw p)

```
    comparison   cell     diff  p_wilcoxon  p_mean
         B < A pooled   9.0147      0.9999  0.9998
         B < A   near   3.3579      1.0000  1.0000
         B < A    mid  17.0193      0.9853  0.9990
         B < A    far   1.8947      0.2476  0.7694
         D < A pooled  38.4418      1.0000  1.0000
         D < A   near  33.4142      1.0000  1.0000
         D < A    mid  41.5161      1.0000  1.0000
         D < A    far  40.1937      1.0000  1.0000
         C < A pooled   2.6599      1.0000  0.9998
         C < A   near   1.9011      0.9959  0.9995
         C < A    mid   2.5073      1.0000  0.9990
         C < A    far   4.1577      1.0000  1.0000
  D < random_D pooled  -0.6133      0.0019  0.0683
  D < random_D   near  -0.8374      0.0796  0.1901
  D < random_D    mid   1.7349      0.7270  0.9540
  D < random_D    far  -4.9574      0.0001  0.0008
        B < D* pooled   1.3203      0.4761  0.9879
        B < D*   near  -0.7227      0.0243  0.1806
        B < D*    mid   3.8516      0.9936  1.0000
        B < D*    far  -0.5319      0.0164  0.0687
        D* < A pooled   7.6944      1.0000  0.9999
        D* < A   near   4.0806      1.0000  1.0000
        D* < A    mid  13.1678      0.9958  0.9990
        D* < A    far   2.4267      0.8976  0.9187
D* < random_D* pooled  -0.4711      0.0000  0.0454
D* < random_D*   near   0.2036      0.2819  0.6006
D* < random_D*    mid   0.6509      0.0000  0.8238
D* < random_D*    far  -3.7751      0.0000  0.0000
  C < random_C pooled   3.4554      1.0000  1.0000
  C < random_C   near   2.0470      0.5751  0.9986
  C < random_C    mid   3.0558      0.9987  0.9998
  C < random_C    far   6.4678      1.0000  1.0000
B' < random_B' pooled   0.4978      0.0000  0.8926
B' < random_B'   near   0.1425      0.8568  0.8609
B' < random_B'    mid   2.6059      0.0000  0.9696
B' < random_B'    far  -3.1602      0.0000  0.0000
        B' < D pooled -28.2766      0.0000  0.0004
        B' < D   near -29.2930      0.0000  0.0000
        B' < D    mid -23.6273      0.0001  0.0123
        B' < D    far -35.9778      0.0000  0.0008
```

## ViT label rate per CSF (near/mid/far cells)

```
csf
ViM                     0.46
pNML                    0.29
KPCA RecError global    0.25
Residual                0.25
GE                      0.17
GradNorm                0.12
MahaPP                  0.08
PE                      0.08
NNGuide                 0.08
NCI                     0.08
MLS                     0.08
Maha                    0.04
NeCo                    0.04
PCA RecError global     0.04
Energy                  0.00
PCE                     0.00
MSR                     0.00
REN                     0.00
GEN                     0.00
fDBD                    0.00
CTM                     0.00
```

## Shortlist inclusion rate per arm

```
arm                      A     B    B'     C    D    D*
csf                                                    
CTM                   0.52  0.00  0.00  0.82  0.0  0.00
Energy                0.48  0.00  0.00  0.45  0.0  0.00
GE                    0.48  0.38  0.38  0.48  0.0  0.62
GEN                   0.48  0.00  0.00  0.62  0.0  0.00
GradNorm              0.50  0.00  0.00  0.30  0.0  0.28
KPCA RecError global  0.28  0.72  0.72  0.35  0.0  0.72
MLS                   0.50  0.52  0.50  0.50  0.0  0.28
MSR                   0.45  0.00  0.00  0.40  0.0  0.00
Maha                  0.22  0.12  0.12  0.15  0.0  0.00
MahaPP                0.38  0.38  0.68  0.38  0.0  0.00
NCI                   0.60  0.38  0.68  0.85  0.0  0.00
NNGuide               0.35  0.52  0.50  0.50  0.0  0.12
NeCo                  0.50  0.12  0.12  0.50  0.0  0.00
PCA RecError global    NaN  0.02  0.02  0.02  0.0  0.00
PCE                   0.32  0.00  0.00  0.38  0.0  0.00
PE                    0.35  0.00  0.00  0.32  0.0  0.00
REN                   0.48  0.00  0.00  0.75  0.0  0.00
Residual              0.18  0.65  0.72  0.00  0.0  0.75
ViM                   0.28  0.12  0.12  0.00  0.5  0.92
fDBD                  0.72  0.00  0.00  0.57  0.0  0.00
pNML                  0.30  0.38  0.62  0.28  0.0  0.62
```
