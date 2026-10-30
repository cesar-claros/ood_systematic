# G1/G2 gate: resnet50.tv_in1k

Accuracy: 76.15 vs 76.18 (tol 0.5) PASS

| CSF | near ours | near ref | far ours | far ref | gate |
|---|---|---|---|---|---|
| MSR | 75.41 | 76.02 | 84.01 | 85.23 | PASS |
| Energy | 76.05 | 75.89 | 89.53 | 89.47 | PASS |
| ViM | 72.40 | 72.08 | 91.26 | 92.68 | report-only (within 2.0) |
| Maha | 55.37 | 55.44 | 74.12 | 74.25 | report-only (within 2.0) |

Extra sets (no OpenOOD reference), AUROC per CSF:

```
eval_dataset          imagenet_o  places    sun
csf                                            
CTM                        64.04   80.13  83.03
Energy                     41.50   84.00  86.55
GE                         31.45   82.84  84.73
GEN                        28.14   80.25  81.21
GradNorm                   21.73   64.85  63.89
KPCA RecError global       69.85   78.95  83.66
MLS                        40.74   84.03  86.44
MSR                        28.14   79.72  80.67
Maha                       81.10   51.55  51.77
MahaPP                     82.66   78.54  83.73
NCI                        59.71   87.94  90.94
NNGuide                    48.49   88.37  91.36
NeCo                       72.42   48.20  46.12
PCA RecError global        73.99   69.69  69.27
PCE                        28.14   80.41  81.42
PE                         28.58   81.71  83.00
REN                        28.66   81.80  83.11
Residual                   81.84   45.04  44.91
ViM                        71.13   78.30  80.97
fDBD                       60.46   82.83  85.73
pNML                       42.76   74.27  75.19
```

G2 sanity, iid failure-detection AUGRC (lower better):

```
                 csf      augrc
                 GEN  53.106559
                 PCE  53.119452
                 MSR  53.162918
                  PE  53.735631
                 REN  53.818682
                  GE  56.060329
                 NCI  65.716798
                fDBD  66.988015
                 MLS  68.975792
                 CTM  69.807321
              Energy  71.088705
             NNGuide  71.675691
                 ViM  74.588294
                pNML  86.355533
              MahaPP  92.135610
 PCA RecError global  92.299391
KPCA RecError global  92.692049
            GradNorm  96.209149
                Maha 106.189872
            Residual 117.402235
                NeCo 126.814545
```
