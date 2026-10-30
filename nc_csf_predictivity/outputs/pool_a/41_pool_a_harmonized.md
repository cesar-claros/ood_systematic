# Pool A harmonized rerun (paper-faithful protocol, 21 CSFs)

**Source:** `x8_pool_a/pool_a_harmonized.py`. Gates: probe accuracies identical; invariant CSFs (Maha, MahaPP, CTM, fDBD) reproduce the pilot (max |dAUGRC| 0.0042). Protocol changes and BO settings in the module docstring.

## Regime-free selector, pooled cells (drift vs submitted GLiC-2)

| roster | config | near | mid | far | submitted |
|---|---|---|---|---|---|
| 18 (submitted) | source_nr | 7.22 | 5.44 | 11.75 | 4.96/3.99/5.07 |
| 18 (submitted) | none_nr | 2.32 | 2.02 | 1.73 | 1.59/1.38/0.59 |
| 21 (revision) | source_nr | 7.07 | 4.68 | 11.70 | 4.96/3.99/5.07 |
| 21 (revision) | none_nr | 2.15 | 2.02 | 1.66 | 1.59/1.38/0.59 |

## Per-CSF pooled OOD AUGRC drift (harmonized - pilot)

```
                      harmonized   pilot  delta
csf                                            
PCA RecError global       176.93  188.51 -11.59
NeCo                      197.05  206.17  -9.12
Residual                  190.41  194.67  -4.26
NNGuide                   177.57  178.49  -0.92
pNML                      316.96  317.84  -0.88
ViM                       168.98  169.20  -0.22
Maha                      190.34  190.34   0.00
CTM                       174.15  174.15   0.00
PE                        183.63  183.63   0.00
PCE                       184.74  184.73   0.00
MSR                       185.33  185.33  -0.00
MahaPP                    185.05  185.05  -0.00
fDBD                      171.94  171.94   0.00
Energy                    185.10  185.10  -0.00
GE                        183.09  183.09   0.00
MLS                       184.55  184.55   0.00
NCI                       170.42  170.31   0.11
GradNorm                  229.40  229.23   0.18
GEN                       185.59  183.89   1.70
REN                       186.47  182.73   3.75
KPCA RecError global      199.23     NaN    NaN
```

## H1 Mantel (harmonized, 21-CSF ranks)

```
                       vector  n_models      r      p
             classical NC (8)        40 0.7240 0.0001
extended (8 + X8 descriptors)        40 0.7871 0.0001
```

## Cliques: 11 of 24 (encoder, source, regime) cells identical to the newcsfs pilot cliques
