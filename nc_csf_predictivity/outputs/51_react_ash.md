# ReAct and ASH-S on the CNN pools (camera-ready item 4.2)

**Source:** `nc_csf_predictivity/evaluation/react_ash_table.py`. Paired AUGRC x1000 (wrapper - base, positive = worse) over the OOD rows of the paper's VGG13 and ResNet18 models; Wilcoxon two-sided.

Gates GW and GC passed. Wrapped CSFs: Energy, GE, GEN, GradNorm, MLS, MSR, PCE, PE, REN.

## Per CSF

```
architecture            arm      csf   rows     base  mean_diff  median_diff  wrapper_better  p_wilcoxon
    ResNet18 ASH (ash_s@65)   Energy  448.0 222.7113    -0.0346      -0.0623          0.6920      0.0000
    ResNet18 ASH (ash_s@65)       GE  448.0 223.0897    -0.0236      -0.0556          0.6786      0.0000
    ResNet18 ASH (ash_s@65)      GEN  448.0 224.0374     0.1338      -0.0138          0.5558      0.3143
    ResNet18 ASH (ash_s@65) GradNorm  448.0 254.6032     1.1015       0.0381          0.4353      0.0000
    ResNet18 ASH (ash_s@65)      MLS  448.0 222.4400    -0.0099      -0.0459          0.6518      0.0000
    ResNet18 ASH (ash_s@65)      MSR  448.0 223.6423     0.0198      -0.0155          0.5871      0.0063
    ResNet18 ASH (ash_s@65)      PCE  448.0 223.1243     0.0055      -0.0235          0.5893      0.0000
    ResNet18 ASH (ash_s@65)       PE  448.0 222.5589    -0.0144      -0.0337          0.6429      0.0000
    ResNet18 ASH (ash_s@65)      REN  448.0 224.4950     0.0871      -0.0173          0.5558      0.2102
    ResNet18          ReAct   Energy  448.0 222.7113     8.9460       6.3485          0.2143      0.0000
    ResNet18          ReAct       GE  448.0 223.0897    12.5015       5.0373          0.2076      0.0000
    ResNet18          ReAct      GEN  448.0 224.0374     6.0454       3.7571          0.2812      0.0000
    ResNet18          ReAct GradNorm  448.0 254.6032    84.1038      83.9820          0.0067      0.0000
    ResNet18          ReAct      MLS  448.0 222.4400    13.1758       5.1431          0.2143      0.0000
    ResNet18          ReAct      MSR  448.0 223.6423    13.1628       4.5597          0.2009      0.0000
    ResNet18          ReAct      PCE  448.0 223.1243    10.3923       4.1225          0.2076      0.0000
    ResNet18          ReAct       PE  448.0 222.5589     9.9057       4.0048          0.2188      0.0000
    ResNet18          ReAct      REN  448.0 224.4950     5.4322       2.8662          0.3571      0.0000
       VGG13 ASH (ash_s@65)   Energy 2240.0 229.1183     0.0447      -0.0745          0.6076      0.0000
       VGG13 ASH (ash_s@65)       GE 2240.0 229.4895     0.0190      -0.0690          0.6045      0.0000
       VGG13 ASH (ash_s@65)      GEN 2240.0 229.4604     0.0247      -0.0228          0.5478      0.0254
       VGG13 ASH (ash_s@65) GradNorm 2240.0 252.4248     0.0843      -0.0463          0.5424      0.7832
       VGG13 ASH (ash_s@65)      MLS 2240.0 228.6443     0.0256      -0.0492          0.5906      0.0000
       VGG13 ASH (ash_s@65)      MSR 2240.0 229.9226     0.0171      -0.0209          0.5571      0.0021
       VGG13 ASH (ash_s@65)      PCE 2240.0 229.3447     0.0225      -0.0329          0.5835      0.0000
       VGG13 ASH (ash_s@65)       PE 2240.0 228.7527     0.0285      -0.0545          0.6009      0.0000
       VGG13 ASH (ash_s@65)      REN 2240.0 228.9805    -0.1028       0.0374          0.4795      0.0001
       VGG13          ReAct   Energy 2240.0 229.1183     6.4462       1.1445          0.3263      0.0000
       VGG13          ReAct       GE 2240.0 229.4895     6.2886       0.9435          0.3165      0.0000
       VGG13          ReAct      GEN 2240.0 229.4604     3.7284       1.5893          0.3286      0.0000
       VGG13          ReAct GradNorm 2240.0 252.4248    43.3903      15.9130          0.2085      0.0000
       VGG13          ReAct      MLS 2240.0 228.6443     4.3383       1.2762          0.3031      0.0000
       VGG13          ReAct      MSR 2240.0 229.9226     4.7133       1.9248          0.2379      0.0000
       VGG13          ReAct      PCE 2240.0 229.3447     4.2663       1.7840          0.2504      0.0000
       VGG13          ReAct       PE 2240.0 228.7527     4.5204       1.4072          0.2750      0.0000
       VGG13          ReAct      REN 2240.0 228.9805     3.8146       1.8487          0.3018      0.0000
```

## Per regime (nine CSFs pooled)

```
architecture            arm regime   rows     base  mean_diff  median_diff  wrapper_better  p_wilcoxon
    ResNet18 ASH (ash_s@65)    far  900.0 211.2587     0.0859      -0.0371          0.6244      0.0000
    ResNet18 ASH (ash_s@65)    mid 1800.0 237.3712     0.1359      -0.0336          0.6333      0.0000
    ResNet18 ASH (ash_s@65)   near 1332.0 222.8480     0.1839      -0.0121          0.5345      0.4835
    ResNet18          ReAct    far  900.0 211.2587    15.5645       4.3544          0.2711      0.0000
    ResNet18          ReAct    mid 1800.0 237.3712    15.9084       3.4132          0.2422      0.0000
    ResNet18          ReAct   near 1332.0 222.8480    23.0323      10.6154          0.1314      0.0000
       VGG13 ASH (ash_s@65)    far 4500.0 216.7473     0.0389      -0.0453          0.5769      0.0017
       VGG13 ASH (ash_s@65)    mid 9000.0 243.4389    -0.1066      -0.0495          0.5869      0.0000
       VGG13 ASH (ash_s@65)   near 6660.0 226.2216     0.1728      -0.0211          0.5371      0.8174
       VGG13          ReAct    far 4500.0 216.7473     3.5590       0.2414          0.4416      0.0000
       VGG13          ReAct    mid 9000.0 243.4389     5.8628       0.9008          0.3242      0.0000
       VGG13          ReAct   near 6660.0 226.2216    17.0861       8.5365          0.1206      0.0000
```

## Per source (nine CSFs pooled)

```
architecture            arm        source   rows     base  mean_diff  median_diff  wrapper_better  p_wilcoxon
    ResNet18 ASH (ash_s@65)       cifar10  864.0 170.2895     0.4473      -0.0198          0.6377      0.0001
    ResNet18 ASH (ash_s@65)      cifar100 1008.0 218.0946     0.0950       0.0032          0.4782      0.3403
    ResNet18 ASH (ash_s@65) supercifar100 1296.0 265.3473     0.0259      -0.0782          0.6821      0.0000
    ResNet18 ASH (ash_s@65)  tinyimagenet  864.0 235.3877     0.0591      -0.0469          0.5752      0.0020
    ResNet18          ReAct       cifar10  864.0 170.2895     5.1422       0.1151          0.3843      0.0000
    ResNet18          ReAct      cifar100 1008.0 218.0946    26.0279      10.1419          0.0238      0.0000
    ResNet18          ReAct supercifar100 1296.0 265.3473    14.5714       5.1059          0.3580      0.0000
    ResNet18          ReAct  tinyimagenet  864.0 235.3877    27.4984      14.8795          0.0405      0.0000
       VGG13 ASH (ash_s@65)       cifar10 4320.0 168.1554    -0.1168      -0.0148          0.5674      0.0000
       VGG13 ASH (ash_s@65)      cifar100 5040.0 227.7770    -0.0926      -0.0627          0.5623      0.0000
       VGG13 ASH (ash_s@65) supercifar100 6480.0 275.1397    -0.0781      -0.1640          0.5829      0.0000
       VGG13 ASH (ash_s@65)  tinyimagenet 4320.0 235.0961     0.4270      -0.0285          0.5539      0.0000
       VGG13          ReAct       cifar10 4320.0 168.1554     5.1018       0.3562          0.1831      0.0000
       VGG13          ReAct      cifar100 5040.0 227.7770     5.0598       1.3080          0.2962      0.0000
       VGG13          ReAct supercifar100 6480.0 275.1397     1.4277      -0.0151          0.5002      0.3018
       VGG13          ReAct  tinyimagenet 4320.0 235.0961    29.1160      14.7472          0.0424      0.0000
```

## Share of rows where the best wrapped variant beats the 22-CSF oracle

```
architecture  arm           
ResNet18      ASH (ash_s@65)    0.1094
              ReAct             0.0804
VGG13         ASH (ash_s@65)    0.0987
              ReAct             0.1326
```

## VGG13 top cliques, swap design: cells (of 36) with a gained variant, a lost base CSF, or a side effect

```
                gained  lost  side_effects
wrapper                                   
ASH (ash_s@65)       4     4             7
ReAct                8     8            12
```

```
       wrapper     panel       source regime                                   gained                             lost                       side_effects
ASH (ash_s@65) confidnet      cifar10    far                                                               Energy, MLS                              -NeCo
         ReAct   devries      cifar10   near                                                                                        +NCI, +NNGuide, +NeCo
         ReAct   devries      cifar10    far                                                                       MLS                           +NNGuide
ASH (ash_s@65)   devries      cifar10    far                                                                                                     +NNGuide
         ReAct        dg      cifar10   near                                                                       MLS                                   
         ReAct        dg      cifar10    far                                                                       MLS                                   
ASH (ash_s@65)        dg      cifar10    far                                                                       MLS                                   
         ReAct confidnet     cifar100   near                                                                        PE                           -NNGuide
ASH (ash_s@65) confidnet     cifar100   near                                                                        PE                                   
         ReAct confidnet     cifar100    mid                                                                                                        -NeCo
ASH (ash_s@65) confidnet     cifar100    mid                              MLS + ASH-S                                                   +MahaPP, +NNGuide
         ReAct   devries     cifar100   near                           Energy + ReAct                     MSR, PCE, PE                               -CTM
         ReAct   devries     cifar100    mid Energy + ReAct, MLS + ReAct, REN + ReAct                                                                +CTM
         ReAct        dg     cifar100   near  Energy + ReAct, GE + ReAct, MLS + ReAct GEN, GradNorm, MSR, PCE, PE, REN -CTM, -NCI, -NNGuide, -NeCo, -fDBD
         ReAct        dg     cifar100    mid              Energy + ReAct, MLS + ReAct                                               -NCI, -NNGuide, -fDBD
         ReAct confidnet   supercifar   near                  GE + ReAct, MLS + ReAct                         GradNorm                                   
ASH (ash_s@65) confidnet   supercifar   near                              MLS + ASH-S                                                                    
         ReAct confidnet   supercifar    mid               Energy + ReAct, GE + ReAct                                                                -CTM
ASH (ash_s@65) confidnet   supercifar    mid                                                                                               +KPCA RecError
         ReAct confidnet   supercifar    far               Energy + ReAct, GE + ReAct                                                                    
         ReAct   devries   supercifar   near                                                                        PE                                   
ASH (ash_s@65)   devries   supercifar   near                                                                        PE                                   
ASH (ash_s@65)   devries   supercifar    mid                                                                                                  +CTM, +NeCo
         ReAct   devries   supercifar    far                           Energy + ReAct                                                                    
ASH (ash_s@65)        dg   supercifar   near                               PE + ASH-S                                                                    
         ReAct        dg   supercifar    mid                                                                               -Maha, -NeCo, -Residual, -pNML
ASH (ash_s@65)        dg   supercifar    mid                              MLS + ASH-S                                                                +ViM
         ReAct confidnet tinyimagenet    mid                                                                                                        +fDBD
ASH (ash_s@65) confidnet tinyimagenet    mid                                                                                                        +fDBD
         ReAct confidnet tinyimagenet    far                                                                                                        +fDBD
```
