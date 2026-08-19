# Pilot 1 Registered Outcome Report (scale: 1 - AUROC_f (primary))

| endpoint | agreement on committed-material cells | pooled aligned delta (sd) | p (one-sided) | p Holm |
|---|---|---|---|---|
| E1 MLS vs Maha | 0.842 | +0.0633 (0.0171) | 0.0026 | 0.0103 |
| E2 CTM vs CTM_mean | 0.429 | -0.0037 (0.0013) | 0.9947 | 1.0000 |
| E4 Energy vs MLS | 0.188 | -0.0014 (0.0015) | 0.9232 | 1.0000 |
| E5 A1- variance inflation | median sd ratio 1.08 | - | 0.9257 | 1.0000 |

## E1: MLS vs Maha

| arm | agree/material | per-set observed delta (committed sign) |
|---|---|---|
| A1- | 4/5 | isun +0.0058(+*); lsun_cropped +0.0110(+*); lsun_resize -0.0007(+*); places365 -0.0031(+); svhn +0.0122(+*); textures -0.0018(+); ti +0.0040(+*); sncs -0.0140(-) |
| A1+ | 3/4 | isun +0.0368(+*); lsun_cropped +0.0151(+); lsun_resize +0.0332(+*); places365 +0.0161(+); svhn -0.0073(+*); textures +0.0100(-); ti +0.0415(+*); sncs +0.0000(-) |
| A1++ | 1/2 | isun +0.0313(-); lsun_cropped +0.0086(+); lsun_resize +0.0257(+); places365 +0.0262(+); svhn -0.0054(+*); textures +0.0150(-); ti +0.0385(+*); sncs +0.0107(-) |
| A2 | 8/8 | isun -0.1465(-*); lsun_cropped -0.1400(-*); lsun_resize -0.1269(-*); places365 -0.1408(-*); svhn -0.1326(-*); textures -0.0453(-*); ti -0.1224(-*); sncs -0.1779(-*) |

## E2: CTM vs CTM_mean

| arm | agree/material | per-set observed delta (committed sign) |
|---|---|---|
| A1- | 2/2 | isun +0.0050(-); lsun_cropped +0.0142(+*); lsun_resize +0.0035(-); places365 +0.0045(-); svhn +0.0075(-); textures +0.0059(+); ti +0.0040(-); sncs +0.0006(+*) |
| A1+ | 0/0 | isun -0.0022(+); lsun_cropped -0.0023(-); lsun_resize -0.0041(+); places365 -0.0015(-); svhn -0.0081(-); textures -0.0038(-); ti -0.0001(+); sncs -0.0032(-) |
| A1++ | 2/5 | isun -0.0032(+*); lsun_cropped -0.0027(-); lsun_resize -0.0051(+*); places365 -0.0002(+); svhn -0.0085(-*); textures -0.0033(+); ti -0.0013(+*); sncs -0.0036(-*) |
| A2 | 2/7 | isun +0.0160(-*); lsun_cropped +0.0360(+); lsun_resize +0.0086(-*); places365 -0.0026(-*); svhn +0.0142(-*); textures +0.0415(-*); ti +0.0113(-*); sncs -0.0199(-*) |

## E4: Energy vs MLS

| arm | agree/material | per-set observed delta (committed sign) |
|---|---|---|
| A1- | 2/2 | isun +0.0017(-); lsun_cropped +0.0009(+*); lsun_resize +0.0013(+); places365 +0.0001(-); svhn +0.0014(+*); textures +0.0004(-); ti +0.0021(-); sncs -0.0001(+) |
| A1+ | 0/4 | isun +0.0017(-*); lsun_cropped +0.0008(+); lsun_resize +0.0010(-*); places365 +0.0003(-); svhn -0.0013(+*); textures +0.0008(+); ti +0.0020(-*); sncs +0.0002(+) |
| A1++ | 0/5 | isun +0.0011(-*); lsun_cropped -0.0001(+*); lsun_resize +0.0007(-*); places365 +0.0010(-); svhn -0.0006(+*); textures +0.0006(+); ti +0.0017(-*); sncs +0.0005(+) |
| A2 | 1/5 | isun +0.0027(-*); lsun_cropped +0.0028(+); lsun_resize +0.0040(-*); places365 +0.0026(+); svhn +0.0038(+*); textures +0.0082(-*); ti +0.0041(-*); sncs +0.0017(+) |

## E3 nulls (TOST; margin = 2 x baseline seed SD)

| score | A1- | A1+ | A1++ | A2 |
|---|---|---|---|---|
| Maha | 2/8 eq | 2/8 eq | 4/8 eq | 0/8 eq |
| CTM_mean | 3/8 eq | 5/8 eq | 4/8 eq | 0/8 eq |
| PCA_RecError | 4/8 eq | 1/8 eq | 2/8 eq | 0/8 eq |
| Residual | 1/8 eq | 0/8 eq | 3/8 eq | 0/8 eq |

## Exploratory

- X-f A2 variance inflation: median sd ratio 0.94, pooled one-sided p 0.6649
- X-a fDBD-CTM (A1-): fDBD gains in 4/8 sets
- X-a fDBD-CTM (A1+): fDBD gains in 0/8 sets
- X-a fDBD-CTM (A1++): fDBD gains in 0/8 sets
- X-a fDBD-CTM (A2): fDBD gains in 6/8 sets


---

# Pilot 1 Registered Outcome Report (scale: AUGRC (secondary))

| endpoint | agreement on committed-material cells | pooled aligned delta (sd) | p (one-sided) | p Holm |
|---|---|---|---|---|
| E1 MLS vs Maha | 0.842 | +0.0147 (0.0043) | 0.0031 | 0.0124 |
| E2 CTM vs CTM_mean | 0.429 | -0.0009 (0.0003) | 0.9948 | 1.0000 |
| E4 Energy vs MLS | 0.188 | -0.0004 (0.0003) | 0.9393 | 1.0000 |
| E5 A1- variance inflation | median sd ratio 1.02 | - | 0.8132 | 1.0000 |

## E1: MLS vs Maha

| arm | agree/material | per-set observed delta (committed sign) |
|---|---|---|
| A1- | 4/5 | isun +0.0014(+*); lsun_cropped +0.0027(+*); lsun_resize -0.0002(+*); places365 -0.0008(+); svhn +0.0020(+*); textures -0.0005(+); ti +0.0009(+*); sncs -0.0034(-) |
| A1+ | 3/4 | isun +0.0090(+*); lsun_cropped +0.0037(+); lsun_resize +0.0080(+*); places365 +0.0039(+); svhn -0.0012(+*); textures +0.0025(-); ti +0.0100(+*); sncs +0.0000(-) |
| A1++ | 1/2 | isun +0.0077(-); lsun_cropped +0.0021(+); lsun_resize +0.0062(+); places365 +0.0063(+); svhn -0.0009(+*); textures +0.0037(-); ti +0.0093(+*); sncs +0.0026(-) |
| A2 | 8/8 | isun -0.0359(-*); lsun_cropped -0.0337(-*); lsun_resize -0.0305(-*); places365 -0.0339(-*); svhn -0.0217(-*); textures -0.0112(-*); ti -0.0294(-*); sncs -0.0428(-*) |

## E2: CTM vs CTM_mean

| arm | agree/material | per-set observed delta (committed sign) |
|---|---|---|
| A1- | 2/2 | isun +0.0012(-); lsun_cropped +0.0034(+*); lsun_resize +0.0008(-); places365 +0.0011(-); svhn +0.0012(-); textures +0.0015(+); ti +0.0010(-); sncs +0.0001(+*) |
| A1+ | 0/0 | isun -0.0005(+); lsun_cropped -0.0006(-); lsun_resize -0.0010(+); places365 -0.0004(-); svhn -0.0013(-); textures -0.0009(-); ti -0.0000(+); sncs -0.0008(-) |
| A1++ | 2/5 | isun -0.0008(+*); lsun_cropped -0.0006(-); lsun_resize -0.0012(+*); places365 -0.0000(+); svhn -0.0014(-*); textures -0.0008(+); ti -0.0003(+*); sncs -0.0009(-*) |
| A2 | 2/7 | isun +0.0039(-*); lsun_cropped +0.0087(+); lsun_resize +0.0021(-*); places365 -0.0006(-*); svhn +0.0023(-*); textures +0.0103(-*); ti +0.0027(-*); sncs -0.0048(-*) |

## E4: Energy vs MLS

| arm | agree/material | per-set observed delta (committed sign) |
|---|---|---|
| A1- | 2/2 | isun +0.0004(-); lsun_cropped +0.0002(+*); lsun_resize +0.0003(+); places365 +0.0000(-); svhn +0.0002(+*); textures +0.0001(-); ti +0.0005(-); sncs -0.0000(+) |
| A1+ | 0/4 | isun +0.0004(-*); lsun_cropped +0.0002(+); lsun_resize +0.0002(-*); places365 +0.0001(-); svhn -0.0002(+*); textures +0.0002(+); ti +0.0005(-*); sncs +0.0000(+) |
| A1++ | 0/5 | isun +0.0003(-*); lsun_cropped -0.0000(+*); lsun_resize +0.0002(-*); places365 +0.0002(-); svhn -0.0001(+*); textures +0.0001(+); ti +0.0004(-*); sncs +0.0001(+) |
| A2 | 1/5 | isun +0.0007(-*); lsun_cropped +0.0007(+); lsun_resize +0.0010(-*); places365 +0.0006(+); svhn +0.0006(+*); textures +0.0020(-*); ti +0.0010(-*); sncs +0.0004(+) |

## E3 nulls (TOST; margin = 2 x baseline seed SD)

| score | A1- | A1+ | A1++ | A2 |
|---|---|---|---|---|
| Maha | 2/8 eq | 2/8 eq | 4/8 eq | 0/8 eq |
| CTM_mean | 3/8 eq | 5/8 eq | 4/8 eq | 1/8 eq |
| PCA_RecError | 4/8 eq | 1/8 eq | 2/8 eq | 0/8 eq |
| Residual | 2/8 eq | 0/8 eq | 3/8 eq | 0/8 eq |

## Exploratory

- X-f A2 variance inflation: median sd ratio 1.06, pooled one-sided p 0.4872
- X-a fDBD-CTM (A1-): fDBD gains in 4/8 sets
- X-a fDBD-CTM (A1+): fDBD gains in 0/8 sets
- X-a fDBD-CTM (A1++): fDBD gains in 0/8 sets
- X-a fDBD-CTM (A2): fDBD gains in 6/8 sets
