# Stage 2b: committed directional predictions (frozen emp plug-in)

Gap = L_A - L_B (L = 1 - AUROC); negative delta = A gains on B vs the paired baseline. Committed before any detector outcome exists (manifest Addendum A item 3).

## E1: MLS vs Maha

| arm | majority sign (material cells) | material / total | per-set signed deltas |
|---|---|---|---|
| A1- | + (A loses ground) | 5/8 | isun +0.0065*; lsun_cropped +0.0056*; lsun_resize +0.0026*; places365 +0.0000; svhn +0.0095*; textures +0.0000; ti +0.0043*; sncs -0.0000 |
| A1+ | + (A loses ground) | 4/8 | isun +0.0047*; lsun_cropped +0.0018; lsun_resize +0.0054*; places365 +0.0000; svhn +0.0062*; textures -0.0000; ti +0.0044*; sncs -0.0000 |
| A1++ | + (A loses ground) | 2/8 | isun -0.0003; lsun_cropped +0.0014; lsun_resize +0.0003; places365 +0.0000; svhn +0.0027*; textures -0.0000; ti +0.0032*; sncs -0.0001 |
| A2 | - (MLS gains) | 8/8 | isun -0.5139*; lsun_cropped -0.2012*; lsun_resize -0.5139*; places365 -0.1576*; svhn -0.3174*; textures -0.0750*; ti -0.4797*; sncs -0.4149* |

## E2: CTM_head vs CTM_mean

| arm | majority sign (material cells) | material / total | per-set signed deltas |
|---|---|---|---|
| A1- | + (A loses ground) | 2/8 | isun -0.0009; lsun_cropped +0.0044*; lsun_resize -0.0007; places365 -0.0001; svhn -0.0003; textures +0.0004; ti -0.0017; sncs +0.0051* |
| A1+ | none material | 0/8 | isun +0.0015; lsun_cropped -0.0018; lsun_resize +0.0019; places365 -0.0001; svhn -0.0016; textures -0.0000; ti +0.0003; sncs -0.0017 |
| A1++ | + (A loses ground) | 5/8 | isun +0.0046*; lsun_cropped -0.0028; lsun_resize +0.0044*; places365 +0.0001; svhn -0.0033*; textures +0.0003; ti +0.0028*; sncs -0.0077* |
| A2 | - (CTM_head gains) | 7/8 | isun -0.0185*; lsun_cropped +0.0031; lsun_resize -0.0247*; places365 -0.0018*; svhn -0.0106*; textures -0.0042*; ti -0.0175*; sncs -0.0072* |

## E4: Energy vs MLS

| arm | majority sign (material cells) | material / total | per-set signed deltas |
|---|---|---|---|
| A1- | + (A loses ground) | 2/8 | isun -0.0000; lsun_cropped +0.0018*; lsun_resize +0.0001; places365 -0.0000; svhn +0.0042*; textures -0.0000; ti -0.0000; sncs +0.0000 |
| A1+ | - (Energy gains) | 4/8 | isun -0.0004*; lsun_cropped +0.0005; lsun_resize -0.0005*; places365 -0.0000; svhn +0.0016*; textures +0.0000; ti -0.0005*; sncs +0.0000 |
| A1++ | - (Energy gains) | 5/8 | isun -0.0008*; lsun_cropped +0.0030*; lsun_resize -0.0006*; places365 -0.0000; svhn +0.0047*; textures +0.0000; ti -0.0007*; sncs +0.0001 |
| A2 | - (Energy gains) | 5/8 | isun -0.0009*; lsun_cropped +0.0011; lsun_resize -0.0010*; places365 +0.0000; svhn +0.0025*; textures -0.0007*; ti -0.0013*; sncs +0.0001 |

`*` = material (|delta| >= 2 se_gap). Unblinding of detector outcomes is permitted only after this file is committed.