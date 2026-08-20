# Pilot 2 Transport Report (scale: 1 - AUROC_f (primary))

Margin (frozen pre-errors): 0.00466; train cells 192, held-out cells 288.

| arm | held-out MAE | vs plugin |
|---|---|---|
| plugin | 0.01860 | +0.00000 |
| nuisance_pc | 0.02005 | +0.00145 |
| nc_pc | 0.02021 | +0.00161 |
| cellmean | 0.02962 | +0.01102 |
| nuisance_pc_logitscale | 0.01981 | +0.00121 |

**Margin condition: EXPAND** (plugin must beat nuisance_pc AND dose by > margin).
**Sign condition: FAIL** (86/153 material cells, fraction 0.562, threshold 0.75).
**Registered Pilot 2 verdict: EXPAND**

## Per-pair held-out MAE

| arm | E1 | E2 | E4 |
|---|---|---|---|
| plugin | 0.04197 | 0.00723 | 0.00660 |
| nuisance_pc | 0.04351 | 0.00646 | 0.01018 |
| nc_pc | 0.04325 | 0.00447 | 0.01290 |
| cellmean | 0.07741 | 0.00969 | 0.00175 |
| nuisance_pc_logitscale | 0.04340 | 0.00582 | 0.01021 |

## Not-one-dataset / not-one-seed (pass condition 4)

- plugin beats nuisance_pc in 8/8 sets
- plugin beats nuisance_pc in 4/4 seeds

## On-support diagnostic (R10 / Addendum A item 4)

- held-out cells on/off support: 207/81 (threshold 1.13, held-out distance range 0.92-1.41)
- plugin MAE on-support: 0.018351644483732117, off-support: 0.0192400193522687

## Mechanism label (pass condition 3, LOSO CV)

- CV MAE without label 0.01657, with label 0.01653, gain +0.00005

## Calibration fits (2 params per arm)

- plugin: alpha -0.00332, beta +0.4180
- nuisance_pc: alpha -0.02797, beta -0.0105
- nc_pc: alpha -0.02797, beta -0.0071
- nuisance_pc_logitscale: alpha -0.02797, beta -0.0094


---

# Pilot 2 Transport Report (scale: AUGRC (secondary))

Margin (frozen pre-errors): 0.00130; train cells 192, held-out cells 288.

| arm | held-out MAE | vs plugin |
|---|---|---|
| plugin | 0.00434 | +0.00000 |
| nuisance_pc | 0.00473 | +0.00039 |
| nc_pc | 0.00482 | +0.00047 |
| cellmean | 0.00691 | +0.00257 |
| nuisance_pc_logitscale | 0.00468 | +0.00034 |

**Margin condition: EXPAND** (plugin must beat nuisance_pc AND dose by > margin).
**Sign condition: FAIL** (88/153 material cells, fraction 0.575, threshold 0.75).
**Registered Pilot 2 verdict: EXPAND**

## Per-pair held-out MAE

| arm | E1 | E2 | E4 |
|---|---|---|---|
| plugin | 0.00978 | 0.00161 | 0.00164 |
| nuisance_pc | 0.01014 | 0.00150 | 0.00256 |
| nc_pc | 0.01009 | 0.00114 | 0.00322 |
| cellmean | 0.01807 | 0.00226 | 0.00040 |
| nuisance_pc_logitscale | 0.01012 | 0.00135 | 0.00258 |

## Not-one-dataset / not-one-seed (pass condition 4)

- plugin beats nuisance_pc in 8/8 sets
- plugin beats nuisance_pc in 4/4 seeds

## On-support diagnostic (R10 / Addendum A item 4)

- held-out cells on/off support: 207/81 (threshold 1.13, held-out distance range 0.92-1.41)
- plugin MAE on-support: 0.004330990456146167, off-support: 0.004378356647593686

## Mechanism label (pass condition 3, LOSO CV)

- CV MAE without label 0.00385, with label 0.00384, gain +0.00001

## Calibration fits (2 params per arm)

- plugin: alpha -0.00089, beta +0.0986
- nuisance_pc: alpha -0.00671, beta -0.0024
- nc_pc: alpha -0.00671, beta -0.0016
- nuisance_pc_logitscale: alpha -0.00671, beta -0.0022
