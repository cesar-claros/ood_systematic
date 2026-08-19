# Pilot 2 Transport Report (scale: 1 - AUROC_f (primary))

Margin (frozen pre-errors): 0.00936; train cells 384, held-out cells 96.

| arm | held-out MAE | vs plugin |
|---|---|---|
| plugin | 0.17193 | +0.00000 |
| nuisance_pc | 0.09126 | -0.08067 |
| nc_pc | 0.07232 | -0.09961 |
| dose | 0.05361 | -0.11833 |
| cellmean | 0.05361 | -0.11833 |
| nuisance_pc_logitscale | 0.11960 | -0.05234 |

**Margin condition: FAIL** (plugin must beat nuisance_pc AND dose by > margin).
**Sign condition: FAIL** (50/84 material cells, fraction 0.595, threshold 0.75).
**Registered Pilot 2 verdict: FAIL**

## Per-pair held-out MAE

| arm | E1 | E2 | E4 |
|---|---|---|---|
| plugin | 0.47823 | 0.02446 | 0.01311 |
| nuisance_pc | 0.08655 | 0.09341 | 0.09383 |
| nc_pc | 0.10781 | 0.05437 | 0.05479 |
| dose | 0.13872 | 0.01892 | 0.00318 |
| cellmean | 0.13872 | 0.01892 | 0.00318 |
| nuisance_pc_logitscale | 0.06479 | 0.14679 | 0.14721 |

## Not-one-dataset / not-one-seed (pass condition 4)

- plugin beats nuisance_pc in 2/8 sets
- plugin beats nuisance_pc in 0/4 seeds

## On-support diagnostic (R10 / Addendum A item 4)

- held-out cells on/off support: 0/96 (threshold 1.55, held-out distance range 70.12-71.40)
- plugin MAE on-support: None, off-support: 0.17193295971323108

## Mechanism label (pass condition 3, LOSO CV)

- CV MAE without label 0.01657, with label 0.01653, gain +0.00005

## Calibration fits (2 params per arm)

- plugin: alpha -0.00445, beta +1.8361
- nuisance_pc: alpha -0.00594, beta +0.0011
- nc_pc: alpha -0.00594, beta -0.0005
- nuisance_pc_logitscale: alpha -0.00594, beta +0.0007
- dose: alpha -0.00700, beta +0.0035


---

# Pilot 2 Transport Report (scale: AUGRC (secondary))

Margin (frozen pre-errors): 0.00242; train cells 384, held-out cells 96.

| arm | held-out MAE | vs plugin |
|---|---|---|
| plugin | 0.04308 | +0.00000 |
| nuisance_pc | 0.02194 | -0.02114 |
| nc_pc | 0.01809 | -0.02499 |
| dose | 0.01247 | -0.03061 |
| cellmean | 0.01247 | -0.03061 |
| nuisance_pc_logitscale | 0.03267 | -0.01041 |

**Margin condition: FAIL** (plugin must beat nuisance_pc AND dose by > margin).
**Sign condition: FAIL** (50/84 material cells, fraction 0.595, threshold 0.75).
**Registered Pilot 2 verdict: FAIL**

## Per-pair held-out MAE

| arm | E1 | E2 | E4 |
|---|---|---|---|
| plugin | 0.12006 | 0.00595 | 0.00323 |
| nuisance_pc | 0.02017 | 0.02277 | 0.02288 |
| nc_pc | 0.02408 | 0.01504 | 0.01515 |
| dose | 0.03224 | 0.00444 | 0.00074 |
| cellmean | 0.03224 | 0.00444 | 0.00074 |
| nuisance_pc_logitscale | 0.01594 | 0.04098 | 0.04109 |

## Not-one-dataset / not-one-seed (pass condition 4)

- plugin beats nuisance_pc in 2/8 sets
- plugin beats nuisance_pc in 0/4 seeds

## On-support diagnostic (R10 / Addendum A item 4)

- held-out cells on/off support: 0/96 (threshold 1.55, held-out distance range 70.12-71.40)
- plugin MAE on-support: None, off-support: 0.043081667433931915

## Mechanism label (pass condition 3, LOSO CV)

- CV MAE without label 0.00385, with label 0.00384, gain +0.00001

## Calibration fits (2 params per arm)

- plugin: alpha -0.00120, beta +0.4539
- nuisance_pc: alpha -0.00157, beta +0.0003
- nc_pc: alpha -0.00157, beta -0.0001
- nuisance_pc_logitscale: alpha -0.00157, beta +0.0002
- dose: alpha -0.00185, beta +0.0009
