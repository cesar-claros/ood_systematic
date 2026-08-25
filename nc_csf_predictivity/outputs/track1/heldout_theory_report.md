# Held-out theory-to-real validation (Stage 2, frozen design)

Coords loaded for 280/280 checkpoints; problems: none. Materiality |gap| >= 10.0 (AUGRC x 1000). Theory arm has no fitted parameters.

## Checkpoint-held-out (grouped 5-fold)

cells 2240, material 718, theory coverage 1.000

| arm | sign acc (material) | balanced acc |
|---|---|---|
| theory | 0.099 | 0.104 |
| severity | 0.684 | 0.608 |
| geometry | 0.774 | 0.737 |
| flexible | 0.776 | 0.738 |
| mean | 0.600 | 0.500 |
| src_id | 0.784 | 0.741 |

theory - severity: -0.585, cluster CI95 [-0.6626694093429475, -0.5068934538638667]

## Leave-one-source-out

cells 2240, material 718, theory coverage 1.000

| arm | sign acc (material) | balanced acc |
|---|---|---|
| theory | 0.099 | 0.104 |
| severity | 0.607 | 0.544 |
| geometry | 0.703 | 0.663 |
| flexible | 0.649 | 0.594 |
| mean | 0.600 | 0.500 |

theory - severity: -0.508, cluster CI95 [-0.5750043103448277, -0.44182242147844686]

| held-out source | theory | severity | n material |
|---|---|---|---|
| cifar10 | 0.065 | 0.259 | 170 |
| cifar100 | 0.022 | 0.989 | 90 |
| supercifar100 | 0.182 | 0.682 | 314 |
| tinyimagenet | 0.007 | 0.618 | 144 |

## Gates (claim contract)

- Gate 1 (theory beats severity-only, clustered CI > 0): FAIL (point -0.585, CI [-0.6626694093429475, -0.5068934538638667])
- Gate 2 (improvement not carried by one source): FAIL (theory>severity on [])
- Gate 3 (strata handoff ordering retained held-out): report the per-tertile first-handoff comparison from the crossing pipeline on held-out folds (descriptive; see report)
- Gate 4 (nothing tuned on evaluation folds): PASS by construction (theory arm has no fitted parameters; baselines fold-fitted)
- Mode indication: ORGANIZATIONAL (theory does not beat empirical baselines held-out) or THEORY-FIRST per contract
## Diagnosis appendix (2026-08-25, 400-cell sample, seed 3)

The theory arm's 0.099 is saturation, not inversion: 92-94% of cells have both predicted AUROCs > 0.99 (median 1.0000), 93% of predicted winner margins are < 0.01 and 82% are exactly zero (sign(0) scores as incorrect against either outcome; flipped-sign accuracy is 0.056, ruling out a sign-convention error). Among cells with a nonzero predicted sign, Spearman(predicted gap, observed gap) = +0.26. Mechanism: measured population-mean coordinates (gamma a in 0.13-0.42 across all sets, with dictionary SNR 23-120) place every benchmark cell in the phase map's saturated corner, while empirical performance is far from ceiling; the population-mean convention understates per-sample difficulty (direction-mixing shrinkage, documented at pilot0). Consequence per the frozen gates: Gate 1 and Gate 2 FAIL; mode = ORGANIZATIONAL (ordering robust, first handoff visible, frozen formulas at population coordinates do not predict per-cell winners). Positive held-out finding: the geometry-only regression beats severity-only in both fold modes (0.774 vs 0.684 checkpoint-held-out; 0.703 vs 0.607 source-held-out), and unlike the source-identity baseline it remains applicable to unseen sources.
