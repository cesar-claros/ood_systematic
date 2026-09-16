# Headroom decomposition and regime-free executable policies (descriptive, 2026-09-15)

Global fixed rule chosen on all VGG13 OOD rows: CTM 

## Global CTM regret by regime

- ResNet18: regret near/mid/far 2.07 / 3.52 / 2.4; oracle level 213.1 / 226.1 / 200.9
- ViT: regret near/mid/far 8.13 / 17.86 / 11.42; oracle level 167.7 / 184.8 / 141.5
- probes (harmonized): regret near/mid/far 10.03 / 10.18 / 13.37; oracle level 161.8 / 177.4 / 137.6

## Decomposition (all OOD rows pooled; checkpoint = one trained model)

- ResNet18: rows 448, detectors 20; total 2.790 = fixed-switch (CTM) 0.000 + checkpoint-specific 1.555 + shift-specific 1.235; one-detector-per-checkpoint ceiling over CTM 1.555
- ViT: rows 320, detectors 19; total 13.109 = fixed-switch (KPCA RecError global) 5.606 + checkpoint-specific 3.042 + shift-specific 4.461; one-detector-per-checkpoint ceiling over CTM 8.648
- probes: rows 320, detectors 21; total 10.826 = fixed-switch (ViM) 5.167 + checkpoint-specific 2.766 + shift-specific 2.894; one-detector-per-checkpoint ceiling over CTM 7.932

Illustrative AUROC-equivalent (Traub et al. 2024, Eq. 7, at failure fraction 0.5): one displayed unit = 0.004 AUROC.

## Regime-free exploratory ridge loss policy (one head set on all VGG13 OOD rows; one detector per checkpoint; alpha = 1)

- ResNet18, train_only: pooled 6.18 (near/mid/far 5.57 / 7.18 / 5.11); CTM pooled 2.79
- ResNet18, target_pool: pooled 16.10 (near/mid/far 14.9 / 18.22 / 13.64); CTM pooled 2.79
- ResNet18, per-paradigm metadata rule (best VGG13 detector of the same paradigm, CTM if none): pooled 2.70
- ViT, train_only: pooled 34.22 (near/mid/far 54.68 / 25.48 / 19.53); CTM pooled 13.11
- ViT, target_pool: pooled 14.37 (near/mid/far 10.27 / 19.91 / 9.73); CTM pooled 13.11
- ViT, per-paradigm metadata rule (best VGG13 detector of the same paradigm, CTM if none): pooled 13.11

## Per-checkpoint abstention on ResNet18 (one decision per checkpoint; fallback CTM; thresholds not chosen on training-side data)

          config  tau  coverage_checkpoints  policy_pooled  CTM_pooled
none_nr_marginal  0.5                  1.00           7.27        2.79
none_nr_marginal  0.6                  0.96           6.69        2.79
none_nr_marginal  0.7                  0.84           5.01        2.79
none_nr_marginal  0.8                  0.82           4.98        2.79
none_nr_marginal  0.9                  0.75           5.09        2.79 

            config  tau  coverage_checkpoints  policy_pooled  CTM_pooled
source_nr_marginal  0.5                  1.00           7.25        2.79
source_nr_marginal  0.6                  0.93           7.26        2.79
source_nr_marginal  0.7                  0.91           7.25        2.79
source_nr_marginal  0.8                  0.77           7.25        2.79
source_nr_marginal  0.9                  0.46           4.30        2.79 

