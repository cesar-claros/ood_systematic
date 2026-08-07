# X6 real-spectra campaign (deployment-facing modules)

Campaign code for the X6 projection-filtering study. Theory, verification appendix, and protocol: `documentation/X6_rmt_projection_filtering.md` (section 8). The synthetic verification scripts (pass2-pass4) stay with the notes in `documentation/x6_spectral_scripts/`; this directory holds only what the measurement campaign runs.

## Contents

- `spectral_diagnostics.py` — ID-side estimators: MP-median-corrected bulk with deflation, outlier-map spike inversion, per-class within-class censuses (valid for D > N_c), split-half projector stability (with its k/D null), per-spike alignments, class-subspace heterogeneity (split-sample, debiased), common-mode and weight dials. Census default is centered covariance (implementation-faithful: the pipeline's `TorchStandardScaler` only centers); `standardize=True` is a robustness arm only. Self-test: `../.venv/bin/python spectral_diagnostics.py`.
- `spectra_campaign_harness.py` — two-tier prediction protocol. Tier A (ID-only): recovery/stability statements plus the one-sided falsifiable claim "not recoverable implies no significant benefit"; benefit signs are never emitted from ID data (orientation identifiability, X6 section 2.5). Tier B (deployment-batch adaptation, explicitly OOD-side): per-operator crossing rules from a small unlabeled batch (kept-space `a_hat > a*`, complement `a_hat < a_flip`, logit by row-space alignment). Self-test: `../.venv/bin/python spectra_campaign_harness.py`.
- `projection_targets.csv` — development-tier Delta-AUGRC targets per (arch, base_csf, variant). Currently aggregated over ALL VGG paradigms; must be regenerated for the restricted pool (gate 1 below) before calibration.

## Campaign pool (decided 2026-08-07)

- Development tier (calibrate and freeze rules; never counts as validation): **ConfidNet VGG13** checkpoints and **all ViT** checkpoints, across CIFAR-10/100, SuperCIFAR-100, TinyImageNet (all runs, dropout, reward settings).
- Excluded from the campaign: DeVries and DG VGG pools (nonstandard heads: DG carries extra logits, DeVries a confidence branch). They remain available as an optional secondary held-out set, since frozen rules will not have seen them.
- Held-out tier (validation only; do not inspect outcome tables before the freeze): ResNet18 checkpoints (`../scores_risk_resnet18`), CLIP/SSL probe pools (`../clip_scores`, extraction machinery in `../x8_pool_a`), held-out source datasets.

## Kickoff gates (before freezing rules)

1. Regenerate `projection_targets.csv` restricted to ConfidNet VGG13 + ViT cells (via `retrieve_scores.py` + `stats_eval.py` with a paradigm filter), and pin the Delta-baseline semantics per CSF family in writing.
2. Adjudicate the ambiguous entries of `FAMILY_OPERATOR` in the harness (kept vs complement vs logit) with one-line justifications.
3. Write the HPC extraction script (activations, labels, head weights, ID val accuracy per checkpoint; effective N after correct-only filtering), following the `x8_pool_a/extract_features.py` pattern.
4. Freeze this directory (tag the code repo) before any held-out outcome table is opened.
