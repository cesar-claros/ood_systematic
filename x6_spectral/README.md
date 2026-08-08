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

## Running stage 1 (Tier-A measurement; safe before the freeze)

Stage 1 reads checkpoints and datasets only, never outcome tables, so it can run before the rule freeze. Inside the campaign container (`container/Dockerfile`; `docker pull cesarclaros/systematic_ood:cuda11-7`), from the `code/` repo root on the HPC (`.env` provides `EXPERIMENT_ROOT_DIR` and `DATASET_ROOT_DIR`; the driver sources it):

```bash
# smoke test: one fast VGG cell, then one ViT cell
python x6_spectral/measure_checkpoint.py \
    --model_path=cifar10_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2 --use_cuda
python x6_spectral/measure_checkpoint.py \
    --model_path=vit/cifar100_modelvit_bbvit_lr0.01_bs128_run0_do0_rew0 --use_cuda

# full development pool (80 cells: 40 ConfidNet VGG13 + 40 ViT), resumable
CSF_NUM_WORKERS=12 nohup bash x6_spectral/run_x6_measure.sh > x6_measure.log 2>&1 &
```

Outputs land in `x6_spectral/outputs/`: `<cell>.json` (diagnostics in three arms: correct-only = implementation-faithful, all-sample, standardized robustness; Tier A predictions; alignments; runtimes) and `<cell>.npz` (means, top eigenvectors, eigenvalues, head weights: what Tier B needs without re-forwarding). The driver skips cells whose JSON exists and appends failures to `outputs/failures.log`. The forward pass dominates runtime (ViT at 384x384 over the train split); `CSF_BATCH_SIZE` overrides the batch size as in `csf_fit.py`. `--het_splits` (default 2) controls the class-heterogeneity splits; raise it for the final run on small-C sources. `outputs/` is gitignored: measurements and analysis stay on the HPC.

After the sweep, aggregate (still measurement-only, freeze-safe):

```bash
python x6_spectral/aggregate_tier_a.py            # add --arm all / all_standardized for the robustness arms
```

writes `outputs/tier_a_summary.csv` (one row per cell) and `outputs/tier_a_report.md` (grouped mean +- sd across runs: recovery margins, stability vs null, heterogeneity, dials; Tier-A prediction tallies with the one-sided no-benefit claims to score post-freeze; arm-consistency deltas; manifest coverage; flagged cells; headline ViT-vs-VGG contrasts over structurally trustworthy cells).

## Freeze gates (status in `FREEZE.md`)

1. Done: Delta-baseline semantics pinned in `FREEZE.md`; `make_projection_targets.py` builds `projection_targets_dev.csv` (ConfidNet VGG13 + ViT, 120 rows) by importing the paper's own generator. One open flag: confirm the plain (not `_fix-config`) score files are the source of record.
2. Done: `FAMILY_OPERATOR` adjudicated with one-line justifications in `FREEZE.md` (fDBD re-adjudicated kept -> logit; hybrids assigned to their dominant operator).
3. Done: stage-1 measurement executed 2026-08-08 (80/80 dev cells; Tier-A rule r2 calibrated on dev measurements only).
4. Tag the code repo once the fix-config flag is resolved; only then open held-out outcome tables.
