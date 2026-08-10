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

## Running stage 2 (Tier-B orientation; dev pool = calibration)

Stage 2 needs the stage-1 NPZs and reads evaluation images only (no outcome tables). Protocol constants and rules are pre-registered in `FREEZE.md` (r2-tierB). From `code/` inside the container:

```bash
# smoke test one cell, then the pool
python x6_spectral/measure_orientation.py \
    --model_path=cifar10_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2 --use_cuda
CSF_NUM_WORKERS=12 nohup bash x6_spectral/run_x6_orientation.sh > x6_orientation.log 2>&1 &

# after the sweep: score rules + batch-trial against cell-level dev deltas
python x6_spectral/score_tier_b.py
```

Outputs: `outputs/orientation/<cell>.json` (per OOD dataset: a_hat/lam_hat per draw, r2-tierB signs, batch-trial AUROCs), then `outputs/tier_b_dev_scoring.csv` + `outputs/tier_b_dev_report.md` (cell-level accuracy per operator class vs the three nulls, material-cells cut, coverage). Runtime is dominated by the one validation forward pass per checkpoint; OOD batches are 640 samples each.

## Running the held-out tier (ResNet18; frozen r3 protocol)

Strict order: measurement first (outcome-free), scoring second (opens the held-out outcome tables for the first time). From `code/` inside the container:

```bash
# 1. measurement (outcome-free): smoke-test one cell, then the 8-cell pool
python x6_spectral/measure_checkpoint.py \
    --model_path=cifar10_paper_sweep/confidnet_bbresnet18_do0_run1_rew2.2 --use_cuda
bash x6_spectral/run_x6_measure.sh x6_spectral/manifest_heldout_resnet18.txt
bash x6_spectral/run_x6_orientation.sh x6_spectral/manifest_heldout_resnet18.txt

# 2. scoring (opens held-out outcomes; only after step 1 completes)
python x6_spectral/make_projection_targets.py --pool resnet18
python x6_spectral/score_tier_b.py --pool resnet18
```

Outputs: `projection_targets_heldout_resnet18.csv` (pooled Wilcoxon table under the pinned within-slice semantics) and `outputs/tier_b_heldout_resnet18_{scoring.csv,report.md}` (cell-level trial-arm verdict; rule arms ride along as the pre-registered negative control). Single run per cell: no run averaging or checkpoint-level uncertainty, as pinned in `FREEZE.md`. The `--pool dev` paths of both scripts regenerate the frozen dev artifacts byte-identically (regression-checked), so the generalization does not touch gate-1 semantics. Tier-A one-sided claims for ResNet18, if stage 1 emits any, are checked against the pooled held-out table.

**r8 re-measurement (see FREEZE: rule version r8).** Out-of-sample ID reference: refit statistics use the second val half, ID blocks the first; targets the r7 Maha residual. To re-measure under r8: `mv x6_spectral/outputs/orientation x6_spectral/outputs/orientation_r7`, then rerun both orientation sweeps and rescore both pools (same commands as below with the new archive name).

**r7 re-measurement (see FREEZE: rule version r7).** The trial now loads the DEPLOYED PF estimators per checkpoint (exact components, tuned k, per-class projectors; no stage-1b train re-forward needed), uses batch-level AUGRC as the primary statistic (AUROC demoted to a CEILING-flagged diagnostic), and covers the r6 families (GradNorm closed-form, Maha class-pred, PCA RecError class-pred and class). Dev + ResNet18 are the calibration set; CLIP/SSL is the pristine tier. To re-measure under r7:

```bash
mv x6_spectral/outputs/orientation x6_spectral/outputs/orientation_r5   # keep the r5 record
bash x6_spectral/run_x6_orientation.sh                                  # dev pool
bash x6_spectral/run_x6_orientation.sh x6_spectral/manifest_heldout_resnet18.txt
python x6_spectral/score_tier_b.py --pool dev
python x6_spectral/score_tier_b.py --pool resnet18
```

Per-checkpoint runtime grows somewhat (per-class back-projections and three tied-covariance refits, the largest a D=2048 pinv) but stays forward-dominated. If a checkpoint logs "Deployed global PF params missing", its `pf_params` flag records the stage-1 fallback; treat those cells as reduced-fidelity.

## Freeze gates (status in `FREEZE.md`)

1. Done: Delta-baseline semantics pinned in `FREEZE.md`; `make_projection_targets.py` builds `projection_targets_dev.csv` (ConfidNet VGG13 + ViT, 120 rows) by importing the paper's own generator. One open flag: confirm the plain (not `_fix-config`) score files are the source of record.
2. Done: `FAMILY_OPERATOR` adjudicated with one-line justifications in `FREEZE.md` (fDBD re-adjudicated kept -> logit; hybrids assigned to their dominant operator).
3. Done: stage-1 measurement executed 2026-08-08 (80/80 dev cells; Tier-A rule r2 calibrated on dev measurements only).
4. Tag the code repo once the fix-config flag is resolved; only then open held-out outcome tables.
