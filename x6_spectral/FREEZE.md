# X6 campaign freeze document (gates 1 and 2)

Pinned 2026-08-08, before any held-out outcome table is inspected. Rule version: r2 (adjusted-stability Tier-A gate; see `spectra_campaign_harness.py`). Freezing happens by tagging the code repo after the one open flag below is resolved; from that tag onward, rule or semantics changes require a new version number and a written note here.

## Gate 1: Delta-baseline semantics (pinned)

Executable form: `make_projection_targets.py`, which imports the paper's own generator (`projection_filtering_analysis.py`) so the semantics are identical by construction, and writes `projection_targets_dev.csv` (120 family-variant rows, 29 significant improvements).

1. Source data: `scores_risk/scores_AUGRC_MCD-False_{Conv,ViT}_{source}.csv` (plain files, MCD-False; AUGRC x1000, run-averaged rows per paradigm, dropout, reward, method).
2. Pool restriction: Conv rows filtered to `model == "confidnet"`; ViT files are single-paradigm (`modelvit`) and used whole.
3. Pairing unit: (paradigm, dropout, reward) x source x OOD evaluation column, with the per-source OOD column lists of the generator; the ID `test` column is excluded.
4. Delta = AUGRC(base) - AUGRC(variant); positive means the variant improves (AUGRC is lower-better). `delta_augrc` is the mean over paired cells; `median_diff` and `n_total` are carried for transparency.
5. Baselines per family: the raw (unprojected) method for every family except PCA RecError and KPCA RecError, which exist only under projection and are baselined at their own `global` variant.
6. Significance: two-sided Wilcoxon signed-rank on the pooled paired deltas, alpha = 0.05, no multiplicity correction (matching the paper's generator). `significant_improvement` = significant AND mean delta > 0. Held-out scoring adds null baselines and multiplicity control per X6 section 8; the dev targets deliberately mirror the paper instead.
7. Variant coverage: `global`, `class`, `class pred`, `class avg` as available per family. Tier rules cover global / class pred / class avg; the `class` variant (max-over-class reconstruction, unrouted) is scored with the class-pred rule minus the routing screen.
8. Thin support after the paradigm filter is visible in `n_total` (e.g. VGG13 Maha rows rest on 8 paired cells); rows with `n_total` < 16 should be read with that in mind.
9. `projection_targets_dev.csv` supersedes the mixed-paradigm `projection_targets.csv` for all campaign scoring; the old file stays as the historical paper-wide table.

**Flag resolved (2026-08-08): the plain files are the source of record.** The `_fix-config` variants are the hyperparameter-locked exports of `retrieve_scores.py --fix-config` (every (dropout, reward) slice retained; built for the intervention dose-response protocol and the NC-predictor Track 1, not for the projection analyses). Empirical verification: recomputing the mixed-paradigm Conv deltas from the plain files reproduces the paper's projection table to rounding on all four reference cells (Maha global +24.14 vs +24.1; PCA RecError class pred +38.64 vs +38.6; Maha class pred +18.66 vs +18.7; pNML class pred +6.73 vs +6.73), while the fix-config files give materially different values (+13.62, +28.22, +20.77, +6.82). The stale README stage-3 note claiming projection analyses consume fix-config was corrected the same day.

**Held-out ResNet18 semantics (pinned 2026-08-08, revised same day):** `scores_risk_resnet18/` contains only `_fix-config` exports and the pool is single-run (`run1` per config; confidnet/devries at rew2.2, dg at rewards 2.2-20, each at do0/do1). No regeneration is needed, for a verified reason: for ConfidNet the hyperparameter grid is the single reward 2.2, so the plain files' cross-config selection reduces to choosing a dropout slice per method and the values are unchanged; on the dev VGG13 files every ConfidNet plain row merges onto its fix-config counterpart with max absolute difference 0 (83/83 rows). Pinned protocol: score held-out ResNet18 on the existing fix-config exports restricted to `model == "confidnet"`, pairing base and variant WITHIN each (dropout, reward=2.2) slice (both slices contribute; value-identical to plain semantics per slice), Wilcoxon as in gate 1. Documented deviation from the dev pairing: the dev generator pairs only where base and variant selected the same dropout (a validation-side selection not reproducible from these exports); within-slice pairing is the value-identical replacement. Single-seed caveats: no run averaging and no checkpoint-level uncertainty on ResNet18. Protocol design used composition metadata only (model, dropout, reward row counts); no ResNet18 outcome values were read before the freeze.

**Dev-target transparency note:** because the plain files report one selected dropout per method, gate-1 pairing forms only where base and variant selected the same dropout; `n_total` therefore reflects dropout-selection agreement (e.g. VGG13 Maha n=8 means one source in agreement), not missing evaluations.

## Gate 2: FAMILY_OPERATOR adjudication (one-line justifications)

Adjudicated against the implementations in `code/src/csfs/`; the map itself lives in `spectra_campaign_harness.py`. Projection variants consume ProjectionFiltering back-projections (features) or back-projected logits, so "kept coordinates" below means the span retained by the fitted projector.

| Family | Operator | Justification (from the implementation) |
|---|---|---|
| Maha | kept | min-over-class Mahalanobis distance on back-projected features; the section 2.3 plug-in reassignment lemma applies |
| CTM | kept | max cosine between back-projected features and classifier weight rows: prototype similarity inside the kept span |
| CTMmean | kept | CTM with class-mean prototypes instead of weight rows |
| NNGuide | kept (hybrid) | kNN inner product against an energy-weighted ID feature bank, times energy; the bank proximity lives in projected feature space and carries the orientation sensitivity |
| NeCo | kept (ratio) | norm ratio of top-subspace projection to full feature: normalized-kept form, subject to the section 2.3 ratio caveats |
| PCA RecError | complement | deployed normalized residual \|h - recon\|/\|h\|; the section 2.3 inversion caveat applies for in-span shifts |
| KPCA RecError | complement | reconstruction error in kernel feature space: nonlinear complement, scoped as an analogy rather than covered by the linear theorem |
| Residual | complement | norm of the centered feature projected onto the principal complement |
| ViM | complement (hybrid) | energy minus alpha times the complement norm: the virtual logit is complement-dominant by design |
| MLS, Energy, MSR, GEN, GE, PE, PCE, REN | logit | scores of back-projected logits |
| fDBD | logit (re-adjudicated from kept) | mean normalized logit margins \|l_max - l_c\|/\|w_max - w_c\|: row-space geometry, with a scalar feature-distance normalizer |
| GradNorm | logit | last-layer gradient norm of KL to uniform: softmax-margin driven, with a feature-norm factor |
| pNML | logit | per-class regret over softmax probabilities with feature-leverage regularization |
| Confidence | logit | external confidence heads (ConfidNet/DeVries branches) |
| MahalanobisPP, NCI | out of scope | pilot families; adjudicate if and when they appear in an outcome table |

## Tier-B protocol addendum (r2-tierB, pre-registered 2026-08-08 before any orientation measurement)

Stage 2 (`measure_orientation.py` + `run_x6_orientation.sh`) reads evaluation IMAGES only, never outcome tables; dev-pool stage 2 is Tier-B calibration, held-out stage 2 runs only after the tag.

**Constants.** Batch per draw 128; 5 draws taken as disjoint consecutive blocks of the first 640 samples in deterministic loader order; ID reference = validation features (first 4000; also supplies delta_hat, the bulk variance for lam_hat, and the val-estimated class means for the NCC trial, all flagged as val-side).

**Projector convention.** Primary rank q = min(q90, k_save), where q90 is the smallest rank reaching 90 percent explained variance on the stage-1 correct-only spectrum (the pipeline's documented default threshold) and k_save = C-1+64 is what the stage-1 NPZ stores; when capped, the flag and the actually reached coverage are recorded. Secondary rank q = C-1 (class-mean scale) is recorded alongside.

**Rules (one per operator class).** Undetermined (all signs 0) when lam_hat < sqrt(D): below that even the oracle raw detector is weak. Kept: +1 iff a_hat > a*(lam_hat, q, D). Complement: +1 iff a_hat < a_flip(lam_hat, q, D). Logit: +1 iff |W P delta_hat| >= 0.8 |W delta_hat| (tolerance for estimation shrinkage of the kept response), undetermined when the displacement is classifier-invisible (|W delta_hat| below 5 percent of its maximum attainable value); row-space alignment is a diagnostic, not a rule. Variant mapping: `global` rows take the operator sign; `class` and `class pred` rows of raw-baselined families reuse the operator sign (the Tier-A class conditions hold across the dev pool); `class avg` rows are always -1 (Tier-A theory claim); PCA/KPCA RecError rows (variant-vs-global baseline) are DEFERRED to stage 2b, which requires per-class projector artifacts not extracted in stage 1.

**Trial arm (deployment-batch trial, scored alongside the rules).** Registered score set computable from stage-1 artifacts: MLS, Energy, MSR raw vs globally projected; NCC (Euclidean nearest class centroid, a labeled proxy for the deployed tied-covariance Mahalanobis); global reconstruction error in deployed-normalized and unnormalized forms (no raw counterpart). Trial signs = sign of the projected-minus-raw AUROC delta against the val reference; mapped to families {MLS, Energy, MSR, Maha(proxy)}, global variant only.

**Aggregation and scoring.** Signs aggregate by majority across draws within a checkpoint, then across the runs of a (backbone, source, dropout) group, with agreement fractions recorded. Scoring granularity is the cell: (arch, source, dropout-slice, OOD dataset, family, variant), with the cell delta rebuilt under the frozen gate-1 semantics (`score_tier_b.py`). Reported against three nulls (always +1, always -1, per-(arch, family, variant) majority), on all cells and on material cells (|delta| > 1.0 AUGRC x1000), with undetermined and deferred counts as coverage. MSR note: MSR is provably immune to common-mode logit shifts, so the shared logit rule is expected to be weakest there; per-family refinements are allowed only as a new rule version after dev calibration.

## Status

- Gate 1: done (this document + `make_projection_targets.py` + `projection_targets_dev.csv`); the fix-config flag is resolved above with empirical evidence.
- Gate 2: done (this document + the map in `spectra_campaign_harness.py`).
- Gate 3: done (stage-1 measurement executed 2026-08-08, 80/80 cells; rule r2 calibrated on dev measurements only).
- Gate 4: ready to tag. Commit, tag the code repo (suggested: `x6-freeze-r2`), and only after the tag open held-out outcome tables (ResNet18 after the plain regeneration pinned above, CLIP/SSL probes, held-out sources).
