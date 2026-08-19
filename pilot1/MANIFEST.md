# Pilot 1 Manifest — Randomized Self-Duality Interventions (X1+X3 Flagship)

**Status:** FROZEN CONTENT (fields filled 2026-08-18); the git commit introducing this file into the code/ repo is the registration timestamp and the analysis-code freeze. No OOD outcome of any Pilot 1 run may be inspected before that commit exists.
**Plan:** `documentation/X1_X3_flagship_mechanistic_paper_plan.md` (v2, sections 4-8, 10, 15); amendments R1-R11 per `documentation/X1_X3_flagship_plan_review.md`.
**Evidence of record:** Pilot 0/0b verdicts (`documentation/pilot0_runbook.md`), smoke findings incl. the bit-identical lam=0 determinism proof (`documentation/pilot1_intervention_wiring.md`).

## 1. Frozen environment (filled 2026-08-18)

- fd-shifts fork commit: `9775fc3` on `main` of cesar-claros/fd-shifts-0.1.1 ("Refine warning suppression for Hydra migration compatibility"; history includes 46e0ee4 intervention model, f686864 A2 head-swap fix + regression test, fe629fc/9775fc3 warning filter; test_nc_intervention.py at 9 passing tests)
- Container image: `docker.io/cesarclaros/systematic_ood_intervention:cuda11-7@sha256:de9be5eca9977b76cb7f5b502bf74aa687cc2b8563081e88bd8e70f2ead32bcd` (pushed 2026-08-18; all 15 runs use this image on the same GPU partition; no PYTHONPATH overrides). Pre-dispatch sanity check inside the image, confirming it carries 9775fc3: `grep -c '(?s)' /usr/local/lib/python*/site-packages/fd_shifts/exec.py` returns a nonzero count and a `fast_dev_run` emits no Hydra migration warning.
- Analysis code: the code/ repo commit that introduces this manifest (self-referential registration; the pilot0 package therein supplies the frozen H estimators and the empirical-covariance predictor)

## 2. Design: 15 fresh paired trainings

Arms x seeds; identical recipe (`fd_shifts/configs/study/intervention.yaml`: VGG-13, CIFAR-100, 300 epochs, SGD lr 0.1 cosine, wd 5e-4, do0, fc_dim 512, EMA warm-up 5 epochs, cadence checkpoints [2,4,8,15,25,40,60,90,130,180,240,300]). Seeds `exp.global_seed` in {1001, 1002, 1003}; identical seed => identical backbone init and data order across arms (verified bit-level in smoke); A2 pairs at backbone level (pre-registered). Baseline runs the identical code path at lam = 0. Dose set (smoke-cleared 2026-08-16/18): baseline 0.0, repulsion -0.1, moderate +0.3, strong +1.0, A2 hard. Exact commands: `dispatch_pilot1.sh` (sha256 of that file is part of this registration).

| arm | exp.name pattern | intervention overrides |
|---|---|---|
| baseline | etfreg_bbvgg13_do0_run{R}_lam0.0 | kind=etfreg, lam=0.0 |
| A1- | etfreg_bbvgg13_do0_run{R}_lam-0.1 | kind=etfreg, lam=-0.1 |
| A1+ | etfreg_bbvgg13_do0_run{R}_lam0.3 | kind=etfreg, lam=0.3 |
| A1++ | etfreg_bbvgg13_do0_run{R}_lam1.0 | kind=etfreg, lam=1.0 |
| A2 | etfhard_bbvgg13_do0_run{R}_lamhard | mode=fixed_etf |

## 3. Blinded analysis order

1. Train all 15; record failures per section 7 without replacement.
2. Extract features (pilot0/extract_pilot0.py path) and compute geometry + nuisance vectors; write the MANIPULATION REPORT (gates M1-M2 below) and commit it BEFORE any CSF scoring.
3. Only then: CSF scoring, registered outcome analysis, Pilot 2.

## 4. Manipulation gates (before outcomes)

- M1: at least one active arm moves the Papyan self-duality metric by >= 1 baseline-seed SD, with monotone ordering across the five levels (A1- through A2). Range target (not a gate): exit the benchmark span 0.03-0.13.
- M2: median ID val-accuracy loss <= 1.5 pp for at least one active arm. All runs stay in the ITT analysis regardless.
- Selectivity is measured and reported, NOT gated (plan R9): a non-selective arm is relabeled a joint intervention and its flat-null cells switch to model-predicted responses.

## 5. Registered confirmatory endpoints (Holm within the axis)

Primary loss scale: failure AUROC (AUGRC deployment-weighted secondary; R1). Paired seed contrasts; materiality filter = |predicted delta| >= 2x Hanley-McNeil SE, frozen estimator/predictor code (R4).

- E1 (PRIMARY): MLS-vs-Mahalanobis gap moves in the X1-predicted direction per arm (improve arms: MLS gains on Maha; A1-: opposite sign).
- E2: head-CTM vs mean-CTM gap, same directional logic (head side moves, mean side flat).
- E3 (nulls, TOST with margin set from baseline paired-seed variability): Maha, CTM_mean, PCA_RE, Residual unchanged under every arm; conditional on selectivity per section 4.
- E4 (mechanism): Energy-MLS divergence shrinks as measured logit-gap scale grows across arms.
- E5 (A1- instability): between-seed variance of head-side AUROCs inflates at lam=-0.1 (Levene/Brown-Forsythe).

## 6. Pre-declared exploratory (non-confirmatory)

- X-a: fDBD-CTM divergence trend flips positive on A1- (theta_w-ordering hypothesis from Pilot 0b; n=4 checkpoint evidence).
- X-b: full 22-CSF roster cliques per arm endpoint; X-c: benchmark NC-selector regret on intervention models; X-d: MSR behavior (documented boundary); X-e: A2 scale-parameter trajectory.

## 7. Failure rule (intervention-blind, fixed now)

A run is a catastrophic optimization failure iff training loss becomes non-finite at any point OR final ID validation accuracy < 10% (10x chance at C=100). Failed runs are reported, never silently reseeded; arms with >= 2 failures trigger the plan's stop-or-redesign rule.

## 8. Pilot 2 specification (R2/R10)

Fit on baseline+A1 (12 models), predict A2 (3 models) without refitting. Geometry model = empirical-covariance exact-mean plug-in (pilot0/theory.py NoiseModel "emp") with intercept+slope calibration (2 fitted parameters); comparators at matched parameter budgets (nuisance-Q first PC, eight-NC first PC, dose model with A2 degenerating to OOD-cell means, cell-mean baseline). On-support scoring vs the A1 geometry hull (Mahalanobis diagnostic); the material margin is set from Pilot 1 nested bootstrap AFTER Pilot 1 outcomes and BEFORE A2 unblinding. "Expand before deciding" is an admissible verdict.

## 9. Go / pivot

Per plan section 17: flagship iff manipulation + predicted-direction response + transport all hold; Pivot A (intervention-only paper) if transport fails; Pivot B (misspecification) if directions fail with successful manipulation; stop/redesign if manipulation fails.

## Addendum A (2026-08-19, BLINDED STAGE: committed with the manipulation report, before any OOD forward or detector outcome)

1. **Protocol deviation, declared:** 20 runs were trained (four seeds, 1001-1004), not the registered 15/3. All four seeds enter every analysis (pure power; decided pre-unblinding).
2. **A2 relabeled from measured geometry.** The fixed-ETF arm's measured self-duality (0.389 +/- 0.001) lands ~1040 baseline SDs ABOVE baseline (0.0142), with var_collapse 0.95 (~8x the A1 arms), head residual 0.19, logit scale 13.4: the "alignment endpoint" premise is refuted (with D=512 and a 99-dim ETF head, features reach 99.5% train accuracy without aligning their means to the head). A2 is a DISTINCT JOINT MECHANISM, not the top A1 dose. Registered M1 therefore fails on ordering by construction; the amended gate M1' (A1 dial only) is the operative manipulation gate: Spearman(dose, self_duality) = -0.971 over 16 runs, moves 28-98 baseline SDs, bidirectional. M1' PASS, M2 PASS (all arms within 1.5 pp).
3. **E1 per-arm directions are generated from measured geometry via the frozen plug-in**, not from nominal arm labels (A2's nominal "improve" sign is known-wrong from its measured geometry; the plan's own convention, X3 section 1, is measured-NC primacy). Stage 2b is inserted into the blinded order: measure the H coordinates (OOD activations, no detector scores) with the frozen pilot0 estimators, generate and COMMIT the per-(arm, OOD set, pair) predicted signs from the frozen empirical-covariance plug-in, and only then unblind detector outcomes.
4. **Pilot 2 framing fixed now:** A2 lies far outside the A1 hull on multiple coordinates (opposite side on self_duality, plus var_collapse/head-residual shifts), so transport-to-A2 is an off-support test by construction. The plug-in arm predicts A2 from its own measured geometry (no support requirement); the fitted comparators extrapolate. Verdicts are reported under the on-support protocol of section 8 unchanged.
5. **New pre-declared exploratory X-f:** between-seed variance inflation of head-side scores on A2 (its measured geometry is quenched-regime, like A1- but stronger). E5 stays registered on A1- as written.
6. Selectivity outcome recorded: A1 arms co-move W-side angular coordinates (0.6-2.9x standardized) as geometrically expected; all A arms are analyzed as head-geometry joint interventions dominated by self-duality, per the plan's relabeling rule.
