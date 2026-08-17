# Wiring the X3 interventions into the fd-shifts fork

**STATUS 2026-08-14: WIRED.** Implemented in the local fork checkout (`fd-shifts-0.1.1/`) as a single `intervention_model` paradigm carrying all three penalty kinds plus the A2 fixed-ETF head and the negative-dose arm; see `documentation/pilot1_intervention_wiring.md` for the change list, smoke commands, and dispatch matrix. The plan below is the original design note (superseded where it differs: one paradigm module instead of three clones).

Target repo: `cesar-claros/fd-shifts-0.1.1`. Three additions, mirroring how the
`dg` paradigm carries its `rew` scalar.

1. **Loss modules.** Copy `nc_penalties.py` into `fd_shifts/loss/` (or the
   fork's equivalent). Each lightning module gains: an `EMAClassMeans` buffer
   sized to the encoder output; `update()` called on each training batch's
   penultimate features; `loss = ce + lam * <penalty>` per the paradigm map in
   `nc_penalties.PENALTIES`.
2. **Paradigm registration.** Clone the vanilla-CE lightning module three times
   (`etfreg`, `varreg`, `eqnreg`) the same way `devries`/`dg` are registered;
   `lam` rides the experiment name suffix exactly like `rew`
   (`etfreg_bbvgg13_do0_run1_lam0.3`). Hard-mode I1: swap the head for a fixed
   simplex-ETF matrix (`requires_grad=False`) plus a learnable scalar scale;
   register as `lam` value `hard`.
3. **Dispatch.** Sweep per the design doc (5 dials x 3 seeds x {CIFAR-100,
   TinyImageNet} x VGG-13, shared baselines). Analysis side needs no changes:
   `NeuralCollapseMetrics` supplies the manipulation check, `csf_pipeline`
   consumes checkpoints as-is, and `analysis_jt_tost.py` runs the pre-registered
   tests on the resulting long table joined with the dial value.

CPU verification status (`verify_numpy.py`): etfreg drives self-duality -100%
(selective), eqnreg drives equinorm -87% (selective), varreg drives the NC1
proxy -97% with moderate coupling (+66% self-duality, +43% equinorm) -- the
design's Section 5 selectivity gate adjudicates on real training; if coupling
persists, the arm is reported as a joint intervention per the design.
