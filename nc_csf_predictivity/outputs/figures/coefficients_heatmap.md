# Per-CSF binary logistic regression coefficients — heatmap caption

**Files:** `coefficients_heatmap_xarch.{pdf,png}`, `coefficients_heatmap_lopo.{pdf,png}`
**Generator:** `code/nc_csf_predictivity/evaluation/coefficients_heatmap.py`
**Date:** 2026-05-04

## Caption (paper-quality, ~120 words)

Coefficients of the per-CSF binary logistic-regression heads (label rule =
within_eps_rank, with-source one-hot encoding, internal 5-fold CV for
regularization). Columns: 20 CSFs split by side, with feature-side CSFs on
the left and head-side on the right. Rows: 8 standardized Papyan NC
metrics, 3 source one-hot levels (cifar10 dropped as reference), 3 regime
one-hot levels (`all` regime dropped as reference). Cell values are signed
coefficients on the standardized-feature scale; blue = positive (feature
value increases the predicted competitive probability), red = negative.
Sign reversals across CSFs on `equiangular_wc` (Energy +4.08, fDBD −3.50)
and `self_duality` (head-side mostly positive, Maha −3.11) confirm the
predictor is learning per-CSF specificity rather than a generic
"OOD-easy model" axis.

## Key patterns to call out in figure-2 prose

1. **NC sign reversals across CSFs on `equiangular_wc` and `self_duality`** —
   the predictor learns CSF identity, not a single quality axis.
   Single most striking row: `equiangular_wc` flips between Energy (+4.08),
   GE (+2.78), CTM (+3.05) (positive) and fDBD (−3.50), GradNorm (−3.32),
   Maha (−1.33), NNGuide (−1.62) (negative).

2. **`var_collapse` is uniformly faint.** NC1, the canonical NC metric,
   carries little predictive signal in this Ridge after the other 7
   metrics are accounted for. Likely a regularization-collinearity effect:
   var_collapse is correlated with several other Papyan metrics in the
   training pool, and the L2 penalty pushes its coefficient to zero in
   favor of the more discriminative metrics.

3. **CTM's column is uniformly near-zero.** Not because NC carries no
   signal in general, but because CTM's binary "competitive" label is
   broadly positive (53.1% of training cells) without any NC-distinguishable
   sub-population. The CV-tuned C drops to 0.0001 (max regularization),
   leaving the head as a constant marginal predictor (≈0.53). This is a
   correct decision by the CV — CTM is the broadly-best baseline (Always-CTM),
   and the NC predictor's value comes from identifying *which other* CSFs
   are competitive *alongside* CTM on a given model.

4. **Maha's column is the most NC-discriminative.** CV selected C = 2.78
   (essentially no regularization), and the resulting coefficients are large:
   `equinorm_wc` +1.63, `equinorm_uc` +1.22, `self_duality` −3.11,
   `max_equiangular_wc` −1.61. Mechanistically: Maha works when activations
   are equinormed (NC2-like) but the activation-to-weight alignment is
   broken (NC3-violated). When NC3 holds, the classifier weights already
   carry the same separation information and Maha's structural advantage
   disappears.

5. **`self_duality` shows the side-asymmetry.** Mostly positive on head-side
   (PE +2.16, GEN +2.19, MLS +2.01, GE +1.99, PCE +2.34, Confidence +1.39),
   negative or small on feature-side (Maha −3.11, Residual −1.75). Mechanism:
   head-side CSFs use logits = activations × W; benefit from
   activation-weight alignment. Covariance-based feature-side CSFs benefit
   when alignment is broken because that leaves them a unique signal to
   exploit.

6. **Source one-hots carry concentrated source-dataset shifts.**
   `source_supercifar100` is +2.13 for Maha (Maha thrives on the coarsened
   class structure of supercifar100). `source_tinyimagenet` is broadly
   negative on head-side (head-side competitive sets shrink under the
   200-class problem). `source_cifar100` is positive for CTM (+1.59) and
   suppresses some others.

7. **Regime one-hots track the OOD-severity gradient.** `regime_far` is
   positive for feature-side methods (Maha +1.92, fDBD +0.41) — feature-side
   methods become more relevant as severity grows. `regime_near` is positive
   for head-side methods (Confidence band +2.13) — head-side wins on near-OOD
   where the test distribution is closer to the training distribution.

## How the regularization-pattern reading works

The CV-chosen C value per CSF is itself a readout of how predictable that
CSF's competitive label is from NC:

| chosen C | interpretation | example CSFs |
|---|---|---|
| 1e-4 | NC carries no usable signal — predictor reduces to marginal | CTM (0.0001), NNGuide (0.0008) |
| 5e-3 to 5e-2 | NC carries moderate signal | most CSFs |
| 0.36 | NC carries strong signal | Confidence, PCA RecError global |
| 2.78 | NC carries very strong signal — model wants no regularization | Maha |

This regularization-as-signal-readout is itself the mechanistic story for
the headline win: the NC predictor *abstains* (constant prediction) on the
CSFs where it can't help and *learns* on the CSFs where it can. The
empty-set imputation in step 14 turns this into a legitimate always-predict
predictor that beats Always-CTM by 30–48% raw AUGRC on feature-side and
all-CSF cells.

## Suggested figure placement in the paper

If the paper uses 3 main figures + a coefficient heatmap as supporting:

- Fig 1 (hero): Mantel scatter
- Fig 2: cross-arch regret bars by side
- Fig 3: this coefficient heatmap (xarch version)
- Appendix Fig A1: lopo coefficient heatmap (averaged across 4 folds)

Or, if the paper has a "Methods + Mechanistic Interpretation" structure,
the coefficient heatmap fits naturally in the latter section as the
"what did the predictor learn" figure.
