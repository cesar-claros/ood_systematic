# Theorem / code ledger (plan v3, carried v2 section 6.6)

| Result | Assumptions | Class | Implementation | Independent check |
|---|---|---|---|---|
| AUGRC = pi^2/2 + pi(1-pi)(1 - A^f); Delta^G = pi(1-pi) Delta^F | common examples, residuals, half-credit ties | exact identity (established, Traub et al.) | `set_outcomes` + analysis bridge test | float64 identity to 1e-10 (test 1) |
| NC1 low-rank form sum_i u_i' Sigma_W u_i / s_i^2 with cutoff sqrt(1e-6) s_max | Sigma_B = MM'/C | exact identity | `papyan_metrics` (direct pinv) | phase-1 bootstrap arithmetic asserted equal at unit weights on 96 checkpoints |
| s_dict = (C-1)/sqrt(C NC1) = R/sigma | isotropic equal-radius exact ETF only | exact identity under stated model | `record_params` | ETF-EXACT-v2 `check()` on 5 grid points |
| ETF-EXACT-v2 constructor identities (rank, cosines, alignment a and -a/(C-1), L_par, L_base = t R, self-duality 2(1-cos theta), complement fraction sin^2 theta) | D >= 2C+1, a in [0,1], theta in [0, pi/2) | exact identities | `etf_constructor.construct` | `measure()` independent of the constructor's algebra |
| Historical decoder realizes requested coordinates | none | approximation with documented discrepancies | `build_config_model` (unchanged) | this round-trip report: selected-class vs max alignment, own-class rotation, L cos(theta) scale, eta draws |
| Taylor moments m_f + tr(H Sigma)/2, grad' Sigma grad + tr(H Sigma H Sigma)/2 | Gaussian component, smooth fixed branch | exact for the quadratic surrogate, approximation for the score | `gaussian_diagnostics.taylor_moments` | finite-difference gradients/Hessians; MC agreement (test 5, 6) |
| Energy AUROC via binormal mixture of Taylor moments | fixed softmax branch, Gaussian components | approximation (curvature + distribution) | `A0/A1-TAYLOR` | G0/G1-MC discrepancy reported separately from observed |
| Max-cosine (CTM) AUROC via fixed-branch Taylor | no branch switching, r bounded away from 0 | approximation (branch switching + norm fluctuation) | `A0/A1-TAYLOR`, undefined at ties | switching rate and norm concentration diagnostics reported |
| Fixed-index MLS argmax bound | ID and OOD switching probabilities | bound (coupling), NOT an exact error rate | manuscript statement | qualification stated in text |
| Mahalanobis normal-CDF expression | fixed-prototype model | chi-square-difference approximation | manuscript statement | labeled approximate |
| Delete-family jackknife SE and t reference | approximately independent families | approximate inference candidate | `icml_campaign_analysis`-style jackknife | qualified only in the enumerated simulations (phase 4) |
| Partial conjunction 2 p_(3) | valid marginal p-values | exact under stated validity | retired with DIST | n/a |

Repair record 2026-09-09: the previous closing sentence of this ledger asserted that every approximate row was written as approximate in the manuscript; that was false (the union bound was called the ID error rate, the Energy and Mahalanobis AUROCs were written as equalities, and Theorem 1(i) was stated for every profile with a given maximum alignment). After the 2026-09-09 repair the manuscript states the fixed-index surrogate and its switching bound on the canonical single-alignment profile, records the tied-profile counterexample (exact MLS AUROC 0.290 at gamma a = 1 for C = 3, a = 1/2, gamma = 2, s = 4), writes Energy and Mahalanobis as approximations, and calls the union bound a bound. Verified by `tests/test_thm_norm_repair_20260909.py`.

| MLS chance crossing (Theorem 1(i)) | canonical single-alignment profile; fixed-index surrogate | exact for the surrogate; coupling bound p_ID + p_OOD for the exact score; FALSE for tied profiles | manuscript theorems.tex after repair | counterexample and bound reproduced in tests/test_thm_norm_repair_20260909.py |
