# Stage-2 source-expansion report (census + Q3 theory diagnostics)

Analysis-only; frozen closed forms, registered clamps, no recalibration. Materiality = |balanced gap| >= 0.01 (protocol amendment 5c); gap = AUGRC_Energy - AUGRC_CTM, negative = Energy-favored.

Census: 88 checkpoints, 508 cells, 0 failures, all suites complete.

## Source: breeds
- n_cells: 28
- n_material: 18
- frac_positive_material: 0.0
- median_abs_gap_balanced: 0.0116
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.02, 0.037, 0.054]
- s_dict_q: [4.8, 5.5, 6.1]
- rho_q: [1.26, 1.38, 1.54]
- median_runtime_sec: 354.9
- theory_sign_acc_material: 1.0
- n_material_pred_zero: 0
- spearman_absmargin_vs_absgap: 0.869

## Source: svhn
- n_cells: 480
- n_material: 0
- frac_positive_material: None
- median_abs_gap_balanced: 0.0043
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.026, 0.064, 0.126]
- s_dict_q: [7.1, 7.6, 8.6]
- rho_q: [0.81, 1.02, 1.73]
- median_runtime_sec: 28.8
- spearman_absmargin_vs_absgap: 0.468

## breeds / confidnet
- n_cells: 4
- n_material: 3
- frac_positive_material: 0.0
- median_abs_gap_balanced: 0.0121
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.023, 0.026, 0.037]
- s_dict_q: [4.8, 5.3, 5.5]
- rho_q: [1.25, 1.33, 1.4]
- median_runtime_sec: 647.0
- theory_sign_acc_material: 1.0
- n_material_pred_zero: 0
- spearman_absmargin_vs_absgap: None

## breeds / devries
- n_cells: 4
- n_material: 1
- frac_positive_material: 0.0
- median_abs_gap_balanced: 0.0056
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.045, 0.049, 0.051]
- s_dict_q: [5.5, 6.1, 6.3]
- rho_q: [1.29, 1.45, 1.46]
- median_runtime_sec: 355.9
- theory_sign_acc_material: 1.0
- n_material_pred_zero: 0
- spearman_absmargin_vs_absgap: None

## breeds / dg
- n_cells: 20
- n_material: 14
- frac_positive_material: 0.0
- median_abs_gap_balanced: 0.0119
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.02, 0.035, 0.055]
- s_dict_q: [4.9, 5.5, 6.1]
- rho_q: [1.28, 1.38, 1.57]
- median_runtime_sec: 354.4
- theory_sign_acc_material: 1.0
- n_material_pred_zero: 0
- spearman_absmargin_vs_absgap: 0.863

## Notes

- Raw-vs-balanced: sign(gap_raw) == sign(gap_balanced) on 0.996 of cells (balancing rescales, it does not flip).
- ViT inventory (recorded, EXCLUDED per protocol section 1): the release also ships ViT sweeps (svhn 105, breeds 48, wilds_animals 150, plus openset variants); admissible later only as exploratory cross-regime evidence by dated amendment.
