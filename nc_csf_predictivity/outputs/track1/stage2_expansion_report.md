# Stage-2 source-expansion report (census + Q3 theory diagnostics)

Analysis-only; frozen closed forms, registered clamps, no recalibration. Materiality = |balanced gap| >= 0.01 (protocol amendment 5c); gap = AUGRC_Energy - AUGRC_CTM, negative = Energy-favored.

Census: 91 checkpoints, 523 cells, 0 failures, all suites complete.

## Source: breeds
- n_cells: 28
- n_material: 17
- frac_positive_material: 0.0
- median_abs_gap_balanced: 0.0111
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.02, 0.037, 0.054]
- s_dict_q: [4.8, 5.5, 6.2]
- rho_q: [1.26, 1.39, 1.54]
- median_runtime_sec: 354.9
- theory_sign_acc_material: 1.0
- n_material_pred_zero: 0
- spearman_absmargin_vs_absgap: 0.851

## Source: imagenet200
- n_cells: 15
- n_material: 0
- frac_positive_material: None
- median_abs_gap_balanced: 0.0055
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.071, 0.147, 0.215]
- s_dict_q: [13.9, 14.0, 14.1]
- rho_q: [1.13, 1.26, 1.36]
- median_runtime_sec: 278.3
- spearman_absmargin_vs_absgap: 0.411

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
- median_abs_gap_balanced: 0.0114
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.024, 0.026, 0.036]
- s_dict_q: [4.8, 5.3, 5.6]
- rho_q: [1.25, 1.33, 1.4]
- median_runtime_sec: 648.7
- theory_sign_acc_material: 1.0
- n_material_pred_zero: 0
- spearman_absmargin_vs_absgap: None

## breeds / devries
- n_cells: 4
- n_material: 1
- frac_positive_material: 0.0
- median_abs_gap_balanced: 0.0052
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.045, 0.05, 0.052]
- s_dict_q: [5.5, 6.1, 6.3]
- rho_q: [1.29, 1.45, 1.46]
- median_runtime_sec: 355.9
- theory_sign_acc_material: 1.0
- n_material_pred_zero: 0
- spearman_absmargin_vs_absgap: None

## breeds / dg
- n_cells: 20
- n_material: 13
- frac_positive_material: 0.0
- median_abs_gap_balanced: 0.0112
- frac_margin_zero: 0.0
- frac_both_above_099: 1.0
- frac_energy_material_side: 0.0
- frac_ctm_material_side: 0.0
- ga_q: [0.02, 0.035, 0.055]
- s_dict_q: [5.0, 5.5, 6.1]
- rho_q: [1.28, 1.39, 1.57]
- median_runtime_sec: 354.6
- theory_sign_acc_material: 1.0
- n_material_pred_zero: 0
- spearman_absmargin_vs_absgap: 0.853

## ImageNet-200 per-set score AUROCs (mean over 3 seeds, [min, max])

- **ssb_hard** (near); MSR 0.804 [0.803, 0.804]; MLS 0.802 [0.801, 0.802]; Energy 0.798 [0.798, 0.799]; CTM 0.794 [0.792, 0.796]; Maha 0.584 [0.576, 0.589]; fDBD 0.806 [0.805, 0.807]; gap_bal +0.0009 [-0.0002, +0.0016]
- **ninco** (near); MSR 0.863 [0.862, 0.864]; MLS 0.857 [0.856, 0.858]; Energy 0.852 [0.851, 0.853]; CTM 0.868 [0.867, 0.870]; Maha 0.655 [0.650, 0.661]; fDBD 0.880 [0.878, 0.881]; gap_bal +0.0052 [+0.0047, +0.0056]
- **textures** (far); MSR 0.884 [0.883, 0.885]; MLS 0.906 [0.904, 0.908]; Energy 0.908 [0.906, 0.910]; CTM 0.944 [0.943, 0.945]; Maha 0.792 [0.790, 0.797]; fDBD 0.929 [0.928, 0.930]; gap_bal +0.0075 [+0.0068, +0.0083]
- **inaturalist** (far); MSR 0.928 [0.925, 0.931]; MLS 0.931 [0.926, 0.937]; Energy 0.926 [0.920, 0.932]; CTM 0.959 [0.958, 0.961]; Maha 0.750 [0.741, 0.759]; fDBD 0.957 [0.956, 0.959]; gap_bal +0.0061 [+0.0055, +0.0067]
- **openimage_o** (far); MSR 0.892 [0.892, 0.893]; MLS 0.896 [0.895, 0.899]; Energy 0.892 [0.890, 0.896]; CTM 0.913 [0.912, 0.914]; Maha 0.699 [0.697, 0.700]; fDBD 0.917 [0.916, 0.919]; gap_bal +0.0052 [+0.0050, +0.0056]

ImageNet-200 ID error rates: [0.1361, 0.1354, 0.1373] (OpenOOD reports ~86% RN18 accuracy: harmonization sanity).

## Notes

- Raw-vs-balanced: sign(gap_raw) == sign(gap_balanced) on 0.996 of cells (balancing rescales, it does not flip).
- ViT inventory (recorded, EXCLUDED per protocol section 1): the release also ships ViT sweeps (svhn 105, breeds 48, wilds_animals 150, plus openset variants); admissible later only as exploratory cross-regime evidence by dated amendment.
