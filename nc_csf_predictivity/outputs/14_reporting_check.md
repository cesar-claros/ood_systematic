# Step 16 — Reporting figures and tables

**Date:** 2026-05-04
**Source:** `code/nc_csf_predictivity/evaluation/report.py`

## Generated artifacts

- `figures/competitive_heatmap_xarch.pdf`  (23 KB)
- `figures/competitive_heatmap_xarch.png`  (77 KB)
- `figures/mantel_scatter.pdf`  (42 KB)
- `figures/mantel_scatter.png`  (149 KB)
- `figures/nc_feature_importance.pdf`  (19 KB)
- `figures/nc_feature_importance.png`  (96 KB)
- `figures/regret_by_side.pdf`  (23 KB)
- `figures/regret_by_side.png`  (63 KB)
- `figures/regret_table.md`  (11 KB)
- `figures/wilcoxon_summary.md`  (112 KB)

## Notes

All PDF/PNG pairs are saved at 150 dpi for PDF (vector) and PNG. `regret_table.md` and `wilcoxon_summary.md` are markdown extracts of the underlying parquet aggregates from steps 14–15. The Figure-1 candidate is `competitive_heatmap_xarch.{pdf,png}`; key support figure for the activation-vs-weight finding is `nc_feature_importance.{pdf,png}`. Mantel scatter visualizes the step-15 Mantel statistic for the xarch ResNet18 test pool.
