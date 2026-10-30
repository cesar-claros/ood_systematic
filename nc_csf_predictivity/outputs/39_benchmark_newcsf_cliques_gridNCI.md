# 39 — Per-config benchmark cliques with MahaPP + NCI_gridNCI

**Input:** `new_csfs_pilot_summary_long_gridNCI.csv`; alpha=0.05; unadjusted Conover-Friedman (paper convention); metrics AUGRC+AURC negated; top clique = first greedy layer.

Experiments: 320 parsed (280 VGG13, 40 ViT, 0 ResNet18); 0 unparsed.

## Per-config cliques (VGG-13, extended roster)

Cells: 280; top-clique size mean 2.34 (min 1, max 12).

- MahaPP in top clique in 9 per-config cells (of 280).
- NCI in top clique in 60 per-config cells (of 280).
