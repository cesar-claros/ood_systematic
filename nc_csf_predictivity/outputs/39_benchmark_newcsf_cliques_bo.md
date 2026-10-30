# 39 — Per-config benchmark cliques with MahaPP + NCI_bo

**Input:** `new_csfs_pilot_summary_long.csv`; alpha=0.05; unadjusted Conover-Friedman (paper convention); metrics AUGRC+AURC negated; top clique = first greedy layer.

Experiments: 320 parsed (280 VGG13, 40 ViT, 0 ResNet18); 0 unparsed.

## Per-config cliques (VGG-13, extended roster)

Cells: 280; top-clique size mean 2.31 (min 1, max 11).

- MahaPP in top clique in 8 per-config cells (of 280).
- NCI in top clique in 61 per-config cells (of 280).
