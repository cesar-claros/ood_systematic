# Holm vs unadjusted Conover cliques (step-5 granularity)

| pool      |   cells | identical_top_clique   | unadjusted_subset_of_holm   | strictly_grew   |   mean_jaccard |   min_jaccard |   mean_size_unadj |   mean_size_holm | winner_overlap_nonempty   |
|:----------|--------:|:-----------------------|:----------------------------|:----------------|---------------:|--------------:|------------------:|-----------------:|:--------------------------|
| VGG13 CNN |     224 | 35.3%                  | 100.0%                      | 64.7%           |          0.709 |          0.2  |              2.44 |             3.87 | 100.0%                    |
| ViT       |      32 | 25.0%                  | 100.0%                      | 75.0%           |          0.615 |          0.25 |              2.03 |             3.91 | 100.0%                    |
