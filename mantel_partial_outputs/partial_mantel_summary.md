# Partial Mantel: NC vs CSF rankings, controlling convergence proxies

Permutations: 9999, seed 42, Spearman flavor, residual-permutation method (Legendre).

Covariates: iid_msr / iid_mean = IID test AUGRC (MSR / mean over retained CSFs); lr = learning rate (varies only for ViT); dataset = source-identity distance. CNN strata share one training schedule per dataset by design, so schedule/epoch covariates are constants there.

| backbone   | study     |   n_blocks | control                   |       r |      p |
|:-----------|:----------|-----------:|:--------------------------|--------:|-------:|
| Conv       | confidnet |         20 | (none, replication)       |  0.6436 | 0.0001 |
| Conv       | confidnet |         20 | (none, partial-code path) |  0.6436 | 0.0001 |
| Conv       | confidnet |         20 | iid_msr                   |  0.5806 | 0.0001 |
| Conv       | confidnet |         20 | iid_mean                  |  0.4586 | 0.0001 |
| Conv       | confidnet |         20 | dataset                   |  0.3262 | 0.0004 |
| Conv       | confidnet |         20 | iid_msr+dataset           |  0.4691 | 0.0001 |
| Conv       | devries   |         20 | (none, replication)       |  0.6792 | 0.0001 |
| Conv       | devries   |         20 | (none, partial-code path) |  0.6792 | 0.0001 |
| Conv       | devries   |         20 | iid_msr                   |  0.6393 | 0.0001 |
| Conv       | devries   |         20 | iid_mean                  |  0.6199 | 0.0001 |
| Conv       | devries   |         20 | dataset                   |  0.4741 | 0.0001 |
| Conv       | devries   |         20 | iid_msr+dataset           |  0.627  | 0.0001 |
| Conv       | dg        |         20 | (none, replication)       |  0.3093 | 0.0034 |
| Conv       | dg        |         20 | (none, partial-code path) |  0.3093 | 0.0031 |
| Conv       | dg        |         20 | iid_msr                   |  0.2156 | 0.0102 |
| Conv       | dg        |         20 | iid_mean                  |  0.2395 | 0.0089 |
| Conv       | dg        |         20 | dataset                   | -0.1749 | 0.9806 |
| Conv       | dg        |         20 | iid_msr+dataset           |  0.1495 | 0.056  |
| ViT        | vit       |         20 | (none, replication)       |  0.5897 | 0.0001 |
| ViT        | vit       |         20 | (none, partial-code path) |  0.5897 | 0.0001 |
| ViT        | vit       |         20 | iid_msr                   |  0.4289 | 0.0001 |
| ViT        | vit       |         20 | iid_mean                  |  0.4249 | 0.0001 |
| ViT        | vit       |         20 | dataset                   |  0.1886 | 0.0109 |
| ViT        | vit       |         20 | iid_msr+dataset           |  0.1788 | 0.0153 |
| ViT        | vit       |         20 | iid_msr+lr                |  0.1789 | 0.0204 |
| ViT        | vit       |         20 | iid_msr+lr+dataset        | -0.0422 | 0.7239 |
