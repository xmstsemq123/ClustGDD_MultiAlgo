# Spearman correlation with Δ accuracy vs. the same-seed K-Means run

Statistics are relative to the same-seed baseline (rel_* = ratio, d_* = difference), because H differs between seeds. Runs pooled over methods and seeds.

| dataset   | ratio   | without_random   |   n |   rel_wcss |   d_size_cv |   rel_fid |   d_purity |
|:----------|:--------|:-----------------|----:|-----------:|------------:|----------:|-----------:|
| Cora      | 1.3%    | False            |  21 |      -0.47 |        0.36 |     -0.41 |       0.32 |
| Cora      | 2.6%    | False            |  21 |      -0.17 |        0.24 |     -0.65 |       0.46 |
| Cora      | 5.2%    | False            |  21 |      -0.43 |        0.44 |     -0.36 |       0.65 |
| Cora      | 1.3%    | True             |  18 |      -0.17 |        0.05 |     -0.07 |      -0.08 |
| Cora      | 2.6%    | True             |  18 |       0.32 |       -0.17 |     -0.44 |       0.14 |
| Cora      | 5.2%    | True             |  18 |      -0.1  |        0.11 |      0.01 |       0.44 |
| Citeseer  | 0.9%    | False            |  21 |      -0.36 |        0.19 |     -0.67 |       0.73 |
| Citeseer  | 1.8%    | False            |  21 |      -0.24 |        0.39 |     -0.38 |       0.27 |
| Citeseer  | 3.6%    | False            |  21 |      -0.51 |        0.34 |     -0.32 |       0.7  |
| Citeseer  | 0.9%    | True             |  18 |       0.02 |       -0.24 |     -0.47 |       0.56 |
| Citeseer  | 1.8%    | True             |  18 |       0.2  |        0.04 |     -0.03 |      -0.15 |
| Citeseer  | 3.6%    | True             |  18 |      -0.22 |       -0.05 |      0.08 |       0.53 |
| arXiv     | 0.05%   | False            |  15 |      -0.45 |        0.87 |     -0.51 |       0.94 |
| arXiv     | 0.25%   | False            |  15 |      -0.23 |        0.56 |     -0.81 |       0.87 |
| arXiv     | 0.5%    | False            |  15 |      -0.32 |        0.19 |     -0.69 |       0.78 |
| arXiv     | 0.05%   | True             |  12 |       0.06 |        0.83 |     -0.24 |       0.89 |
| arXiv     | 0.25%   | True             |  12 |       0.48 |        0.14 |     -0.64 |       0.75 |
| arXiv     | 0.5%    | True             |  12 |       0.31 |       -0.56 |     -0.43 |       0.57 |
| Flickr    | 0.1%    | False            |  15 |      -0.5  |        0.14 |     -0.21 |       0.4  |
| Flickr    | 0.5%    | False            |  15 |      -0.35 |        0.19 |     -0.29 |       0.19 |
| Flickr    | 1%      | False            |  15 |      -0.75 |        0.61 |      0.15 |       0.1  |
| Flickr    | 0.1%    | True             |  12 |      -0.03 |       -0.4  |     -0.12 |      -0.15 |
| Flickr    | 0.5%    | True             |  12 |       0.25 |       -0.57 |     -0.15 |      -0.56 |
| Flickr    | 1%      | True             |  12 |      -0.52 |        0.24 |      0.53 |      -0.73 |
| Reddit    | 0.05%   | False            |  15 |      -0.9  |        0.44 |     -0.7  |       0.98 |
| Reddit    | 0.1%    | False            |  15 |      -0.93 |        0.32 |     -0.72 |       0.86 |
| Reddit    | 0.5%    | False            |  15 |      -0.44 |        0.5  |     -0.35 |       0.72 |
| Reddit    | 0.05%   | True             |  12 |      -0.82 |        0.06 |     -0.43 |       0.97 |
| Reddit    | 0.1%    | True             |  12 |      -0.87 |       -0.06 |     -0.47 |       0.74 |
| Reddit    | 0.5%    | True             |  12 |       0.08 |        0.09 |     -0.04 |       0.46 |