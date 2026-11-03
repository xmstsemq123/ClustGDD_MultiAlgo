#!/bin/bash
# Overnight queue, most important comparisons first. Resumable: rerun to continue.
cd "$(dirname "$0")"
P=/home/ess/miniconda3/envs/clustgdd/bin/python
# 1) class-aware methods on the small graphs (the rest of the small grid is done)
$P run_experiments.py --datasets cora citeseer
# 2) large graphs: baseline vs. our methods + negative control
$P run_experiments.py --datasets ogbn-arxiv flickr reddit \
   --methods baseline class_balanced_kmeans balanced_kmeans class_kmeans random_balanced
# 3) balance-tolerance ablation (eps) on the small graphs
$P run_experiments.py --datasets cora citeseer --methods balanced_kmeans class_balanced_kmeans --eps 0.1 0.3 0.5 1.0
# 4) GMM on the large graphs (slowest, last)
$P run_experiments.py --datasets ogbn-arxiv flickr reddit --methods gmm
echo ALL_DONE
