"""Pluggable clustering back-ends for ClustGDD's clustering-based condensation step.

ClustGDD partitions the node representations H (MLP outputs, dim = #classes) into
exactly n clusters; each cluster becomes one synthetic node. Every back-end here
therefore has to return a label vector with exactly n non-empty clusters.

Methods
-------
kmeans            sklearn KMeans (the original paper; n_init=10 as in sklearn 1.3.2)
minibatch_kmeans  sklearn MiniBatchKMeans (the original paper on ogbn-arxiv / reddit)
balanced_kmeans   [ours] capacity-constrained Lloyd iterations: every cluster holds at most
                  ceil((1+eps) * N / n) nodes. Motivated by Theorem 3.1 of the paper,
                  whose FID bound shrinks with the variance of the cluster sizes.
random_balanced   negative control: random partition into equal-size clusters
gmm               Gaussian mixture (diagonal covariance)
ward              agglomerative clustering, Ward linkage (O(N^2) memory, small graphs only)
spectral          spectral clustering on a kNN affinity of H (small graphs only)
"""
import math
import time

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

METHODS = ['kmeans', 'minibatch_kmeans', 'balanced_kmeans', 'random_balanced', 'gmm', 'ward', 'spectral']
QUADRATIC_METHODS = {'ward', 'spectral'}
MAX_QUADRATIC_N = 20000
LARGE_DATASETS = {'ogbn-arxiv', 'reddit'}


def default_method(dataset):
    """The clustering the original code uses for this dataset."""
    return 'minibatch_kmeans' if dataset in LARGE_DATASETS else 'kmeans'


def cluster_means(H, labels, n):
    sizes = np.bincount(labels, minlength=n).astype(np.float64)
    sums = np.zeros((n, H.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, H)
    return (sums / np.maximum(sizes, 1)[:, None]).astype(np.float32)


def ensure_nonempty(H, labels, n):
    """Give every empty cluster the point farthest from the centre of the largest cluster."""
    labels = labels.copy()
    for _ in range(n):
        sizes = np.bincount(labels, minlength=n)
        empty = np.flatnonzero(sizes == 0)
        if empty.size == 0:
            break
        big = sizes.argmax()
        members = np.flatnonzero(labels == big)
        centre = H[members].mean(0)
        far = members[np.argmax(((H[members] - centre) ** 2).sum(1))]
        labels[far] = empty[0]
    return labels


def _sq_dists(H_t, C_t):
    return ((H_t ** 2).sum(1, keepdim=True) - 2 * H_t @ C_t.T + (C_t ** 2).sum(1)).clamp_min_(0)


def _capacity_assign(D, cap):
    """Greedy capacity-constrained assignment.

    Points are processed in decreasing order of regret (2nd-best minus best distance), so the
    points that lose most by not getting their nearest centre are served first; each point takes
    its nearest centre that still has room.
    """
    N, n = D.shape
    best = D.argmin(1)
    two = np.partition(D, 1, axis=1)[:, :2] if n > 1 else np.zeros((N, 2), D.dtype)
    order = np.argsort(-(two[:, 1] - two[:, 0]), kind='stable')
    labels = np.empty(N, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    full = np.zeros(n, dtype=bool)
    for i in order:
        j = best[i]
        if full[j]:
            row = np.where(full, np.inf, D[i])
            j = int(row.argmin())
        labels[i] = j
        counts[j] += 1
        if counts[j] >= cap:
            full[j] = True
    return labels


def balanced_kmeans(H, n, init_labels, eps=0.0, max_iter=30, device='cuda'):
    """Lloyd iterations with a hard upper bound on cluster size, started from a K-Means solution."""
    N = H.shape[0]
    cap = max(math.ceil((1.0 + eps) * N / n), math.ceil(N / n))
    H_t = torch.as_tensor(H, dtype=torch.float32, device=device)
    centers = cluster_means(H, init_labels, n)
    best_labels, best_wcss, it_used = None, np.inf, 0
    labels = None
    for it in range(max_iter):
        D = _sq_dists(H_t, torch.as_tensor(centers, device=device)).cpu().numpy()
        new = _capacity_assign(D, cap)
        it_used = it + 1
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        centers = cluster_means(H, labels, n)
        wcss = ((H - centers[labels]) ** 2).sum()
        if wcss < best_wcss:
            best_wcss, best_labels = wcss, labels.copy()
    return best_labels, {'balance_cap': cap, 'balance_iters_used': it_used}


def run_clustering(H, n, method, seed, dataset=None, minibatch=1000, eps=0.0,
                   max_iter=30, device='cuda'):
    """Return (labels[N] with exactly n non-empty clusters, extra info dict)."""
    H = np.ascontiguousarray(H, dtype=np.float32)
    N = H.shape[0]
    info = {}
    if method in QUADRATIC_METHODS and N > MAX_QUADRATIC_N:
        raise ValueError(f'{method} needs O(N^2) memory; N={N} > {MAX_QUADRATIC_N}')

    if method == 'kmeans':
        labels = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(H).labels_
    elif method == 'minibatch_kmeans':
        labels = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=seed,
                                 batch_size=minibatch).fit(H).labels_
    elif method == 'balanced_kmeans':
        init, _ = run_clustering(H, n, default_method(dataset), seed, dataset, minibatch)
        labels, info = balanced_kmeans(H, n, init, eps=eps, max_iter=max_iter, device=device)
    elif method == 'random_balanced':
        rng = np.random.RandomState(seed)
        labels = np.empty(N, dtype=np.int64)
        labels[rng.permutation(N)] = np.arange(N) % n
    elif method == 'gmm':
        gmm = GaussianMixture(n_components=n, covariance_type='diag', random_state=seed,
                              reg_covar=1e-4, max_iter=100)
        labels = gmm.fit(H).predict(H)
        info['gmm_converged'] = bool(gmm.converged_)
    elif method == 'ward':
        labels = AgglomerativeClustering(n_clusters=n, linkage='ward').fit(H).labels_
    elif method == 'spectral':
        labels = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', n_neighbors=10,
                                    assign_labels='kmeans', random_state=seed).fit(H).labels_
    else:
        raise ValueError(f'unknown clustering method {method}; choose from {METHODS}')

    labels = np.asarray(labels, dtype=np.int64)
    n_empty = int((np.bincount(labels, minlength=n) == 0).sum())
    info['n_empty_repaired'] = n_empty
    if n_empty:
        labels = ensure_nonempty(H, labels, n)
    return labels, info


def frechet_distance(X, Y):
    """FID between the row distributions of X and Y (float64, scipy sqrtm)."""
    from scipy.linalg import sqrtm
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    mu1, mu2 = X.mean(0), Y.mean(0)
    s1 = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    s2 = np.cov(Y, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    covmean = sqrtm(s1 @ s2)
    covmean = covmean.real if np.iscomplexobj(covmean) else covmean
    mean_term = float(((mu1 - mu2) ** 2).sum())
    cov_term = float(np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
    return mean_term + cov_term, mean_term, cov_term


def cluster_stats(H, labels, n, y_true, nclass):
    """Quality measures of a partition, used for the analysis in the report.

    H       : representations that were clustered (N x K)
    y_true  : ground-truth labels of those N nodes (analysis only, never used for training)
    """
    H = np.asarray(H, dtype=np.float64)
    N = H.shape[0]
    sizes = np.bincount(labels, minlength=n)
    centers = cluster_means(H, labels, n).astype(np.float64)
    syn_label = centers.argmax(1)

    # majority-label purity and agreement between node label and its synthetic node's label
    conf = np.zeros((n, nclass), dtype=np.int64)
    np.add.at(conf, (labels, y_true), 1)
    purity = conf.max(1).sum() / N
    agreement = float((syn_label[labels] == y_true).mean())

    # class distribution of synthetic labels vs. of the clustered nodes
    p_true = np.bincount(y_true, minlength=nclass) / N
    p_syn = np.bincount(syn_label, minlength=nclass) / n
    fid, fid_mean, fid_cov = frechet_distance(H, centers)
    return {
        'wcss': float(((H - centers[labels]) ** 2).sum() / N),
        'size_min': int(sizes.min()), 'size_max': int(sizes.max()),
        'size_cv': float(sizes.std() / sizes.mean()),
        'thm31_bound': float(((N / n - sizes) ** 2).sum() / N ** 2),
        'purity': float(purity),
        'label_agreement': agreement,
        'syn_classes_covered': int((p_syn > 0).sum()),
        'label_dist_l1': float(np.abs(p_true - p_syn).sum()),
        'fid': fid, 'fid_mean_term': fid_mean, 'fid_cov_term': fid_cov,
    }


def timed(fn, *a, **kw):
    t = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t


def log_result(args, cluster_info, res, times, path):
    """Append one run (config + clustering stats + downstream accuracy) as a JSON line."""
    import json
    import os
    res = np.asarray(res)
    row = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool)) or v is None}
    row.update(cluster_info)
    row.update(times)
    row.update({'acc_train_mean': float(res[:, 0].mean()), 'acc_test_mean': float(res[:, 1].mean()),
                'acc_test_std': float(res[:, 1].std()), 'acc_test_runs': res[:, 1].tolist(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print('[result]', json.dumps({k: row[k] for k in ('dataset', 'reduction_rate', 'cluster_method', 'seed',
                                                      'acc_test_mean', 'acc_test_std', 'wcss', 'size_cv')}))
