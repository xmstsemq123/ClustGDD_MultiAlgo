"""Run the clustering-method comparison grid with the authors' hyper-parameters.

Each (dataset, ratio) uses exactly the settings of main_transduct.sh / main_induct.sh; only the
clustering method (and seed / balance_eps) changes. Finished runs are read back from the
results file and skipped, so the script can be stopped and restarted.

Examples
  python run_experiments.py --datasets cora citeseer --methods kmeans balanced_kmeans --seeds 15 16 17
  python run_experiments.py --datasets cora --methods balanced_kmeans --eps 0 0.1 0.3 0.5 1.0
"""
import argparse
import json
import os
import subprocess
import sys
import time

from clustering import MAX_QUADRATIC_N, QUADRATIC_METHODS, default_method

EPS_METHODS = {'balanced_kmeans', 'class_balanced_kmeans'}

HERE = os.path.dirname(os.path.abspath(__file__))

TRANSDUCTIVE = {'cora', 'citeseer', 'ogbn-arxiv'}
# Number of nodes that get clustered (all nodes when transductive, training nodes when inductive).
CLUSTERED_N = {'cora': 2708, 'citeseer': 3327, 'ogbn-arxiv': 169343, 'flickr': 44625, 'reddit': 153932}

HP = {  # copied from main_transduct.sh / main_induct.sh
    ('cora', 1.0): '--prop_num 20 --alpha 0.8 --predropout 0.6 --sp_ratio 0.4 --preep 80 --postep 2000 --postprop_num 2 --frcoe 0.01 --predcoe 1.0',
    ('cora', 0.5): '--prop_num 5 --postprop_num 2 --alpha 0.8 --predropout 0.6 --sp_ratio 0.4 --preep 80 --postep 2000 --frcoe 0.01 --predcoe 1.0',
    ('cora', 0.25): '--prop_num 5 --postprop_num 2 --alpha 0.8 --predropout 0.7 --sp_ratio 0.06 --preep 80 --postep 2000 --frcoe 0.01 --predcoe 1.0',
    ('citeseer', 0.25): '--prop_num 2 --postprop_num 1 --alpha 0.8 --predropout 0.7 --sp_ratio 0.06 --preep 120 --postep 80 --frcoe 0.01 --predcoe 1.0',
    ('citeseer', 0.5): '--prop_num 2 --postprop_num 1 --alpha 0.5 --predropout 0.8 --prewd 0.0002 --sp_ratio 0.21 --preep 100 --postep 200 --hidden 128 --frcoe 0.01 --predcoe 0.05',
    ('citeseer', 1.0): '--prop_num 2 --postprop_num 1 --alpha 0.5 --predropout 0.7 --sp_ratio 0.2 --preep 200 --postep 200 --frcoe 0.01 --predcoe 0.9 --w1 0.1',
    ('ogbn-arxiv', 0.001): '--prop_num 20 --postprop_num 10 --alpha 0.92 --predropout 0.6 --sp_ratio 0.1 --preep 1000 --postep 1000 --frcoe 1.4 --predcoe 0.0125',
    ('ogbn-arxiv', 0.005): '--prop_num 18 --postprop_num 10 --alpha 0.91 --predropout 0.6 --sp_ratio 0.1 --preep 1000 --postep 1000 --frcoe 1.9 --predcoe 0.025',
    ('ogbn-arxiv', 0.01): '--prop_num 20 --postprop_num 10 --alpha 0.92 --predropout 0.6 --sp_ratio 0.1 --preep 1000 --postep 1000 --frcoe 1.0 --predcoe 0.025',
    ('flickr', 0.001): '--prop_num 2 --postprop_num 2 --alpha 0.8 --predropout 0.6 --sp_ratio 0.5 --preep 2000 --postep 2000 --frcoe 0.2 --predcoe 0.8 --w1 0.8',
    ('flickr', 0.005): '--prop_num 2 --postprop_num 1 --alpha 0.8 --predropout 0.6 --sp_ratio 0.5 --preep 2000 --postep 2000 --frcoe 0.2 --predcoe 0.8 --w1 1.0',
    ('flickr', 0.01): '--prop_num 2 --postprop_num 2 --alpha 0.8 --predropout 0.6 --sp_ratio 0.5 --preep 2000 --postep 2000 --frcoe 0.2 --predcoe 0.8 --w1 0.8',
    ('reddit', 0.0005): '--prop_num 10 --postprop_num 7 --alpha 0.92 --predropout 0.6 --sp_ratio 0.1 --preep 800 --postep 600 --hidden 512 --frcoe 3.5 --predcoe 0.038 --w1 0.015',
    ('reddit', 0.001): '--prop_num 10 --postprop_num 7 --alpha 0.92 --predropout 0.6 --sp_ratio 0.1 --preep 800 --postep 600 --hidden 512 --frcoe 3.3 --predcoe 0.024 --w1 0.015',
    ('reddit', 0.005): '--prop_num 20 --postprop_num 10 --alpha 0.95 --predropout 0.6 --sp_ratio 0.1 --preep 1000 --postep 1000 --frcoe 0.8 --predcoe 0.05 --w1 0.01',
}


def key(dataset, rate, method, seed, eps):
    return (dataset, float(rate), method, int(seed), float(eps) if method in EPS_METHODS else 0.0)


def done_keys(path):
    out = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out.add(key(r['dataset'], r['reduction_rate'], r['cluster_method'], r['seed'], r.get('balance_eps', 0.0)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['cora', 'citeseer'])
    p.add_argument('--rates', nargs='*', type=float, default=None, help='default: all three of the dataset')
    p.add_argument('--methods', nargs='+', default=['baseline', 'balanced_kmeans', 'class_kmeans', 'class_balanced_kmeans',
                                                    'random_balanced', 'gmm', 'ward', 'spectral'],
                   help="'baseline' = the clustering the original code uses for the dataset")
    p.add_argument('--seeds', nargs='+', type=int, default=[15, 16, 17])
    p.add_argument('--eps', nargs='+', type=float, default=[0.0], help='balance_eps values for balanced_kmeans')
    p.add_argument('--results_file', default='results/results.jsonl')
    p.add_argument('--log_dir', default='logs/runs')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()

    os.chdir(HERE)
    os.makedirs(a.log_dir, exist_ok=True)
    finished = done_keys(a.results_file)
    jobs = []
    for d in a.datasets:
        rates = a.rates or sorted(r for (ds, r) in HP if ds == d)
        for r in rates:
            for m in a.methods:
                m = default_method(d) if m == 'baseline' else m  # the paper's own clustering
                if m in QUADRATIC_METHODS and CLUSTERED_N[d] > MAX_QUADRATIC_N:
                    continue
                for e in (a.eps if m in EPS_METHODS else [0.0]):
                    for s in a.seeds:
                        if key(d, r, m, s, e) not in finished:
                            jobs.append((d, r, m, s, e))
    print(f'{len(jobs)} runs to do ({len(finished)} already in {a.results_file})', flush=True)

    for i, (d, r, m, s, e) in enumerate(jobs, 1):
        script = 'train_clustgdd_transduct.py' if d in TRANSDUCTIVE else 'train_clustgdd_induct.py'
        cmd = [sys.executable, script, '--gpu_id', '0', '--dataset', d, '--reduction_rate', str(r),
               *HP[(d, r)].split(), '--cluster_method', m, '--seed', str(s), '--balance_eps', str(e),
               '--results_file', a.results_file]
        log = os.path.join(a.log_dir, f'{d}_{r}_{m}_eps{e}_s{s}.log')
        print(f'[{i}/{len(jobs)}] {d} r={r} {m} eps={e} seed={s}', flush=True)
        if a.dry_run:
            print('   ', ' '.join(cmd))
            continue
        t = time.time()
        with open(log, 'w') as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f'    rc={rc} {time.time() - t:.0f}s  log={log}', flush=True)


if __name__ == '__main__':
    main()
