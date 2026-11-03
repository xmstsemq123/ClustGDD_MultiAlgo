"""Summarise results/results.jsonl into tables and figures for the report.

Outputs (results/):
  summary_acc.md / .csv     test accuracy mean ± std over seeds, per dataset / ratio / method
  summary_metrics.csv       clustering statistics (WCSS, size CV, purity, FID, time) per cell
  correlation.md            Spearman correlation between each clustering statistic and accuracy
  figs/delta_acc.png        accuracy change vs. the paper's clustering, per method
  figs/metric_vs_acc.png    WCSS / size-CV / FID / purity against accuracy
  figs/eps_ablation.png     balance tolerance eps vs. accuracy
  figs/cluster_sizes.png    sorted cluster sizes of baseline vs. our methods
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from clustering import default_method  # noqa: E402

DATASETS = ['cora', 'citeseer', 'ogbn-arxiv', 'flickr', 'reddit']
DNAME = {'cora': 'Cora', 'citeseer': 'Citeseer', 'ogbn-arxiv': 'arXiv', 'flickr': 'Flickr', 'reddit': 'Reddit'}
# paper's ratio n/N (transductive) or n/N_train (inductive) for each reduction_rate
RATIO = {('cora', 0.25): '1.3%', ('cora', 0.5): '2.6%', ('cora', 1.0): '5.2%',
         ('citeseer', 0.25): '0.9%', ('citeseer', 0.5): '1.8%', ('citeseer', 1.0): '3.6%',
         ('ogbn-arxiv', 0.001): '0.05%', ('ogbn-arxiv', 0.005): '0.25%', ('ogbn-arxiv', 0.01): '0.5%',
         ('flickr', 0.001): '0.1%', ('flickr', 0.005): '0.5%', ('flickr', 0.01): '1%',
         ('reddit', 0.0005): '0.05%', ('reddit', 0.001): '0.1%', ('reddit', 0.005): '0.5%'}
ORDER = ['baseline', 'balanced_kmeans', 'class_kmeans', 'class_balanced_kmeans', 'gmm', 'ward', 'spectral',
         'random_balanced']
MNAME = {'baseline': 'K-Means (paper)', 'balanced_kmeans': 'Balanced K-Means', 'class_kmeans': 'Class K-Means',
         'class_balanced_kmeans': 'Class-Balanced K-Means', 'gmm': 'GMM', 'ward': 'Ward', 'spectral': 'Spectral',
         'random_balanced': 'Random balanced'}
PAIRED = ['rel_wcss', 'd_size_cv', 'rel_fid', 'd_purity']
ROLE = {'baseline': 'baseline', 'balanced_kmeans': 'ours', 'class_kmeans': 'ours', 'class_balanced_kmeans': 'ours',
        'gmm': 'comparison', 'ward': 'comparison', 'spectral': 'comparison', 'random_balanced': 'control'}

# reference palette (light): role colours, recessive neutrals
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
COLOR = {'ours': '#2a78d6', 'control': '#eb6834', 'baseline': INK2, 'comparison': MUTED}


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'legend.frameon': False,
    })


def load(path):
    df = pd.read_json(path, lines=True)
    df['balance_eps'] = df.get('balance_eps', 0.0).fillna(0.0)
    df['acc'] = df['acc_test_mean'] * 100
    # 'baseline' = whatever clustering the original code used for this dataset
    df['method'] = [('baseline' if m == default_method(d) else m) for m, d in zip(df.cluster_method, df.dataset)]
    df['ratio'] = [RATIO.get((d, r), str(r)) for d, r in zip(df.dataset, df.reduction_rate)]
    df = df.drop_duplicates(['dataset', 'reduction_rate', 'method', 'balance_eps', 'seed'], keep='last')
    # H (the clustered MLP output) differs per seed, so raw WCSS / FID are only comparable within a seed.
    # Pair every run with the baseline run of the same dataset / ratio / seed.
    base = df[df.method == 'baseline'].set_index(['dataset', 'reduction_rate', 'seed'])
    idx = list(zip(df.dataset, df.reduction_rate, df.seed))
    has = [i in base.index for i in idx]
    b = base.reindex(idx)
    df['d_acc'] = np.where(has, df.acc.values - b.acc.values, np.nan)
    df['rel_wcss'] = np.where(has, df.wcss.values / b.wcss.values, np.nan)
    df['rel_fid'] = np.where(has, df.fid.values / b.fid.values, np.nan)
    df['d_size_cv'] = np.where(has, df.size_cv.values - b.size_cv.values, np.nan)
    df['d_purity'] = np.where(has, df.purity.values - b.purity.values, np.nan)
    return df


def main_grid(df):
    """eps = 0 runs only (the eps ablation is reported separately)."""
    return df[df.balance_eps == 0]


def tables(df, out):
    g = main_grid(df).groupby(['dataset', 'reduction_rate', 'ratio', 'method'])
    acc = g.acc.agg(['mean', 'std', 'size']).reset_index()
    acc.to_csv(f'{out}/summary_acc.csv', index=False)
    met = g[['wcss', 'size_cv', 'thm31_bound', 'purity', 'label_agreement', 'fid', 'cluster_time']].mean().reset_index()
    met.to_csv(f'{out}/summary_metrics.csv', index=False)

    lines = ['# Test accuracy (%) — mean ± std over seeds\n']
    for d in [d for d in DATASETS if d in set(acc.dataset)]:
        sub = acc[acc.dataset == d]
        methods = [m for m in ORDER if m in set(sub.method)]
        lines.append(f'\n## {DNAME[d]}\n')
        lines.append('| ratio | ' + ' | '.join(MNAME[m] for m in methods) + ' |')
        lines.append('|---' * (len(methods) + 1) + '|')
        for r in sorted(set(sub.reduction_rate)):
            row = sub[sub.reduction_rate == r].set_index('method')
            best = row.loc[[m for m in methods if m != 'random_balanced' and m in row.index], 'mean'].idxmax()
            cells = []
            for m in methods:
                if m not in row.index:
                    cells.append('–')
                    continue
                c = f"{row.loc[m, 'mean']:.1f} ± {row.loc[m, 'std']:.1f}"
                if row.loc[m, 'size'] < 3:
                    c += f" (n={int(row.loc[m, 'size'])})"
                cells.append(f'**{c}**' if m == best else c)
            lines.append(f"| {RATIO.get((d, r), r)} | " + ' | '.join(cells) + ' |')
    open(f'{out}/summary_acc.md', 'w').write('\n'.join(lines) + '\n')

    # Spearman correlation of each statistic with accuracy, per dataset, across all (method, ratio, seed) runs;
    # computed with and without the random negative control (which dominates any correlation).
    rows = []
    for d in [d for d in DATASETS if d in set(df.dataset)]:
        for excl in (False, True):
            sub = main_grid(df)[main_grid(df).dataset == d]
            if excl:
                sub = sub[sub.method != 'random_balanced']
            for r in sorted(set(sub.reduction_rate)):
                s = sub[sub.reduction_rate == r]
                if len(s) < 5:
                    continue
                s = s[s.method != 'baseline'].dropna(subset=['d_acc'])
                if len(s) < 5:
                    continue
                row = {'dataset': DNAME[d], 'ratio': RATIO.get((d, r), r), 'without_random': excl, 'n': len(s)}
                for k in PAIRED:
                    row[k] = spearmanr(s[k], s.d_acc).statistic
                rows.append(row)
    cor = pd.DataFrame(rows)
    with open(f'{out}/correlation.md', 'w') as f:
        f.write('# Spearman correlation with Δ accuracy vs. the same-seed K-Means run\n\n'
                'Statistics are relative to the same-seed baseline (rel_* = ratio, d_* = difference), '
                'because H differs between seeds. Runs pooled over methods and seeds.\n\n')
        f.write(cor.round(2).to_markdown(index=False) if len(cor) else 'not enough runs\n')
    return acc


def fig_delta(acc, out):
    """Accuracy change vs. the paper's clustering, one small multiple per dataset."""
    methods = [m for m in ORDER if m not in ('baseline', 'random_balanced') and m in set(acc.method)]
    ds = [d for d in DATASETS if acc[(acc.dataset == d) & acc.method.isin(methods)].shape[0]]
    fig, axes = plt.subplots(1, len(ds), figsize=(3.2 * len(ds), 3.6), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], ds):
        sub = acc[acc.dataset == d]
        rates = sorted(set(sub.reduction_rate))
        for j, m in enumerate(methods):
            for k, r in enumerate(rates):
                cell = sub[(sub.reduction_rate == r)].set_index('method')
                if m not in cell.index or 'baseline' not in cell.index:
                    continue
                delta = cell.loc[m, 'mean'] - cell.loc['baseline', 'mean']
                y = j + (k - (len(rates) - 1) / 2) * 0.22
                ax.errorbar(delta, y, xerr=cell.loc[m, 'std'], fmt='o', ms=6, color=COLOR[ROLE[m]],
                            mec=SURFACE, mew=1.5, elinewidth=1.2, capsize=0, alpha=1 - 0.25 * k)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(DNAME[d])
        ax.set_xlabel('Δ accuracy vs. K-Means (pp)')
        ax.grid(axis='y', visible=False)
    axes[0][0].set_yticks(range(len(methods)), [MNAME[m] for m in methods])
    axes[0][0].invert_yaxis()
    fig.suptitle('Accuracy change relative to the paper\'s clustering (dots: ratios small→large, bars: ±1 std over seeds)',
                 fontsize=10, color=INK2, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(f'{out}/figs/delta_acc.png', dpi=200)
    plt.close(fig)


def fig_metrics(df, out):
    """Paired view: change of each clustering statistic vs. change of accuracy, both relative to the
    same-seed K-Means run. One row per dataset, all ratios pooled; random control left out."""
    from matplotlib.ticker import MaxNLocator
    sub = main_grid(df)
    sub = sub[(sub.method != 'baseline') & (sub.method != 'random_balanced')].dropna(subset=['d_acc'])
    ds = [d for d in DATASETS if d in set(sub.dataset)]
    if not ds:
        return
    keys = [('rel_wcss', 'WCSS ÷ K-Means WCSS'), ('d_size_cv', 'Δ cluster-size CV'),
            ('rel_fid', 'FID ÷ K-Means FID'), ('d_purity', 'Δ purity')]
    fig, axes = plt.subplots(len(ds), len(keys), figsize=(2.9 * len(keys), 2.5 * len(ds) + 0.6), squeeze=False)
    for i, d in enumerate(ds):
        s = sub[sub.dataset == d]
        for j, (k, lab) in enumerate(keys):
            ax = axes[i][j]
            ax.axhline(0, color=AXIS, lw=1, zorder=0)
            for role, name in [('comparison', 'GMM / Ward / Spectral'), ('ours', 'ours')]:
                p = s[s.method.map(ROLE) == role]
                ax.scatter(p[k], p.d_acc, s=34, color=COLOR[role], edgecolor=SURFACE, linewidth=1.5, label=name)
            rho = spearmanr(s[k], s.d_acc).statistic if len(s) >= 5 else np.nan
            ax.set_title(f'ρ = {rho:.2f}', fontsize=9, fontweight='normal', color=INK2, loc='right')
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel(lab)
            if j == 0:
                ax.set_ylabel(f'{DNAME[d]}\nΔ accuracy (pp)')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper left', ncol=2, fontsize=9, bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    fig.savefig(f'{out}/figs/metric_vs_acc.png', dpi=200)
    plt.close(fig)


def fig_eps(df, out):
    eps_df = df[df.method.isin(['balanced_kmeans', 'class_balanced_kmeans'])]
    if eps_df.balance_eps.nunique() < 2:
        return
    ds = [d for d in DATASETS if d in set(eps_df.dataset) and eps_df[eps_df.dataset == d].balance_eps.nunique() > 1]
    rates = {d: sorted(set(eps_df[eps_df.dataset == d].reduction_rate)) for d in ds}
    fig, axes = plt.subplots(len(ds), max(len(v) for v in rates.values()), figsize=(10, 2.8 * len(ds)), squeeze=False)
    styles = {'balanced_kmeans': ('#2a78d6', '-', 'Balanced K-Means'),
              'class_balanced_kmeans': ('#1baf7a', '-', 'Class-Balanced K-Means')}
    for i, d in enumerate(ds):
        for j, r in enumerate(rates[d]):
            ax = axes[i][j]
            base = main_grid(df)[(main_grid(df).dataset == d) & (main_grid(df).reduction_rate == r) & (main_grid(df).method == 'baseline')]
            if len(base):
                ax.axhline(base.acc.mean(), color=INK2, lw=1, ls='--')
                ax.text(0.99, base.acc.mean(), 'K-Means', color=INK2, fontsize=8, ha='right', va='bottom',
                        transform=ax.get_yaxis_transform())
            for m, (c, ls, lab) in styles.items():
                g = eps_df[(eps_df.dataset == d) & (eps_df.reduction_rate == r) & (eps_df.method == m)]
                g = g.groupby('balance_eps').acc.agg(['mean', 'std']).reset_index()
                if not len(g):
                    continue
                ax.plot(g.balance_eps, g['mean'], ls, color=c, lw=2, marker='o', ms=6, mec=SURFACE, mew=1.5, label=lab)
                ax.fill_between(g.balance_eps, g['mean'] - g['std'], g['mean'] + g['std'], color=c, alpha=0.12, lw=0)
            ax.set_title(f'{DNAME[d]} {RATIO.get((d, r), r)}')
            ax.set_xlabel('ε (max cluster size = (1+ε)·N/n)')
            if j == 0:
                ax.set_ylabel('accuracy (%)')
    axes[0][0].legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{out}/figs/eps_ablation.png', dpi=200)
    plt.close(fig)


def fig_sizes(df, out):
    if 'cluster_sizes' not in df:
        return
    s = main_grid(df).dropna(subset=['cluster_sizes'])
    s = s[s.method.isin(['baseline', 'balanced_kmeans', 'class_balanced_kmeans'])]
    ds = [d for d in DATASETS if s[s.dataset == d].method.nunique() == 3]
    if not ds:
        return
    styles = {'baseline': (INK2, 'K-Means (paper)'), 'balanced_kmeans': ('#2a78d6', 'Balanced K-Means'),
              'class_balanced_kmeans': ('#1baf7a', 'Class-Balanced K-Means')}
    fig, axes = plt.subplots(1, len(ds), figsize=(3.4 * len(ds), 3.0), squeeze=False)
    for ax, d in zip(axes[0], ds):
        sd = s[s.dataset == d]
        r = min(sd.reduction_rate)
        for m, (c, lab) in styles.items():
            row = sd[(sd.reduction_rate == r) & (sd.method == m)].iloc[0]
            ax.plot(np.arange(1, len(row.cluster_sizes) + 1), row.cluster_sizes, color=c, lw=2, label=lab)
        ax.set_title(f'{DNAME[d]} {RATIO.get((d, r), r)}')
        ax.set_xlabel('cluster rank')
        ax.set_ylabel('nodes in cluster')
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{out}/figs/cluster_sizes.png', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_file', default='results/results.jsonl')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(f'{a.out}/figs', exist_ok=True)
    style()
    df = load(a.results_file)
    acc = tables(df, a.out)
    fig_delta(acc, a.out)
    fig_metrics(df, a.out)
    fig_eps(df, a.out)
    fig_sizes(df, a.out)
    print(open(f'{a.out}/summary_acc.md').read())
    print(open(f'{a.out}/correlation.md').read())


if __name__ == '__main__':
    main()
