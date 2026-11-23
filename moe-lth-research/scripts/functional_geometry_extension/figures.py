import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from fge_common import *

FIG = OUT / 'figures'
COL = {7: 'C0', 17: 'C1', 29: 'C2', 43: 'C3', 61: 'C4'}
WIN = {(7, .8), (7, .85), (43, .8), (61, .8)}


def xs(ax):
    ax.axvline(2500, color='gray', ls=':', lw=1)
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel('step (dotted: optimizer reset at 2500)')


def lines(df, ycol, title, fname, split='sparsity', extra=None):
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, sp in zip(axes, (.8, .85)):
        for seed, g in df[df.sparsity == sp].groupby('seed'):
            g = g.sort_values('step')
            win = (seed, sp) in WIN
            ax.plot(g.step, g[ycol], color=COL[seed], marker='o' if win else 'x', ms=4, ls='-' if win else '--', label=f'seed {seed}' + (' (matches dense CE)' if win else ''))
        xs(ax)
        ax.set_title(f'sparsity {sp}')
        ax.grid(alpha=.3)
    axes[0].set_ylabel(ycol)
    axes[0].legend(fontsize=7)
    fig.suptitle(title + '  [every seed shown; solid = matches/beats matched dense at 5,000 attempted steps]', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def main():
    c = pd.read_csv(OUT / 'consolidated_existing_metrics/coupling_trajectories_rerun.csv')
    c = c[(c['mode'] == 'fixed_source') & (c.selection == 'all_expert')]
    sp = c[c.model_role == 'sparse']
    lines(sp, 'pearson', 'Existing: router-score/post-GELU-activation Pearson (fixed-source, all experts; n=128 positions)', 'coupling_sparse_by_seed.png')
    de = c[c.model_role == 'dense'].drop_duplicates(['seed', 'step'])
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, g in de.groupby('seed'):
        ax.plot(g.sort_values('step').step, g.sort_values('step').pearson, color=COL[s], label=f'dense seed {s}')
    xs(ax); ax.legend(fontsize=7); ax.set_title('Existing: dense coupling by seed'); ax.set_ylabel('pearson')
    FIG.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(FIG / 'coupling_dense_by_seed.png', dpi=130); plt.close(fig)
    A = OUT / 'activation_geometry/per_seed_summary.csv'
    if A.exists():
        a = pd.read_csv(A)
        for q in ('hidden', 'output'):
            for m in ('fixed_source', 'dense_native'):
                d = a[(a.quantity == q) & (a['mode'] == m)]
                if len(d):
                    for y in ('cka', 'relative_error', 'feat_overlap_r16'):
                        lines(d, y, f'New: sparse vs same-step dense {q}, input mode {m}', f'activation_{q}_{m}_{y}.png')
    Dp = OUT / 'fixed_gate_distortion/per_seed_summary.csv'
    if Dp.exists():
        d = pd.read_csv(Dp)
        for (g, im), x in d.groupby(['gate_reference', 'input_mode']):
            lines(x, 'rel_full', f'New: fixed-gate distortion D/||ref||, gates={g}, inputs={im}', f'distortion_{g}_{im}.png')
    G = OUT / 'gradient_geometry/per_seed_summary.csv'
    if G.exists():
        g = pd.read_csv(G)
        g = g[g.kind == 'task']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, mat in zip(axes, ('fc1', 'fc2')):
            for (seed, sp_), x in g[g.matrix == mat].groupby(['seed', 'sparsity']):
                x = x.sort_values('step')
                ax.plot(x.step, x.energy_excess_over_random, color=COL[seed], ls='-' if sp_ == .8 else '--', marker='o', ms=3)
            ax.axhline(0, color='k', lw=.8); ax.set_title(f'{mat}: IMP-mask energy retention minus random-mask null (solid .80, dashed .85)'); ax.title.set_fontsize(8)
            ax.set_xlabel('checkpoint step')
        fig.tight_layout(); fig.savefig(FIG / 'gradient_energy_excess_over_random.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
