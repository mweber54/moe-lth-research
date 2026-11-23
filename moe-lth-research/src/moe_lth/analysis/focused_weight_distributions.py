"""Focused, layer-pooled distributions for E32 seed 7 at 80% and 85% sparsity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import weight_distributions as analysis

CASES = analysis.DEFAULTS[:2]
COLORS = ['#606060', '#e69f00', '#0072b2', '#009e73']
LABELS = ['Dense source (2500)', 'Sparse initial (2500)', 'Sparse final (5000)', 'Dense final (5000)']
TITLES = ['Histogram', 'Signed CDF', '|weight| CDF']


def rendering_profile(x, edges):
    """Exact counts first; only the displayed CDF staircase is compressed."""
    signed, counts = np.unique(x, return_counts=True)
    absolute, abs_counts = np.unique(np.abs(x), return_counts=True)
    return dict(signed=signed, counts=counts, absolute=absolute, abs_counts=abs_counts,
                histogram=np.histogram(x, bins=edges)[0])


def display_cdf(values, counts, *, nonzero=False):
    if nonzero:
        selected = values != 0
        values, counts = values[selected], counts[selected]
    if not len(values):
        return np.array([]), np.array([])
    cdf = np.cumsum(counts, dtype=np.float64) / counts.sum()
    zero = np.searchsorted(values, 0)
    indices = np.unique(np.r_[np.linspace(0, len(values)-1, min(2500,len(values))).astype(int),
                              np.clip(zero+np.array([-1,0,1]), 0,len(values)-1)])
    return values[indices], cdf[indices]


def draw(axes, profiles, labels, colors, edges, limit, nonzero_flags):
    for profile, label, color, nonzero in zip(profiles, labels, colors, nonzero_flags):
        counts = profile['histogram'].copy()
        if nonzero:
            # Zero is a point mass in one bin, not every weight in that bin.
            zero_count = profile['counts'][profile['signed'] == 0].sum()
            zero_bin = np.searchsorted(edges, 0, side='right')-1
            counts[zero_bin] -= zero_count
        total = counts.sum()
        if total:
            axes[0].stairs(counts/(total*np.diff(edges)), edges, label=label, color=color, lw=1.3)
        for col, values, multiplicities in [(1, profile['signed'], profile['counts']),
                                            (2, profile['absolute'], profile['abs_counts'])]:
            x,y = display_cdf(values, multiplicities, nonzero=nonzero)
            if len(x):
                # Start the staircase at zero probability, preserving the zero atom.
                axes[col].step(np.r_[(-limit if col==1 else 0),x,limit], np.r_[0,y,1],
                               where='post', label=label, color=color, lw=1.3)
    for col, ax in enumerate(axes):
        ax.set_xlim((0,limit) if col==2 else (-limit,limit))
        if col: ax.set_ylim(0,1.01)
        ax.set_xlabel('|weight|' if col==2 else 'Weight')
        ax.set_ylabel('Density' if col==0 else 'Cumulative probability')
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)


def pooled_arrays(cache, layer, matrix):
    names = sorted(cache.glob(f'dense_source_blocks.{layer}.moe.*.weight.npy'))
    names = [p.name.removeprefix('dense_source_').removesuffix('.npy') for p in names
             if (m := analysis.PATTERN.fullmatch(p.name.removeprefix('dense_source_').removesuffix('.npy')))
             and (m.group(3) or 'router') == matrix]
    analysis.require(len(names) == (1 if matrix=='router' else 32), 'Unexpected pooled matrix coverage')
    return [np.concatenate([np.load(cache/f'{stage}_{name}.npy',mmap_mode='r').ravel() for name in names])
            for stage in analysis.STAGES]


def save(fig, output, stem):
    fig.tight_layout(rect=(0,0,.999,.925))
    fig.savefig(output/f'{stem}.png',dpi=300)
    fig.savefig(output/f'{stem}.pdf')
    plt.close(fig)


def focused_figures(output, reports):
    analysis.require([r['case'] for r in reports] == CASES, 'Focused analysis requires exactly the two seed-7 cases')
    analysis.require(reports[0]['config'] == reports[1]['config'], 'Cross-sparsity architecture mismatch')
    for stage in ('dense_source','dense_final'):
        hashes = [next(c['sha256'] for c in r['checkpoints'] if c['stage']==stage) for r in reports]
        analysis.require(hashes[0] == hashes[1], f'Cross-sparsity {stage} identity mismatch')
    settings=[]
    layers=range(reports[0]['config']['num_layers'])
    for matrix in ('router','fc1','fc2'):
        subsets=['all'] if matrix=='router' else ['all','nonzero']
        panels={}
        for subset in subsets:
            fig,axes=plt.subplots(len(layers),6,figsize=(18,10.8),squeeze=False)
            fig.suptitle(f'{matrix}: four-stage comparison'+(' — sparse exact zeros excluded' if subset=='nonzero' else ' — all weights'),fontsize=15)
            panels[('stages',subset)]=(fig,axes)
            fig,axes=plt.subplots(len(layers),3,figsize=(11,10.8),squeeze=False)
            fig.suptitle(f'{matrix}: 80% vs 85% sparse final'+(' — exact zeros excluded' if subset=='nonzero' else ' — all weights'),fontsize=15)
            panels[('final',subset)]=(fig,axes)
        for layer in layers:
            data=[pooled_arrays(output/r['case']/'cache',layer,matrix) for r in reports]
            limit=max(float(np.abs(x).max()) for arrays in data for x in arrays)*1.02 or 1.
            edges=np.linspace(-limit,limit,101)
            settings.append(dict(matrix=matrix,layer=layer,signed_limits=[-limit,limit],absolute_limits=[0,limit],bin_edges=edges.tolist()))
            profiles=[[rendering_profile(x,edges) for x in arrays] for arrays in data]
            del data
            for subset in subsets:
                stage_axes=panels[('stages',subset)][1]
                for index in range(2):
                    axes=stage_axes[layer,index*3:index*3+3]
                    draw(axes,profiles[index],LABELS,COLORS,edges,limit,
                         [False,subset=='nonzero',subset=='nonzero',False])
                    for col,ax in enumerate(axes): ax.set_title(f'Layer {layer} · {80+5*index}% · {TITLES[col]}',fontsize=10)
                ymax=max(stage_axes[layer,0].get_ylim()[1],stage_axes[layer,3].get_ylim()[1])
                for index in (0,3): stage_axes[layer,index].set_ylim(0,ymax)
                axes=panels[('final',subset)][1][layer]
                draw(axes,[profiles[0][2],profiles[1][2]],['80% sparse final','85% sparse final'],
                     [COLORS[1],COLORS[2]],edges,limit,[subset=='nonzero']*2)
                for col,ax in enumerate(axes): ax.set_title(f'Layer {layer} · {TITLES[col]}',fontsize=10)
            del profiles
            print(f'Focused {matrix} layer {layer} complete',flush=True)
        for (kind,subset),(fig,axes) in panels.items():
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.957),ncol=len(labels),frameon=False,fontsize=10)
            save(fig,output,f'{matrix}_{kind}_{subset}')
    (output/'plot_settings.json').write_text(json.dumps(settings,indent=2))


README = '''# Focused E32 seed-7 weight distributions

Only `e32/seed_7/final_s0.80` and `e32/seed_7/final_s0.85` are included. Run:

```bash
PYTHONPATH=src .venv-highres/bin/python -m moe_lth.analysis.focused_weight_distributions
```

## Ten figures, each in 300-dpi PNG and vector PDF

- `router_stages_all`, `fc1_stages_all`, `fc2_stages_all`: one row per transformer layer; the left three columns show 80% sparsity, the right three show 85%. Columns are signed density histogram, signed empirical CDF, and absolute empirical CDF. Gray = dense source, orange = sparse initial, blue = sparse final, green = matched dense final.
- `fc1_stages_nonzero`, `fc2_stages_nonzero`: the same comparisons after removing exact zeros from sparse curves only; dense curves retain all weights.
- `router_final_all`, `fc1_final_all`, `fc2_final_all`: direct 80%- versus 85%-sparse final distributions, orange versus blue, with one layer per row.
- `fc1_final_nonzero`, `fc2_final_nonzero`: direct final comparisons after excluding exact zeros from both sparse models.

All 32 experts are pooled within each layer/projection. No per-expert pages, geometry figures, or other configurations are generated. fc1 is the GELU input/up projection and fc2 the output/down projection; no gate matrix exists. Routers remain separate and unpruned.

## Interpretation and controls

Weights are raw and never standardized. For each matrix type and layer, the same 100 histogram bins and weight limits cover every stage of both configurations, including the nonzero and direct-comparison versions. Histogram density integrates to one within each displayed distribution. Histogram y limits also match between the two sparsity columns. CDFs use F(t)=count(w<=t)/N, or |w| for magnitudes, with common probability limits. A sparse histogram's zero spike is a finite-bin representation of a point mass. No Gaussian model or smooth density fit is assumed. `plot_settings.json` records all limits and bin edges.

All weights contribute to counts, summaries, and distributional distances. For display only, CDFs retain up to 2500 ordered support locations plus zero and adjacent support points; probabilities are calculated from complete empirical counts. Removing exact zeros is not the same as selecting mask survivors: naturally zero surviving coordinates are excluded in nonzero figures, while fixed-mask survivor statistics include them.

## Provenance and statistics

The original analysis verifier and statistics implementation are reused unchanged: canonical dense matching, checkpoint SHA256s, seeds, architecture, source/target steps, original and continuation data hashes, continuation protocols, mask hashes, shapes, and masked-zero integrity are checked. Checkpoints are mapped sequentially on CPU; scratch extracted arrays are removed after success. The two cases must also share identical dense source and dense-final checkpoint hashes.

Each case directory contains verified `provenance.json` and the existing CSV/JSON `statistics`, `distances`, `growth`, and `heterogeneity` tables, including per-expert statistics but no per-expert figures. All-weight, mask-surviving, and nonzero statistics remain distinct. Signed/absolute Wasserstein and KS comparisons retain all entries. See the parent analysis README for full statistical definitions. Dense source and sparse initial refer to step 2500; final endpoints refer to step 5000. Sparse initial is the start of extended continuation, not immediately after pruning. Continuations use fresh AdamW/GradScaler state. Distributional similarity does not establish geometric or subspace alignment.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,default=Path('results/weight_distribution_analysis/focused_winners'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); torch.set_num_threads(1)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for case in CASES:
        try:
            report,_,_=analysis.analyze(args.root.resolve(),output,case,args.workers,make_plots=False,keep_cache=True)
            reports.append(report)
        except (FileNotFoundError,ValueError,KeyError) as exc:
            reports.append(dict(case=case,status='missing_or_incompatible',error=str(exc)))
        analysis.export(output,'checkpoint_audit',reports)
    (output/'README.md').write_text(README)
    if any(r['status']!='complete' for r in reports): raise SystemExit(1)
    focused_figures(output,reports)
    for case in CASES:
        cache=output/case/'cache'
        for path in cache.glob('*.npy'): path.unlink()
        cache.rmdir()


if __name__=='__main__': main()
