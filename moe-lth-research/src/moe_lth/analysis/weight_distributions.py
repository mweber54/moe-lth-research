"""CPU-only, read-only checkpoint distribution analysis. See generated README."""
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

PATTERN = re.compile(r'blocks\.(\d+)\.moe\.(?:router\.projection|experts\.(\d+)\.(fc1|fc2))\.weight$')
DEFAULTS = ['e32/seed_7/final_s0.80', 'e32/seed_7/final_s0.85', 'e32/seed_17/final_s0.80', 'e16/seed_17/final_s0.80']
STAGES = ['dense_source', 'sparse_initial', 'sparse_final', 'dense_final']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mask_hash(masks):
    h = hashlib.sha256()
    for name in sorted(masks):
        h.update(name.encode()); h.update(masks[name].to(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def statistics(x):
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return {'count': 0}
    q = np.quantile(np.abs(x), [.25, .5, .75, .9, .95, .99])
    return dict(count=x.size, mean=float(x.mean()), std=float(x.std()), median=float(np.median(x)),
                minimum=float(x.min()), maximum=float(x.max()), zero_fraction=float(np.mean(x == 0)),
                l1=float(np.abs(x).sum()), l2=float(np.linalg.norm(x)),
                **{f'abs_q{int(p*100)}': float(v) for p, v in zip([.25,.5,.75,.9,.95,.99], q)})


def distances(a, b):
    if not len(a) or not len(b):
        return {'wasserstein': None, 'ks': None, 'absolute_wasserstein': None, 'absolute_ks': None}
    return dict(wasserstein=float(wasserstein_distance(a, b)),
                ks=float(ks_2samp(a, b, method='asymp').statistic),
                absolute_wasserstein=float(wasserstein_distance(np.abs(a), np.abs(b))),
                absolute_ks=float(ks_2samp(np.abs(a), np.abs(b), method='asymp').statistic))



def empirical_profile(x):
    profiles = []
    for values in (x, np.abs(x)):
        ordered = np.sort(values)
        # CDF at each order statistic; searchsorted handles ties exactly.
        cdf = np.searchsorted(ordered, ordered, side='right') / max(len(ordered), 1)
        profiles.append((ordered, cdf))
    return profiles


def profile_distances(first, second):
    result = {}
    for prefix, (a, fa), (b, fb) in zip(('', 'absolute_'), first, second):
        require(len(a) == len(b), 'Paired empirical profiles must have equal counts')
        if not len(a):
            result[prefix+'wasserstein'] = result[prefix+'ks'] = None
            continue
        # Equal-sized empirical measures: W1 is the mean difference of order statistics.
        result[prefix+'wasserstein'] = float(np.mean(np.abs(a.astype(np.float64)-b)))
        result[prefix+'ks'] = float(max(np.max(np.abs(fa-np.searchsorted(b,a,side='right')/len(b))),
                                         np.max(np.abs(fb-np.searchsorted(a,b,side='right')/len(a)))))
    return result


def changes(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    norm = np.linalg.norm(a)
    ma = np.abs(a).mean() if a.size else 0
    delta = np.abs(b) - np.abs(a)
    return dict(normalized_drift=float(np.linalg.norm(b-a)/norm) if norm else None,
                mean_abs_change=float(delta.mean()) if a.size else None,
                mean_abs_growth_ratio=float(np.abs(b).mean()/ma) if ma else None,
                fraction_magnitudes_grown=float(np.mean(delta > 0)) if a.size else None,
                mean_abs_change_per_attempted_step=float(delta.mean()/2500) if a.size else None)


def export(directory, name, rows):
    (directory / f'{name}.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (directory / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)


def curves(ax, arrays, magnitude=False, histogram=False, limit=None):
    colors = ['#606060', '#e69f00', '#0072b2', '#009e73']
    for label, x, color in zip(STAGES, arrays, colors):
        if not x.size:
            continue
        if histogram:
            ax.hist(x, bins=np.linspace(-limit, limit, 81), density=True,
                    histtype='step', lw=.9, color=color, label=label)
        else:
            values, counts = np.unique(np.abs(x) if magnitude else x, return_counts=True)
            y = np.cumsum(counts)/x.size
            # Retain the zero atom and extrema; reduce rendering only, never statistics.
            indices = np.unique(np.r_[0, np.linspace(0, len(values)-1, min(1500,len(values))).astype(int),
                                          np.clip(np.searchsorted(values, 0)+np.array([-1,0,1]), 0,len(values)-1), len(values)-1])
            ax.step(values[indices], y[indices], where='post', color=color, lw=.9, label=label)
        ax.set_xlim((0, limit) if magnitude else (-limit, limit))
    ax.set_xlabel('|weight|' if magnitude else 'weight')
    ax.set_ylabel('density' if histogram else 'empirical probability')
    ax.grid(alpha=.18)


def plot_groups(destination, groups, arrays, masks, title):
    """One row per expert (or pooled/router matrix), six distribution panels."""
    for offset in range(0, len(groups), 4):
        chunk = groups[offset:offset+4]
        fig, axes = plt.subplots(len(chunk), 6, figsize=(18, 2.7*len(chunk)), squeeze=False)
        # Shared limits across experts, stages, and masks in this matrix/layer group.
        limit = max(float(np.max(np.abs(a))) for data in arrays.values() for a in data) * 1.02
        limit = limit or 1
        for row, group in enumerate(chunk):
            data, keep = arrays[group], masks[group]
            match = PATTERN.fullmatch(group)
            label = ('router' if match.group(2) is None else f'expert {match.group(2)}') if match else group
            for col, subset in enumerate(('all', 'nonzero')):
                selected = data if subset == 'all' else [a[a != 0] if i in (1,2) else a for i,a in enumerate(data)]
                for j in range(3):
                    ax = axes[row, col*3+j]
                    curves(ax, selected, histogram=j == 0, magnitude=j == 2, limit=limit)
                    ax.set_title(f'{label} | {subset} '+['histogram','signed CDF','absolute CDF'][j], fontsize=8)
            axes[row,0].legend(fontsize=6)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0,0,1,.97))
        stem = destination / f'page_{offset//4:02d}'
        fig.savefig(stem.with_suffix('.png'), dpi=220)
        fig.savefig(stem.with_suffix('.pdf'))
        plt.close(fig)


def analyze(root, output, case, workers=4, *, make_plots=True, keep_cache=False):
    extension = root/'results/imp_extended_recovery'/case
    sm = read(extension/'extension_manifest.json')
    dm = read(extension.parent/'dense/extension_manifest.json')
    identity = sm['identity']; did = dm['identity']
    experts, seed = int(case.split('/')[0][1:]), int(case.split('/')[1][5:])
    sparsity = float(case.split('final_s')[1])
    for m, condition in [(sm, 'sparse'), (dm, 'dense')]:
        require(m['num_experts'] == experts and m['seed'] == seed and m['condition'] == condition, 'Configuration mismatch')
        require(m['source_step'] == 2500 and m['target_step'] == 5000, 'Unexpected continuation steps')
        require(m['identity']['optimizer_protocol'] == 'fresh_adamw_and_grad_scaler_from_step_2500', 'Unexpected protocol')
    require(sm['sparsity'] == sparsity, 'Sparsity mismatch')
    for key in ['training_batch_hash','validation_batch_hash','source_step','target_step','optimizer_protocol','evaluation_interval']:
        require(identity[key] == did[key], f'Dense/sparse protocol mismatch: {key}')
    with (root/'results/final_dataset_a/canonical/dataset_a_canonical.csv').open() as f:
        canonical = [r for r in csv.DictReader(f) if int(r['num_experts']) == experts and int(r['seed']) == seed and r['router_age'] == '0' and float(r['sparsity']) == sparsity]
    require(len(canonical) == 1, 'Missing or ambiguous canonical dense match')
    canonical = canonical[0]
    require(Path(dm['source_checkpoint']) == Path(canonical['dense_artifact_path'])/'checkpoints/final_recovered.pt', 'Canonical dense path mismatch')
    require(dm['source_checkpoint_sha256'] == canonical['dense_final_checkpoint_sha256'], 'Canonical dense hash mismatch')
    source = Path(sm['source_checkpoint'])
    original = read(source.parent.parent/'stage.json')
    require(original['training_batch_sequence_hash'] == canonical['data_hash'], 'Original training batches mismatch canonical comparator')
    require(original['validation_batch_sequence_hash'] == canonical['validation_hash'] == identity['validation_batch_hash'], 'Original validation batches mismatch')
    mask_path = source.parent.parent.parent/f'mask_{8 if sparsity == .8 else 9:02d}.pt'
    masks = torch.load(mask_path, map_location='cpu', weights_only=True)
    mh = mask_hash(masks)
    require(mh == sm['mask_hash'] == original['mask_hash'], 'Mask hash mismatch')
    require(original['seed'] == seed and original['sparsity'] == sparsity and original['optimizer_step_attempts'] == 2500, 'Original recovery mismatch')
    paths = [Path(dm['source_checkpoint']), source, extension/'checkpoints/step_5000.pt', extension.parent/'dense/checkpoints/step_5000.pt']
    expected = [dm['source_checkpoint_sha256'], sm['source_checkpoint_sha256'], sm['step_5000_checkpoint_sha256'], dm['step_5000_checkpoint_sha256']]
    require(expected[1] == original['checkpoint_sha256'], 'Original source SHA mismatch')
    destination = output/case
    destination.mkdir(parents=True, exist_ok=True)
    # Extract only relevant matrices to disk, releasing each checkpoint before the next.
    cache = destination/'cache'; cache.mkdir(exist_ok=True)
    names = None; config = None; training = None; shapes = {}; audit = []
    for stage, path, digest in zip(STAGES, paths, expected):
        require(sha256(path) == digest, f'SHA mismatch: {path}')
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        require(payload['step'] == (2500 if stage.endswith(('source','initial')) else 5000), f'Step mismatch: {path}')
        if stage in ('sparse_final','dense_final'):
            m = sm if stage == 'sparse_final' else dm
            require(payload['identity'] == m['identity'] and payload['mask_hash'] == m['mask_hash'], 'Endpoint identity mismatch')
        else:
            require(payload['config']['seed'] == seed, 'Checkpoint seed mismatch')
            cfg = payload['config']['model']
            protocol = {k: payload['config']['training'][k] for k in ['batch_size','learning_rate','weight_decay','grad_clip','precision']}
            if training is None: training = protocol
            require(training == protocol, 'Source training protocol mismatch')
            require(cfg['num_experts'] == experts, 'Architecture mismatch')
            if config is None: config = cfg
            require(config == cfg, 'Dense/sparse architecture mismatch')
        state = payload['model']
        selected = sorted(n for n in state if PATTERN.fullmatch(n))
        if names is None: names = selected
        require(names == selected, 'Tensor identity mismatch')
        for name in names:
            x = state[name].detach().numpy()
            if stage == 'dense_source': shapes[name] = x.shape
            require(shapes[name] == x.shape, f'Tensor shape mismatch: {name}')
            require(np.isfinite(x).all(), f'Nonfinite weights: {name}')
            if stage.startswith('sparse') and name in masks:
                require(tuple(masks[name].shape) == x.shape, 'Mask shape mismatch')
                require(np.all(x[~masks[name].numpy()] == 0), 'Masked positions are nonzero')
            np.save(cache/f'{stage}_{name}.npy', x)
        audit.append(dict(stage=stage, path=str(path), sha256=digest, step=payload['step']))
        del state, payload
    require(set(masks) == {n for n in names if '.experts.' in n}, 'Unexpected mask coverage')
    summaries=[]; comparisons=[]; growth=[]; heterogeneity=[]
    grouped={}
    for name in names:
        match=PATTERN.fullmatch(name); layer, expert, matrix=match.groups()
        grouped.setdefault((layer, matrix or 'router'), []).append(name)
    jobs = [(case, destination, cache, {n:masks[n] for n in members if n in masks}, layer, matrix, members, make_plots)
            for (layer,matrix),members in grouped.items()]
    if workers == 1:
        results = map(analyze_group, jobs)
        for ss, dd, gg, hh in results:
            summaries.extend(ss); comparisons.extend(dd); growth.extend(gg); heterogeneity.extend(hh)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for ss, dd, gg, hh in executor.map(analyze_group, jobs):
                summaries.extend(ss); comparisons.extend(dd); growth.extend(gg); heterogeneity.extend(hh)
    for label, rows in [('statistics',summaries),('distances',comparisons),('growth',growth),('heterogeneity',heterogeneity)]: export(destination,label,rows)
    gap=sm['extension_final_validation_cross_entropy']-dm['extension_final_validation_cross_entropy']
    result=dict(case=case, status='complete', config=config, checkpoints=audit, mask_path=str(mask_path), mask_hash=mh,
                original_recovery_router_checkpoint=original['router_checkpoint'], original_recovery_router_step=original['router_step'],
                protocol=identity, training_hyperparameters=training, canonical_dense_condition=canonical['dense_condition_id'], sparse_minus_dense_ce_5000=gap, matches_or_beats_dense=gap<=0,
                sparse_ce=sm['extension_final_validation_cross_entropy'], dense_ce=dm['extension_final_validation_cross_entropy'])
    (destination/'provenance.json').write_text(json.dumps(result,indent=2))
    # Cache is a reproducible scratch artifact, not needed after analysis.
    if not keep_cache:
        for p in cache.glob('*.npy'): p.unlink()
        cache.rmdir()
    return result, growth, heterogeneity



def analyze_group(job):
    case, destination, cache, masks, layer, matrix, members, make_plots = job
    summaries=[]; comparisons=[]; growth=[]; heterogeneity=[]
    arrays={}; keeps={}
    for name in members:
        arrays[name]=[np.load(cache/f'{s}_{name}.npy', mmap_mode='r').ravel() for s in STAGES]
        keeps[name]=masks[name].numpy().ravel() if name in masks else np.ones(arrays[name][0].size, bool)
    # Layer pooling respects matrix type and includes every expert with equal entry weight.
    if matrix != 'router':
        arrays['pooled']= [np.concatenate([arrays[n][i] for n in members]) for i in range(4)]
        keeps['pooled']=np.concatenate([keeps[n] for n in members])
    local_growth=[]
    for name, data in arrays.items():
        keep=keeps[name]; base=dict(case=case, layer=int(layer), matrix=matrix, parameter=name)
        for i,(stage,x) in enumerate(zip(STAGES,data)):
            for subset, values in [('all',x),('surviving', x[keep]),('nonzero',x[x!=0])]:
                summaries.append(dict(base, stage=stage, subset=subset, masked_fraction=float((~keep).mean()) if i in (1,2) else 0., natural_zero_count=int(np.sum((x==0)&keep)), **statistics(values)))
        profiles = {subset: [empirical_profile(a if subset == 'all' else a[keep]) for a in data] for subset in ('all','surviving')}
        for i,j in itertools.combinations(range(4),2):
            for subset in ('all','surviving'):
                comparisons.append(dict(base, first=STAGES[i], second=STAGES[j], subset=subset, **profile_distances(profiles[subset][i],profiles[subset][j])))
        del profiles
        for i,j in [(1,2),(0,3)]:
            row=dict(base, first=STAGES[i], second=STAGES[j], subset='surviving', **changes(data[i][keep],data[j][keep]))
            growth.append(row)
            if i==1 and name!='pooled': local_growth.append(row)
    for field in ['mean_abs_growth_ratio','normalized_drift','mean_abs_change']:
        vals=np.array([r[field] for r in local_growth if r[field] is not None])
        if vals.size:
            heterogeneity.append(dict(case=case, layer=int(layer), matrix=matrix, measure=field, expert_count=len(vals), mean=float(vals.mean()), std=float(vals.std()), minimum=float(vals.min()), maximum=float(vals.max())))
    if make_plots:
        plots=destination/f'layer_{layer}_{matrix}'; plots.mkdir(exist_ok=True)
        plot_groups(plots, list(arrays), arrays, keeps, f'{case} — layer {layer}, {matrix}; raw weights, shared axes')
    print(f'{case}: layer {layer} {matrix} complete', flush=True)
    del arrays, keeps
    return summaries, comparisons, growth, heterogeneity



def router_outcome_plots(output, reports):
    complete = [r for r in reports if r['status']=='complete']
    if not complete: return
    extracted = {}
    for report in complete:
        for stage in ('sparse_initial','sparse_final'):
            path = next(r['path'] for r in report['checkpoints'] if r['stage']==stage)
            payload = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            extracted[(report['case'],stage)] = {int(PATTERN.fullmatch(n).group(1)):t.numpy().ravel().copy()
                for n,t in payload['model'].items() if '.moe.router.projection.weight' in n and PATTERN.fullmatch(n)}
            del payload
    layers = sorted(set(layer for values in extracted.values() for layer in values))
    for stage in ('sparse_initial','sparse_final'):
        fig, axes = plt.subplots(len(layers),3,figsize=(12,2.8*len(layers)),squeeze=False)
        for row,layer in enumerate(layers):
            limit=max(float(np.max(np.abs(v[layer]))) for v in extracted.values() if layer in v)*1.02
            bins=np.linspace(-limit,limit,81)
            for report in complete:
                values=extracted[(report['case'],stage)]
                if layer not in values: continue
                x=values[layer]
                label=report['case'] + (' (beats dense)' if report['matches_or_beats_dense'] else ' (higher CE)')
                axes[row,0].hist(x,bins=bins,density=True,histtype='step',lw=1,label=label)
                for col, data in ((1,x),(2,np.abs(x))):
                    ordered,counts=np.unique(data,return_counts=True)
                    axes[row,col].step(ordered,np.cumsum(counts)/len(data),where='post',lw=1)
            for col in range(3):
                axes[row,col].set(xlim=(0,limit) if col==2 else (-limit,limit),
                                  xlabel='|weight|' if col==2 else 'weight',ylabel='density' if col==0 else 'empirical probability',
                                  title=f'layer {layer}: '+['histogram','signed CDF','absolute CDF'][col])
                axes[row,col].grid(alpha=.2)
            axes[row,0].legend(fontsize=5)
        fig.suptitle(f'Router outcome comparison: {stage}; raw weights, shared axes')
        fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(output/f'router_outcomes_{stage}.png',dpi=220)
        fig.savefig(output/f'router_outcomes_{stage}.pdf'); plt.close(fig)


def summarize_outcomes(output, reports):
    """Describe selected cases without treating layers/experts as independent trials."""
    rows = []; router_totals = []
    for report in reports:
        if report['status'] != 'complete': continue
        directory = output/report['case']
        stats = read(directory/'statistics.json')
        growth = read(directory/'growth.json')
        shifts = read(directory/'distances.json')
        for first in ('sparse_initial', 'dense_source'):
            initial_norms = {r['parameter']:r['l2'] for r in stats if r['matrix']=='router' and r['stage']==first and r['subset']=='all'}
            components = [r for r in growth if r['matrix']=='router' and r['first']==first]
            delta_squared = sum((r['normalized_drift']*initial_norms[r['parameter']])**2 for r in components)
            norm_squared = sum(v*v for v in initial_norms.values())
            normalized = float(np.sqrt(delta_squared/norm_squared)) if norm_squared else None
            endpoint = next(r['path'] for r in report['checkpoints'] if r['stage']==('sparse_final' if first=='sparse_initial' else 'dense_final'))
            manifest = read(Path(endpoint).parent.parent/'extension_manifest.json')
            recorded = manifest['trainer_record']['router_parameter_drift_normalized_final']
            require(np.isclose(normalized,recorded,rtol=1e-6,atol=1e-8), 'Combined router drift disagrees with manifest')
            router_totals.append(dict(case=report['case'], first=first, normalized_router_drift=normalized,
                                      absolute_router_drift=float(np.sqrt(delta_squared)), manifest_normalized_drift=recorded,
                                      success=report['matches_or_beats_dense'], ce_gap=report['sparse_minus_dense_ce_5000']))
        for matrix in ('router','fc1','fc2'):
            valid = lambda r: r['matrix'] == matrix and r['parameter'] != 'pooled'
            initial = [r['l1']/r['count'] for r in stats if valid(r) and r['stage']=='sparse_initial' and r['subset']=='surviving' and r['count']]
            changes_ = [r for r in growth if valid(r) and r['first']=='sparse_initial']
            distances_ = [r for r in shifts if valid(r) and r['first']=='sparse_initial' and r['second']=='sparse_final' and r['subset']=='surviving']
            row = dict(case=report['case'], matrix=matrix, success=report['matches_or_beats_dense'], ce_gap=report['sparse_minus_dense_ce_5000'])
            row['mean_initial_survivor_magnitude'] = float(np.mean(initial))
            for label, values in [('growth_ratio',[r['mean_abs_growth_ratio'] for r in changes_]),
                                  ('normalized_drift',[r['normalized_drift'] for r in changes_]),
                                  ('survivor_wasserstein',[r['wasserstein'] for r in distances_]),
                                  ('survivor_ks',[r['ks'] for r in distances_])]:
                values = [v for v in values if v is not None]
                row['mean_'+label] = float(np.mean(values)) if values else None
                row['sd_'+label] = float(np.std(values)) if values else None
            rows.append(row)
    export(output, 'exploratory_summary', rows)
    export(output, 'router_total_drift', router_totals)
    notes = ['# Exploratory outcome comparisons', '',
             'Each row averages observed layer/expert matrices within one configuration. Standard deviations describe heterogeneity; they are not uncertainty estimates. No weights or experts are treated as independent experimental replicates.', '',
             '| Case | Matrix | CE gap | Initial mean survivor magnitude | Mean magnitude growth ratio | Mean normalized drift | Mean survivor W1 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        notes.append(f"| {r['case']} | {r['matrix']} | {r['ce_gap']:.6f} | {r['mean_initial_survivor_magnitude']:.6g} | {r['mean_growth_ratio']:.4f} | {r['mean_normalized_drift']:.4f} | {r['mean_survivor_wasserstein']:.6g} |")
    notes += ['', 'Negative CE gap denotes a sparse endpoint that beats its matched dense endpoint. These are descriptive comparisons of four selected configurations. Architecture, seed and sparsity differ across the outcome groups, so apparent differences in survivor magnitudes, growth, shifts or heterogeneity cannot be attributed to success itself. See the per-layer heterogeneity and distance tables for variation hidden by these averages. Distributional results do not establish geometric alignment.']
    (output/'exploratory_findings.md').write_text('\n'.join(notes)+'\n')
    router_outcome_plots(output, reports)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,default=Path('results/weight_distribution_analysis'))
    parser.add_argument('--workers',type=int,default=4,help='CPU workers after sequential checkpoint extraction')
    parser.add_argument('--cases',nargs='+',default=DEFAULTS)
    args=parser.parse_args(); torch.set_num_threads(1)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    reports=[]; router=[]; experts=[]
    for case in args.cases:
        try:
            result,growth,heterogeneity=analyze(args.root.resolve(),output,case,args.workers)
            reports.append(result)
            router.extend(dict(r, success=result['matches_or_beats_dense'], ce_gap=result['sparse_minus_dense_ce_5000']) for r in growth if r['matrix']=='router')
            experts.extend(dict(r, success=result['matches_or_beats_dense'], ce_gap=result['sparse_minus_dense_ce_5000']) for r in heterogeneity if r['matrix']!='router')
        except (FileNotFoundError, ValueError, KeyError) as exc:
            reports.append(dict(case=case,status='missing_or_incompatible',error=str(exc)))
            print(f'{case}: {exc}',flush=True)
        export(output,'checkpoint_audit',reports)
    export(output,'router_comparison',router); export(output,'expert_heterogeneity_comparison',experts)
    if router:
        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        for r in router:
            if r['first'] != 'sparse_initial': continue
            color = '#0072b2' if r['success'] else '#d55e00'
            axes[0].scatter(r['ce_gap'], r['normalized_drift'], color=color, s=25)
        axes[0].set(xlabel='sparse − dense CE at 5000', ylabel='router normalized drift')
        for r in experts:
            if r['measure'] != 'mean_abs_growth_ratio': continue
            axes[1].scatter(r['ce_gap'], r['std'], color='#0072b2' if r['success'] else '#d55e00', marker='o' if r['matrix']=='fc1' else '^', s=25)
        axes[1].set(xlabel='sparse − dense CE at 5000', ylabel='across-expert SD of magnitude growth ratio')
        for ax in axes: ax.axvline(0, color='gray', ls=':', lw=1); ax.grid(alpha=.2)
        fig.suptitle('Exploratory outcomes: blue matches/beats dense; orange higher CE')
        fig.tight_layout(); fig.savefig(output/'outcome_comparison.png',dpi=220); fig.savefig(output/'outcome_comparison.pdf'); plt.close(fig)
    (output/'README.md').write_text(README + '\n\nRun results:\n\n' + '\n'.join(f"- `{r['case']}`: {r['status']}" + (f"; sparse − dense CE = {r['sparse_minus_dense_ce_5000']:.6f}" if r['status']=='complete' else f"; {r['error']}") for r in reports))
    summarize_outcomes(output, reports)
    if any(r['status']!='complete' for r in reports): raise SystemExit(1)


README = '''# Router and expert weight distributions

Run from repository root with `.venv-highres/bin/python -m moe_lth.analysis.weight_distributions` (set `PYTHONPATH=src` if not installed). Override `--cases`, `--root`, or `--output` as needed. Requires NumPy, SciPy, Matplotlib and PyTorch. Checkpoints are read-only; only one checkpoint is mapped at a time. Four CPU workers process extracted layer/matrix groups; use `--workers 1` for minimum memory. Scratch extracted matrices are deleted after each case. Statistics and distances use every weight, without sampling or rescaling.

## Checkpoints and verification

Each case uses its extension manifest and the matched dense manifest in the same architecture/seed directory. `provenance.json` records exact paths and verified SHA256s: dense source at step 2500, sparse beginning of extension at step 2500, sparse endpoint at 5000, and matched dense endpoint at 5000. The original IMP trajectory stage and mask_08 (80%) or mask_09 (85%) are checked against the source and mask hashes. The sparse beginning here is the recovered subnetwork before *extended continuation*, not immediately after pruning. Original recovery router provenance is also recorded. These are fresh AdamW/GradScaler continuations, not exact optimizer resumes. Training/validation sequence hashes and protocol identities must match. Success means sparse validation CE <= matched dense validation CE at 5000, with no tolerance or statistical-significance claim. Missing or incompatible inputs are reported, never replaced.

## Plots

Each layer/matrix directory contains numbered PNG/PDF sheets, with one row per expert (and a pooled row), or the layer router. The actual architecture uses fc1 = up projection followed by GELU and fc2 = down projection; there is no gated FFN or gate matrix. Biases and router temperature buffers are excluded. Experts are identified by layer and index, never matched by weight similarity.

Columns 1–3 show normalized signed histograms, signed empirical CDFs, and absolute empirical CDFs. Columns 4–6 repeat these with sparse exact zeros excluded (dense curves retain all entries). All four stages are overlaid, including initial/final sparse and final dense/sparse comparisons. Histogram area is one; sparse zero atoms are represented by a finite-width bin and must not be read as continuous density. CDF F(t) = count(w <= t)/N; absolute CDF uses |w|. No Gaussian assumption or KDE is used because sparse distributions contain point masses. CDF rendering retains up to 1500 ordered points plus zero and endpoints; computed statistics and distances are exact empirical calculations. Axes and histogram bins are shared across stages and experts within each layer/matrix type; raw weights are never standardized. Pooling is within layer and matrix type. Router distributions remain separate from expert distributions.

## Tables and interpretation

Each case exports CSV and JSON statistics, distances, growth, and heterogeneity. Standard deviation is population SD; absolute quantiles are 25/50/75/90/95/99 percentiles. L1 = sum |w| and L2 = sqrt(sum w²). All, surviving (mask true, including natural zeros), and nonzero (exact w != 0) subsets are distinct. Dense surviving subsets use the paired sparse mask coordinates; masked_fraction is zero for dense weights. natural_zero_count counts exact zeros at kept coordinates, not pruned zeros.

Distances cover every pair of stages for comparable layer/matrix/expert identities: Wasserstein-1 in weight units and KS = sup_t |F(t)-G(t)|, for signed and absolute weights. All-weight and fixed-mask-survivor comparisons are explicitly labeled. No KS p-values or independent-weight inference is claimed. Growth uses paired surviving coordinates: mean(|final|-|initial|), ratio of mean absolute magnitudes, fraction that grew, and change per 2500 attempted steps. Normalized parameter drift = ||final-initial||₂/||initial||₂; undefined zero-denominator quantities are null. Dense growth is also measured at the same sparse-mask coordinates. Router drift uses all router coordinates.

Heterogeneity summarizes population SD and range across experts of growth and drift, separately per layer and projection. Root router and expert comparison tables attach matched CE outcomes; `router_total_drift` combines all router layers and cross-checks normalized drift against the training manifests. `exploratory_findings.md` and `exploratory_summary` report configuration-level descriptive comparisons. The outcome plot shows layer router drift and per-layer expert growth heterogeneity against matched CE gaps. Router outcome sheets overlay initial and final router distributions across the selected cases, with shared raw weight limits. All comparisons are exploratory. Small selected configuration counts, dependent weights, different sparsities, and architecture/seed confounding prevent causal claims. Weight distribution similarity is distinct from geometric or subspace alignment and supplies no evidence of alignment.
'''

if __name__=='__main__': main()
