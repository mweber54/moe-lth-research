"""Read-only retrospective RAMP–IMP analysis. CPU inference only; no trainer imports.

Run from repo root: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python
scripts/ramp_imp_correspondence_v1/run.py --output results/ramp_imp_correspondence_v1
Existing output is resumable only when registered settings and input hashes match.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from moe_lth.analysis.ramp import profile, neuron_mask_statistics
from moe_lth.data import load_token_splits, TokenBlockDataset
from moe_lth.pruning.router_age import load_model_from_checkpoint, state_dict_hash

PRIMARY = ROOT / 'results/imp_80_85/final_recovery_metrics.csv'
MASKS = ROOT / 'results/imp_80_85/imp_mask_manifest.csv'
TIMES = (0, 250, 1250, 2500)
FEATURES = ['ramp', 'random', 'utilization', 'activation_only', 'down_norm', 'routing_activation',
            'initial_magnitude', 'checkpoint_magnitude', 'ramp_per_dispatch', 'ramp_per_route_energy']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def records(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def register(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'Registration mismatch; use a new output directory: {path}')
    else:
        write_json(path, value)


def curve(path):
    return {int(r['step']): float(r['loss']) for line in Path(path).read_text().splitlines()
            if 'loss' in (r := json.loads(line))}


def calibrate(config, out, sequences, seed):
    """Canonical block coordinates; exact byte-sequence validation disjointness."""
    if sequences % 2 or sequences < 2:
        raise ValueError('Use an even calibration sequence count')
    dc = config['data']
    if dc['tokenizer'] != 'byte':
        raise ValueError('This calibration adapter requires the audited byte tokenizer')
    train, validation = load_token_splits(dc)
    length = int(dc['seq_len'])
    ds = TokenBlockDataset(train, length)
    validation_bytes = bytes(validation)
    meta = json.loads((ROOT / 'data/multidomain_balanced/metadata.json').read_text())
    sources = {domain: Path(meta['train'][key]) for domain, key in [('tinystories', 'tiny_path'), ('wikitext', 'wiki_path')]}
    source_bytes = {domain: path.read_text(encoding='utf-8').encode('utf-8') for domain, path in sources.items()}
    rng = np.random.default_rng(seed)
    selected, counts, seen = [], dict.fromkeys(sources, 0), set()
    for index in rng.permutation(len(ds)):
        x, _ = ds[int(index)]
        raw = bytes(x.tolist())
        if raw in validation_bytes or raw in seen:
            continue
        membership = [(domain, data.find(raw)) for domain, data in source_bytes.items() if raw in data]
        if len(membership) != 1:
            continue
        domain, offset = membership[0]
        if counts[domain] >= sequences // 2:
            continue
        selected.append(dict(block_index=int(index), byte_start=int(index) * length,
                             domain=domain, source_path=str(sources[domain]), source_byte_offset=offset,
                             sequence_sha256=hashlib.sha256(raw).hexdigest(), token_ids=x.tolist()))
        counts[domain] += 1
        seen.add(raw)
        if len(selected) == sequences:
            break
    if len(selected) != sequences:
        raise RuntimeError('Insufficient nonoverlapping unambiguous calibration blocks')
    # Fixed, balanced batches; domain labels do not trigger separate capacity passes.
    selected = [selected[i] for i in rng.permutation(len(selected))]
    batch_size = min(32, sequences)
    tensors = [torch.tensor([r['token_ids'] for r in selected[i:i + batch_size]]) for i in range(0, sequences, batch_size)]
    labels = [[r['domain'] for r in selected[i:i + batch_size]] for i in range(0, sequences, batch_size)]
    manifest = dict(schema=1, dataset='canonical Dataset A mixed training split', selection_seed=seed,
                    calibration_split='train', evaluation_split='validation', selection='uniform block permutation; equal source quotas; no outcome access',
                    tokenizer='moe_lth.data.ByteTokenizer UTF-8 byte IDs 0..255', tokenizer_version_sha256=sha(ROOT / 'src/moe_lth/data.py'),
                    sequence_length=length, sequences=sequences, tokens=sequences * length, batch_size=batch_size,
                    domain_counts=counts, data_config=dc, source_files={str(p): sha(p) for p in [Path(dc['train_path']), Path(dc['validation_path']), *sources.values()]},
                    full_validation_sequence_overlap_count=0, overlap_rule='reject exact input sequence anywhere in full validation byte stream',
                    token_sha256=hashlib.sha256(b''.join(t.numpy().tobytes() for t in tensors)).hexdigest(),
                    domain_semantics='exact unique source substring membership; no semantic specialization inference',
                    samples=selected)
    register(out / 'calibration_manifest.json', manifest)
    return tensors, labels, manifest


def check_pair(row, maskrow):
    init = Path(row['original_initialization_checkpoint'])
    dense = Path(row['matched_dense_checkpoint'])
    sparse = Path(row['trained_checkpoint'])
    mask = Path(maskrow['mask_path'])
    for path in (init, dense, sparse, mask, Path(maskrow['trained_checkpoint'])):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha(init) != row['initial_sha256']:
        raise RuntimeError('Original initialization file hash mismatch')
    loaded = torch.load(mask, map_location='cpu', weights_only=True)
    if state_dict_hash(loaded) != row['mask_hash']:
        raise RuntimeError('Mask tensor hash mismatch')
    dense_meta = json.loads((dense.parent.parent / 'metadata.json').read_text())
    sparse_stage = json.loads((sparse.parent.parent / 'stage.json').read_text())
    if sha(sparse) != sparse_stage['checkpoint_sha256']:
        raise RuntimeError('Sparse checkpoint hash mismatch')
    for k in ['training_batch_sequence_hash', 'validation_batch_sequence_hash']:
        if dense_meta[k] != row[k] or sparse_stage[k] != row[k]:
            raise RuntimeError(f'Unmatched protocol: {k}')
    a = curve(row['curves_path'])[2500]
    b = curve(dense.parent.parent / 'metrics.jsonl')[2500]
    if abs(a - float(row['final_validation_loss'])) > 1e-8 or abs(b - float(row['dense_validation_ce'])) > 1e-8:
        raise RuntimeError('Curve endpoint differs from result table')
    return loaded, dict(architecture=f"E{row['num_experts']}", seed=int(row['seed']), sparsity=float(row['sparsity']),
                        mask_id=row['mask_hash'], initialization_id=row['initial_sha256'], rewind_step=0,
                        discovery_run=f"dataset-a-imp-r0-v1/E{row['num_experts']}/seed{row['seed']}",
                        recovery_run=str(sparse.parent.parent), mask_path=str(mask), initial_checkpoint=str(init),
                        sparse_checkpoint=str(sparse), paired_dense_checkpoint=str(dense),
                        sparse_loss=a, dense_loss=b, recovery_gap=(a-b)/b,
                        mask_file_sha256=sha(mask), sparse_file_sha256=sha(sparse), dense_file_sha256=sha(dense),
                        discovery_source_checkpoint=maskrow['trained_checkpoint'], train_hash=row['training_batch_sequence_hash'],
                        validation_hash=row['validation_batch_sequence_hash'], protocol=row['protocol'])


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(stats.rankdata(x[ok]), stats.rankdata(y[ok]))[0, 1])


def group_metrics(frame, identity, scope, layer, expert):
    rows, deciles = [], []
    targets = ['retention', 'up_retention', 'down_retention']
    for feature in FEATURES:
        x = frame[feature].to_numpy()
        r = dict(identity, scope=scope, layer=layer, expert=expert, feature=feature, neurons=len(x))
        r.update({target + '_rho': correlation(x, frame[target]) for target in targets})
        # Ties receive shared midrank deciles and fractional top-decile inclusion.
        # This avoids assigning arbitrary neuron IDs to high-importance groups.
        if scope == 'within_layer':
            y = frame.retention.to_numpy()
            order = stats.rankdata(x) / len(x)
            high = order >= .9
            truth = y >= np.quantile(y, .9)
            r['high_retention_auc'] = roc_auc_score(truth, x) if 0 < truth.sum() < len(truth) else np.nan
            r['high_retention_average_precision'] = average_precision_score(truth, x) if truth.any() else np.nan
            r['high_retention_prevalence'] = truth.mean()
            r['top_decile_enrichment'] = y[high].mean() / y.mean() if high.any() and y.mean() else np.nan
            bins = np.minimum(9, np.floor((stats.rankdata(x) - .5) / len(x) * 10).astype(int))
            for decile in range(10):
                sub = frame[bins == decile]
                deciles.append(dict(identity, scope=scope, layer=layer, feature=feature, decile=decile + 1,
                                    neurons=len(sub), retention=sub.retention.mean(), up_retention=sub.up_retention.mean(), down_retention=sub.down_retention.mean()))
        rows.append(r)
    return rows, deciles


def analyze_pair(frame, identity):
    groups, deciles = [], []
    for layer, layer_df in frame.groupby('layer', sort=True):
        r, d = group_metrics(layer_df, identity, 'within_layer', layer, -1)
        groups.extend(r); deciles.extend(d)
        for expert, expert_df in layer_df.groupby('expert', sort=True):
            r, _ = group_metrics(expert_df, identity, 'within_expert', layer, expert)
            groups.extend(r)
    gs = pd.DataFrame(groups)
    summary = dict(identity)
    ret = frame.retention.to_numpy()
    count = frame.surviving_weights.to_numpy()
    for feature in FEATURES:
        x = frame[feature].to_numpy()
        den = x.sum()
        summary[feature + '_weighted_retention'] = float(x @ ret / den) if den > 0 else np.nan
        summary[feature + '_retained_mean'] = float(x @ count / count.sum()) if count.sum() else np.nan
        summary[feature + '_strong_retention_share'] = float(x[ret >= .5].sum() / den) if den else np.nan
        for scope in ('within_layer', 'within_expert'):
            q = gs[(gs.feature == feature) & (gs.scope == scope)]
            summary[feature + '_' + scope + '_rho'] = q.retention_rho.mean()
            summary[feature + '_' + scope + '_valid_groups'] = int(q.retention_rho.notna().sum())
    exp = frame.groupby(['layer', 'expert']).ramp.sum()
    summary['expert_ramp_cv'] = exp.std(ddof=0) / exp.mean() if exp.mean() else np.nan
    summary['unexposed_expert_fraction'] = float((frame.groupby(['layer', 'expert']).utilization.first() == 0).mean())
    return gs, pd.DataFrame(deciles), summary


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    rows = records(PRIMARY)
    if len(rows) != 30:
        raise RuntimeError('Expected all 30 prespecified historical recoveries')
    maskrows = {r['mask_hash']: r for r in records(MASKS)}
    registry = dict(experiment='ramp_imp_correspondence_v1', calibration_sequences=args.sequences,
                    calibration_seed=20261113, profiling_device='cpu', profiling_precision='fp32 activations / fp64 reductions',
                    torch_version=torch.__version__, numpy_version=np.__version__, times=list(TIMES),
                    features=FEATURES, training_enabled=False, primary_table_sha256=sha(PRIMARY), mask_table_sha256=sha(MASKS),
                    source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'src/moe_lth/analysis/ramp.py', ROOT/'src/moe_lth/models/moe_layer.py', ROOT/'src/moe_lth/models/experts.py', ROOT/'src/moe_lth/models/transformer.py', ROOT/'src/moe_lth/models/router.py']},
                    h2='descriptive only; n=5 source seeds per architecture/sparsity; no fitted multivariable predictor',
                    feature_timing='checkpoint_magnitude uses current checkpoint; no later weight feature enters early profiles; mask-conditioned H2 is retrospective',
                    h1_unit='equal-expert or equal-layer mean per run, then seed uncertainty; no neuron p-values',
                    h3='prepared only, never executed')
    register(out/'registration.json', registry)
    groups = {}
    for row in rows:
        groups.setdefault((int(row['num_experts']), int(row['seed'])), []).append(row)
    first = torch.load(rows[0]['original_initialization_checkpoint'], map_location='cpu', weights_only=True)
    batches, labels, calibration = calibrate(first['config'], out, args.sequences, registry['calibration_seed'])
    del first
    verified = []
    for (experts, seed), pairrows in sorted(groups.items()):
        folder = out / f'e{experts}_seed{seed}'
        folder.mkdir(exist_ok=True)
        masks, identities = [], []
        for row in sorted(pairrows, key=lambda r: float(r['sparsity'])):
            mask, identity = check_pair(row, maskrows[row['mask_hash']])
            masks.append(mask); identities.append(identity); verified.append(identity)
        register(folder/'input_manifest.json', identities)
        initial = torch.load(pairrows[0]['original_initialization_checkpoint'], map_location='cpu', weights_only=True)
        if initial['step'] != 0 or initial['config']['data'] != calibration['data_config']:
            raise RuntimeError('Initial step or data configuration mismatch')
        config = initial['config']
        if config['model']['num_experts'] != experts or config['seed'] != seed:
            raise RuntimeError('Saved model identity mismatch')
        plans = [(f'reference_step_{t}', Path(identities[0]['initial_checkpoint']).parent/f'step_{t}.pt', t) for t in TIMES]
        # Direct unpruned ancestor of IMP, distinct from the reference trajectory.
        plans.append(('discovery_dense_final', Path(identities[0]['mask_path']).parent/'round_00/checkpoints/final_recovered.pt', 2500))
        checkpoint_inventory = []
        for name, path, step in plans:
            if not path.is_file():
                checkpoint_inventory.append(dict(name=name, path=str(path), available=False))
                continue
            digest = sha(path)
            checkpoint_inventory.append(dict(name=name, path=str(path), available=True, sha256=digest, step=step))
            dest = folder / name
            dest.mkdir(exist_ok=True)
            register(dest/'checkpoint.json', dict(path=str(path), sha256=digest, step=step, model_config=config['model'], calibration_sha256=calibration['token_sha256']))
            if (dest/'complete.json').exists():
                complete = json.loads((dest/'complete.json').read_text())
                for fname, expected in complete['files'].items():
                    if sha(dest/fname) != expected:
                        raise RuntimeError(f'Corrupt completed result: {dest/fname}')
                print('verified cached', experts, seed, name, flush=True)
                continue
            print('profiling', experts, seed, name, flush=True)
            model = load_model_from_checkpoint(config['model'], str(path), torch.device('cpu'))
            accumulation = profile(model, batches, labels)
            current = model.state_dict()
            # Repeated checkpoint/calibration determinism is validated on the first real checkpoint.
            if experts == 8 and seed == 7 and step == 0:
                repeat = profile(model, batches, labels)
                if any(not torch.equal(a.weighted_square, repeat[k].weighted_square) for k, a in accumulation.items()):
                    raise RuntimeError('Historical checkpoint repeat profiling is not bit deterministic')
            profile_frames = []
            for (domain, layer, expert), a in sorted(accumulation.items()):
                down = current[f'blocks.{layer}.moe.experts.{expert}.fc2.weight']
                scores = {k: v.numpy() for k, v in a.scores(down).items()}
                size = len(scores['ramp'])
                profile_frames.append(pd.DataFrame(dict(domain=domain, layer=layer, expert=expert,
                    neuron=np.arange(size), utilization=a.count, routing_exposure=a.probability_sum,
                    route_energy=a.probability_square, **scores)))
            profiles = pd.concat(profile_frames, ignore_index=True)
            profiles.to_parquet(dest/'profiles.parquet', index=False)
            h1, decile, runrows = [], [], []
            for mask, identity in zip(masks, identities):
                mapping = neuron_mask_statistics(mask, initial['model'], current)
                mapping_df = pd.concat([pd.DataFrame(dict(layer=l, expert=e, neuron=np.arange(len(v['retention'])), **{k: a.numpy() for k, a in v.items()})) for (l,e),v in mapping.items()], ignore_index=True)
                mapping_df = mapping_df.rename(columns={'final_magnitude': 'checkpoint_magnitude'})
                for domain, pdf in profiles.groupby('domain', sort=True):
                    merged = pdf.merge(mapping_df, on=['layer', 'expert', 'neuron'], validate='one_to_one')
                    # Same deterministic random ranking for a neuron at every timing/mask/domain.
                    random = np.random.default_rng(20261113 + experts * 100 + seed).random(len(merged))
                    merged['random'] = random
                    ident = dict(identity, profile_name=name, profile_step=step, profile_checkpoint=str(path),
                                 profile_checkpoint_sha256=digest, domain=domain, calibration_sha256=calibration['token_sha256'])
                    gs, ds, rs = analyze_pair(merged, ident)
                    h1.append(gs); decile.append(ds); runrows.append(rs)
                    # Consolidated pairing rows; domain subsets reside in profiles and H1/H2 tables.
                    if domain == 'pooled':
                        for k in ['architecture', 'seed', 'sparsity', 'mask_id', 'initialization_id', 'rewind_step', 'discovery_run', 'recovery_run', 'profile_name', 'profile_step', 'profile_checkpoint']:
                            merged[k] = ident[k]
                        merged.to_parquet(dest/f"neurons_s{identity['sparsity']:.2f}.parquet", index=False)
            pd.concat(h1, ignore_index=True).to_parquet(dest/'h1_groups.parquet', index=False)
            pd.concat(decile, ignore_index=True).to_parquet(dest/'deciles.parquet', index=False)
            pd.DataFrame(runrows).to_parquet(dest/'mask_runs.parquet', index=False)
            files = {p.name: sha(p) for p in dest.glob('*.parquet')}
            write_json(dest/'complete.json', dict(files=files, native_routing=True))
            del model, accumulation, current
        register(folder/'checkpoint_inventory.json', checkpoint_inventory)
    pd.DataFrame(verified).to_csv(out/'verified_recovery_pairs.csv', index=False)
    summarize(out)


def summarize(out):
    runpaths = sorted(out.glob('e*_seed*/*/mask_runs.parquet'))
    runs = pd.concat([pd.read_parquet(p) for p in runpaths], ignore_index=True)
    runs.to_csv(out/'mask_run_results.csv', index=False)
    all_groups = pd.concat([pd.read_parquet(p) for p in sorted(out.glob('e*_seed*/*/h1_groups.parquet'))], ignore_index=True)
    all_groups.to_parquet(out/'h1_groups.parquet', index=False)
    keys = ['architecture', 'sparsity', 'profile_name', 'profile_step', 'domain', 'scope', 'feature']
    run_group = all_groups.groupby(keys+['seed'], dropna=False)[['retention_rho','up_retention_rho','down_retention_rho','top_decile_enrichment','high_retention_auc','high_retention_average_precision']].mean().reset_index()
    run_group.to_csv(out/'h1_seed_results.csv', index=False)
    summaries = []
    for key, df in run_group.groupby(keys, dropna=False):
        ident = dict(zip(keys, key))
        for metric in ['retention_rho','up_retention_rho','down_retention_rho','top_decile_enrichment','high_retention_auc','high_retention_average_precision']:
            values = df[metric].dropna().to_numpy()
            n = len(values)
            mean = values.mean() if n else np.nan
            half = stats.t.ppf(.975, n-1) * values.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan
            summaries.append(dict(ident, metric=metric, independent_seeds=n, mean=mean, ci95_low=mean-half, ci95_high=mean+half,
                                  uncertainty='descriptive seed t interval; not multiplicity-adjusted'))
    pd.DataFrame(summaries).to_csv(out/'h1_seed_summary.csv', index=False)
    # Paired seed-level RAMP improvements over baselines.
    paired = []
    for key, df in run_group.groupby(keys[:-1], dropna=False):
        rr = df[df.feature=='ramp'].set_index('seed').retention_rho
        for baseline in FEATURES[1:]:
            bb = df[df.feature==baseline].set_index('seed').retention_rho
            diff = (rr-bb).dropna()
            n=len(diff); mean=diff.mean()
            half=stats.t.ppf(.975,n-1)*diff.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
            paired.append(dict(zip(keys[:-1],key), baseline=baseline, independent_seeds=n, mean_rho_difference=mean, ci95_low=mean-half,ci95_high=mean+half))
    pd.DataFrame(paired).to_csv(out/'h1_baseline_differences.csv', index=False)
    # Fixed univariate summaries, stratified protocols. No model fitting with five seeds.
    assoc=[]
    for key, df in runs.groupby(['architecture','sparsity','profile_name','profile_step','domain']):
        for feature in FEATURES:
            for suffix in ['weighted_retention','retained_mean','strong_retention_share','within_expert_rho','within_layer_rho']:
                column=feature+'_'+suffix
                assoc.append(dict(zip(['architecture','sparsity','profile_name','profile_step','domain'],key), feature=column,
                                  independent_seeds=df.seed.nunique(), spearman_rho=correlation(df[column],df.recovery_gap),
                                  loo_rho_min=min_finite([correlation(df[df.seed!=s][column],df[df.seed!=s].recovery_gap) for s in df.seed]),
                                  loo_rho_max=max_finite([correlation(df[df.seed!=s][column],df[df.seed!=s].recovery_gap) for s in df.seed]),
                                  interpretation='descriptive n5 association and leave-one-seed-out sensitivity; no validated predictor'))
    pd.DataFrame(assoc).to_csv(out/'h2_descriptive_associations.csv',index=False)
    dec=pd.concat([pd.read_parquet(p) for p in sorted(out.glob('e*_seed*/*/deciles.parquet'))], ignore_index=True)
    dec.to_parquet(out/'retention_deciles.parquet',index=False)
    stability=[]
    for folder in sorted(out.glob('e*_seed*')):
        paths=sorted(folder.glob('*/profiles.parquet'))
        by_name={p.parent.name:pd.read_parquet(p).query("domain == 'pooled'").sort_values(['layer','expert','neuron']) for p in paths}
        final=by_name['reference_step_2500']
        for name,df in by_name.items():
            for layer in sorted(df.layer.unique()):
                stability.append(dict(group=folder.name,profile_name=name,layer=int(layer),final_reference='reference_step_2500',ramp_rank_stability=correlation(df[df.layer==layer].ramp,final[final.layer==layer].ramp)))
    pd.DataFrame(stability).to_csv(out/'checkpoint_rank_stability.csv',index=False)
    from reporting import make_figures, make_report, mechanistic_controls
    mechanistic_controls(out, runs)
    make_figures(out, runs, pd.DataFrame(summaries), dec, pd.DataFrame(stability))
    make_report(out, runs, pd.DataFrame(summaries), pd.DataFrame(assoc))
    write_json(out/'completion.json',dict(historical_recoveries=30,independent_architecture_seed_groups=15,
        completed_profiles=len(runpaths),mask_profile_domain_rows=len(runs),failed_validation_checks=0,
        training_launched=False,neurons_layout='e*_seed*/*/neurons_s*.parquet (one pooled neuron/checkpoint/mask row)',
        calibration_sequence_count=json.loads((out/'calibration_manifest.json').read_text())['sequences']))


def min_finite(a):
    a=np.array(a);return float(np.nanmin(a)) if np.isfinite(a).any() else np.nan


def max_finite(a):
    a=np.array(a);return float(np.nanmax(a)) if np.isfinite(a).any() else np.nan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/ramp_imp_correspondence_v1')
    parser.add_argument('--sequences',type=int,default=64)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:
        summarize(args.output.resolve())
    else:
        run(args)
