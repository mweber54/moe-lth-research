"""Dataset A IMP orchestration; existing recovery trainer owns all optimization."""
import argparse
import csv
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/fixed_budget_dataset')]
import torch
import common
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.pruning.imp import SPARSITIES, nested_expert_local_masks
from moe_lth.pruning.masks import expert_weight_parameters, load_masks, save_masks
from moe_lth.pruning.router_age import (load_model_from_checkpoint, parameter_group,
    selected_experts_per_batch, state_dict_hash)
from moe_lth.utils import seed_everything, configure_device

CANONICAL = ROOT / 'results/final_dataset_a/canonical/dataset_a_canonical.csv'
OUT = ROOT / 'results/imp_80_85'


def write_json(path, value):
    common.jwrite(path, value)


def write_csv(path, rows, fields):
    fields = list(dict.fromkeys(fields + [k for row in rows for k in row]))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def canonical_pairs():
    with CANONICAL.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['router_age'] == '0' and float(r['sparsity']) in (.8, .85)]
    pairs = {}
    for row in rows:
        key = (int(row['num_experts']), int(row['seed']), float(row['sparsity']))
        if key in pairs or row['condition_valid'] != 'True':
            raise ValueError(f'Invalid canonical pairing: {key}')
        pairs[key] = row
    if len(pairs) != 50:
        raise ValueError('Expected exactly 50 canonical R0 comparators')
    for e in common.ES:
        for seed in common.SEEDS:
            a, b = pairs[e, seed, .8], pairs[e, seed, .85]
            for field in ('source_checkpoint_path', 'source_checkpoint_hash', 'dense_artifact_path',
                          'dense_final_checkpoint_sha256', 'dense_final_validation_loss', 'data_hash', 'validation_hash'):
                if a[field] != b[field]:
                    raise ValueError(f'80/85 pairing differs: E{e} seed{seed} {field}')
    return pairs


def preflight(pairs, groups):
    inventory = []
    for e, seed in groups:
        row = pairs[e, seed, .8]
        init = Path(row['source_checkpoint_path'])
        dense = Path(row['dense_artifact_path']) / 'checkpoints/final_recovered.pt'
        source = init.parent / 'step_2500.pt'
        checks = {}
        for kind, path, expected in [('initial', init, row['source_checkpoint_hash']),
                                     ('dense', dense, row['dense_final_checkpoint_sha256']),
                                     ('source_final', source, None)]:
            ok = path.is_file()
            digest = common.sha(path) if ok else None
            checks[kind] = {'path': str(path), 'exists': ok, 'sha256': digest,
                            'verified': ok and (expected is None or digest == expected)}
        inventory.append({'num_experts': e, 'seed': seed, 'checkpoints': checks})
    return inventory


def run_stage(config, init, masks, train, val, reference, device, dest, dense_ce, identity):
    if (dest / 'stage.json').exists():
        saved = common.jread(dest / 'stage.json')
        if saved['identity'] != identity or common.sha(dest / 'checkpoints/final_recovered.pt') != saved['checkpoint_sha256']:
            raise RuntimeError(f'Resume identity/checkpoint mismatch: {dest}')
        return saved
    if dest.exists() and any(dest.iterdir()) and not (dest / 'interruption.pt').exists():
        raise RuntimeError(f'Incomplete stage at {dest}; preserved. Resume only completed stage boundaries; do not silently restart.')
    initial = load_model_from_checkpoint(config['model'], str(init), torch.device('cpu'))
    state = {n: t.detach().clone() * masks[n] if n in masks else t.detach().clone()
             for n, t in initial.state_dict().items()}
    del initial
    group_hash = lambda g: state_dict_hash({n: t for n, t in state.items() if parameter_group(n) == g})
    started = time.perf_counter()
    record = recovery._run_recovery_condition(
        config=config, condition_name=dest.name, pruned_base_state=state,
        router_checkpoint=str(init), router_age_percent=0, router_step=0,
        masks=masks, expert_hash=group_hash('expert'), shared_hash=group_hash('shared'),
        mask_hash=recovery._mask_hash(masks), reference_selected=reference,
        calibration_batches=recovery._calibration_batches(val), train_batches=train,
        validation_batches=val, train_batch_hash=recovery._batch_sequence_hash(train),
        validation_batch_hash=recovery._batch_sequence_hash(val), device=device,
        recovery_steps=len(train), dense_loss=dense_ce, output_dir=dest,
        confidence_control=False, target_confidence=None, seed=config['seed'],
        sparsity=identity['sparsity'], router_mode='trainable',
        interruption_checkpoint=dest / 'interruption.pt')
    gradients = [json.loads(line) for line in (dest / 'gradient_stats/gradient_stats.jsonl').read_text().splitlines()]
    record['final_training_loss_including_aux'] = gradients[-1]['train_loss']
    record['final_training_ce'] = gradients[-1].get('train_cross_entropy')
    record['original_initialization_checkpoint'] = str(init)
    record['actual_sparsity'] = 1 - sum(int(m.sum()) for m in masks.values()) / sum(m.numel() for m in masks.values())
    record['compute_device'] = str(device)
    record['gpu_name'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else None
    record.update(identity=identity, duration_seconds=time.perf_counter() - started,
                  checkpoint_sha256=common.sha(dest / 'checkpoints/final_recovered.pt'))
    write_json(dest / 'stage.json', record)
    (dest / 'interruption.pt').unlink(missing_ok=True)
    return record


def trajectory(e, seed, pairs, out, device):
    pair = pairs[e, seed, .8]
    init = Path(pair['source_checkpoint_path'])
    payload = common.load(init)
    config = payload['config']
    if payload['step'] != 0 or common.protocol_errors(config, e, seed):
        raise ValueError('Initialization/protocol mismatch')
    if not torch.__version__.startswith('2.7.0'):
        raise RuntimeError('Canonical Dataset A requires torch 2.7.0')
    seed_everything(seed)
    tr, va = build_dataloaders(config['data'], 128, seed)
    train = recovery._materialize_batches(tr, 2500)
    val = recovery._materialize_validation_batches(va, 32)
    if recovery._batch_sequence_hash(train) != pair['data_hash'] or recovery._batch_sequence_hash(val) != pair['validation_hash']:
        raise RuntimeError('Canonical data hashes differ')
    model = load_model_from_checkpoint(config['model'], str(init), device)
    masks = {n: torch.ones_like(p, dtype=torch.bool, device='cpu') for n, p in expert_weight_parameters(model).items()}
    # Diagnostic reference only; never used to initialize any training stage.
    reference_model = load_model_from_checkpoint(config['model'], str(init.parent / 'step_2500.pt'), device)
    reference = selected_experts_per_batch(reference_model, recovery._calibration_batches(val), device)
    del reference_model, model
    root = out / f'trajectories/e{e}/seed_{seed}'
    root.mkdir(parents=True, exist_ok=True)
    identity_base = {'num_experts': e, 'seed': seed, 'initial_sha256': common.sha(init),
                     'protocol': 'dataset-a-imp-r0-v1', 'steps': 2500, 'torch_version': torch.__version__,
                     'schedule': list(SPARSITIES)}
    for k, target in enumerate(SPARSITIES[1:]):
        dest = root / f'round_{k:02d}'
        identity = dict(identity_base, round=k, sparsity=SPARSITIES[k], mask_hash=recovery._mask_hash(masks))
        record = run_stage(config, init, masks, train, val, reference, device, dest,
                           float(pair['dense_final_validation_loss']), identity)
        trained = load_model_from_checkpoint(config['model'], str(dest / 'checkpoints/final_recovered.pt'), torch.device('cpu'))
        new = nested_expert_local_masks(trained, masks, target)
        mask_path = root / f'mask_{k+1:02d}.pt'
        if mask_path.exists():
            if recovery._mask_hash(load_masks(mask_path)) != recovery._mask_hash(new):
                raise RuntimeError('Resume mask differs')
        else:
            save_masks(new, mask_path)
        write_json(root / f'mask_{k+1:02d}.json', dict(identity_base, pruning_round=k+1,
            target_sparsity=target, actual_sparsity=1-sum(int(m.sum()) for m in new.values())/sum(m.numel() for m in new.values()),
            mask_hash=recovery._mask_hash(new), parent_mask_hash=recovery._mask_hash(masks),
            keep_count=sum(int(m.sum()) for m in new.values()), eligible_count=sum(m.numel() for m in new.values()),
            final_training_ce=record.get('final_training_ce'),
            mask_path=str(mask_path), original_initialization_checkpoint=str(init),
            trained_checkpoint=str(dest / 'checkpoints/final_recovered.pt'),
            stage_path=str(dest), **{key: record[key] for key in ('duration_seconds', 'optimizer_step_attempts', 'optimizer_steps_applied', 'final_validation_loss', 'trainable_router_gradient_observed', 'router_parameter_drift_final')}))
        masks = new
        del trained
    for index in (8, 9):
        target = SPARSITIES[index]
        mask = load_masks(root / f'mask_{index:02d}.pt')
        row = pairs[e, seed, target]
        run_stage(config, init, mask, train, val, reference, device, root / f'final_s{target:.2f}',
                  float(row['dense_final_validation_loss']), dict(identity_base, final=True,
                    sparsity=target, mask_hash=recovery._mask_hash(mask)))


def aggregate(out, pairs):
    rounds, masks, finals, comparisons, checkpoints = [], [], [], [], []
    for (e, seed, sparsity), pair in pairs.items():
        if sparsity != .8:
            continue
        for role, path, digest in [('original_initialization', pair['source_checkpoint_path'], pair['source_checkpoint_hash']),
                ('matched_dense', str(Path(pair['dense_artifact_path']) / 'checkpoints/final_recovered.pt'), pair['dense_final_checkpoint_sha256'])]:
            checkpoints.append({'num_experts': e, 'seed': seed, 'role': role, 'path': path, 'sha256': digest})
    for path in sorted((out / 'trajectories').glob('e*/seed_*/mask_*.json')):
        masks.append(common.jread(path))
    for path in sorted((out / 'trajectories').glob('e*/seed_*/*/stage.json')):
        r = common.jread(path)
        row = {**r['identity'], **{k: v for k, v in r.items() if isinstance(v, (str, int, float, bool))}}
        row['trained_checkpoint'] = str(path.parent / 'checkpoints/final_recovered.pt')
        row['curves_path'] = str(path.parent / 'metrics.jsonl')
        checkpoints.append({'path': row['trained_checkpoint'], 'sha256': r['checkpoint_sha256'], 'role': 'final' if r['identity'].get('final') else 'discovery'})
        if r['identity'].get('final'):
            pair = pairs[row['num_experts'], row['seed'], row['sparsity']]
            dense = float(pair['dense_final_validation_loss'])
            gap = r['final_validation_loss'] - dense
            row.update(dense_validation_ce=dense, absolute_ce_gap=gap, relative_ce_gap=gap/dense,
                       matched_dense_checkpoint=str(Path(pair['dense_artifact_path']) / 'checkpoints/final_recovered.pt'),
                       original_initialization_checkpoint=pair['source_checkpoint_path'])
            finals.append(row)
            comparisons.append({'num_experts': row['num_experts'], 'seed': row['seed'], 'sparsity': row['sparsity'],
                'imp_gap': gap, 'omp_gap': float(pair['sparse_final_validation_loss'])-dense,
                'paired_improvement': float(pair['sparse_final_validation_loss'])-r['final_validation_loss'],
                'omp_curves_path': str(Path(pair['sparse_artifact_path'])/'metrics.jsonl'),
                'imp_curves_path': row['curves_path'], 'dense_curves_path': str(Path(pair['dense_artifact_path'])/'metrics.jsonl')})
        else:
            mask_metadata = path.parent.parent / f"mask_{row['round']+1:02d}.json"
            if mask_metadata.exists():
                child = common.jread(mask_metadata)
                row.update(next_target_sparsity=child['target_sparsity'], next_actual_sparsity=child['actual_sparsity'],
                           next_mask_hash=child['mask_hash'], parent_mask_hash=child['parent_mask_hash'])
            rounds.append(row)
    for name, rows, fields in [('imp_round_metrics', rounds, ['num_experts','seed','round','sparsity','mask_hash','original_initialization_checkpoint','trained_checkpoint','final_training_ce','final_validation_loss','optimizer_step_attempts','optimizer_steps_applied','trainable_router_gradient_observed','router_parameter_drift_final','duration_seconds','training_batch_sequence_hash','validation_batch_sequence_hash','curves_path']),
            ('imp_mask_manifest', masks, ['num_experts','seed','pruning_round','target_sparsity','actual_sparsity','mask_hash','parent_mask_hash','keep_count','eligible_count','mask_path','original_initialization_checkpoint','trained_checkpoint','final_training_ce','final_validation_loss','optimizer_step_attempts','optimizer_steps_applied','duration_seconds']),
            ('final_recovery_metrics', finals, ['num_experts','seed','sparsity','mask_hash','trained_checkpoint','original_initialization_checkpoint','matched_dense_checkpoint','dense_validation_ce','final_validation_loss','absolute_ce_gap','relative_ce_gap','curves_path','training_batch_sequence_hash','validation_batch_sequence_hash','optimizer_step_attempts','optimizer_steps_applied']),
            ('imp_vs_omp_comparison', comparisons, ['num_experts','seed','sparsity','imp_gap','omp_gap','paired_improvement']),
            ('checkpoint_manifest', checkpoints, ['path','sha256','role'])]:
        write_csv(out / (name + '.csv'), rows, fields)
    import statistics
    summaries = []
    for e in common.ES:
        for s in (.8, .85):
            rows = [r for r in comparisons if r['num_experts']==e and r['sparsity']==s]
            if rows:
                summaries.append({'num_experts':e,'sparsity':s,'count':len(rows),
                    **{f'{metric}_{stat}': (statistics.mean([r[metric] for r in rows]) if stat=='mean' else statistics.stdev([r[metric] for r in rows]) if len(rows)>1 else None)
                    for metric in ('imp_gap','omp_gap','paired_improvement') for stat in ('mean','sd')}})
    write_json(out/'screening.json', {'completed':len(finals),'expected':50,
        'within_relative_ce':{str(t):sum(r['relative_ce_gap'] <= t for r in finals) for t in (.02,.05,.1)},
        'omp_within_relative_ce':{str(t):sum(r['omp_gap']/float(pairs[r['num_experts'],r['seed'],r['sparsity']]['dense_final_validation_loss']) <= t for r in comparisons) for t in (.02,.05,.1)},
        'discovery_seconds':sum(r['duration_seconds'] for r in rounds), 'paired_summary':summaries,
        'interpretation':'Descriptive validation screens only; expert-only LTH adaptation. No confirmed winning-ticket or statistical equivalence claim.'})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--groups', nargs='+', default=[f'{e}:{s}' for e in common.ES for s in common.SEEDS])
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--execute', action='store_true', help='Explicitly launch selected trajectories')
    p.add_argument('--resume', action='store_true', help='Verify and reuse completed stages; refuse incomplete stages')
    args = p.parse_args()
    groups = [tuple(map(int, x.split(':'))) for x in args.groups]
    if any(e not in common.ES or s not in common.SEEDS for e,s in groups) or len(set(groups))!=len(groups):
        p.error('Invalid or duplicate architecture/seed group')
    out = args.output.resolve()
    if out == ROOT/'results' or OUT not in (out, *out.parents):
        p.error('Output must be results/imp_80_85 or a subdirectory')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pairs = canonical_pairs()
        inventory = preflight(pairs, groups)
        write_json(out/'preflight.json', inventory)
        aggregate(out, pairs)
        if not args.execute:
            print(json.dumps(inventory, indent=2))
            return
        if not all(c['verified'] for r in inventory for c in r['checkpoints'].values()):
            raise RuntimeError('Missing or mismatched prerequisite; see preflight.json')
        if (out/'trajectories').exists() and not args.resume:
            raise FileExistsError('Existing trajectories require --resume')
        device = torch.device('cuda')
        configure_device(device)
        torch.set_num_threads(4)
        for e, seed in groups:
            trajectory(e, seed, pairs, out, device)
            aggregate(out, pairs)

if __name__ == '__main__':
    main()
