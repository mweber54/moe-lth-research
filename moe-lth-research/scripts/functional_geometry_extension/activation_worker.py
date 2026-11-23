"""Activation geometry and fixed-gate distortion for one source seed over the frozen trajectory.

Usage: activation_worker.py --seed 7 [--steps 0,25,...] [--limit-experts N] [--out-name activation_geometry]
Resumable per (seed, step). No optimizer, no training, no writes outside OUT.
"""
import argparse, time, os
from fge_common import *
from fge_function import *

MODES = ('fixed_source', 'dense_native')


def sparsity_ok(state, mask):
    return all(bool((state[n][~m] == 0).all()) for n, m in mask.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', default=','.join(map(str, STEPS)))
    ap.add_argument('--out-name', default='activation_geometry')
    ap.add_argument('--limit-experts', type=int, default=0)
    args = ap.parse_args()
    configure_precision()
    seed = args.seed
    steps = [int(s) for s in args.steps.split(',')]
    base = OUT / args.out_name / f'seed_{seed}'
    attempt = f'attempt_{int(time.time())}_{os.getpid()}'
    pr = probe(seed)
    pos = pr['positions']
    masks = {s: {k: v for k, v in load(mask_for(seed, s)).items()} for s in SPARSITIES}
    F_state = load(source_final_path(seed))['model']
    mf = build_model(F_state, seed)
    fin, _ = native_inputs(mf, pr['tokens'])
    src_dev = max(float((fin[l] - pr['source_full_inputs'][l]).abs().max()) for l in range(4))
    impl = implementation_hash('fge_common.py', 'fge_metrics.py', 'fge_function.py', 'activation_worker.py')
    with heartbeat(OUT / args.out_name / f'heartbeat_seed_{seed}.json', seed=seed, attempt=attempt, implementation=impl) as beat:
        for step in steps:
            dest = base / f'step_{step:04d}'
            if (dest / 'completed.json').exists():
                continue
            t0 = time.time()
            beat(step=step, phase='load')
            dense_p = snapshot_path(cond_id(seed), step)
            dense_state = load(dense_p)['model']
            md = build_model(dense_state, seed)
            sparse = {}
            for s in SPARSITIES:
                p = snapshot_path(cond_id(seed, s), step)
                st = load(p)['model']
                sparse[s] = dict(path=p, state=st, model=build_model(st, seed), mask_enforced=sparsity_ok(st, masks[s]))
            dn, dense_routes = native_inputs(md, pr['tokens'])
            inputs = {'fixed_source': pr['source_full_inputs'], 'dense_native': dn}
            geo_rows, dist_rows, token_arrays = [], [], {}
            nl = len(md.blocks)
            for mode in MODES:
                for l in range(nl):
                    beat(step=step, phase=f'geometry_{mode}_L{l}')
                    x = inputs[mode][l].reshape(-1, inputs[mode][l].shape[-1])[pos]
                    ne = len(md.blocks[l].moe.experts)
                    for e in range(ne if not args.limit_experts else min(ne, args.limit_experts)):
                        ed = md.blocks[l].moe.experts[e]
                        spectra = None
                        for s in SPARSITIES:
                            rows, spectra = expert_pair_geometry(sparse[s]['model'].blocks[l].moe.experts[e], ed, x, spectra)
                            for quantity, row in rows.items():
                                geo_rows.append(dict(seed=seed, sparsity=s, step=step, mode=mode, layer=l, expert=e, quantity=quantity, **row))
            for l in range(nl):
                beat(step=step, phase=f'distortion_L{l}')
                for s in SPARSITIES:
                    ms = sparse[s]['model']
                    for mode in MODES:
                        g = gates(md, l, inputs[mode][l], pos)
                        rec = distortion(l, md.blocks[l].moe.experts, ms.blocks[l].moe.experts, inputs[mode][l], pos, g)
                        dist_rows.append(dict(seed=seed, sparsity=s, step=step, gate_reference='same_step_dense', input_mode=mode,
                                              layer=l, **distortion_summary(rec)))
                        token_arrays[f'sd_{s:.2f}_{mode}_L{l}_d2'] = rec['d2'].cpu().numpy()
                        token_arrays[f'sd_{s:.2f}_{mode}_L{l}_r2'] = rec['r2'].cpu().numpy()
                        token_arrays[f'sd_{s:.2f}_{mode}_L{l}_accepted'] = rec['accepted'].cpu().numpy()
                    g = gates(mf, l, pr['source_full_inputs'][l], pos)
                    rec = distortion(l, md.blocks[l].moe.experts, ms.blocks[l].moe.experts, pr['source_full_inputs'][l], pos, g)
                    dist_rows.append(dict(seed=seed, sparsity=s, step=step, gate_reference='fixed_source_final', input_mode='fixed_source',
                                          layer=l, **distortion_summary(rec)))
                    token_arrays[f'sf_{s:.2f}_L{l}_d2'] = rec['d2'].cpu().numpy()
                    token_arrays[f'sf_{s:.2f}_L{l}_accepted'] = rec['accepted'].cpu().numpy()
            identities = dict(source_inputs_max_dev_from_F_native=src_dev)
            if step == 0:
                A = load(initial_path(seed))['model']
                identities['dense_step0_equals_original_init'] = all(torch.equal(dense_state[k], A[k]) for k in A)
                for s in SPARSITIES:
                    identities[f'sparse{s:.2f}_step0_equals_masked_original'] = all(
                        torch.equal(sparse[s]['state'][k], A[k] * masks[s][k] if k in masks[s] else A[k]) for k in A)
            table(dest / 'expert_geometry.csv', geo_rows)
            table(dest / 'distortion.csv', dist_rows)
            np.savez_compressed(dest / 'distortion_tokens.npz', **token_arrays)
            save(dest / 'completed.json', dict(seed=seed, step=step, attempt=attempt, implementation=impl, seconds=time.time() - t0,
                                               dense_snapshot=str(dense_p), dense_snapshot_sha256_in_json=read(dense_p.with_suffix('.json')).get('snapshot_sha256'),
                                               sparse_snapshots={f'{s:.2f}': str(v['path']) for s, v in sparse.items()},
                                               mask_enforced={f'{s:.2f}': v['mask_enforced'] for s, v in sparse.items()},
                                               precision='fp32, TF32 off, eval mode, no autocast', probe_positions=int(len(pos)),
                                               probe_sha256=read(HSC / f'probes/seed_{seed}.json')['sha256'], identities=identities,
                                               limit_experts=args.limit_experts))
            del md, sparse
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
