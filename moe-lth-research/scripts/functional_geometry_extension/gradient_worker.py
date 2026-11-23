"""Gradient energy and subspace geometry at steps 0/2500/5000 on 16 fixed training minibatches.

Native forward/backward for dense and sparse models on identical raw batches. No optimizer step. Task (CE) and
auxiliary-loss gradients are kept separate. Uncentered Gram-based subspaces (ranks 1/2/4/8). Random-mask
energy controls preserve exact per-tensor keep counts (fixed seeds).
"""
import argparse, time, os
import torch.nn.functional as F
from fge_common import *
from fge_metrics import energy_retention, cosine, gram_subspace_overlap

GSTEPS = (0, 2500, 5000)
NB = 16
BATCH_STRIDE = 20
RANKS = (1, 2, 4, 8)
DRAWS = 10


def expert_names(layer):
    return [f'blocks.{layer}.moe.experts.{e}.{m}.weight' for e in range(32) for m in ('fc1', 'fc2')]


def batches(seed):
    """Deterministic, outcome-independent: training-loader batches 0,20,...,300 of epoch 1 (batch size 128)."""
    from moe_lth.data import build_dataloaders
    from moe_lth.experiments import run_router_age_recovery as R
    c = condition(cond_id(seed))['config']
    os.chdir(ROOT)
    R.seed_everything(seed)
    tr, _ = build_dataloaders(c['data'], int(c['training']['batch_size']), seed)
    idx = [i * BATCH_STRIDE for i in range(NB)]
    got = R._materialize_batches(tr, max(idx) + 1)
    return [got[i] for i in idx], idx


def grads(model, tokens, targets, aux_weight):
    """Returns {'task': {name: grad}, 'aux': {name: grad}} for expert fc weights (flattened per layer/matrix)."""
    params = {n: p for n, p in model.named_parameters() if '.moe.experts.' in n and n.endswith('.weight')}
    for p in model.parameters():
        p.grad = None
    out = model(tokens)
    ce = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), targets.reshape(-1))
    names = list(params)
    task = torch.autograd.grad(ce, [params[n] for n in names], retain_graph=True, allow_unused=True)
    aux = torch.autograd.grad(aux_weight * out.auxiliary_loss, [params[n] for n in names], allow_unused=True)
    z = lambda g, n: g if g is not None else torch.zeros_like(params[n])
    return float(ce), {n: z(g, n) for n, g in zip(names, task)}, {n: z(g, n) for n, g in zip(names, aux)}


def stack(gs, layer, matrix):
    return torch.stack([torch.stack([g[f'blocks.{layer}.moe.experts.{e}.{matrix}.weight'].flatten() for e in range(32)]) for g in gs])  # (B,32,P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out-name', default='gradient_geometry')
    ap.add_argument('--nb', type=int, default=NB)
    args = ap.parse_args()
    configure_precision()
    seed = args.seed
    dest = OUT / args.out_name / f'seed_{seed}'
    if (dest / 'completed.json').exists():
        return
    impl = implementation_hash('fge_common.py', 'fge_metrics.py', 'gradient_worker.py')
    bt, idx = batches(seed)
    bt = bt[:args.nb]
    aux_w = float(condition(cond_id(seed))['config']['routing']['aux_loss_weight'])
    masks = {s: load(mask_for(seed, s)) for s in SPARSITIES}
    energy_rows, cos_rows, norm_rows, sub_rows, ctrl_rows, ce_rows = [], [], [], [], [], []
    with heartbeat(OUT / args.out_name / f'heartbeat_seed_{seed}.json', seed=seed, implementation=impl) as beat:
        for step in GSTEPS:
            beat(step=step, phase='dense')
            md = build_model(load(snapshot_path(cond_id(seed), step))['model'], seed)
            gd_task, gd_aux = [], []
            for tok, tgt in bt:
                ce, t, a = grads(md, tok.cuda(), tgt.cuda(), aux_w)
                gd_task.append(t); gd_aux.append(a); ce_rows.append(dict(seed=seed, sparsity=0.0, step=step, batch=len(gd_task) - 1, ce=ce))
            for s in SPARSITIES:
                beat(step=step, phase=f'sparse_{s}')
                ms = build_model(load(snapshot_path(cond_id(seed, s), step))['model'], seed)
                gs_task, gs_aux = [], []
                for i, (tok, tgt) in enumerate(bt):
                    ce, t, a = grads(ms, tok.cuda(), tgt.cuda(), aux_w)
                    gs_task.append(t); gs_aux.append(a); ce_rows.append(dict(seed=seed, sparsity=s, step=step, batch=i, ce=ce))
                for l in range(4):
                    for mat in ('fc1', 'fc2'):
                        M = torch.stack([masks[s][f'blocks.{l}.moe.experts.{e}.{mat}.weight'].flatten() for e in range(32)]).cuda()  # (32,P)
                        for kind, gdl, gsl in (('task', gd_task, gs_task), ('aux', gd_aux, gs_aux)):
                            Gd = stack(gdl, l, mat).flatten(1)  # (B, 32P)
                            Gs = stack(gsl, l, mat).flatten(1)
                            Mf = M.flatten().to(Gd.dtype)
                            Gp = Gd * Mf
                            Gsm = Gs * Mf
                            for b in range(Gd.shape[0]):
                                energy_rows.append(dict(seed=seed, sparsity=s, step=step, layer=l, matrix=mat, kind=kind, batch=b,
                                                        energy_retention=energy_retention(Gd[b], Mf), dense_norm=float(Gd[b].double().norm()),
                                                        sparse_norm=float(Gsm[b].double().norm()), sparse_unrestricted_norm=float(Gs[b].double().norm()),
                                                        projected_dense_norm=float(Gp[b].double().norm()), keep_fraction=float(Mf.mean())))
                                cos_rows.append(dict(seed=seed, sparsity=s, step=step, layer=l, matrix=mat, kind=kind, batch=b,
                                                     cosine_projected_dense_vs_sparse=cosine(Gp[b], Gsm[b]),
                                                     cosine_dense_vs_sparse_unrestricted=cosine(Gd[b], Gs[b])))
                            if kind == 'task':
                                Ka = Gp.double() @ Gp.double().T
                                Kb = Gsm.double() @ Gsm.double().T
                                Kab = Gp.double() @ Gsm.double().T
                                for r in RANKS:
                                    ov, status = gram_subspace_overlap(Ka, Kb, Kab, r)
                                    sub_rows.append(dict(seed=seed, sparsity=s, step=step, layer=l, matrix=mat, rank=r, overlap=ov, status=status,
                                                         centered=False, n_batches=Gd.shape[0]))
                                for d in range(DRAWS):
                                    gen = torch.Generator(device='cuda').manual_seed(seed * 1000 + int(s * 100) * 10 + d + 7 * l + (1 if mat == 'fc2' else 0))
                                    keep = M.sum(1)
                                    R = torch.rand(M.shape, device='cuda', generator=gen)
                                    thresh = R.sort(1).values.gather(1, (keep - 1).clamp_min(0)[:, None].long())
                                    RM = (R <= thresh).flatten().to(Gd.dtype)
                                    den = Gd.square().sum(1).clamp_min(1e-300)
                                    ctrl_rows.append(dict(seed=seed, sparsity=s, step=step, layer=l, matrix=mat, draw=d,
                                                          mean_energy_retention=float(((Gd * RM).square().sum(1) / den).mean()),
                                                          imp_mean_energy_retention=float((Gp.square().sum(1) / den).mean())))
                del ms, gs_task, gs_aux
                torch.cuda.empty_cache()
            del md, gd_task, gd_aux
            torch.cuda.empty_cache()
        table(dest / 'energy_retention_and_norms.csv', energy_rows)
        table(dest / 'cosines.csv', cos_rows)
        table(dest / 'subspace_overlap.csv', sub_rows)
        table(dest / 'random_mask_energy_control.csv', ctrl_rows)
        table(dest / 'batch_ce.csv', ce_rows)
        save(dest / 'completed.json', dict(seed=seed, implementation=impl, batches_train_loader_indices=idx[:args.nb], aux_loss_weight=aux_w,
                                           precision='fp32, TF32 off, eval mode, native forward/backward, no optimizer', random_draws=DRAWS,
                                           ranks=RANKS, centered=False, steps=GSTEPS))


if __name__ == '__main__':
    main()
