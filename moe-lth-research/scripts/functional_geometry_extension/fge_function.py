"""Expert-function geometry on identical inputs and fixed-gate functional distortion.

Gates are frozen from an explicitly identified reference; both expert sets are evaluated on the same hidden
vectors. Capacity decisions always use the reference model's full 16,384-token batch, then are restricted to
the registered 2,048 probe positions.
"""
import torch
import torch.nn.functional as F
from fge_metrics import (cka_centered, spectrum, rank_status, retained_energy, feature_overlap,
                         sample_overlap, pair_vector_stats)

RANKS = (8, 16, 32)


@torch.no_grad()
def native_inputs(model, tokens):
    """Per-layer MoE input (B,T,d) in the model's own forward pass, plus route traces."""
    out = model(tokens)
    return [h for h in out.pre_router_hidden_states], out.routes


@torch.no_grad()
def gates(ref_model, layer, x_full, positions):
    """Reference gates on a full batch (capacity over the whole batch), restricted to probe positions."""
    moe = ref_model.blocks[layer].moe
    _, _, tr = moe(x_full)
    k = moe.top_k
    return dict(idx=tr.selected_expert_indices.reshape(-1, k)[positions],
                prob=tr.selected_probabilities.reshape(-1, k)[positions],
                acc=tr.accepted_mask[positions])


@torch.no_grad()
def distortion(layer, ref_experts, other_experts, x_full, positions, g):
    """D = mean_x ||sum_e a_e(x)[f_other,e(x) - f_ref,e(x)]||^2 with frozen gates g.

    Returns per-token arrays over the probe positions (zero where a token is not accepted) and reference
    MoE-output squared norms under the same frozen gates and the reference expert set.
    """
    x = x_full.reshape(-1, x_full.shape[-1])[positions]
    n, k = g['idx'].shape
    d2 = torch.zeros(n, dtype=torch.float64, device=x.device)
    r2 = torch.zeros_like(d2)
    diff = torch.zeros(n, x.shape[1], dtype=torch.float64, device=x.device)
    ref = torch.zeros_like(diff)
    for slot in range(k):
        for e in range(len(ref_experts)):
            sel = ((g['idx'][:, slot] == e) & g['acc'][:, slot]).nonzero().squeeze(1)
            if sel.numel() == 0:
                continue
            a = g['prob'][sel, slot].double()[:, None]
            xe = x[sel]
            fr = ref_experts[e](xe).double()
            fo = other_experts[e](xe).double()
            diff[sel] += a * (fo - fr)
            ref[sel] += a * fr
    d2 = diff.square().sum(1)
    r2 = ref.square().sum(1)
    accepted = g['acc'].any(1)
    return dict(d2=d2, r2=r2, accepted=accepted)


def distortion_summary(rec):
    d2, r2, acc = rec['d2'], rec['r2'], rec['accepted']
    na = int(acc.sum())
    full_d, full_r = float(d2.mean()), float(r2.mean())
    out = dict(n_tokens=int(len(d2)), n_accepted=na, D_full=full_d, ref_norm_full=full_r,
               rel_full=full_d / full_r if full_r > 0 else None)
    if na:
        ad, ar = float(d2[acc].mean()), float(r2[acc].mean())
        out.update(D_accepted=ad, ref_norm_accepted=ar, rel_accepted=ad / ar if ar > 0 else None)
    else:
        out.update(D_accepted=None, ref_norm_accepted=None, rel_accepted=None)
    return out


def expert_acts(expert, x):
    h = F.gelu(expert.fc1(x))
    return h, expert.fc2(h)


@torch.no_grad()
def expert_pair_geometry(es, ed, x, dense_spectra=None):
    """Compare a sparse and a dense expert on the same inputs x (N,d).

    Returns row dicts per quantity (post-GELU hidden 'hidden', complete output 'output') and the dense
    spectra (cacheable, they depend only on the dense expert and x).
    """
    hs, ys = expert_acts(es, x)
    hd, yd = expert_acts(ed, x)
    rows = {}
    spectra = dict(hidden=None, output=None) if dense_spectra is None else dense_spectra
    for name, a, b in (('hidden', hs, hd), ('output', ys, yd)):
        row = dict(cka=cka_centered(a, b), **pair_vector_stats(a, b))
        row['sparse_energy'] = float(a.double().square().mean())
        row['dense_energy'] = float(b.double().square().mean())
        sa = spectrum(a)
        if spectra[name] is None:
            spectra[name] = spectrum(b)
        sb = spectra[name]
        for k in RANKS:
            ra, rb = rank_status(sa, k), rank_status(sb, k)
            ok = ra['valid'] and rb['valid']
            row[f'feat_overlap_r{k}'] = feature_overlap(sa, sb, k)[0] if ok else None
            row[f'feat_min_cos_r{k}'] = feature_overlap(sa, sb, k)[1] if ok else None
            row[f'samp_overlap_r{k}'] = sample_overlap(sa, sb, k)[0] if ok else None
            row[f'sparse_retained_r{k}'] = retained_energy(sa, k)
            row[f'dense_retained_r{k}'] = retained_energy(sb, k)
            row[f'rank_status_r{k}'] = 'ok' if ok else f"sparse:{ra['reason']};dense:{rb['reason']}"
            row[f'degenerate_cutoff_r{k}'] = bool(ra['degenerate'] or rb['degenerate'])
        row['sparse_numerical_rank'] = rank_status(sa, 1)['numerical_rank']
        row['dense_numerical_rank'] = rank_status(sb, 1)['numerical_rank']
        rows[name] = row
    return rows, spectra
