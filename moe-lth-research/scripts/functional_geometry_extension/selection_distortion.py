"""Immediate pruning distortion: direct IMP state C versus diagnostic copy D = C with the selected mask.

Gates come from C (the unmasked selection state). Also reports probe CE and prediction KL(C||D), whole-model
and with the mask applied to a single layer's experts only (isolated diagnostic copies; no files written).
"""
import argparse
import torch.nn.functional as F
from fge_common import *
from fge_function import *


def masked_state(state, mask, layers=None):
    out = {k: v.clone() for k, v in state.items()}
    for k, m in mask.items():
        if layers is None or int(k.split('.')[1]) in layers:
            out[k] = out[k] * m
    return out


@torch.no_grad()
def ce_kl(mc, md, pr):
    oc = mc(pr['tokens']).logits.double()
    od = md(pr['tokens']).logits.double()
    t = pr['targets']
    ce_c = F.cross_entropy(oc.reshape(-1, oc.shape[-1]), t.reshape(-1))
    ce_d = F.cross_entropy(od.reshape(-1, od.shape[-1]), t.reshape(-1))
    kl = F.kl_div(F.log_softmax(od, -1).reshape(-1, od.shape[-1]), F.log_softmax(oc, -1).reshape(-1, oc.shape[-1]),
                  log_target=True, reduction='batchmean')
    return float(ce_c), float(ce_d), float(kl)


def main():
    configure_precision()
    rows, ce_rows = [], []
    for seed in SEEDS:
        pr = probe(seed)
        pos = pr['positions']
        for s in SPARSITIES:
            mask = load(mask_for(seed, s))
            cpath = selection_state_path(seed, s)
            cst = load(cpath)['model']
            mc = build_model(cst, seed)
            dst = masked_state(cst, mask)
            md = build_model(dst, seed)
            kept = sum(int(m.sum()) for m in mask.values())
            total = sum(m.numel() for m in mask.values())
            nat, _ = native_inputs(mc, pr['tokens'])
            for mode, inp in (('C_native', nat), ('fixed_source', pr['source_full_inputs'])):
                for l in range(4):
                    g = gates(mc, l, inp[l], pos)
                    rec = distortion(l, mc.blocks[l].moe.experts, md.blocks[l].moe.experts, inp[l], pos, g)
                    rows.append(dict(seed=seed, sparsity=s, gate_reference='C_selection_state', input_mode=mode, layer=l,
                                     mask_keep_fraction=kept / total, selection_state=str(cpath), **distortion_summary(rec)))
            ce_c, ce_d, kl = ce_kl(mc, md, pr)
            ce_rows.append(dict(seed=seed, sparsity=s, masked_layers='all', ce_C=ce_c, ce_D=ce_d, ce_increase=ce_d - ce_c, kl_C_to_D=kl))
            for l in range(4):
                mdl = build_model(masked_state(cst, mask, {l}), seed)
                ce_c, ce_d, kl = ce_kl(mc, mdl, pr)
                ce_rows.append(dict(seed=seed, sparsity=s, masked_layers=str(l), ce_C=ce_c, ce_D=ce_d, ce_increase=ce_d - ce_c, kl_C_to_D=kl))
            print(seed, s, flush=True)
    table(OUT / 'fixed_gate_distortion/selection_state_pruning_distortion.csv', rows)
    table(OUT / 'fixed_gate_distortion/selection_state_pruning_ce_kl.csv', ce_rows)


if __name__ == '__main__':
    main()
