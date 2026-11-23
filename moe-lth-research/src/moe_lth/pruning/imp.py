"""Nested expert-local IMP, with deterministic ties and integer keep budgets."""
from collections import defaultdict
import torch
from .masks import expert_weight_parameters

SPARSITIES = (0., .2, .36, .488, .5904, .67232, .737856, .7902848, .8, .85)


def validate_masks(model, masks):
    parameters = expert_weight_parameters(model)
    if not parameters or set(parameters) != set(masks):
        raise ValueError('Masks must cover exactly the expert weight matrices')
    for name, p in parameters.items():
        m = masks[name]
        if m.shape != p.shape or m.dtype != torch.bool:
            raise ValueError(f'Invalid binary mask: {name}')


def nested_expert_local_masks(model, masks, sparsity):
    if not 0 <= sparsity < 1:
        raise ValueError('Invalid sparsity')
    validate_masks(model, masks)
    groups = defaultdict(list)
    # Parameter ordering matches canonical OMP. Stable ascending ties prune the
    # lowest flattened coordinate first (OMP topk does not specify tie order).
    for name, p in expert_weight_parameters(model).items():
        groups[name.rsplit('.experts.', 1)[0] + '.experts.' + name.split('.experts.')[1].split('.')[0]].append((name, p))
    result = {}
    for parameters in groups.values():
        old = torch.cat([masks[n].cpu().flatten() for n, _ in parameters])
        values = torch.cat([p.detach().float().abs().cpu().flatten() for _, p in parameters])
        keep = max(1, round(old.numel() * (1 - sparsity)))
        active = old.nonzero().flatten()
        if keep > active.numel():
            raise ValueError('Target would reactivate pruned weights')
        if not torch.isfinite(values[active]).all():
            raise ValueError('Nonfinite surviving magnitudes')
        order = torch.argsort(values[active], stable=True)
        new = old.clone()
        new[active[order[:active.numel() - keep]]] = False
        offset = 0
        for name, p in parameters:
            result[name] = new[offset:offset + p.numel()].reshape(p.shape)
            offset += p.numel()
    return result
