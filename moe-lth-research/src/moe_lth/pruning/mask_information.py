"""Uniform fixed-count null masks for IMP connection-placement experiments."""
from collections import defaultdict
import torch
from .imp import validate_masks

CONDITIONS = ('global', 'expert', 'matrix')


def groups(masks, condition):
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown allocation: {condition}')
    result = defaultdict(list)
    for name in sorted(masks):
        key = 'all' if condition == 'global' else name.rsplit('.', 2)[0] if condition == 'expert' else name
        result[key].append(name)
    return list(result.values())


def assert_control(model, original, candidate, condition):
    validate_masks(model, original)
    validate_masks(model, candidate)
    for names in groups(original, condition):
        a = torch.cat([original[n].cpu().flatten() for n in names])
        b = torch.cat([candidate[n].cpu().flatten() for n in names])
        assert a.numel() == b.numel()
        assert int(a.sum()) == int(b.sum()), f'Keep count differs: {names}'
    assert sum(int(x.sum()) for x in original.values()) == sum(int(x.sum()) for x in candidate.values())
    forced = all(sum(int(original[n].sum()) for n in ns) in (0, sum(original[n].numel() for n in ns)) for ns in groups(original, condition))
    if not forced and all(torch.equal(original[n].cpu(), candidate[n].cpu()) for n in original):
        raise ValueError('Control equals original despite nontrivial sampling space')


def randomize(model, original, condition, seed):
    """Sample uniformly, conditional on differing from IMP if equality is possible.

    Rejection rather than a forced bit swap avoids biasing the conditional null.
    Layer identity is part of the expert key; expert IDs repeat across layers.
    """
    validate_masks(model, original)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    grouped = groups(original, condition)
    forced = all(sum(int(original[n].sum()) for n in ns) in (0, sum(original[n].numel() for n in ns)) for ns in grouped)
    while True:
        result = {}
        for names in grouped:
            size = sum(original[n].numel() for n in names)
            keep = sum(int(original[n].sum()) for n in names)
            flat = torch.zeros(size, dtype=torch.bool)
            if keep == size:
                flat.fill_(True)
            elif keep:
                flat[torch.randperm(size, generator=generator)[:keep]] = True
            offset = 0
            for name in names:
                n = original[name].numel()
                result[name] = flat[offset:offset+n].reshape(original[name].shape).clone()
                offset += n
        if forced or any(not torch.equal(original[n].cpu(),result[n]) for n in original):
            assert_control(model, original, result, condition)
            return result
