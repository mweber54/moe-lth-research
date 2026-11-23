"""Mask structure from existing IMP masks (CPU only). Exact-count random controls use fixed seeds."""
import numpy as np
from fge_common import *

DRAWS = 5


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return None
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def unit_stats(m1, m2):
    """m1: fc1 mask (H,D) rows=hidden units; m2: fc2 mask (D,H) columns=hidden units."""
    r = m1.sum(1).astype(float)
    c = m2.sum(0).astype(float)
    top = max(1, int(round(0.1 * len(r))))
    corr = float(np.corrcoef(r, c)[0, 1]) if r.std() > 0 and c.std() > 0 else None
    return dict(fc1_inputs_mean=r.mean(), fc1_inputs_min=r.min(), fc1_inputs_max=r.max(), fc2_outputs_mean=c.mean(),
                fc2_outputs_min=c.min(), fc2_outputs_max=c.max(),
                dead_no_inputs=int((r == 0).sum()), dead_no_outputs=int((c == 0).sum()), dead_both=int(((r == 0) & (c == 0)).sum()),
                no_inputs_but_has_outputs=int(((r == 0) & (c > 0)).sum()),
                gini_fc1=gini(r), gini_fc2=gini(c), top10pct_share_fc1=float(np.sort(r)[::-1][:top].sum() / r.sum()) if r.sum() else None,
                top10pct_share_fc2=float(np.sort(c)[::-1][:top].sum() / c.sum()) if c.sum() else None, fc1_fc2_channel_corr=corr)


def random_like(m, rng):
    flat = np.zeros(m.size, dtype=bool)
    flat[rng.permutation(m.size)[:int(m.sum())]] = True
    return flat.reshape(m.shape)


def main():
    tensor_rows, expert_rows, nest_rows, cross_rows = [], [], [], []
    masks = {}
    for seed in SEEDS:
        for s in SPARSITIES:
            masks[(seed, s)] = {k: v.numpy() for k, v in load(mask_for(seed, s)).items()}
    for (seed, s), mk in masks.items():
        for k, m in mk.items():
            tensor_rows.append(dict(seed=seed, sparsity=s, tensor=k, numel=m.size, kept=int(m.sum()), tensor_sparsity=1 - m.mean()))
        for l in range(4):
            for e in range(32):
                m1, m2 = mk[f'blocks.{l}.moe.experts.{e}.fc1.weight'], mk[f'blocks.{l}.moe.experts.{e}.fc2.weight']
                expert_rows.append(dict(seed=seed, sparsity=s, layer=l, expert=e, control='imp', **unit_stats(m1, m2)))
                for d in range(DRAWS):
                    rng = np.random.default_rng([seed, int(s * 100), l, e, d])
                    expert_rows.append(dict(seed=seed, sparsity=s, layer=l, expert=e, control=f'random_exact_count_{d}',
                                            **unit_stats(random_like(m1, rng), random_like(m2, rng))))
    for seed in SEEDS:
        a, b = masks[(seed, .80)], masks[(seed, .85)]
        inter = sum(int((a[k] & b[k]).sum()) for k in a)
        ka, kb = sum(int(a[k].sum()) for k in a), sum(int(b[k].sum()) for k in a)
        nest_rows.append(dict(seed=seed, kept_080=ka, kept_085=kb, intersection=inter, frac_085_inside_080=inter / kb,
                              jaccard=inter / (ka + kb - inter), expected_jaccard_if_nested=kb / ka,
                              note='nesting is a structural consequence of IMP, not a discovery'))
    for s in SPARSITIES:
        for i, sa in enumerate(SEEDS):
            for sb in SEEDS[i + 1:]:
                a, b = masks[(sa, s)], masks[(sb, s)]
                inter = sum(int((a[k] & b[k]).sum()) for k in a)
                ka, kb = sum(int(a[k].sum()) for k in a), sum(int(b[k].sum()) for k in a)
                n = sum(a[k].size for k in a)
                pa, pb = ka / n, kb / n
                cross_rows.append(dict(sparsity=s, seed_a=sa, seed_b=sb, jaccard=inter / (ka + kb - inter),
                                       expected_jaccard_independent=pa * pb / (pa + pb - pa * pb),
                                       note='raw coordinate overlap only; expert coordinates are not semantically aligned across seeds'))
    d = OUT / 'mask_structure'
    table(d / 'tensor_sparsity.csv', tensor_rows)
    table(d / 'expert_unit_structure.csv', expert_rows)
    table(d / 'within_seed_nesting.csv', nest_rows)
    table(d / 'cross_seed_coordinate_overlap.csv', cross_rows)


if __name__ == '__main__':
    main()
