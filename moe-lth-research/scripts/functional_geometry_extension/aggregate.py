"""Aggregate new outputs: expert -> layer -> seed (equal weights), then across seeds (seed is the unit)."""
import glob
import pandas as pd
from fge_common import *


def cat(pattern):
    fs = sorted(glob.glob(str(OUT / pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def hier(df, keys, layer_keys, vals):
    """mean within layer over remaining rows, then mean over layers -> per-seed value."""
    num = df.groupby(layer_keys, dropna=False)[vals].mean().reset_index()
    return num.groupby(keys, dropna=False)[vals].mean().reset_index()


def across(per_seed, keys, vals):
    g = per_seed.groupby(keys, dropna=False)
    out = g[vals].agg(['mean', 'min', 'max']).reset_index()
    out.columns = ['_'.join(c).strip('_') for c in out.columns]
    out['n_seeds'] = g['seed'].nunique().values
    return out


def main():
    A = cat('activation_geometry/seed_*/step_*/expert_geometry.csv')
    if len(A):
        vals = ['cka', 'relative_error', 'mean_cosine'] + [f'{p}_r{r}' for r in (8, 16, 32) for p in ('feat_overlap', 'samp_overlap', 'sparse_retained', 'dense_retained')]
        for r in (8, 16, 32):
            A[f'valid_r{r}'] = (A[f'rank_status_r{r}'] == 'ok').astype(float)
        vals += [f'valid_r{r}' for r in (8, 16, 32)]
        k = ['seed', 'sparsity', 'step', 'mode', 'quantity']
        ps = hier(A, k, k + ['layer'], vals)
        table(OUT / 'activation_geometry/per_seed_summary.csv', ps.to_dict('records'))
        table(OUT / 'activation_geometry/cross_seed_summary.csv', across(ps, ['sparsity', 'step', 'mode', 'quantity'], vals).to_dict('records'))
    D = cat('activation_geometry/seed_*/step_*/distortion.csv')
    if len(D):
        vals = ['rel_full', 'rel_accepted']
        k = ['seed', 'sparsity', 'step', 'gate_reference', 'input_mode']
        ps = D.groupby(k)[vals].mean().reset_index()
        table(OUT / 'fixed_gate_distortion/per_seed_summary.csv', ps.to_dict('records'))
        table(OUT / 'fixed_gate_distortion/cross_seed_summary.csv', across(ps, ['sparsity', 'step', 'gate_reference', 'input_mode'], vals).to_dict('records'))
    E_ = cat('gradient_geometry/seed_*/energy_retention_and_norms.csv')
    C = cat('gradient_geometry/seed_*/cosines.csv')
    R = cat('gradient_geometry/seed_*/random_mask_energy_control.csv')
    S = cat('gradient_geometry/seed_*/subspace_overlap.csv')
    if len(E_):
        m = E_.merge(C, on=['seed', 'sparsity', 'step', 'layer', 'matrix', 'kind', 'batch'])
        vals = ['energy_retention', 'keep_fraction', 'cosine_projected_dense_vs_sparse', 'cosine_dense_vs_sparse_unrestricted']
        k = ['seed', 'sparsity', 'step', 'matrix', 'kind']
        ps = hier(m, k, k + ['layer'], vals)
        rn = R.groupby(['seed', 'sparsity', 'step', 'matrix', 'layer', 'draw'])['mean_energy_retention'].mean().reset_index()
        rn = rn.groupby(['seed', 'sparsity', 'step', 'matrix']).mean_energy_retention.mean().reset_index().rename(columns={'mean_energy_retention': 'random_mask_energy_retention'})
        ps = ps.merge(rn, on=['seed', 'sparsity', 'step', 'matrix'], how='left')
        ps['energy_excess_over_random'] = ps.energy_retention - ps.random_mask_energy_retention
        ps['energy_excess_over_keep_fraction'] = ps.energy_retention - ps.keep_fraction
        table(OUT / 'gradient_geometry/per_seed_summary.csv', ps.to_dict('records'))
        v2 = vals + ['random_mask_energy_retention', 'energy_excess_over_random', 'energy_excess_over_keep_fraction']
        table(OUT / 'gradient_geometry/cross_seed_summary.csv', across(ps, ['sparsity', 'step', 'matrix', 'kind'], v2).to_dict('records'))
    if len(S):
        S['valid'] = (S.status == 'ok').astype(float)
        k = ['seed', 'sparsity', 'step', 'matrix', 'rank']
        ps = hier(S, k, k + ['layer'], ['overlap', 'valid'])
        table(OUT / 'gradient_geometry/subspace_per_seed_summary.csv', ps.to_dict('records'))
        table(OUT / 'gradient_geometry/subspace_cross_seed_summary.csv', across(ps, ['sparsity', 'step', 'matrix', 'rank'], ['overlap', 'valid']).to_dict('records'))
    print('ok', len(A), len(D), len(E_))


if __name__ == '__main__':
    main()
