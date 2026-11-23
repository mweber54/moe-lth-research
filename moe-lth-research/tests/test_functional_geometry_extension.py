"""Focused numerical checks for the functional geometry extension metrics (small, analytically known cases)."""
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/functional_geometry_extension'))
from fge_metrics import (cka_centered, spectrum, rank_status, feature_overlap, sample_overlap, pair_vector_stats,
                         energy_retention, cosine, gram_subspace_overlap, retained_energy)
from consolidate import EXISTING_TABLES


def test_expert_function_export_name_does_not_claim_sparse_native_only():
    assert ('expert_function_trajectories_n128', 'expert_function_trajectories') in EXISTING_TABLES
    assert not any('sparse_native' in output for output, _ in EXISTING_TABLES)


def test_cka_identical_and_scale_invariant():
    x = torch.randn(50, 6, dtype=torch.float64)
    assert cka_centered(x, x) == pytest.approx(1.0)
    assert cka_centered(x, 3 * x) == pytest.approx(1.0)


def test_cka_centering_removes_offset():
    x = torch.randn(50, 6, dtype=torch.float64)
    assert cka_centered(x, x + 100.0) == pytest.approx(1.0)


def test_cka_orthogonal_rotation_invariant_and_zero_norm_undefined():
    x = torch.randn(50, 6, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
    assert cka_centered(x, x @ q) == pytest.approx(1.0)
    assert cka_centered(x, torch.zeros(50, 6)) is None
    assert cka_centered(x, torch.ones(50, 6)) is None


def test_feature_overlap_known_subspaces():
    n = 100
    a = torch.zeros(n, 6, dtype=torch.float64)
    a[:, 0] = torch.randn(n) * 3
    a[:, 1] = torch.randn(n) * 2
    b = a.clone()
    sa, sb = spectrum(a), spectrum(b)
    assert feature_overlap(sa, sb, 2)[0] == pytest.approx(1.0)
    c = torch.zeros(n, 6, dtype=torch.float64)
    c[:, 2] = a[:, 0]
    c[:, 3] = a[:, 1]
    sc = spectrum(c)
    assert feature_overlap(sa, sc, 2)[0] == pytest.approx(0.0, abs=1e-12)
    # sample-space subspaces coincide even though feature subspaces are orthogonal
    assert sample_overlap(sa, sc, 2)[0] == pytest.approx(1.0)


def test_rank_deficiency_and_n_minus_1_limit():
    x = torch.randn(40, 3, dtype=torch.float64) @ torch.randn(3, 10, dtype=torch.float64)
    sp = spectrum(x)
    assert rank_status(sp, 3)['valid'] and rank_status(sp, 3)['numerical_rank'] == 3
    assert not rank_status(sp, 4)['valid']
    small = spectrum(torch.randn(5, 10, dtype=torch.float64))
    assert rank_status(small, 4)['valid'] and not rank_status(small, 5)['valid']
    assert not rank_status(spectrum(torch.zeros(10, 4)), 1)['valid']


def test_retained_energy_sums_to_one_at_full_rank():
    sp = spectrum(torch.randn(30, 5, dtype=torch.float64))
    assert retained_energy(sp, 5) == pytest.approx(1.0)


def test_degenerate_cutoff_flag():
    x = torch.diag(torch.tensor([3.0, 2.0, 2.0, 1.0], dtype=torch.float64))
    x = torch.cat([x, -x])
    assert rank_status(spectrum(x), 2)['degenerate']
    assert not rank_status(spectrum(x), 1)['degenerate']


def test_pair_vector_stats_identical_and_zero_rows():
    a = torch.randn(20, 4)
    s = pair_vector_stats(a, a)
    assert s['relative_error'] == pytest.approx(0.0) and s['mean_cosine'] == pytest.approx(1.0)
    z = pair_vector_stats(torch.zeros(5, 4), torch.zeros(5, 4))
    assert z['relative_error'] is None and z['mean_cosine'] is None and z['zero_exclusions'] == 5


def test_energy_retention_and_cosine():
    g = torch.tensor([1.0, 2.0, 2.0, 0.0])
    m = torch.tensor([1, 0, 1, 1])
    assert energy_retention(g, m) == pytest.approx((1 + 4) / 9)
    assert energy_retention(torch.zeros(4), m) is None
    assert cosine(g, g * m) == pytest.approx(((1 + 4) / (3 * (5 ** 0.5))))
    assert cosine(torch.zeros(3), g[:3]) is None


def test_gram_subspace_overlap_matches_explicit_projection():
    torch.manual_seed(0)
    xa, xb = torch.randn(6, 40, dtype=torch.float64), torch.randn(6, 40, dtype=torch.float64)
    got, status = gram_subspace_overlap(xa @ xa.T, xb @ xb.T, xa @ xb.T, 3)
    ua = torch.linalg.svd(xa, full_matrices=False).Vh[:3].T
    ub = torch.linalg.svd(xb, full_matrices=False).Vh[:3].T
    want = float(torch.linalg.svdvals(ua.T @ ub).square().sum() / 3)
    assert status == 'ok' and got == pytest.approx(want)
    same, _ = gram_subspace_overlap(xa @ xa.T, xa @ xa.T, xa @ xa.T, 4)
    assert same == pytest.approx(1.0)
    none, status = gram_subspace_overlap(xa @ xa.T, xb @ xb.T, xa @ xb.T, 7)
    assert none is None and status.startswith('rank_exceeds')
    none, status = gram_subspace_overlap(torch.zeros(6, 6), xb @ xb.T, torch.zeros(6, 6), 1)
    assert none is None and status == 'zero_gradient'
