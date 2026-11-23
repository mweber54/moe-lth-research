import numpy as np
import pytest
import torch

from moe_lth.analysis.weight_distributions import PATTERN, changes, distances, mask_hash, statistics
from moe_lth.pruning.router_age import state_dict_hash


def test_empirical_distances_capture_zero_atom_and_signed_shift():
    result = distances(np.array([0., 0., 1., 1.]), np.array([1., 1., 2., 2.]))
    assert result['wasserstein'] == pytest.approx(1.)
    assert result['ks'] == pytest.approx(.5)
    signed = distances(np.array([-1., -2.]), np.array([1., 2.]))
    assert signed['absolute_wasserstein'] == 0
    assert signed['ks'] == 1


def test_surviving_coordinates_include_natural_zeros():
    weights = np.array([0., 0., 2., 4.])
    mask = np.array([False, True, True, True])
    assert statistics(weights)['zero_fraction'] == .5
    assert statistics(weights[mask])['zero_fraction'] == pytest.approx(1/3)
    assert statistics(weights[weights != 0])['zero_fraction'] == 0
    assert statistics(weights)['l2'] == pytest.approx(np.sqrt(20))
    assert statistics(weights)['std'] == pytest.approx(np.std(weights))


def test_drift_and_growth_are_paired_and_handle_zero_denominators():
    result = changes(np.array([-1., 2.]), np.array([-2., 4.]))
    assert result['normalized_drift'] == pytest.approx(1)
    assert result['mean_abs_growth_ratio'] == 2
    assert result['mean_abs_change'] == 1.5
    assert changes(np.zeros(2), np.ones(2))['normalized_drift'] is None
    assert statistics(np.array([])) == {'count': 0}
    assert distances(np.array([]), np.ones(2))['ks'] is None


def test_mask_hash_matches_repository_representation():
    masks = {'b': torch.tensor([True, False]), 'a': torch.tensor([False, True])}
    assert mask_hash(masks) == state_dict_hash({k:v.to(torch.uint8) for k,v in masks.items()})
    assert PATTERN.fullmatch('blocks.0.moe.experts.31.fc1.weight')
    assert PATTERN.fullmatch('blocks.3.moe.router.projection.weight')
    assert not PATTERN.fullmatch('blocks.0.moe.experts.0.fc1.bias')


def test_cached_profiles_match_scipy_with_ties_and_signs():
    from moe_lth.analysis.weight_distributions import empirical_profile, profile_distances
    rng = np.random.default_rng(17)
    a = rng.integers(-4, 5, size=100).astype(np.float32)
    b = rng.integers(-4, 5, size=100).astype(np.float32)
    expected = distances(a, b)
    actual = profile_distances(empirical_profile(a), empirical_profile(b))
    assert actual == pytest.approx(expected)
    a = rng.normal(size=1000).astype(np.float32)
    b = rng.normal(size=1000).astype(np.float32)
    assert profile_distances(empirical_profile(a), empirical_profile(b)) == pytest.approx(distances(a,b))


def test_missing_inputs_produce_explicit_audit(tmp_path, monkeypatch):
    import json
    import sys
    from moe_lth.analysis.weight_distributions import main
    destination = tmp_path/'analysis'
    monkeypatch.setattr(sys, 'argv', ['analysis', '--root', str(tmp_path), '--output', str(destination), '--cases', 'e32/seed_7/final_s0.80'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    audit = json.loads((destination/'checkpoint_audit.json').read_text())
    assert len(audit) == 1
    assert audit[0]['status'] == 'missing_or_incompatible'
    assert 'extension_manifest.json' in audit[0]['error']
    assert (destination/'README.md').exists()
