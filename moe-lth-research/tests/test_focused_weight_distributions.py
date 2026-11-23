import matplotlib.pyplot as plt
import numpy as np
import pytest

from moe_lth.analysis.focused_weight_distributions import display_cdf, draw, rendering_profile


def test_nonzero_histogram_preserves_nonzero_weights_in_zero_bin():
    weights = np.array([-.5, 0., 0., .01, .5])
    edges = np.linspace(-1., 1., 5)
    profile = rendering_profile(weights, edges)
    fig, axes = plt.subplots(1, 3)
    try:
        draw(axes, [profile], ['sparse'], ['blue'], edges, 1., [True])
        density = axes[0].patches[0].get_data().values
        expected, _ = np.histogram(weights[weights != 0], bins=edges, density=True)
        np.testing.assert_allclose(density, expected)
        assert np.sum(density*np.diff(edges)) == pytest.approx(1)
        assert axes[0].get_xlim() == (-1., 1.)
        assert axes[2].get_xlim() == (0., 1.)
    finally:
        plt.close(fig)


def test_cdf_uses_complete_counts_and_distinguishes_zero_exclusion():
    values = np.array([-1., 0., 2.])
    counts = np.array([1, 10, 2])
    x,y = display_cdf(values,counts)
    np.testing.assert_array_equal(x, values)
    np.testing.assert_allclose(y, [1/13, 11/13, 1])
    x,y = display_cdf(values,counts,nonzero=True)
    np.testing.assert_array_equal(x, [-1.,2.])
    np.testing.assert_allclose(y, [1/3,1.])
    assert display_cdf(np.array([0.]),np.array([10]),nonzero=True)[0].size == 0
