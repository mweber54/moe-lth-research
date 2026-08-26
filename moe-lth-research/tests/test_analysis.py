import numpy as np

from moe_lth.analysis.expert_specificity import jensen_shannon


def test_jensen_shannon_is_zero_for_identical_distributions():
    values = np.array([1, 2, 3])
    assert jensen_shannon(values, values) == 0.0


def test_jensen_shannon_increases_for_separate_distributions():
    assert jensen_shannon(np.array([1, 0]), np.array([0, 1])) > 0.5

