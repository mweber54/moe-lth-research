import pytest

from moe_lth.experiments.run_routing_init_factorial import summarize_factorial


def test_factorial_summary_decomposes_route_init_and_interaction_variance():
    matrix = {
        "source_seeds": [1, 2],
        "target_seeds": [10, 20],
        "results": [
            {"source_seed": 1, "target_seed": 10, "cross_init_replay_loss": 1.0},
            {"source_seed": 1, "target_seed": 20, "cross_init_replay_loss": 2.0},
            {"source_seed": 2, "target_seed": 10, "cross_init_replay_loss": 3.0},
            {"source_seed": 2, "target_seed": 20, "cross_init_replay_loss": 4.0},
        ],
    }

    report = summarize_factorial(matrix, metric="cross_init_replay_loss")

    assert report["grand_mean"] == pytest.approx(2.5)
    assert report["route_effect_variance"] == pytest.approx(1.0)
    assert report["init_effect_variance"] == pytest.approx(0.25)
    assert report["interaction_variance"] == pytest.approx(0.0)
