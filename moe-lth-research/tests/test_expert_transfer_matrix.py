import pytest

from moe_lth.experiments.run_expert_transfer_matrix import summarize_transfer_matrix


def test_factorial_summary_reports_diagonal_vs_off_diagonal_penalty():
    matrix = {
        "layers": ["block_0"],
        "results": [
            {"layer": "block_0", "source_expert": 0, "target_expert": 0, "loss": 1.0},
            {"layer": "block_0", "source_expert": 0, "target_expert": 1, "loss": 2.0},
            {"layer": "block_0", "source_expert": 1, "target_expert": 0, "loss": 3.0},
            {"layer": "block_0", "source_expert": 1, "target_expert": 1, "loss": 1.5},
        ],
    }

    report = summarize_transfer_matrix(matrix)

    assert report["by_layer"]["block_0"]["diagonal_mean"] == pytest.approx(1.25)
    assert report["by_layer"]["block_0"]["off_diagonal_mean"] == pytest.approx(2.5)
    assert report["by_layer"]["block_0"]["mean_transfer_penalty"] == pytest.approx(1.25)
