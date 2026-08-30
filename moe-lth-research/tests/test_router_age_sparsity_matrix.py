from __future__ import annotations

from pathlib import Path
from statistics import mean

from moe_lth.experiments.run_router_age_sparsity_matrix import (
    ROUTER_AGES,
    SPARSITIES,
    _aggregate,
    _matrix,
    _statistics,
    _write_svg_curves,
    _write_svg_heatmap,
)


def _complete_rows() -> list[dict]:
    rows = []
    for seed in (7, 17, 29):
        for sparsity in SPARSITIES:
            for age in ROUTER_AGES:
                gap = sparsity + age / 10000 + seed / 100000
                rows.append({
                    "reference_seed": seed,
                    "sparsity": sparsity,
                    "router_age": age,
                    "router_step": round(25 * age),
                    "sparse_final_loss": 2 + gap,
                    "dense_final_loss": 2,
                    "ticket_gap": gap,
                    "sparse_initial_validation_loss": 3,
                    "dense_initial_validation_loss": 3,
                    "early_auc_sparse": 4 + gap,
                    "early_auc_dense": 4,
                    "early_auc_gap": gap,
                    "expert_gradient_norm_sparse": 1,
                    "mask_hash": "mask",
                    "router_hash": "router",
                    "shared_state_hash": "shared",
                    "training_sequence_hash": "train",
                    "validation_sequence_hash": "validation",
                    "dense_baseline_reused": True,
                    "audit_passed": True,
                })
    return rows


def test_complete_matrix_aggregation_and_figures(tmp_path: Path):
    rows = _complete_rows()
    matrix = _matrix(rows, mean)

    assert len(rows) == 105
    assert len(matrix) == 5
    assert all(set(row) == {"sparsity", *[f"R{age}" for age in ROUTER_AGES]} for row in matrix)
    assert len(_aggregate(rows)) == 35

    statistics = _statistics(rows)
    assert statistics["n_observations"] == 105
    assert statistics["model"].endswith("seed_fixed_effect")

    heatmap = tmp_path / "heatmap.svg"
    curves = tmp_path / "curves.svg"
    _write_svg_heatmap(heatmap, matrix)
    _write_svg_curves(curves, rows)
    assert "Mean sparse-dense ticket gap" in heatmap.read_text(encoding="utf-8")
    assert "Mean ticket gap" in curves.read_text(encoding="utf-8")
