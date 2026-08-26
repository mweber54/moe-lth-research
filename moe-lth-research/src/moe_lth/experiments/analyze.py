from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from moe_lth.analysis.expert_usage import usage_summary
from moe_lth.analysis.routing_stability import routing_agreement
from moe_lth.analysis.router_geometry import router_vector_similarities
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import mask_jaccard, save_masks
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.visualization.plot_routing import plot_expert_usage
from moe_lth.visualization.plot_masks import plot_similarity_heatmap
from moe_lth.visualization.plot_results import plot_conceptual_diagram, plot_route_mask_scatter
from moe_lth.visualization.plot_routing import plot_routing_stability


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = list((run_dir / "checkpoints").glob("step_*.pt"))
    return max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))


def analyze_suite(suite_dir: str, sparsity: float = 0.8) -> dict:
    root = Path(suite_dir)
    if not root.exists():
        results_root = Path("results")
        available = sorted(
            str(path)
            for path in results_root.rglob("*_suite")
            if path.is_dir()
        ) if results_root.exists() else []
        suggestion = (
            "\nAvailable suite directories:\n  " + "\n  ".join(available)
            if available
            else "\nNo completed suite directories were found. Run run_suite first."
        )
        raise FileNotFoundError(f"Suite directory does not exist: {root}{suggestion}")
    conditions = [path for path in root.iterdir() if path.is_dir() and (path / "resolved_config.yaml").exists()]
    if not conditions:
        raise FileNotFoundError(
            f"No completed condition runs were found in {root}. "
            "Expected condition directories containing resolved_config.yaml."
        )
    report: dict = {"conditions": {}, "pairwise": []}
    masks = {}
    final_routes = {}
    stability_rows = []

    for run_dir in sorted(conditions):
        checkpoint = _latest_checkpoint(run_dir)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = TinyMoELanguageModel(payload["config"]["model"])
        load_checkpoint(checkpoint, model)
        condition_masks = expert_local_magnitude_masks(model, sparsity)
        mask_path = run_dir / "masks" / f"analysis_sparsity_{sparsity}.pt"
        save_masks(condition_masks, mask_path)
        masks[run_dir.name] = condition_masks

        route_files = sorted(
            (run_dir / "logs").glob("validation_routes_step_*.npz"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        stability = routing_agreement(str(route_files[0]), str(route_files[-1])) if len(route_files) >= 2 else {}
        if route_files:
            for route_file in route_files:
                checkpoint_step = int(route_file.stem.split("_")[-1])
                stability_rows.append(
                    {
                        "condition": run_dir.name,
                        "checkpoint": checkpoint_step,
                        "agreement": routing_agreement(str(route_file), str(route_files[-1]))["overall"],
                    }
                )
        if route_files:
            final_routes[run_dir.name] = route_files[-1]
        report["conditions"][run_dir.name] = {
            "usage": usage_summary(str(run_dir / "logs" / "expert_usage.jsonl")),
            "routing_stability_first_to_final": stability,
            "router_geometry": router_vector_similarities(model),
            "checkpoint": str(checkpoint),
            "mask": str(mask_path),
        }
        plot_expert_usage(
            str(run_dir / "logs" / "expert_usage.jsonl"),
            str(root / "figures" / f"{run_dir.name}_expert_usage.png"),
        )

    route_mask_points = []
    for first, second in combinations(sorted(masks), 2):
        mask_similarity = mask_jaccard(masks[first], masks[second])
        route_similarity = routing_agreement(str(final_routes[first]), str(final_routes[second]))["overall"]
        row = {
            "first": first,
            "second": second,
            "mask_jaccard": mask_similarity,
            "routing_agreement": route_similarity,
        }
        report["pairwise"].append(row)
        route_mask_points.append((route_similarity, mask_similarity))
    if len(route_mask_points) >= 2:
        report["routing_history_mask_similarity_correlation"] = float(
            np.corrcoef(np.asarray(route_mask_points).T)[0, 1]
        )

    labels = sorted(masks)
    matrix = np.eye(len(labels))
    for first_index, first in enumerate(labels):
        for second_index, second in enumerate(labels):
            matrix[first_index, second_index] = mask_jaccard(masks[first], masks[second])
    plot_conceptual_diagram(str(root / "figures" / "conceptual_diagram.png"))
    plot_similarity_heatmap(matrix, labels, str(root / "figures" / "mask_similarity.png"), "Condition mask similarity")
    plot_route_mask_scatter(report["pairwise"], str(root / "figures" / "routing_history_vs_mask_similarity.png"))
    plot_routing_stability(stability_rows, str(root / "figures" / "routing_stability.png"))

    destination = root / "tables" / "analysis_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--sparsity", type=float, default=0.8)
    args = parser.parse_args()
    try:
        report = analyze_suite(args.suite_dir, args.sparsity)
    except FileNotFoundError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
