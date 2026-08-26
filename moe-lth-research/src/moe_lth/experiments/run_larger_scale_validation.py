from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from moe_lth.config import load_config, save_config
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import mask_jaccard
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.training.train import train_from_config
from moe_lth.experiments.run_rewind_suite import run_rewind_suite


def _checkpoint(config: dict) -> Path:
    return Path(config["output_dir"]) / "checkpoints" / f"step_{config['training']['steps']}.pt"


def _mask_overlap(config: dict, first_checkpoint: Path, second_checkpoint: Path, sparsity: float) -> float:
    first = TinyMoELanguageModel(config["model"])
    second = TinyMoELanguageModel(config["model"])
    load_checkpoint(first_checkpoint, first)
    load_checkpoint(second_checkpoint, second)
    return float(mask_jaccard(
        expert_local_magnitude_masks(first, sparsity),
        expert_local_magnitude_masks(second, sparsity),
    ))


def run_larger_scale_validation(config: dict) -> dict:
    baseline_config = deepcopy(config)
    baseline_config["routing"].update({"mode": "learned", "replay_path": None})
    baseline_config["output_dir"] = str(Path(config["output_dir"]) / "normal")
    baseline_summary = train_from_config(baseline_config)
    baseline_checkpoint = _checkpoint(baseline_config)
    route_archive = Path(baseline_config["output_dir"]) / "logs" / "train_route_history.npz"

    intervention_config = deepcopy(config)
    intervention_config["routing"].update({
        "mode": "deconfounded_shuffle",
        "replay_path": str(route_archive),
    })
    intervention_config["output_dir"] = str(Path(config["output_dir"]) / "deconfounded_shuffle")
    intervention_config["training"]["checkpoint_steps"] = [int(config["training"]["steps"])]
    intervention_config["training"]["save_optimizer"] = False
    intervention_summary = train_from_config(intervention_config)
    intervention_checkpoint = _checkpoint(intervention_config)

    rewind_results = {}
    for sparsity in config["pruning"]["sparsities"]:
        rewind_results[str(sparsity)] = run_rewind_suite(
            baseline_config,
            str(baseline_checkpoint),
            float(sparsity),
        )

    report = {
        "experiment": "p08_larger_scale",
        "scale": {
            "parameters": baseline_summary["parameters"],
            "model": config["model"],
            "dataset": config["data"],
        },
        "baseline": baseline_summary,
        "count_preserving_intervention": intervention_summary,
        "mask_overlap": {
            str(sparsity): _mask_overlap(
                config,
                baseline_checkpoint,
                intervention_checkpoint,
                float(sparsity),
            )
            for sparsity in config["pruning"]["sparsities"]
        },
        "rewind": rewind_results,
    }
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")
    (output_dir / "larger_scale_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P0.8 larger-scale validation protocol.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_larger_scale_validation(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
