from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from moe_lth.config import load_config
from moe_lth.pruning.evaluate_pruning import evaluate_pruning
from moe_lth.training.train import train_from_config


MINIMUM_VIABLE_CONDITIONS = [
    ("normal", "learned"),
    ("random_every_step", "random_every_step"),
    ("replay", "replay"),
    ("swapped", "swapped"),
]


def run_suite(base_config: dict, with_pruning: bool = False) -> list[dict]:
    suite_root = Path(base_config["output_dir"]).parent / f"{Path(base_config['output_dir']).name}_suite"
    summaries = []
    baseline_history = suite_root / "normal" / "logs" / "train_route_history.npz"

    for condition, mode in MINIMUM_VIABLE_CONDITIONS:
        config = deepcopy(base_config)
        config["routing"]["mode"] = mode
        config["output_dir"] = str(suite_root / condition)
        config["training"]["record_train_routes"] = condition == "normal"
        if condition in {"replay", "swapped"}:
            config["routing"]["replay_path"] = str(baseline_history)
        if condition == "swapped":
            config["routing"]["swap_pairs"] = [[0, 1]]
        summary = train_from_config(config)
        if with_pruning:
            checkpoint = suite_root / condition / "checkpoints" / f"step_{config['training']['steps']}.pt"
            summary["pruning"] = evaluate_pruning(config, str(checkpoint))
        summary["condition"] = condition
        summaries.append(summary)

    destination = suite_root / "suite_summary.json"
    destination.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimum viable causal routing suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--with-pruning", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_suite(load_config(args.config), args.with_pruning), indent=2))


if __name__ == "__main__":
    main()

