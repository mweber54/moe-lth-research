from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from moe_lth.config import load_config
from moe_lth.training.train import train_from_config


CONDITIONS = {
    "normal": {"mode": "learned", "aux_loss_weight": 0.01},
    "fixed_random": {"mode": "fixed_random", "aux_loss_weight": 0.01},
    "random_every_step": {"mode": "random_every_step", "aux_loss_weight": 0.01},
    "replay": {"mode": "replay", "aux_loss_weight": 0.01},
    "swapped": {"mode": "swapped", "aux_loss_weight": 0.01, "swap_pairs": [[0, 1]]},
    "shuffled_usage": {"mode": "shuffled_usage", "aux_loss_weight": 0.01},
    "strong_balance": {"mode": "learned", "aux_loss_weight": 0.1},
}


def run_all_conditions(base_config: dict) -> list[dict]:
    root = Path(base_config["output_dir"]).parent / f"{Path(base_config['output_dir']).name}_all_conditions"
    baseline_history = root / "normal" / "logs" / "train_route_history.npz"
    results = []
    for name, routing in CONDITIONS.items():
        config = deepcopy(base_config)
        config["routing"].update(routing)
        config["output_dir"] = str(root / name)
        config["training"]["record_train_routes"] = name == "normal"
        if name in {"replay", "swapped", "shuffled_usage"}:
            config["routing"]["replay_path"] = str(baseline_history)
        summary = train_from_config(config)
        summary["condition"] = name
        results.append(summary)
    (root / "conditions_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_all_conditions(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()

