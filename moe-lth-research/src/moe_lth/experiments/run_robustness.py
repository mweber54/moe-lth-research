from __future__ import annotations

import argparse
import json
from copy import deepcopy
from itertools import product
from pathlib import Path

from moe_lth.config import load_config
from moe_lth.training.train import train_from_config


DEFAULT_GRID = {
    "seeds": [7, 17, 27],
    "num_experts": [4, 8, 16],
    "num_layers": [4, 8],
    "aux_loss_weights": [0.01, 0.1],
}


def run_robustness(base_config: dict, grid: dict | None = None) -> list[dict]:
    grid = grid or DEFAULT_GRID
    root = Path(base_config["output_dir"]).parent / f"{Path(base_config['output_dir']).name}_robustness"
    results = []
    for seed, experts, layers, aux_weight in product(
        grid["seeds"], grid["num_experts"], grid["num_layers"], grid["aux_loss_weights"]
    ):
        config = deepcopy(base_config)
        config["seed"] = seed
        config["model"]["num_experts"] = experts
        config["model"]["num_layers"] = layers
        config["routing"]["aux_loss_weight"] = aux_weight
        name = f"seed_{seed}_experts_{experts}_layers_{layers}_aux_{aux_weight}"
        config["output_dir"] = str(root / name)
        summary = train_from_config(config)
        summary.update({"seed": seed, "num_experts": experts, "num_layers": layers, "aux_loss_weight": aux_weight})
        results.append(summary)
    (root / "robustness_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid-json")
    args = parser.parse_args()
    grid = json.loads(Path(args.grid_json).read_text(encoding="utf-8")) if args.grid_json else None
    print(json.dumps(run_robustness(load_config(args.config), grid), indent=2))


if __name__ == "__main__":
    main()

