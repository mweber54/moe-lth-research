from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from moe_lth.config import load_config
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import random_masks_like, save_masks
from moe_lth.pruning.train_ticket import train_ticket
from moe_lth.training.checkpoint import load_checkpoint


def _closest_checkpoint(checkpoint_dir: Path, target_step: int) -> Path:
    paths = list(checkpoint_dir.glob("step_*.pt"))
    return min(paths, key=lambda path: abs(int(path.stem.split("_")[-1]) - target_step))


def run_rewind_suite(config: dict, final_checkpoint: str, sparsity: float = 0.8) -> list[dict]:
    checkpoint_path = Path(final_checkpoint)
    if not checkpoint_path.exists():
        available = sorted(str(path) for path in Path("results").rglob("step_*.pt"))
        suggestion = (
            "\nAvailable checkpoints:\n  " + "\n  ".join(available[-20:])
            if available
            else "\nNo checkpoints were found. Run Phase 1 first."
        )
        raise FileNotFoundError(f"Final checkpoint does not exist: {checkpoint_path}{suggestion}")
    config = deepcopy(config)
    config["output_dir"] = str(checkpoint_path.parent.parent)
    model = TinyMoELanguageModel(config["model"])
    load_checkpoint(final_checkpoint, model)
    learned_masks = expert_local_magnitude_masks(model, sparsity)
    mask_dir = Path(config["output_dir"]) / "masks"
    learned_path = mask_dir / f"learned_sparsity_{sparsity}.pt"
    random_path = mask_dir / f"random_sparsity_{sparsity}.pt"
    save_masks(learned_masks, learned_path)
    save_masks(random_masks_like(learned_masks, int(config["seed"])), random_path)

    destination = Path(config["output_dir"]) / "tables" / f"rewind_suite_sparsity_{sparsity}.json"
    existing_results = (
        json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else []
    )
    existing_by_key = {
        (row["condition"], float(row["rewind_fraction"]), float(row["sparsity"])): row
        for row in existing_results
    }

    checkpoint_dir = Path(final_checkpoint).parent
    total_steps = int(config["training"]["steps"])
    results = []
    for fraction in config["pruning"]["rewind_fractions"]:
        rewind = _closest_checkpoint(checkpoint_dir, round(total_steps * float(fraction)))
        conditions = [
            ("learned_mask", learned_path, False, config["routing"]["mode"]),
            ("random_mask", random_path, False, config["routing"]["mode"]),
            ("random_reinit", learned_path, True, config["routing"]["mode"]),
            ("randomized_routing", learned_path, False, "random_every_step"),
        ]
        for condition, mask_path, random_reinit, routing_mode in conditions:
            key = (condition, float(fraction), float(sparsity))
            if key in existing_by_key:
                results.append(existing_by_key[key])
                continue
            ticket_config = deepcopy(config)
            ticket_config["routing"]["mode"] = routing_mode
            ticket_config["output_dir"] = str(
                Path(config["output_dir"])
                / "rewind"
                / f"sparsity_{sparsity:g}"
                / f"{condition}_fraction_{fraction:g}"
            )
            result = train_ticket(ticket_config, str(rewind), str(mask_path), random_reinit)
            result.update({"condition": condition, "rewind_fraction": float(fraction), "sparsity": sparsity})
            results.append(result)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(results, indent=2), encoding="utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--final-checkpoint", required=True)
    parser.add_argument("--sparsity", type=float, default=0.8)
    args = parser.parse_args()
    try:
        results = run_rewind_suite(load_config(args.config), args.final_checkpoint, args.sparsity)
    except FileNotFoundError as error:
        parser.error(str(error))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
