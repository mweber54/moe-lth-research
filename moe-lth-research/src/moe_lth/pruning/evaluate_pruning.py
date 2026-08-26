from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch

from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.models import TinyMoELanguageModel
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.training.evaluate import evaluate_language_model
from moe_lth.training.train import build_controller, build_validation_overrides
from moe_lth.utils import resolve_data_seed, resolve_device, seed_everything

from .magnitude_prune import expert_local_magnitude_masks
from .masks import apply_masks_, random_masks_like, save_masks, transfer_expert_masks


def _randomly_reinitialize_experts(model: torch.nn.Module) -> None:
    for block in model.blocks:
        for expert in block.moe.experts:
            for module in expert.modules():
                reset = getattr(module, "reset_parameters", None)
                if callable(reset):
                    reset()


def evaluate_pruning(config: dict, checkpoint: str) -> list[dict]:
    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    _, validation_loader = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
    )
    dense_model = TinyMoELanguageModel(config["model"]).to(device)
    load_checkpoint(checkpoint, dense_model, map_location=device)
    controller = build_controller(config)
    checkpoint_step = int(Path(checkpoint).stem.split("_")[-1])
    validation_overrides = build_validation_overrides(config, checkpoint_step, device, controller)
    evaluation_controller = controller if config["routing"]["mode"] in {"fixed_random", "random_every_step"} else None
    results = []
    dense = evaluate_language_model(
        dense_model,
        validation_loader,
        device,
        controller=evaluation_controller,
        override_batches=validation_overrides,
        max_batches=config["data"]["validation_blocks"],
        route_step_offset=checkpoint_step * 100003,
    )
    results.append({"condition": "dense", "sparsity": 0.0, "loss": dense["loss"], "perplexity": dense["perplexity"]})

    mask_dir = Path(config["output_dir"]) / "masks"
    for sparsity in config["pruning"]["sparsities"]:
        learned_masks = expert_local_magnitude_masks(dense_model, float(sparsity))
        save_masks(learned_masks, mask_dir / f"magnitude_sparsity_{sparsity}.pt")
        conditions = {
            "magnitude": learned_masks,
            "random_mask": random_masks_like(learned_masks, int(config["seed"])),
            "other_expert_mask": transfer_expert_masks(learned_masks, 0, 1),
        }
        for condition, masks in conditions.items():
            model = deepcopy(dense_model)
            apply_masks_(model, masks)
            metrics = evaluate_language_model(
                model,
                validation_loader,
                device,
                controller=evaluation_controller,
                override_batches=validation_overrides,
                max_batches=config["data"]["validation_blocks"],
                route_step_offset=checkpoint_step * 100003,
            )
            results.append(
                {
                    "condition": condition,
                    "sparsity": float(sparsity),
                    "loss": metrics["loss"],
                    "perplexity": metrics["perplexity"],
                    "expert_local_loss": metrics["expert_local_loss"],
                }
            )
        random_reinit_model = deepcopy(dense_model)
        _randomly_reinitialize_experts(random_reinit_model)
        apply_masks_(random_reinit_model, learned_masks)
        metrics = evaluate_language_model(
            random_reinit_model,
            validation_loader,
            device,
            controller=evaluation_controller,
            override_batches=validation_overrides,
            max_batches=config["data"]["validation_blocks"],
            route_step_offset=checkpoint_step * 100003,
        )
        results.append(
            {
                "condition": "magnitude_mask_random_reinit",
                "sparsity": float(sparsity),
                "loss": metrics["loss"],
                "perplexity": metrics["perplexity"],
                "expert_local_loss": metrics["expert_local_loss"],
            }
        )
    destination = Path(config["output_dir"]) / "tables" / "pruning_results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_pruning(load_config(args.config), args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
