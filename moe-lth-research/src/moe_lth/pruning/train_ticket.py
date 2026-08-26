from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F

from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.training.train import (
    build_controller,
    build_validation_overrides,
    configure_router_trainability,
)
from moe_lth.training.evaluate import evaluate_language_model
from moe_lth.utils import (
    configure_device,
    create_grad_scaler,
    resolve_autocast_dtype,
    resolve_data_seed,
    resolve_device,
    seed_everything,
)

from .masks import apply_masks_, load_masks
from .rewind import prepare_rewound_model, register_mask_gradient_hooks


def train_ticket(
    config: dict,
    rewind_checkpoint: str,
    mask_path: str,
    random_reinitialize_experts: bool = False,
) -> dict:
    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    configure_device(device)
    autocast_dtype = resolve_autocast_dtype(config["training"].get("precision", "fp32"))
    use_scaler = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = create_grad_scaler(use_scaler)
    masks = load_masks(mask_path)
    model = prepare_rewound_model(
        config["model"],
        rewind_checkpoint,
        masks,
        device,
        random_reinitialize_experts=random_reinitialize_experts,
    )
    configure_router_trainability(model, config["routing"]["mode"])
    handles = register_mask_gradient_hooks(model, masks)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    train_loader, validation_loader = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
    )
    controller = build_controller(config)
    iterator = cycle(train_loader)
    model.train()
    for step in range(1, int(config["training"]["steps"]) + 1):
        token_ids, targets = next(iterator)
        token_ids, targets = token_ids.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            output = model(token_ids, controller.overrides(token_ids, step))
            language_loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
            )
            loss = language_loss + float(config["routing"]["aux_loss_weight"]) * output.auxiliary_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        apply_masks_(model, masks)

    for handle in handles:
        handle.remove()
    validation_overrides = build_validation_overrides(
        config, int(config["training"]["steps"]), device, controller
    )
    metrics = evaluate_language_model(
        model,
        validation_loader,
        device,
        controller=controller if config["routing"]["mode"] in {"fixed_random", "random_every_step"} else None,
        override_batches=validation_overrides,
        max_batches=config["data"]["validation_blocks"],
        route_step_offset=int(config["training"]["steps"]) * 100003,
    )
    result = {
        "rewind_checkpoint": rewind_checkpoint,
        "mask_path": mask_path,
        "random_reinitialize_experts": random_reinitialize_experts,
        "loss": metrics["loss"],
        "perplexity": metrics["perplexity"],
        "expert_local_loss": metrics["expert_local_loss"],
    }
    destination = Path(config["output_dir"]) / "tables" / "ticket_result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rewind-checkpoint", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--random-reinitialize-experts", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train_ticket(
                load_config(args.config),
                args.rewind_checkpoint,
                args.mask,
                args.random_reinitialize_experts,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
