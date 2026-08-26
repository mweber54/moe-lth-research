from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F

from moe_lth.config import load_config, save_config
from moe_lth.data import build_dataloaders
from moe_lth.models import TinyMoELanguageModel
from moe_lth.routing.interventions import RoutingController
from moe_lth.routing.log_routes import RoutingLogger, sampled_context_records
from moe_lth.routing.route_history import RouteHistory, load_validation_route_batches, save_validation_routes
from moe_lth.training.checkpoint import save_checkpoint
from moe_lth.training.evaluate import evaluate_language_model
from moe_lth.utils import (
    append_jsonl,
    configure_device,
    parameter_count,
    resolve_autocast_dtype,
    create_grad_scaler,
    resolve_data_seed,
    resolve_device,
    seed_everything,
)


def prepare_run_output(output_dir: Path) -> None:
    """Reset generated training artifacts because this trainer does not resume runs."""
    patterns = [
        "logs/*.jsonl",
        "logs/validation_routes_step_*.npz",
        "logs/train_route_history.npz",
        "logs/rich_train_route_history.npz",
        "checkpoints/step_*.pt",
        "summary.json",
    ]
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink()


def configure_router_trainability(model: torch.nn.Module, routing_mode: str) -> None:
    if routing_mode == "fixed_random":
        for block in model.blocks:
            block.moe.router.requires_grad_(False)


def build_controller(config: dict) -> RoutingController:
    routing = config["routing"]
    replay_path = routing.get("replay_path")
    if replay_path:
        try:
            from moe_lth.routing.rich_trace import RichRouteHistory
            history = RichRouteHistory.load(replay_path)
            if not history.traces:
                history = RouteHistory.load(replay_path)
        except Exception:
            history = RouteHistory.load(replay_path)
    else:
        history = None
    layer_swap_pairs = {
        int(layer_id): pairs
        for layer_id, pairs in (routing.get("layer_swap_pairs") or {}).items()
    }
    layer_cyclic_shifts = {
        int(layer_id): int(shift)
        for layer_id, shift in (routing.get("layer_cyclic_shifts") or {}).items()
    }
    return RoutingController(
        mode=routing["mode"],
        num_layers=int(config["model"]["num_layers"]),
        num_experts=int(config["model"]["num_experts"]),
        seed=int(config.get("seed", 0)),
        history=history,
        swap_pairs=routing.get("swap_pairs"),
        layer_swap_pairs=layer_swap_pairs,
        cyclic_shift=int(routing.get("cyclic_shift") or 0),
        layer_cyclic_shifts=layer_cyclic_shifts,
        corruption_fraction=float(routing.get("corruption_fraction", 0.0)),
    )


def build_validation_overrides(
    config: dict,
    step: int,
    device: torch.device,
    controller: RoutingController,
) -> list[list[torch.Tensor]] | None:
    if config["routing"]["mode"] not in {"replay", "swapped", "shuffled_usage", "deconfounded_shuffle", "graded_corruption"}:
        return None
    replay_path = Path(config["routing"]["replay_path"])
    route_dir = replay_path.parent
    route_path = route_dir / f"validation_routes_step_{step}.npz"
    if not route_path.exists():
        candidates = sorted(
            route_dir.glob("validation_routes_step_*.npz"),
            key=lambda p: int(p.stem.split("step_")[-1]),
        )
        if not candidates:
            raise FileNotFoundError(f"Missing baseline validation routes required for replay: {route_path}")
        candidate_steps = [int(p.stem.split("step_")[-1]) for p in candidates]
        previous_steps = [s for s in candidate_steps if s <= step]
        route_path = (
            route_dir / f"validation_routes_step_{max(previous_steps) if previous_steps else min(candidate_steps)}.npz"
            if previous_steps
            else route_dir / f"validation_routes_step_{min(candidate_steps)}.npz"
        )
        if not route_path.exists():
            raise FileNotFoundError(f"Missing baseline validation routes required for replay: {route_path}")
    batches = load_validation_route_batches(route_path, device)
    return [
        [
            controller.transform_replayed(routes, step + batch_id * 100003, layer_id)
            for layer_id, routes in enumerate(layer_routes)
        ]
        for batch_id, layer_routes in enumerate(batches)
    ]


def train_from_config(config: dict) -> dict:
    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    configure_device(device)
    autocast_dtype = resolve_autocast_dtype(config["training"].get("precision", "fp32"), device)
    use_scaler = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = create_grad_scaler(use_scaler)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_run_output(output_dir)
    save_config(config, output_dir / "resolved_config.yaml")

    train_loader, validation_loader = build_dataloaders(
        config["data"],
        int(config["training"]["batch_size"]),
        resolve_data_seed(config),
        reshuffle_each_epoch=bool(config["data"].get("reshuffle_each_epoch", False)),
    )
    model = TinyMoELanguageModel(config["model"]).to(device)
    configure_router_trainability(model, config["routing"]["mode"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    controller = build_controller(config)
    routing_logger = RoutingLogger(output_dir)
    train_history = RouteHistory()
    if config["training"].get("record_rich_routes"):
        from moe_lth.routing.rich_trace import RichRouteHistory
        rich_train_history = RichRouteHistory()
    else:
        rich_train_history = None
    checkpoint_steps = set(int(step) for step in config["training"]["checkpoint_steps"])
    total_steps = int(config["training"]["steps"])
    aux_weight = float(config["routing"]["aux_loss_weight"])
    if config["data"].get("reshuffle_each_epoch", False):
        train_iterator = iter(train_loader)
    else:
        train_iterator = cycle(train_loader)

    if 0 in checkpoint_steps:
        save_checkpoint(
            output_dir / "checkpoints" / "step_0.pt",
            model,
            optimizer if config["training"]["save_optimizer"] else None,
            0,
            None,
            config,
        )

    model.train()
    for step in range(1, total_steps + 1):
        try:
            token_ids, targets = next(train_iterator)
        except StopIteration:
            if config["data"].get("reshuffle_each_epoch", False):
                train_iterator = iter(train_loader)
                token_ids, targets = next(train_iterator)
            else:
                raise
        token_ids, targets = token_ids.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        overrides = controller.overrides(token_ids, step)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            output = model(token_ids, overrides)
            language_loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
            )
            loss = language_loss + aux_weight * output.auxiliary_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()

        if config["training"].get("record_train_routes"):
            for layer_id, trace in enumerate(output.routes):
                train_history.record(step, layer_id, trace.selected_experts)

        if config["training"].get("record_rich_routes") and rich_train_history is not None:
            for layer_id, trace in enumerate(output.routes):
                rich_train_history.record(step, layer_id, trace, token_ids.shape[0], token_ids.shape[1])

        if step % int(config["training"]["log_interval"]) == 0 or step == 1:
            routing_logger.log_step(step, output)
            append_jsonl(
                output_dir / "logs" / "train_metrics.jsonl",
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "language_loss": float(language_loss.detach().cpu()),
                    "auxiliary_loss": float(output.auxiliary_loss.detach().cpu()),
                },
            )
            for record in sampled_context_records(token_ids, output, step):
                append_jsonl(output_dir / "logs" / "expert_context_samples.jsonl", record)

        should_evaluate = step % int(config["training"]["eval_interval"]) == 0 or step in checkpoint_steps
        validation = None
        if should_evaluate:
            validation_overrides = build_validation_overrides(config, step, device, controller)
            validation = evaluate_language_model(
                model,
                validation_loader,
                device,
                controller=controller if config["routing"]["mode"] in {"fixed_random", "random_every_step"} else None,
                override_batches=validation_overrides,
                max_batches=int(config["data"]["validation_blocks"]),
                route_step_offset=step * 100003,
            )
            append_jsonl(
                output_dir / "logs" / "validation_metrics.jsonl",
                {"step": step, "loss": validation["loss"], "perplexity": validation["perplexity"]},
            )
            save_validation_routes(
                output_dir / "logs" / f"validation_routes_step_{step}.npz",
                step,
                validation["routing_batches"],
            )
            model.train()

        if step in checkpoint_steps or step == total_steps:
            save_checkpoint(
                output_dir / "checkpoints" / f"step_{step}.pt",
                model,
                optimizer if config["training"]["save_optimizer"] else None,
                step,
                None if validation is None else validation["loss"],
                config,
            )

    if config["training"].get("record_train_routes"):
        train_history.save(output_dir / "logs" / "train_route_history.npz")

    if config["training"].get("record_rich_routes") and rich_train_history is not None:
        rich_train_history.save(output_dir / "logs" / "rich_train_route_history.npz")

    summary = {
        "output_dir": str(output_dir),
        "parameters": parameter_count(model),
        "device": str(device),
        "steps": total_steps,
        "precision": config["training"].get("precision", "fp32"),
        "final_validation_loss": None if validation is None else validation["loss"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a routing-conditioned MoE experiment.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(train_from_config(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
