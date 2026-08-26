from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from itertools import cycle
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn.functional as F

from moe_lth.config import load_config, save_config
from moe_lth.data import build_dataloaders
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.masks import MaskDict, apply_masks_, expert_weight_parameters, load_masks, save_masks
from moe_lth.pruning.rewind import prepare_rewound_model, register_mask_gradient_hooks
from moe_lth.training.checkpoint import load_checkpoint, save_checkpoint
from moe_lth.training.evaluate import evaluate_language_model
from moe_lth.training.train import build_controller, build_validation_overrides, configure_router_trainability
from moe_lth.utils import (
    append_jsonl,
    configure_device,
    create_grad_scaler,
    resolve_autocast_dtype,
    resolve_data_seed,
    resolve_device,
    seed_everything,
)


REPRESENTATIVE = {
    "label": "Balanced multi-domain: 8E / top-1 / 4L",
    "dataset_dir": "balanced_multi_domain",
    "variant": "experts_8_topk_1_layers_4",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_dir(phase4_root: Path, seed: int) -> Path:
    return phase4_root / REPRESENTATIVE["dataset_dir"] / f"seed_{seed}" / REPRESENTATIVE["variant"]


def _expert_key(parameter_name: str) -> str:
    prefix, remainder = parameter_name.split(".moe.experts.", maxsplit=1)
    expert_id = remainder.split(".", maxsplit=1)[0]
    return f"{prefix}.moe.experts.{expert_id}"


def _all_expert_masks(model: torch.nn.Module) -> MaskDict:
    return {
        name: torch.ones_like(parameter, dtype=torch.bool, device="cpu")
        for name, parameter in expert_weight_parameters(model).items()
    }


def _masked_expert_local_magnitude_masks(
    model: torch.nn.Module,
    previous_masks: MaskDict,
    sparsity: float,
) -> MaskDict:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("Sparsity must be in [0, 1).")

    grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, parameter in expert_weight_parameters(model).items():
        grouped[_expert_key(name)].append((name, parameter))

    masks: MaskDict = {}
    for parameters in grouped.values():
        total_count = sum(parameter.numel() for _, parameter in parameters)
        keep_count = max(1, int(round(total_count * (1.0 - sparsity))))
        magnitudes = []
        active = []
        for name, parameter in parameters:
            magnitudes.append(parameter.detach().abs().flatten().cpu())
            active.append(previous_masks[name].bool().flatten().cpu())
        flat_magnitudes = torch.cat(magnitudes)
        flat_active = torch.cat(active)
        active_indices = flat_active.nonzero(as_tuple=False).flatten()
        if keep_count > active_indices.numel():
            keep_count = int(active_indices.numel())
        active_magnitudes = flat_magnitudes[active_indices]
        selected = active_indices[active_magnitudes.topk(keep_count, sorted=False).indices]
        flat_mask = torch.zeros(total_count, dtype=torch.bool)
        flat_mask[selected] = True

        offset = 0
        for name, parameter in parameters:
            count = parameter.numel()
            masks[name] = flat_mask[offset : offset + count].reshape_as(parameter)
            offset += count
    return masks


def _round_sparsities(target_sparsity: float, rounds: int) -> list[float]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    final_keep = 1.0 - target_sparsity
    return [1.0 - final_keep ** (index / rounds) for index in range(1, rounds + 1)]


def _closest_checkpoint(checkpoint_dir: Path, target_step: int) -> Path:
    paths = list(checkpoint_dir.glob("step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return min(paths, key=lambda path: abs(int(path.stem.split("_")[-1]) - target_step))


def _train_masked_model(
    config: dict,
    rewind_checkpoint: Path,
    masks: MaskDict,
    output_dir: Path,
) -> tuple[TinyMoELanguageModel, dict]:
    result_path = output_dir / "tables" / "imp_round_result.json"
    checkpoint_path = output_dir / "checkpoints" / "final.pt"
    device = resolve_device(config["device"])

    if result_path.exists() and checkpoint_path.exists():
        model = TinyMoELanguageModel(config["model"]).to(device)
        load_checkpoint(checkpoint_path, model, map_location=device)
        return model, _read_json(result_path)

    seed_everything(int(config["seed"]))
    configure_device(device)
    autocast_dtype = resolve_autocast_dtype(config["training"].get("precision", "fp32"))
    use_scaler = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = create_grad_scaler(use_scaler)

    run_config = deepcopy(config)
    run_config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(run_config, output_dir / "resolved_config.yaml")

    model = prepare_rewound_model(
        run_config["model"],
        str(rewind_checkpoint),
        masks,
        device,
        random_reinitialize_experts=False,
    )
    configure_router_trainability(model, run_config["routing"]["mode"])
    handles = register_mask_gradient_hooks(model, masks)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_config["training"]["learning_rate"]),
        weight_decay=float(run_config["training"]["weight_decay"]),
    )
    train_loader, validation_loader = build_dataloaders(
        run_config["data"],
        int(run_config["training"]["batch_size"]),
        resolve_data_seed(run_config),
    )
    controller = build_controller(run_config)
    iterator = cycle(train_loader)
    total_steps = int(run_config["training"]["steps"])
    aux_weight = float(run_config["routing"]["aux_loss_weight"])
    log_interval = int(run_config["training"]["log_interval"])

    model.train()
    for step in range(1, total_steps + 1):
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
            loss = language_loss + aux_weight * output.auxiliary_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(run_config["training"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        apply_masks_(model, masks)

        if step % log_interval == 0 or step == 1:
            append_jsonl(
                output_dir / "logs" / "train_metrics.jsonl",
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "language_loss": float(language_loss.detach().cpu()),
                    "auxiliary_loss": float(output.auxiliary_loss.detach().cpu()),
                },
            )

    for handle in handles:
        handle.remove()

    validation_overrides = build_validation_overrides(
        run_config,
        total_steps,
        device,
        controller,
    )
    metrics = evaluate_language_model(
        model,
        validation_loader,
        device,
        controller=controller if run_config["routing"]["mode"] in {"fixed_random", "random_every_step"} else None,
        override_batches=validation_overrides,
        max_batches=int(run_config["data"]["validation_blocks"]),
        route_step_offset=total_steps * 100003,
    )
    result = {
        "rewind_checkpoint": str(rewind_checkpoint),
        "loss": metrics["loss"],
        "perplexity": metrics["perplexity"],
        "expert_local_loss": metrics["expert_local_loss"],
    }
    _write_json(result_path, result)
    save_checkpoint(checkpoint_path, model, None, total_steps, metrics["loss"], run_config)
    return model, result


def _load_model_from_checkpoint(config: dict, checkpoint: Path, device: torch.device) -> TinyMoELanguageModel:
    model = TinyMoELanguageModel(config["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    return model


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate(seed_results: list[dict]) -> list[dict]:
    grouped: dict[int, list[float]] = defaultdict(list)
    sparsity_by_round = {}
    for seed_result in seed_results:
        for row in seed_result["rounds"]:
            round_index = int(row["round"])
            grouped[round_index].append(float(row["loss"]))
            sparsity_by_round[round_index] = float(row["sparsity"])
    return [
        {
            "round": round_index,
            "sparsity": sparsity_by_round[round_index],
            "loss": _summary(values),
        }
        for round_index, values in sorted(grouped.items())
    ]


def _write_report(result: dict, output_dir: Path) -> Path:
    report_path = output_dir / "imp_representative_results.md"
    summary_path = output_dir / "imp_representative_summary.json"
    _write_json(summary_path, result)

    dense = _summary([row["dense_loss"] for row in result["seed_results"]])
    round_rows = [
        f"| {row['round']} | {row['sparsity']:.2%} | {row['loss']['mean']:.4f} | "
        f"{row['loss']['std']:.4f} | {(row['loss']['mean'] / dense['mean'] - 1.0) * 100.0:+.2f}% |"
        for row in result["aggregate"]
    ]
    seed_rows = []
    for seed_result in result["seed_results"]:
        for row in seed_result["rounds"]:
            seed_rows.append(
                f"| {seed_result['seed']} | {row['round']} | {row['sparsity']:.2%} | "
                f"{row['loss']:.4f} | {row['perplexity']:.4f} |"
            )

    markdown = f"""# Iterative Magnitude Pruning Representative Results

Representative: {REPRESENTATIVE["label"]}

Seeds: {", ".join(str(seed) for seed in result["seeds"])}

This suite performs expert-local iterative magnitude pruning. Each round prunes
the currently surviving expert weights by magnitude, rewinds surviving weights
to the selected checkpoint, retrains for the original schedule, and then uses
the trained sparse model to choose the next round's mask.

Dense representative loss: `{dense['mean']:.4f} +/- {dense['std']:.4f}`.

## Aggregate

| IMP round | Sparsity | Mean loss | Std | Delta vs dense |
|---:|---:|---:|---:|---:|
{chr(10).join(round_rows)}

## Per-Seed Rows

| Seed | IMP round | Sparsity | Loss | Perplexity |
|---:|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

Raw aggregate: [imp_representative_summary.json](imp_representative_summary.json)
"""
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


def run_imp_representative(
    phase4_root: str,
    output_dir: str,
    seeds: list[int],
    target_sparsity: float,
    rounds: int,
    rewind_fraction: float,
) -> dict:
    source_root = Path(phase4_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "imp_representative_status.json"
    round_sparsities = _round_sparsities(target_sparsity, rounds)
    seed_results = []

    for seed in seeds:
        source_run_dir = _run_dir(source_root, seed)
        resolved_path = source_run_dir / "resolved_config.yaml"
        summary_path = source_run_dir / "summary.json"
        if not resolved_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing representative dense run: {source_run_dir}")

        config = load_config(resolved_path)
        dense_summary = _read_json(summary_path)
        total_steps = int(config["training"]["steps"])
        rewind_step = round(total_steps * rewind_fraction)
        rewind_checkpoint = _closest_checkpoint(source_run_dir / "checkpoints", rewind_step)
        dense_checkpoint = source_run_dir / "checkpoints" / f"step_{total_steps}.pt"
        if not dense_checkpoint.exists():
            raise FileNotFoundError(f"Missing dense final checkpoint: {dense_checkpoint}")

        device = resolve_device(config["device"])
        current_model = _load_model_from_checkpoint(config, dense_checkpoint, device)
        current_masks = _all_expert_masks(current_model)
        rounds_out = []
        seed_dir = destination / f"seed_{seed}"
        save_config(config, seed_dir / "source_resolved_config.yaml")

        for round_index, sparsity in enumerate(round_sparsities, start=1):
            round_dir = seed_dir / f"round_{round_index:02d}_sparsity_{sparsity:.4f}"
            mask_path = round_dir / "masks" / "imp_mask.pt"
            if mask_path.exists():
                current_masks = load_masks(mask_path)
            else:
                current_masks = _masked_expert_local_magnitude_masks(
                    current_model,
                    current_masks,
                    sparsity,
                )
                save_masks(current_masks, mask_path)

            print(
                f"[imp] seed={seed} round={round_index}/{rounds} "
                f"sparsity={sparsity:.4f} rewind={rewind_checkpoint.name}",
                flush=True,
            )
            current_model, metrics = _train_masked_model(
                config,
                rewind_checkpoint,
                current_masks,
                round_dir,
            )
            row = {
                "round": round_index,
                "sparsity": sparsity,
                "rewind_fraction": rewind_fraction,
                "rewind_checkpoint": str(rewind_checkpoint),
                "mask_path": str(mask_path),
                "output_dir": str(round_dir),
                "loss": float(metrics["loss"]),
                "perplexity": float(metrics["perplexity"]),
            }
            rounds_out.append(row)
            partial = {
                "representative": REPRESENTATIVE,
                "phase4_root": phase4_root,
                "seeds": seeds,
                "target_sparsity": target_sparsity,
                "rounds": rounds,
                "rewind_fraction": rewind_fraction,
                "seed_results": seed_results
                + [
                    {
                        "seed": seed,
                        "source_run_dir": str(source_run_dir),
                        "dense_loss": float(dense_summary["final_validation_loss"]),
                        "rounds": rounds_out,
                    }
                ],
                "partial": True,
            }
            partial["aggregate"] = _aggregate(partial["seed_results"])
            _write_json(status_path, partial)

        seed_results.append(
            {
                "seed": seed,
                "source_run_dir": str(source_run_dir),
                "dense_loss": float(dense_summary["final_validation_loss"]),
                "rounds": rounds_out,
            }
        )

    result = {
        "representative": REPRESENTATIVE,
        "phase4_root": phase4_root,
        "seeds": seeds,
        "target_sparsity": target_sparsity,
        "rounds": rounds,
        "rewind_fraction": rewind_fraction,
        "seed_results": seed_results,
        "aggregate": _aggregate(seed_results),
    }
    report = _write_report(result, destination)
    result["report"] = str(report)
    _write_json(status_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reduced IMP on the balanced multi-domain representative cell."
    )
    parser.add_argument("--phase4-root", default="results/phase4_robustness")
    parser.add_argument("--output-dir", default="results/imp_representative")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--target-sparsity", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--rewind-fraction", type=float, default=0.0)
    args = parser.parse_args()
    result = run_imp_representative(
        args.phase4_root,
        args.output_dir,
        args.seeds,
        args.target_sparsity,
        args.rounds,
        args.rewind_fraction,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
