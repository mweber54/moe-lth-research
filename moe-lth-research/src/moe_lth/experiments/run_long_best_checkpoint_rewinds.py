from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev

from moe_lth.config import load_config, save_config
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import random_masks_like, save_masks
from moe_lth.pruning.train_ticket import train_ticket
from moe_lth.training.checkpoint import load_checkpoint


DEFAULT_CONDITIONS = ("learned_mask",)
ALL_CONDITIONS = ("learned_mask", "random_mask", "random_reinit", "randomized_routing")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _checkpoint_steps(run_dir: Path) -> set[int]:
    return {
        int(path.stem.split("_")[-1])
        for path in (run_dir / "checkpoints").glob("step_*.pt")
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _suite_dir(config: dict) -> Path:
    output = Path(config["output_dir"])
    return output.parent / f"{output.name}_suite"


def _closest_checkpoint(checkpoint_dir: Path, target_step: int) -> Path:
    checkpoints = list(checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return min(checkpoints, key=lambda path: abs(int(path.stem.split("_")[-1]) - target_step))


def _best_saved_normal_record(config_path: str) -> dict:
    base_config = load_config(config_path)
    seed = int(base_config["seed"])
    run_dir = _suite_dir(base_config) / "normal"
    validation_path = run_dir / "logs" / "validation_metrics.jsonl"
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing validation log for seed {seed}: {validation_path}")

    rows = _read_jsonl(validation_path)
    saved_steps = _checkpoint_steps(run_dir)
    saved_rows = [row for row in rows if int(row["step"]) in saved_steps]
    if not saved_rows:
        raise RuntimeError(f"No validation rows correspond to saved checkpoints in {run_dir}")

    best_saved = min(saved_rows, key=lambda row: float(row["loss"]))
    best_observed = min(rows, key=lambda row: float(row["loss"]))
    summary = _read_json(run_dir / "summary.json")
    checkpoint = run_dir / "checkpoints" / f"step_{int(best_saved['step'])}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing best saved checkpoint: {checkpoint}")

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "resolved_config": str(run_dir / "resolved_config.yaml"),
        "checkpoint": str(checkpoint),
        "best_saved_step": int(best_saved["step"]),
        "best_saved_loss": float(best_saved["loss"]),
        "best_observed_step": int(best_observed["step"]),
        "best_observed_loss": float(best_observed["loss"]),
        "final_loss": float(summary["final_validation_loss"]),
    }


def _prepare_masks(config: dict, checkpoint: str, output_dir: Path, sparsity: float) -> dict[str, str]:
    mask_dir = output_dir / "masks"
    learned_path = mask_dir / f"learned_sparsity_{sparsity:g}.pt"
    random_path = mask_dir / f"random_sparsity_{sparsity:g}.pt"
    if learned_path.exists() and random_path.exists():
        return {"learned": str(learned_path), "random": str(random_path)}

    model = TinyMoELanguageModel(config["model"])
    load_checkpoint(checkpoint, model)
    learned_masks = expert_local_magnitude_masks(model, sparsity)
    save_masks(learned_masks, learned_path)
    save_masks(random_masks_like(learned_masks, int(config["seed"])), random_path)
    return {"learned": str(learned_path), "random": str(random_path)}


def _condition_spec(condition: str, masks: dict[str, str], routing_mode: str) -> tuple[str, bool, str]:
    if condition == "learned_mask":
        return masks["learned"], False, routing_mode
    if condition == "random_mask":
        return masks["random"], False, routing_mode
    if condition == "random_reinit":
        return masks["learned"], True, routing_mode
    if condition == "randomized_routing":
        return masks["learned"], False, "random_every_step"
    raise ValueError(f"Unknown rewind condition: {condition}")


def _run_ticket_if_needed(
    base_config: dict,
    output_dir: Path,
    rewind_checkpoint: Path,
    mask_path: str,
    random_reinit: bool,
    routing_mode: str,
) -> dict:
    result_path = output_dir / "tables" / "ticket_result.json"
    if result_path.exists():
        return _read_json(result_path)

    config = deepcopy(base_config)
    config["routing"]["mode"] = routing_mode
    config["output_dir"] = str(output_dir)
    save_config(config, output_dir / "resolved_config.yaml")
    return train_ticket(config, str(rewind_checkpoint), mask_path, random_reinit)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["condition"],
                float(record["rewind_fraction"]),
                float(record["sparsity"]),
            )
        ].append(float(record["loss"]))
    return [
        {
            "condition": condition,
            "rewind_fraction": fraction,
            "sparsity": sparsity,
            "loss": _summary(losses),
        }
        for (condition, fraction, sparsity), losses in sorted(grouped.items())
    ]


def _write_report(result: dict, output_dir: Path) -> Path:
    report_path = output_dir / "long_best_checkpoint_rewind_results.md"
    summary_path = output_dir / "long_best_checkpoint_rewind_summary.json"
    _write_json(summary_path, result)

    dense_losses = [row["best_saved_loss"] for row in result["selected_checkpoints"]]
    dense = _summary(dense_losses)

    checkpoint_rows = [
        f"| {row['seed']} | {row['best_observed_step']} | {row['best_observed_loss']:.4f} | "
        f"{row['best_saved_step']} | {row['best_saved_loss']:.4f} | {row['final_loss']:.4f} |"
        for row in result["selected_checkpoints"]
    ]
    aggregate_rows = [
        f"| {row['condition']} | {row['sparsity']:.0%} | {row['rewind_fraction']:.0%} | "
        f"{row['loss']['mean']:.4f} | {row['loss']['std']:.4f} | "
        f"{(row['loss']['mean'] / dense['mean'] - 1.0) * 100.0:+.2f}% |"
        for row in result["aggregate"]
    ]
    best_learned = [
        row for row in result["aggregate"] if row["condition"] == "learned_mask"
    ]
    best_line = ""
    if best_learned:
        best = min(best_learned, key=lambda row: row["loss"]["mean"])
        best_line = (
            "\nBest learned-mask rewind: "
            f"{best['rewind_fraction']:.0%}, loss "
            f"`{best['loss']['mean']:.4f} +/- {best['loss']['std']:.4f}`, "
            f"{(best['loss']['mean'] / dense['mean'] - 1.0) * 100.0:+.2f}% vs best-saved dense.\n"
        )

    figure = (
        "\n![Long-budget best-checkpoint rewind curves](long_best_checkpoint_rewind_curves.png)\n"
        if result.get("figure_written")
        else ""
    )

    markdown = f"""# Long-Budget Best-Checkpoint Rewind Results

Seeds: {", ".join(str(seed) for seed in result["seeds"])}

This reduced suite tests 80% expert-local magnitude masks extracted from the
best saved **normal-routing** checkpoint in each long-budget seed. Rewind points
are initialization, 10%, and 25% of the original 10k-step budget.

Best-saved dense normal loss: `{dense['mean']:.4f} +/- {dense['std']:.4f}`.
{best_line}
## Selected Dense Checkpoints

| Seed | Best observed step | Best observed loss | Best saved step | Best saved loss | Final 10k loss |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(checkpoint_rows)}

## Rewind Aggregate

| Condition | Sparsity | Rewind fraction | Mean loss | Std | Delta vs best-saved dense |
|---|---:|---:|---:|---:|---:|
{chr(10).join(aggregate_rows)}
{figure}

Raw aggregate: [long_best_checkpoint_rewind_summary.json](long_best_checkpoint_rewind_summary.json)
"""
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


def _write_figure(result: dict, output_dir: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axis = plt.subplots(figsize=(7, 4))
    for condition in result["conditions"]:
        rows = [
            row for row in result["aggregate"] if row["condition"] == condition
        ]
        if not rows:
            continue
        rows.sort(key=lambda row: row["rewind_fraction"])
        axis.errorbar(
            [row["rewind_fraction"] for row in rows],
            [row["loss"]["mean"] for row in rows],
            yerr=[row["loss"]["std"] for row in rows],
            marker="o",
            capsize=3,
            label=condition.replace("_", " "),
        )
    dense = mean(row["best_saved_loss"] for row in result["selected_checkpoints"])
    axis.axhline(dense, color="black", linestyle="--", label="best saved dense")
    axis.set_title("Long-budget best-checkpoint rewinds")
    axis.set_xlabel("Rewind fraction")
    axis.set_ylabel("Validation loss")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "long_best_checkpoint_rewind_curves.png", dpi=160)
    plt.close(figure)
    return True


def run_long_best_checkpoint_rewinds(
    config_paths: list[str],
    output_dir: str,
    sparsity: float,
    rewind_fractions: list[float],
    conditions: list[str],
) -> dict:
    invalid = sorted(set(conditions) - set(ALL_CONDITIONS))
    if invalid:
        raise ValueError(f"Unknown conditions: {', '.join(invalid)}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "long_best_checkpoint_rewind_status.json"
    selected = [_best_saved_normal_record(config_path) for config_path in config_paths]
    existing = _read_json(status_path) if status_path.exists() else {}
    records_by_key = {
        (
            int(row["seed"]),
            row["condition"],
            float(row["rewind_fraction"]),
            float(row["sparsity"]),
        ): row
        for row in existing.get("records", [])
    }

    for checkpoint_record in selected:
        seed = int(checkpoint_record["seed"])
        source_run_dir = Path(checkpoint_record["run_dir"])
        base_config = load_config(source_run_dir / "resolved_config.yaml")
        seed_dir = destination / f"seed_{seed}" / "normal"
        save_config(base_config, seed_dir / "source_resolved_config.yaml")
        masks = _prepare_masks(base_config, checkpoint_record["checkpoint"], seed_dir, sparsity)

        for fraction in rewind_fractions:
            target_step = round(int(base_config["training"]["steps"]) * float(fraction))
            rewind_checkpoint = _closest_checkpoint(source_run_dir / "checkpoints", target_step)
            for condition in conditions:
                mask_path, random_reinit, routing_mode = _condition_spec(
                    condition, masks, base_config["routing"]["mode"]
                )
                run_dir = (
                    seed_dir
                    / "rewind"
                    / f"sparsity_{sparsity:g}"
                    / f"{condition}_fraction_{float(fraction):g}"
                )
                print(
                    f"[long-best-rewind] seed={seed} condition={condition} "
                    f"sparsity={sparsity:g} rewind={float(fraction):g} "
                    f"checkpoint={rewind_checkpoint.name}",
                    flush=True,
                )
                result = _run_ticket_if_needed(
                    base_config,
                    run_dir,
                    rewind_checkpoint,
                    mask_path,
                    random_reinit,
                    routing_mode,
                )
                record = {
                    "seed": seed,
                    "condition": condition,
                    "sparsity": sparsity,
                    "rewind_fraction": float(fraction),
                    "rewind_checkpoint": str(rewind_checkpoint),
                    "mask_path": mask_path,
                    "random_reinitialize_experts": random_reinit,
                    "routing_mode": routing_mode,
                    "output_dir": str(run_dir),
                    "loss": float(result["loss"]),
                    "perplexity": float(result["perplexity"]),
                }
                records_by_key[(seed, condition, float(fraction), sparsity)] = record
                records = sorted(
                    records_by_key.values(),
                    key=lambda row: (
                        int(row["seed"]),
                        row["condition"],
                        float(row["rewind_fraction"]),
                        float(row["sparsity"]),
                    ),
                )
                partial = {
                    "seeds": [row["seed"] for row in selected],
                    "sparsity": sparsity,
                    "rewind_fractions": sorted(
                        {float(row["rewind_fraction"]) for row in records} | set(rewind_fractions)
                    ),
                    "conditions": [
                        condition
                        for condition in ALL_CONDITIONS
                        if condition in ({row["condition"] for row in records} | set(conditions))
                    ],
                    "selected_checkpoints": selected,
                    "records": records,
                    "aggregate": _aggregate(records),
                    "partial": True,
                }
                _write_json(status_path, partial)

    records = sorted(
        records_by_key.values(),
        key=lambda row: (
            int(row["seed"]),
            row["condition"],
            float(row["rewind_fraction"]),
            float(row["sparsity"]),
        ),
    )
    result = {
        "seeds": [row["seed"] for row in selected],
        "sparsity": sparsity,
        "rewind_fractions": sorted(
            {float(row["rewind_fraction"]) for row in records} | set(rewind_fractions)
        ),
        "conditions": [
            condition
            for condition in ALL_CONDITIONS
            if condition in ({row["condition"] for row in records} | set(conditions))
        ],
        "selected_checkpoints": selected,
        "records": records,
        "aggregate": _aggregate(records),
    }
    result["figure_written"] = _write_figure(result, destination)
    report_path = _write_report(result, destination)
    result["report"] = str(report_path)
    _write_json(status_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reduced rewinds from long-budget best saved normal checkpoints."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--rewind-fractions", nargs="+", type=float, default=[0.0, 0.1, 0.25])
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    args = parser.parse_args()
    result = run_long_best_checkpoint_rewinds(
        args.configs,
        args.output_dir,
        args.sparsity,
        args.rewind_fractions,
        args.conditions,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
