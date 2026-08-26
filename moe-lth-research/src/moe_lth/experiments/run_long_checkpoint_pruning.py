from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

from moe_lth.config import load_config, save_config
from moe_lth.pruning.evaluate_pruning import evaluate_pruning


CONDITIONS = ("normal", "random_every_step", "fixed_random", "shuffled_usage")


def _suite_dir(config: dict) -> Path:
    output = Path(config["output_dir"])
    return output.parent / f"{output.name}_suite"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _format_delta(value: float, baseline: float) -> str:
    return f"{(value / baseline - 1.0) * 100.0:+.2f}%"


def _checkpoint_steps(run_dir: Path) -> set[int]:
    return {
        int(path.stem.split("_")[-1])
        for path in (run_dir / "checkpoints").glob("step_*.pt")
    }


def _best_rows(run_dir: Path) -> tuple[dict, dict]:
    rows = _read_jsonl(run_dir / "logs" / "validation_metrics.jsonl")
    if not rows:
        raise ValueError(f"No validation metrics found in {run_dir}.")
    observed = min(rows, key=lambda row: float(row["loss"]))
    saved_steps = _checkpoint_steps(run_dir)
    saved_rows = [row for row in rows if int(row["step"]) in saved_steps]
    if not saved_rows:
        raise ValueError(f"No validation rows have saved checkpoints in {run_dir}.")
    saved = min(saved_rows, key=lambda row: float(row["loss"]))
    return observed, saved


def _pruning_lookup(rows: list[dict]) -> dict[str, float]:
    lookup = {}
    for row in rows:
        condition = row["condition"]
        sparsity = float(row["sparsity"])
        lookup[f"{condition}|{sparsity:g}"] = float(row["loss"])
    return lookup


def _run_condition_pruning(
    source_run_dir: Path,
    destination: Path,
    checkpoint_step: int,
) -> list[dict]:
    pruning_path = destination / "tables" / "pruning_results.json"
    if pruning_path.exists():
        return _read_json(pruning_path)

    config = load_config(source_run_dir / "resolved_config.yaml")
    checkpoint = source_run_dir / "checkpoints" / f"step_{checkpoint_step}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")

    config = deepcopy(config)
    config["output_dir"] = str(destination)
    destination.mkdir(parents=True, exist_ok=True)
    save_config(config, destination / "resolved_config.yaml")
    return evaluate_pruning(config, str(checkpoint))


def _aggregate(records: list[dict]) -> dict:
    final = {}
    best_observed = {}
    best_saved = {}
    pruning: dict[tuple[str, str, float], list[float]] = {}

    for row in records:
        condition = row["condition"]
        final.setdefault(condition, []).append(float(row["final_loss"]))
        best_observed.setdefault(condition, []).append(float(row["best_observed_loss"]))
        best_saved.setdefault(condition, []).append(float(row["best_saved_loss"]))
        for pruning_row in row["pruning"]:
            key = (
                condition,
                pruning_row["condition"],
                float(pruning_row["sparsity"]),
            )
            pruning.setdefault(key, []).append(float(pruning_row["loss"]))

    return {
        "final": {condition: _summary(values) for condition, values in sorted(final.items())},
        "best_observed": {
            condition: _summary(values) for condition, values in sorted(best_observed.items())
        },
        "best_saved": {
            condition: _summary(values) for condition, values in sorted(best_saved.items())
        },
        "pruning_at_best_saved": {
            f"{condition}|{mask_condition}|{sparsity:g}": _summary(values)
            for (condition, mask_condition, sparsity), values in sorted(pruning.items())
        },
    }


def _write_report(result: dict, output_dir: Path) -> Path:
    aggregate = result["aggregate"]
    report_path = output_dir / "best_checkpoint_pruning_results.md"
    summary_path = output_dir / "best_checkpoint_pruning_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    normal_best_saved = aggregate["best_saved"]["normal"]["mean"]
    dense_rows = []
    for condition in CONDITIONS:
        final = aggregate["final"][condition]
        observed = aggregate["best_observed"][condition]
        saved = aggregate["best_saved"][condition]
        dense_rows.append(
            f"| {condition} | {final['mean']:.4f} | "
            f"{observed['mean']:.4f} | {saved['mean']:.4f} | "
            f"{_format_delta(saved['mean'], normal_best_saved) if condition != 'normal' else '-'} |"
        )

    checkpoint_rows = []
    for row in result["records"]:
        checkpoint_rows.append(
            f"| {row['seed']} | {row['condition']} | "
            f"{row['best_observed_step']} | {row['best_observed_loss']:.4f} | "
            f"{row['best_saved_step']} | {row['best_saved_loss']:.4f} |"
        )

    pruning_rows = []
    for condition in CONDITIONS:
        values = []
        for sparsity in (0.5, 0.8):
            for mask_condition in ("magnitude", "random_mask", "other_expert_mask"):
                stats = aggregate["pruning_at_best_saved"].get(
                    f"{condition}|{mask_condition}|{sparsity:g}"
                )
                values.append("-" if stats is None else f"{stats['mean']:.4f}")
        pruning_rows.append(
            f"| {condition} | {aggregate['best_saved'][condition]['mean']:.4f} | "
            + " | ".join(values)
            + " |"
        )

    markdown = f"""# Long-Budget Best-Checkpoint Pruning

Seeds: {", ".join(str(seed) for seed in result["seeds"])}

This follow-up reinterprets the 10k-step multi-domain causal-control suite by
selecting the best validation checkpoint for each seed and condition. The
pruning pass uses the best **saved** checkpoint, so it never overwrites the
original final-10k pruning artifacts.

## Dense Checkpoint Comparison

| Condition | Final 10k loss | Best observed loss | Best saved-checkpoint loss | Delta vs normal best saved |
|---|---:|---:|---:|---:|
{chr(10).join(dense_rows)}

## Selected Checkpoints

| Seed | Condition | Best observed step | Best observed loss | Best saved step | Best saved loss |
|---:|---|---:|---:|---:|---:|
{chr(10).join(checkpoint_rows)}

## Direct Pruning at Best Saved Checkpoint

| Routing condition | Dense | 50% magnitude | 50% random | 50% other expert | 80% magnitude | 80% random | 80% other expert |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pruning_rows)}

Raw aggregate: [best_checkpoint_pruning_summary.json](best_checkpoint_pruning_summary.json)
"""
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


def run_long_checkpoint_pruning(config_paths: list[str], output_dir: str) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    seeds = []

    for config_path in config_paths:
        base_config = load_config(config_path)
        seed = int(base_config["seed"])
        seeds.append(seed)
        suite_dir = _suite_dir(base_config)
        for condition in CONDITIONS:
            source_run_dir = suite_dir / condition
            observed, saved = _best_rows(source_run_dir)
            summary = _read_json(source_run_dir / "summary.json")
            checkpoint_step = int(saved["step"])
            destination = output / f"seed_{seed}" / condition
            pruning = _run_condition_pruning(source_run_dir, destination, checkpoint_step)
            records.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "source_run_dir": str(source_run_dir),
                    "best_observed_step": int(observed["step"]),
                    "best_observed_loss": float(observed["loss"]),
                    "best_saved_step": checkpoint_step,
                    "best_saved_loss": float(saved["loss"]),
                    "final_loss": float(summary["final_validation_loss"]),
                    "checkpoint": str(source_run_dir / "checkpoints" / f"step_{checkpoint_step}.pt"),
                    "output_dir": str(destination),
                    "pruning": pruning,
                }
            )

        partial = {
            "seeds": sorted(set(seeds)),
            "records": records,
            "aggregate": _aggregate(records),
            "partial": True,
        }
        (output / "best_checkpoint_pruning_status.json").write_text(
            json.dumps(partial, indent=2),
            encoding="utf-8",
        )

    result = {
        "seeds": sorted(seeds),
        "records": records,
        "aggregate": _aggregate(records),
    }
    report_path = _write_report(result, output)
    result["report"] = str(report_path)
    (output / "best_checkpoint_pruning_status.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pruning at best saved checkpoints for long-budget causal controls."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_long_checkpoint_pruning(args.configs, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
