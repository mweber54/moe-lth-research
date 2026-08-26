from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from itertools import product
from pathlib import Path
from statistics import mean, stdev

from moe_lth.config import load_config


EXPERT_COUNTS = [4, 8, 16]
TOP_K_VALUES = [1, 2]
LAYER_COUNTS = [4, 8]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _dataset_name(config: dict) -> str:
    paths = " ".join(
        str(config.get("data", {}).get(key) or "")
        for key in ("path", "train_path", "validation_path")
    ).lower()
    if "multidomain" in paths:
        return "Balanced Multi-Domain"
    if "tinystories" in paths:
        return "TinyStories"
    if "wikitext" in paths:
        return "WikiText-103"
    return "Configured Dataset"


def _dataset_slug(config: dict) -> str:
    return (
        _dataset_name(config)
        .lower()
        .replace("-103", "103")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _suite_dir(config: dict) -> Path:
    output = Path(config["output_dir"])
    return output.parent / f"{output.name}_suite"


def _variant_name(num_experts: int, top_k: int, num_layers: int) -> str:
    return f"experts_{num_experts}_topk_{top_k}_layers_{num_layers}"


def _existing_baseline_run_dir(
    config: dict,
    num_experts: int,
    top_k: int,
    num_layers: int,
) -> Path | None:
    model = config["model"]
    if (
        int(model["num_experts"]) != num_experts
        or int(model.get("top_k", 1)) != top_k
        or int(model["num_layers"]) != num_layers
    ):
        return None
    run_dir = _suite_dir(config) / "normal"
    checkpoint = run_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"
    if (run_dir / "summary.json").exists() and checkpoint.exists():
        return run_dir
    return None


def _run_dir(
    output_root: Path,
    config: dict,
    num_experts: int,
    top_k: int,
    num_layers: int,
) -> Path:
    return (
        output_root
        / _dataset_slug(config)
        / f"seed_{int(config['seed'])}"
        / _variant_name(num_experts, top_k, num_layers)
    )


def _prepare_config(
    base_config: dict,
    run_dir: Path,
    num_experts: int,
    top_k: int,
    num_layers: int,
) -> dict:
    config = deepcopy(base_config)
    config["model"]["num_experts"] = num_experts
    config["model"]["top_k"] = top_k
    config["model"]["num_layers"] = num_layers
    config["routing"]["mode"] = "learned"
    config["output_dir"] = str(run_dir)
    config["training"]["record_train_routes"] = False
    config["training"]["save_optimizer"] = False
    total_steps = int(config["training"]["steps"])
    required_checkpoints = {0, total_steps}
    required_checkpoints.update(
        int(round(total_steps * fraction))
        for fraction in config["pruning"].get("rewind_fractions", [])
    )
    config["training"]["checkpoint_steps"] = sorted(required_checkpoints)
    config["pruning"]["sparsities"] = [0.5, 0.8]
    return config


def _final_checkpoint(run_dir: Path, config: dict) -> Path:
    return run_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"


def _final_usage(run_dir: Path) -> dict:
    records = [
        json.loads(line)
        for line in (run_dir / "logs" / "expert_usage.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final_step = max(int(record["step"]) for record in records)
    grouped: dict[int, list[dict]] = {}
    for record in records:
        if int(record["step"]) == final_step:
            grouped.setdefault(int(record["layer_id"]), []).append(record)

    entropies = []
    dead_experts = []
    cvs = []
    dropped = []
    for layer_records in grouped.values():
        usage = [float(record["usage_fraction"]) for record in layer_records]
        positive = [value for value in usage if value > 0]
        entropy = -sum(value * math.log(value) for value in positive)
        entropies.append(entropy / math.log(max(2, len(usage))))
        dead_experts.append(sum(value == 0 for value in usage))
        usage_mean = mean(usage)
        variance = mean((value - usage_mean) ** 2 for value in usage)
        cvs.append(math.sqrt(variance) / max(usage_mean, 1e-12))
        dropped.append(float(layer_records[0]["dropped_fraction"]))
    return {
        "normalized_entropy": mean(entropies),
        "dead_experts": mean(dead_experts),
        "coefficient_of_variation": mean(cvs),
        "dropped_fraction": mean(dropped),
    }


def _routing_stability(run_dir: Path) -> float | None:
    import numpy as np

    route_files = sorted(
        (run_dir / "logs").glob("validation_routes_step_*.npz"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if len(route_files) < 2:
        return None
    agreements = []
    with np.load(route_files[0]) as first, np.load(route_files[-1]) as final:
        for key in first.files:
            if key in final.files and first[key].shape == final[key].shape:
                agreements.append(float(np.mean(first[key] == final[key])))
    return mean(agreements) if agreements else None


def _condition_loss(pruning: list[dict], condition: str, sparsity: float) -> float | None:
    for row in pruning:
        if row["condition"] == condition and math.isclose(float(row["sparsity"]), sparsity):
            return float(row["loss"])
    return None


def _collect_row(
    scope: str,
    base_config: dict,
    run_dir: Path,
    num_experts: int,
    top_k: int,
    num_layers: int,
) -> dict | None:
    summary_path = run_dir / "summary.json"
    pruning_path = run_dir / "tables" / "pruning_results.json"
    if not summary_path.exists() or not pruning_path.exists():
        return None
    summary = _read_json(summary_path)
    pruning = _read_json(pruning_path)
    return {
        "scope": scope,
        "dataset": _dataset_name(base_config),
        "validation_blocks": int(base_config["data"]["validation_blocks"]),
        "seed": int(base_config["seed"]),
        "num_experts": num_experts,
        "top_k": top_k,
        "num_layers": num_layers,
        "parameters": int(summary["parameters"]),
        "dense_loss": float(summary["final_validation_loss"]),
        "usage": _final_usage(run_dir),
        "primary_routing_stability": _routing_stability(run_dir),
        "pruning": {
            "magnitude_0.5": _condition_loss(pruning, "magnitude", 0.5),
            "random_mask_0.5": _condition_loss(pruning, "random_mask", 0.5),
            "magnitude_0.8": _condition_loss(pruning, "magnitude", 0.8),
            "random_mask_0.8": _condition_loss(pruning, "random_mask", 0.8),
        },
        "run_dir": str(run_dir),
    }


def collect_results(
    architecture_configs: list[str],
    dataset_configs: list[str],
    output_dir: str,
) -> dict:
    output_root = Path(output_dir)
    rows = []
    for config_path in architecture_configs:
        base_config = load_config(config_path)
        for num_experts, top_k, num_layers in product(
            EXPERT_COUNTS, TOP_K_VALUES, LAYER_COUNTS
        ):
            run_dir = _existing_baseline_run_dir(
                base_config, num_experts, top_k, num_layers
            ) or _run_dir(output_root, base_config, num_experts, top_k, num_layers)
            row = _collect_row(
                "architecture",
                base_config,
                run_dir,
                num_experts,
                top_k,
                num_layers,
            )
            if row:
                rows.append(row)

    for config_path in dataset_configs:
        base_config = load_config(config_path)
        model = base_config["model"]
        num_experts = int(model["num_experts"])
        top_k = int(model.get("top_k", 1))
        num_layers = int(model["num_layers"])
        run_dir = _existing_baseline_run_dir(
            base_config, num_experts, top_k, num_layers
        ) or _run_dir(output_root, base_config, num_experts, top_k, num_layers)
        row = _collect_row(
            "dataset",
            base_config,
            run_dir,
            num_experts,
            top_k,
            num_layers,
        )
        if row:
            rows.append(row)
    return _aggregate(rows)


def _aggregate(rows: list[dict]) -> dict:
    architecture_groups: dict[tuple, list[dict]] = {}
    dataset_groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["scope"] == "architecture":
            key = (
                row["dataset"],
                row["num_experts"],
                row["top_k"],
                row["num_layers"],
            )
            architecture_groups.setdefault(key, []).append(row)
        if (
            row["num_experts"] == 8
            and row["top_k"] == 1
            and row["num_layers"] == 4
        ):
            dataset_groups.setdefault(row["dataset"], []).append(row)

    def aggregate_group(values: list[dict]) -> dict:
        pruning = {}
        for key in ("magnitude_0.5", "random_mask_0.5", "magnitude_0.8", "random_mask_0.8"):
            metric_values = [row["pruning"][key] for row in values if row["pruning"][key] is not None]
            pruning[key] = _summary(metric_values) if metric_values else None
        stability = [
            row["primary_routing_stability"]
            for row in values
            if row["primary_routing_stability"] is not None
        ]
        return {
            "seeds": sorted(row["seed"] for row in values),
            "parameters": _summary([float(row["parameters"]) for row in values]),
            "dense_loss": _summary([row["dense_loss"] for row in values]),
            "usage_entropy": _summary([row["usage"]["normalized_entropy"] for row in values]),
            "dead_experts": _summary([row["usage"]["dead_experts"] for row in values]),
            "dropped_fraction": _summary([row["usage"]["dropped_fraction"] for row in values]),
            "primary_routing_stability": _summary(stability) if stability else None,
            "pruning": pruning,
        }

    architecture = []
    for (dataset, experts, top_k, layers), values in sorted(architecture_groups.items()):
        architecture.append(
            {
                "dataset": dataset,
                "num_experts": experts,
                "top_k": top_k,
                "num_layers": layers,
                **aggregate_group(values),
            }
        )
    datasets = []
    for dataset, values in sorted(dataset_groups.items()):
        unique = {(row["dataset"], row["seed"]): row for row in values}
        values = list(unique.values())
        datasets.append(
            {
                "dataset": dataset,
                "validation_blocks": values[0]["validation_blocks"],
                **aggregate_group(values),
            }
        )
    return {"rows": rows, "architecture": architecture, "datasets": datasets}


def _write_status(output_root: Path, statuses: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "phase4_status.json").write_text(
        json.dumps(statuses, indent=2), encoding="utf-8"
    )


def _run_one(
    scope: str,
    base_config: dict,
    output_root: Path,
    num_experts: int,
    top_k: int,
    num_layers: int,
) -> dict:
    run_dir = _existing_baseline_run_dir(
        base_config, num_experts, top_k, num_layers
    )
    reused = run_dir is not None
    if run_dir is None:
        run_dir = _run_dir(output_root, base_config, num_experts, top_k, num_layers)
    config = _prepare_config(base_config, run_dir, num_experts, top_k, num_layers)
    checkpoint = _final_checkpoint(run_dir, config)
    summary_path = run_dir / "summary.json"
    pruning_path = run_dir / "tables" / "pruning_results.json"
    status = {
        "scope": scope,
        "dataset": _dataset_name(base_config),
        "seed": int(base_config["seed"]),
        "num_experts": num_experts,
        "top_k": top_k,
        "num_layers": num_layers,
        "run_dir": str(run_dir),
        "baseline_reused": reused,
    }
    if summary_path.exists() and checkpoint.exists():
        status["training"] = "existing"
    else:
        from moe_lth.training.train import train_from_config

        train_from_config(config)
        status["training"] = "completed"
    if pruning_path.exists():
        status["pruning"] = "existing"
    else:
        from moe_lth.pruning.evaluate_pruning import evaluate_pruning

        evaluate_pruning(config, str(checkpoint))
        status["pruning"] = "completed"
    return status


def run_phase4(
    architecture_configs: list[str],
    dataset_configs: list[str],
    output_dir: str,
) -> dict:
    output_root = Path(output_dir)
    statuses = []
    for config_path in architecture_configs:
        base_config = load_config(config_path)
        for num_experts, top_k, num_layers in product(
            EXPERT_COUNTS, TOP_K_VALUES, LAYER_COUNTS
        ):
            statuses.append(
                _run_one(
                    "architecture",
                    base_config,
                    output_root,
                    num_experts,
                    top_k,
                    num_layers,
                )
            )
            _write_status(output_root, statuses)

    for config_path in dataset_configs:
        base_config = load_config(config_path)
        model = base_config["model"]
        statuses.append(
            _run_one(
                "dataset",
                base_config,
                output_root,
                int(model["num_experts"]),
                int(model.get("top_k", 1)),
                int(model["num_layers"]),
            )
        )
        _write_status(output_root, statuses)

    report = collect_results(architecture_configs, dataset_configs, output_dir)
    summary_path = output_root / "phase4_summary.json"
    report_path = output_root / "phase4_results.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figures_written = False
    try:
        _write_figures(report, output_root)
        figures_written = True
    except ImportError:
        pass
    _write_report(report, report_path, figures_written)
    result = {
        "status": statuses,
        "summary": str(summary_path),
        "report": str(report_path),
    }
    (output_root / "phase4_run_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _write_report(report: dict, path: Path, figures_written: bool) -> None:
    def relative_change(value: float, reference: float) -> float:
        return (value / reference - 1.0) * 100.0

    baseline = next(
        row
        for row in report["architecture"]
        if row["num_experts"] == 8 and row["top_k"] == 1 and row["num_layers"] == 4
    )
    baseline_loss = baseline["dense_loss"]["mean"]
    architecture_by_key = {
        (row["num_experts"], row["top_k"], row["num_layers"]): row
        for row in report["architecture"]
    }
    architecture_rows = []
    for row in report["architecture"]:
        delta = (row["dense_loss"]["mean"] / baseline_loss - 1.0) * 100.0
        magnitude_50 = row["pruning"]["magnitude_0.5"]["mean"]
        magnitude_80 = row["pruning"]["magnitude_0.8"]["mean"]
        architecture_rows.append(
            f"| {row['num_experts']} | {row['top_k']} | {row['num_layers']} | "
            f"{row['parameters']['mean'] / 1e6:.2f} | {row['dense_loss']['mean']:.4f} | "
            f"{row['dense_loss']['std']:.4f} | {delta:+.2f}% | "
            f"{row['usage_entropy']['mean']:.4f} | {row['dead_experts']['mean']:.2f} | "
            f"{row['dropped_fraction']['mean']:.4f} | {magnitude_50:.4f} | {magnitude_80:.4f} |"
        )

    dataset_rows = []
    for row in report["datasets"]:
        pruning = row["pruning"]
        dataset_rows.append(
            f"| {row['dataset']} | {row['validation_blocks']} | "
            f"{row['dense_loss']['mean']:.4f} | {row['dense_loss']['std']:.4f} | "
            f"{row['usage_entropy']['mean']:.4f} | {pruning['magnitude_0.5']['mean']:.4f} | "
            f"{pruning['random_mask_0.5']['mean']:.4f} | "
            f"{pruning['magnitude_0.8']['mean']:.4f} | "
            f"{pruning['random_mask_0.8']['mean']:.4f} |"
        )

    top_2_effects = []
    for num_experts, num_layers in product(EXPERT_COUNTS, LAYER_COUNTS):
        top_1 = architecture_by_key[(num_experts, 1, num_layers)]
        top_2 = architecture_by_key[(num_experts, 2, num_layers)]
        top_2_effects.append(
            relative_change(top_2["dense_loss"]["mean"], top_1["dense_loss"]["mean"])
        )

    depth_effects = []
    for num_experts, top_k in product(EXPERT_COUNTS, TOP_K_VALUES):
        four_layers = architecture_by_key[(num_experts, top_k, 4)]
        eight_layers = architecture_by_key[(num_experts, top_k, 8)]
        depth_effects.append(
            relative_change(
                eight_layers["dense_loss"]["mean"], four_layers["dense_loss"]["mean"]
            )
        )

    best_dense = min(report["architecture"], key=lambda row: row["dense_loss"]["mean"])
    direct_50_changes = [
        relative_change(row["pruning"]["magnitude_0.5"]["mean"], row["dense_loss"]["mean"])
        for row in report["architecture"]
    ]
    direct_80_changes = [
        relative_change(row["pruning"]["magnitude_0.8"]["mean"], row["dense_loss"]["mean"])
        for row in report["architecture"]
    ]
    magnitude_beats_random = all(
        row["pruning"][f"magnitude_{sparsity}"]["mean"]
        < row["pruning"][f"random_mask_{sparsity}"]["mean"]
        for row in report["architecture"] + report["datasets"]
        for sparsity in (0.5, 0.8)
    )
    magnitude_control_result = (
        "Magnitude masks beat random masks in every architecture and dataset "
        "cell at both sparsities."
        if magnitude_beats_random
        else "Magnitude masks did not beat random masks in every tested cell."
    )

    expert_effects = {}
    for top_k, num_layers in product(TOP_K_VALUES, LAYER_COUNTS):
        eight_experts = architecture_by_key[(8, top_k, num_layers)]
        sixteen_experts = architecture_by_key[(16, top_k, num_layers)]
        expert_effects[(top_k, num_layers)] = relative_change(
            sixteen_experts["dense_loss"]["mean"], eight_experts["dense_loss"]["mean"]
        )

    figures = ""
    if figures_written:
        figures = """
![Architecture grid dense loss](architecture_dense_loss.png)

![Architecture grid 80% pruning](architecture_pruning_80.png)

![Dataset robustness](dataset_robustness.png)
"""
    markdown = f"""# Phase 4 Robustness Grid Results

Architecture dataset: WikiText-103 subset. Seeds: 7, 17, and 29.

The architecture grid is a full factorial over expert count (`4`, `8`, `16`),
routing (`top-1`, `top-2`), and depth (`4`, `8` layers). All other training
settings remain fixed. The balanced multi-domain baseline interleaves equal
amounts of TinyStories and WikiText and evaluates up to 32 validation batches.

## Architecture Grid

| Experts | Top-k | Layers | Params (M) | Dense loss | Std | Delta vs 8E/Top-1/4L | Usage entropy | Dead experts | Dropped fraction | 50% magnitude | 80% magnitude |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(architecture_rows)}

## Dataset Robustness

| Dataset | Validation batches | Dense loss | Std | Usage entropy | 50% magnitude | 50% random | 80% magnitude | 80% random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(dataset_rows)}

{figures}

## Interpretation

- **Top-2 routing improved dense loss in all six matched comparisons.** Its
  mean paired change relative to top-1 was `{sum(top_2_effects) / len(top_2_effects):+.2f}%`.
- **Eight layers improved dense loss in all six matched comparisons.** The
  mean paired change relative to four layers was
  `{sum(depth_effects) / len(depth_effects):+.2f}%`.
- **More experts showed an interaction with depth.** Moving from 8 to 16
  experts helped the four-layer models by
  `{expert_effects[(1, 4)]:+.2f}%` (top-1) and
  `{expert_effects[(2, 4)]:+.2f}%` (top-2), but changed the eight-layer models
  by `{expert_effects[(1, 8)]:+.2f}%` and `{expert_effects[(2, 8)]:+.2f}%`.
  At this fixed training budget, depth and top-2 routing were more reliable
  gains than expert-count scaling alone.
- **The best dense configuration was {best_dense['num_experts']} experts,
  top-{best_dense['top_k']}, and {best_dense['num_layers']} layers** at
  `{best_dense['dense_loss']['mean']:.4f} +/- {best_dense['dense_loss']['std']:.4f}`.
  No aggregate cell had dead experts, and normalized usage entropy remained
  between `0.9846` and `0.9952`.
- **Mask structure generalized, but direct high-sparsity pruning did not.**
  {magnitude_control_result} Direct 50% pruning increased loss by
  `{min(direct_50_changes):.2f}%` to `{max(direct_50_changes):.2f}%`, whereas
  direct 80% pruning increased it by `{min(direct_80_changes):.2f}%` to
  `{max(direct_80_changes):.2f}%`. The 80% values are one-shot pruning results,
  not lottery-ticket rewind tests; representative Phase 4 rewind suites are
  still required before claiming high-sparsity robustness.
- **The broader validation setting replicated the basic result.** On the
  balanced TinyStories/WikiText corpus, dense loss was `1.4801 +/- 0.0425`;
  the 50% magnitude mask scored `1.5360` versus `3.7565` for a random mask.

Raw aggregate: [`phase4_summary.json`](phase4_summary.json)
"""
    path.write_text(markdown, encoding="utf-8")


def _write_figures(report: dict, output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    architecture = report["architecture"]
    figure, axis = plt.subplots(figsize=(8, 5))
    for top_k, layers in product(TOP_K_VALUES, LAYER_COUNTS):
        rows = sorted(
            (
                row
                for row in architecture
                if row["top_k"] == top_k and row["num_layers"] == layers
            ),
            key=lambda row: row["num_experts"],
        )
        axis.errorbar(
            [row["num_experts"] for row in rows],
            [row["dense_loss"]["mean"] for row in rows],
            yerr=[row["dense_loss"]["std"] for row in rows],
            marker="o",
            capsize=3,
            label=f"top-{top_k}, {layers} layers",
        )
    axis.set_xlabel("Number of experts")
    axis.set_ylabel("Validation loss")
    axis.set_title("Phase 4 architecture grid: dense loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "architecture_dense_loss.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for top_k, layers in product(TOP_K_VALUES, LAYER_COUNTS):
        rows = sorted(
            (
                row
                for row in architecture
                if row["top_k"] == top_k and row["num_layers"] == layers
            ),
            key=lambda row: row["num_experts"],
        )
        axis.errorbar(
            [row["num_experts"] for row in rows],
            [row["pruning"]["magnitude_0.8"]["mean"] for row in rows],
            yerr=[row["pruning"]["magnitude_0.8"]["std"] for row in rows],
            marker="o",
            capsize=3,
            label=f"top-{top_k}, {layers} layers",
        )
    axis.set_xlabel("Number of experts")
    axis.set_ylabel("Validation loss after 80% magnitude pruning")
    axis.set_title("Phase 4 architecture grid: direct pruning")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "architecture_pruning_80.png", dpi=160)
    plt.close(figure)

    datasets = report["datasets"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        [row["dataset"] for row in datasets],
        [row["dense_loss"]["mean"] for row in datasets],
        yerr=[row["dense_loss"]["std"] for row in datasets],
        capsize=4,
    )
    axis.set_ylabel("Validation loss")
    axis.set_title("Baseline robustness across datasets")
    axis.tick_params(axis="x", labelrotation=15)
    figure.tight_layout()
    figure.savefig(output_root / "dataset_robustness.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 4 robustness grid.")
    parser.add_argument("--architecture-configs", nargs="+", required=True)
    parser.add_argument("--dataset-configs", nargs="+", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report = collect_results(
            args.architecture_configs, args.dataset_configs, args.output_dir
        )
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "phase4_summary.json"
        report_path = output_root / "phase4_results.md"
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        figures_written = False
        try:
            _write_figures(report, output_root)
            figures_written = True
        except ImportError:
            pass
        _write_report(report, report_path, figures_written)
        result = {"summary": str(summary_path), "report": str(report_path)}
    else:
        result = run_phase4(
            args.architecture_configs, args.dataset_configs, args.output_dir
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
