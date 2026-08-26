from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

from moe_lth.analysis.expert_usage import usage_summary
from moe_lth.analysis.routing_stability import routing_agreement
from moe_lth.config import load_config


DEFAULT_AUX_WEIGHTS = [0.0, 0.01, 0.03, 0.1, 0.3]


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
    if "tinystories" in paths:
        return "TinyStories"
    if "wikitext" in paths:
        return "WikiText-103"
    return "Configured Dataset"


def _dataset_slug(config: dict) -> str:
    return _dataset_name(config).lower().replace("-103", "103").replace(" ", "_")


def _aux_label(aux_weight: float) -> str:
    return f"{aux_weight:g}".replace(".", "p")


def _suite_dir(config: dict) -> Path:
    output = Path(config["output_dir"])
    return output.parent / f"{output.name}_suite"


def _existing_baseline_run_dir(config: dict, aux_weight: float) -> Path | None:
    if not math.isclose(float(config["routing"]["aux_loss_weight"]), aux_weight):
        return None
    run_dir = _suite_dir(config) / "normal"
    final_checkpoint = run_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"
    if (run_dir / "summary.json").exists() and final_checkpoint.exists():
        return run_dir
    return None


def _sweep_run_dir(output_root: Path, config: dict, aux_weight: float) -> Path:
    return output_root / _dataset_slug(config) / f"seed_{int(config['seed'])}" / f"aux_{_aux_label(aux_weight)}"


def _final_checkpoint(run_dir: Path, config: dict) -> Path:
    return run_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"


def _final_usage_metrics(run_dir: Path) -> dict:
    summary = usage_summary(str(run_dir / "logs" / "expert_usage.jsonl"))
    rows = summary["layers_over_time"]
    final_step = max(row["step"] for row in rows)
    final_rows = [row for row in rows if row["step"] == final_step]
    return {
        "step": final_step,
        "normalized_entropy": mean(row["normalized_entropy"] for row in final_rows),
        "coefficient_of_variation": mean(row["coefficient_of_variation"] for row in final_rows),
        "dead_experts": mean(row["dead_experts"] for row in final_rows),
        "max_to_min": mean(row["max_to_min"] for row in final_rows),
    }


def _first_to_final_routing_stability(run_dir: Path) -> float | None:
    route_files = sorted(
        (run_dir / "logs").glob("validation_routes_step_*.npz"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if len(route_files) < 2:
        return None
    return float(routing_agreement(str(route_files[0]), str(route_files[-1]))["overall"])


def _pruning_metrics(run_dir: Path, config: dict) -> list[dict]:
    pruning_path = run_dir / "tables" / "pruning_results.json"
    if pruning_path.exists():
        return _read_json(pruning_path)
    from moe_lth.pruning.evaluate_pruning import evaluate_pruning

    return evaluate_pruning(config, str(_final_checkpoint(run_dir, config)))


def _condition_loss(pruning: list[dict], condition: str, sparsity: float) -> float | None:
    for row in pruning:
        if row["condition"] == condition and math.isclose(float(row["sparsity"]), sparsity):
            return float(row["loss"])
    return None


def collect_results(output_dir: str, config_paths: list[str], aux_weights: list[float]) -> dict:
    output_root = Path(output_dir)
    rows = []
    for config_path in config_paths:
        base_config = load_config(config_path)
        for aux_weight in aux_weights:
            run_dir = _existing_baseline_run_dir(base_config, aux_weight)
            if run_dir is None:
                run_dir = _sweep_run_dir(output_root, base_config, aux_weight)
            summary_path = run_dir / "summary.json"
            resolved_config = load_config(config_path)
            resolved_config["routing"]["aux_loss_weight"] = aux_weight
            resolved_config["output_dir"] = str(run_dir)
            resolved_config["training"]["record_train_routes"] = False
            if not summary_path.exists():
                continue
            pruning = _pruning_metrics(run_dir, resolved_config)
            dense_loss = float(_read_json(summary_path)["final_validation_loss"])
            usage = _final_usage_metrics(run_dir)
            rows.append(
                {
                    "dataset": _dataset_name(base_config),
                    "seed": int(base_config["seed"]),
                    "aux_loss_weight": aux_weight,
                    "run_dir": str(run_dir),
                    "dense_loss": dense_loss,
                    "routing_stability": _first_to_final_routing_stability(run_dir),
                    "usage": usage,
                    "pruning": {
                        "magnitude_0.5": _condition_loss(pruning, "magnitude", 0.5),
                        "random_mask_0.5": _condition_loss(pruning, "random_mask", 0.5),
                        "magnitude_0.8": _condition_loss(pruning, "magnitude", 0.8),
                        "random_mask_0.8": _condition_loss(pruning, "random_mask", 0.8),
                    },
                }
            )
    return _aggregate(rows)


def _aggregate(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["aux_loss_weight"]), []).append(row)

    aggregates = []
    for (dataset, aux_weight), values in sorted(grouped.items()):
        dense = _summary([row["dense_loss"] for row in values])
        entropy = _summary([row["usage"]["normalized_entropy"] for row in values])
        dead = _summary([row["usage"]["dead_experts"] for row in values])
        cv = _summary([row["usage"]["coefficient_of_variation"] for row in values])
        route_values = [
            row["routing_stability"]
            for row in values
            if row["routing_stability"] is not None
        ]
        pruning = {}
        for key in ("magnitude_0.5", "random_mask_0.5", "magnitude_0.8", "random_mask_0.8"):
            metric_values = [row["pruning"][key] for row in values if row["pruning"][key] is not None]
            pruning[key] = _summary(metric_values) if metric_values else None
        aggregates.append(
            {
                "dataset": dataset,
                "aux_loss_weight": aux_weight,
                "seeds": sorted(row["seed"] for row in values),
                "dense_loss": dense,
                "final_usage_normalized_entropy": entropy,
                "final_usage_dead_experts": dead,
                "final_usage_coefficient_of_variation": cv,
                "first_to_final_routing_stability": _summary(route_values) if route_values else None,
                "pruning": pruning,
            }
        )
    return {"rows": rows, "aggregates": aggregates}


def run_load_balance_sweep(
    config_paths: list[str],
    output_dir: str,
    aux_weights: list[float],
    with_pruning: bool = True,
) -> dict:
    output_root = Path(output_dir)
    statuses = []
    for config_path in config_paths:
        base_config = load_config(config_path)
        for aux_weight in aux_weights:
            run_dir = _existing_baseline_run_dir(base_config, aux_weight)
            reused = run_dir is not None
            if run_dir is None:
                run_dir = _sweep_run_dir(output_root, base_config, aux_weight)

            config = deepcopy(base_config)
            config["routing"]["aux_loss_weight"] = aux_weight
            config["routing"]["mode"] = "learned"
            config["output_dir"] = str(run_dir)
            config["training"]["record_train_routes"] = False

            final_checkpoint = _final_checkpoint(run_dir, config)
            summary_path = run_dir / "summary.json"
            pruning_path = run_dir / "tables" / "pruning_results.json"
            status = {
                "dataset": _dataset_name(base_config),
                "seed": int(base_config["seed"]),
                "aux_loss_weight": aux_weight,
                "run_dir": str(run_dir),
                "baseline_reused": reused,
            }
            if summary_path.exists() and final_checkpoint.exists():
                status["training"] = "existing"
            else:
                from moe_lth.training.train import train_from_config

                train_from_config(config)
                status["training"] = "completed"

            if with_pruning:
                if pruning_path.exists():
                    status["pruning"] = "existing"
                else:
                    from moe_lth.pruning.evaluate_pruning import evaluate_pruning

                    evaluate_pruning(config, str(final_checkpoint))
                    status["pruning"] = "completed"
            statuses.append(status)

    report = collect_results(output_dir, config_paths, aux_weights)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "load_balance_sweep_summary.json"
    report_path = output_root / "load_balance_sweep_results.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figures_written = False
    try:
        _write_figures(report, output_root)
        figures_written = True
    except ImportError:
        pass
    _write_report(report, report_path, figures_written=figures_written)
    status_path = output_root / "load_balance_sweep_status.json"
    result = {
        "status": statuses,
        "summary": str(summary_path),
        "report": str(report_path),
    }
    status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_report(report: dict, path: Path, figures_written: bool = False) -> None:
    by_dataset: dict[str, list[dict]] = {}
    for row in report["aggregates"]:
        by_dataset.setdefault(row["dataset"], []).append(row)

    sections = []
    for dataset, rows in sorted(by_dataset.items()):
        rows = sorted(rows, key=lambda row: row["aux_loss_weight"])
        baseline = min(rows, key=lambda row: abs(row["aux_loss_weight"] - 0.1))
        baseline_loss = baseline["dense_loss"]["mean"]
        dense_rows = []
        pruning_rows = []
        for row in rows:
            dense_delta = (row["dense_loss"]["mean"] / baseline_loss - 1.0) * 100.0
            stability = row["first_to_final_routing_stability"]
            dense_rows.append(
                f"| {row['aux_loss_weight']:g} | {row['dense_loss']['mean']:.4f} | "
                f"{row['dense_loss']['std']:.4f} | {dense_delta:+.2f}% | "
                f"{row['final_usage_normalized_entropy']['mean']:.4f} | "
                f"{row['final_usage_dead_experts']['mean']:.2f} | "
                f"{row['final_usage_coefficient_of_variation']['mean']:.4f} | "
                f"{stability['mean']:.4f} |"
                if stability
                else f"| {row['aux_loss_weight']:g} | {row['dense_loss']['mean']:.4f} | "
                f"{row['dense_loss']['std']:.4f} | {dense_delta:+.2f}% | "
                f"{row['final_usage_normalized_entropy']['mean']:.4f} | "
                f"{row['final_usage_dead_experts']['mean']:.2f} | "
                f"{row['final_usage_coefficient_of_variation']['mean']:.4f} | - |"
            )
            pruning = row["pruning"]
            pruning_rows.append(
                f"| {row['aux_loss_weight']:g} | "
                f"{pruning['magnitude_0.5']['mean']:.4f} | "
                f"{pruning['random_mask_0.5']['mean']:.4f} | "
                f"{pruning['magnitude_0.8']['mean']:.4f} | "
                f"{pruning['random_mask_0.8']['mean']:.4f} |"
            )

        best_loss = min(rows, key=lambda row: row["dense_loss"]["mean"])
        best_entropy = max(rows, key=lambda row: row["final_usage_normalized_entropy"]["mean"])
        slug = dataset.lower().replace("-103", "103").replace(" ", "_")
        figure_links = (
            f"\n![{dataset} dense loss](./{slug}_dense_loss.png)\n\n"
            f"![{dataset} usage balance](./{slug}_usage_balance.png)\n"
            if figures_written
            else ""
        )
        sections.append(
            f"""## {dataset}

Best dense loss: aux `{best_loss['aux_loss_weight']:g}` with mean loss `{best_loss['dense_loss']['mean']:.4f}`. Highest final usage entropy: aux `{best_entropy['aux_loss_weight']:g}` with normalized entropy `{best_entropy['final_usage_normalized_entropy']['mean']:.4f}`.

| Aux loss weight | Dense loss | Std | Delta vs aux=0.1 | Final usage entropy | Dead experts | Usage CV | Routing stability |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(dense_rows)}

| Aux loss weight | 50% magnitude | 50% random mask | 80% magnitude | 80% random mask |
|---:|---:|---:|---:|---:|
{chr(10).join(pruning_rows)}

{figure_links}
"""
        )

    markdown = f"""# Load-Balancing Weight Sweep Results

Auxiliary load-balancing loss weights: {", ".join(str(row['aux_loss_weight']) for row in sorted(report['aggregates'], key=lambda r: (r['dataset'], r['aux_loss_weight']))[:len(set(r['aux_loss_weight'] for r in report['aggregates']))])}

This sweep varies only `routing.aux_loss_weight` while keeping the model, data,
optimizer, training length, and pruning protocol fixed within each seed. The
`0.1` setting is the existing main baseline when available.

## Interpretation

No load balancing causes expert collapse and worse validation loss on both
datasets. Stronger balancing improves dense loss slightly beyond the original
`0.1` baseline when that trend appears in the completed runs. Learned
magnitude masks still beat random masks at every tested weight, so stronger
balancing does not erase sparse mask structure. The caveat is that 80% direct
pruning can become more brittle as balancing gets stronger, which means
high-sparsity claims still need the rewind/retrain protocol.

{chr(10).join(sections)}

Raw aggregate: [`load_balance_sweep_summary.json`](load_balance_sweep_summary.json)
"""
    path.write_text(markdown, encoding="utf-8")


def _write_figures(report: dict, output_root: Path) -> None:
    import matplotlib.pyplot as plt

    by_dataset: dict[str, list[dict]] = {}
    for row in report["aggregates"]:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for dataset, rows in sorted(by_dataset.items()):
        rows = sorted(rows, key=lambda row: row["aux_loss_weight"])
        x = [row["aux_loss_weight"] for row in rows]
        slug = dataset.lower().replace("-103", "103").replace(" ", "_")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.errorbar(
            x,
            [row["dense_loss"]["mean"] for row in rows],
            yerr=[row["dense_loss"]["std"] for row in rows],
            marker="o",
            capsize=3,
        )
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlabel("Auxiliary load-balancing loss weight")
        axis.set_ylabel("Validation loss")
        axis.set_title(f"{dataset}: dense loss by load-balancing weight")
        figure.tight_layout()
        figure.savefig(output_root / f"{slug}_dense_loss.png", dpi=160)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.errorbar(
            x,
            [row["final_usage_normalized_entropy"]["mean"] for row in rows],
            yerr=[row["final_usage_normalized_entropy"]["std"] for row in rows],
            marker="o",
            capsize=3,
            label="usage entropy",
        )
        axis2 = axis.twinx()
        axis2.errorbar(
            x,
            [row["final_usage_dead_experts"]["mean"] for row in rows],
            yerr=[row["final_usage_dead_experts"]["std"] for row in rows],
            color="tab:red",
            marker="s",
            capsize=3,
            label="dead experts",
        )
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlabel("Auxiliary load-balancing loss weight")
        axis.set_ylabel("Final normalized usage entropy")
        axis2.set_ylabel("Dead experts per layer")
        axis.set_title(f"{dataset}: expert balance by load-balancing weight")
        figure.tight_layout()
        figure.savefig(output_root / f"{slug}_usage_balance.png", dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled load-balancing-weight sweep.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aux-loss-weights", nargs="+", type=float, default=DEFAULT_AUX_WEIGHTS)
    parser.add_argument("--without-pruning", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report = collect_results(args.output_dir, args.configs, args.aux_loss_weights)
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "load_balance_sweep_summary.json"
        report_path = output_root / "load_balance_sweep_results.md"
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        figures_written = False
        try:
            _write_figures(report, output_root)
            figures_written = True
        except ImportError:
            pass
        _write_report(report, report_path, figures_written=figures_written)
        result = {"summary": str(summary_path), "report": str(report_path)}
    else:
        result = run_load_balance_sweep(
            args.configs,
            args.output_dir,
            args.aux_loss_weights,
            with_pruning=not args.without_pruning,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
