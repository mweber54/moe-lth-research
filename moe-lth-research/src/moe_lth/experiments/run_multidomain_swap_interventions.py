from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

from moe_lth.analysis.routing_stability import routing_agreement
from moe_lth.config import load_config
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.evaluate_pruning import evaluate_pruning
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import mask_jaccard
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.training.train import train_from_config


SWAP_CONDITIONS = {
    "swap_0_1_all": {
        "label": "Global swap 0<->1",
        "routing": {"swap_pairs": [[0, 1]]},
    },
    "swap_0_4_all": {
        "label": "Global swap 0<->4",
        "routing": {"swap_pairs": [[0, 4]]},
    },
    "swap_2_6_all": {
        "label": "Global swap 2<->6",
        "routing": {"swap_pairs": [[2, 6]]},
    },
    "layer0_swap_0_1": {
        "label": "Layer 0 swap 0<->1",
        "routing": {"layer_swap_pairs": {"0": [[0, 1]]}},
    },
    "layer3_swap_0_1": {
        "label": "Layer 3 swap 0<->1",
        "routing": {"layer_swap_pairs": {"3": [[0, 1]]}},
    },
    "cyclic_shift_all": {
        "label": "Global cyclic shift +1",
        "routing": {"cyclic_shift": 1},
    },
}


def _suite_dir(config: dict) -> Path:
    output = Path(config["output_dir"])
    return output.parent / f"{output.name}_suite"


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


def _final_checkpoint(run_dir: Path, config: dict) -> Path:
    return run_dir / "checkpoints" / f"step_{int(config['training']['steps'])}.pt"


def _normal_run_dir(config: dict) -> Path:
    return _suite_dir(config) / "normal"


def _normal_history(config: dict) -> Path:
    return _normal_run_dir(config) / "logs" / "train_route_history.npz"


def _condition_config(base_config: dict, output_root: Path, condition: str) -> dict:
    config = deepcopy(base_config)
    config["routing"]["mode"] = "swapped"
    config["routing"]["replay_path"] = str(_normal_history(base_config))
    config["routing"]["swap_pairs"] = []
    config["routing"]["layer_swap_pairs"] = {}
    config["routing"]["cyclic_shift"] = 0
    config["routing"]["layer_cyclic_shifts"] = {}
    config["routing"].update(SWAP_CONDITIONS[condition]["routing"])
    config["training"]["record_train_routes"] = False
    config["training"]["save_optimizer"] = False
    config["output_dir"] = str(output_root / f"seed_{int(config['seed'])}" / condition)
    return config


def _is_training_complete(config: dict) -> bool:
    run_dir = Path(config["output_dir"])
    return (run_dir / "summary.json").exists() and _final_checkpoint(run_dir, config).exists()


def _load_masks_for_checkpoint(checkpoint: Path) -> dict:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TinyMoELanguageModel(payload["config"]["model"])
    load_checkpoint(checkpoint, model, map_location="cpu")
    return expert_local_magnitude_masks(model, 0.8)


def _normal_pruning(normal_dir: Path) -> list[dict]:
    pruning_path = normal_dir / "tables" / "pruning_results.json"
    if not pruning_path.exists():
        return []
    return _read_json(pruning_path)


def _pruning_table_rows(pruning: list[dict]) -> dict[tuple[str, float], float]:
    rows = {}
    for row in pruning:
        rows[(row["condition"], float(row["sparsity"]))] = float(row["loss"])
    return rows


def _analyze_seed(base_config: dict, output_root: Path) -> dict:
    normal_dir = _normal_run_dir(base_config)
    normal_checkpoint = _final_checkpoint(normal_dir, base_config)
    normal_routes = normal_dir / "logs" / f"validation_routes_step_{int(base_config['training']['steps'])}.npz"
    normal_masks = _load_masks_for_checkpoint(normal_checkpoint)
    normal_summary = _read_json(normal_dir / "summary.json")

    conditions = {
        "normal": {
            "label": "Normal learned routing",
            "final_validation_loss": float(normal_summary["final_validation_loss"]),
            "pruning": _normal_pruning(normal_dir),
            "routing_agreement_to_normal": 1.0,
            "mask_jaccard_to_normal": 1.0,
        }
    }

    for condition, metadata in SWAP_CONDITIONS.items():
        config = _condition_config(base_config, output_root, condition)
        run_dir = Path(config["output_dir"])
        checkpoint = _final_checkpoint(run_dir, config)
        summary = _read_json(run_dir / "summary.json")
        route_path = run_dir / "logs" / f"validation_routes_step_{int(config['training']['steps'])}.npz"
        condition_masks = _load_masks_for_checkpoint(checkpoint)
        conditions[condition] = {
            "label": metadata["label"],
            "final_validation_loss": float(summary["final_validation_loss"]),
            "pruning": _read_json(run_dir / "tables" / "pruning_results.json"),
            "routing_agreement_to_normal": routing_agreement(str(normal_routes), str(route_path))["overall"],
            "mask_jaccard_to_normal": mask_jaccard(normal_masks, condition_masks),
        }

    return {
        "seed": int(base_config["seed"]),
        "normal_run_dir": str(normal_dir),
        "conditions": conditions,
    }


def _aggregate(seed_reports: list[dict]) -> dict:
    dense: dict[str, list[float]] = defaultdict(list)
    route_agreement: dict[str, list[float]] = defaultdict(list)
    mask_jaccard_values: dict[str, list[float]] = defaultdict(list)
    pruning: dict[tuple[str, str, float], list[float]] = defaultdict(list)

    for seed_report in seed_reports:
        for condition, row in seed_report["conditions"].items():
            dense[condition].append(float(row["final_validation_loss"]))
            route_agreement[condition].append(float(row["routing_agreement_to_normal"]))
            mask_jaccard_values[condition].append(float(row["mask_jaccard_to_normal"]))
            for pruning_row in row.get("pruning", []):
                pruning[
                    (
                        condition,
                        pruning_row["condition"],
                        float(pruning_row["sparsity"]),
                    )
                ].append(float(pruning_row["loss"]))

    return {
        "dataset_name": "Balanced Multi-Domain",
        "seeds": sorted(row["seed"] for row in seed_reports),
        "condition_labels": {
            "normal": "Normal learned routing",
            **{condition: metadata["label"] for condition, metadata in SWAP_CONDITIONS.items()},
        },
        "dense": {condition: _summary(values) for condition, values in sorted(dense.items())},
        "routing_agreement_to_normal": {
            condition: _summary(values) for condition, values in sorted(route_agreement.items())
        },
        "mask_jaccard_to_normal": {
            condition: _summary(values) for condition, values in sorted(mask_jaccard_values.items())
        },
        "pruning_by_condition": {
            f"{condition}|{mask_condition}|{sparsity:g}": _summary(values)
            for (condition, mask_condition, sparsity), values in sorted(pruning.items())
        },
        "seed_reports": seed_reports,
    }


def _write_figures(report: dict, output_dir: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    order = ["normal", *SWAP_CONDITIONS.keys()]
    labels = [report["condition_labels"][condition] for condition in order]
    dense = [report["dense"][condition]["mean"] for condition in order]
    dense_std = [report["dense"][condition]["std"] for condition in order]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(labels, dense, yerr=dense_std, capsize=4)
    axis.set_ylabel("Validation loss")
    axis.set_title("Balanced multi-domain swap interventions")
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "swap_dense_loss.png", dpi=160)
    plt.close(figure)

    x = [report["routing_agreement_to_normal"][condition]["mean"] for condition in order[1:]]
    y = [report["mask_jaccard_to_normal"][condition]["mean"] for condition in order[1:]]
    xerr = [report["routing_agreement_to_normal"][condition]["std"] for condition in order[1:]]
    yerr = [report["mask_jaccard_to_normal"][condition]["std"] for condition in order[1:]]
    figure, axis = plt.subplots(figsize=(7, 5))
    for index, condition in enumerate(order[1:]):
        axis.errorbar(x[index], y[index], xerr=xerr[index], yerr=yerr[index], marker="o", capsize=3)
        axis.annotate(condition, (x[index], y[index]), fontsize=8)
    axis.set_xlabel("Routing agreement to normal")
    axis.set_ylabel("80% mask Jaccard to normal")
    axis.set_title("Swap routing agreement versus mask similarity")
    figure.tight_layout()
    figure.savefig(output_dir / "swap_route_mask_similarity.png", dpi=160)
    plt.close(figure)

    return True


def _format_delta(value: float, baseline: float) -> str:
    return f"{(value / baseline - 1.0) * 100.0:+.2f}%"


def _write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "swap_interventions_summary.json"
    report_path = output_dir / "swap_interventions_results.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figures_written = _write_figures(report, output_dir)

    order = ["normal", *SWAP_CONDITIONS.keys()]
    normal_dense = report["dense"]["normal"]["mean"]
    dense_rows = []
    similarity_rows = []
    pruning_rows = []
    for condition in order:
        dense = report["dense"][condition]
        dense_rows.append(
            f"| {condition} | {report['condition_labels'][condition]} | {dense['mean']:.4f} | "
            f"{dense['std']:.4f} | {_format_delta(dense['mean'], normal_dense) if condition != 'normal' else '-'} |"
        )
        route = report["routing_agreement_to_normal"][condition]
        mask = report["mask_jaccard_to_normal"][condition]
        similarity_rows.append(
            f"| {condition} | {route['mean']:.4f} +/- {route['std']:.4f} | "
            f"{mask['mean']:.4f} +/- {mask['std']:.4f} |"
        )
        values = []
        for sparsity in (0.5, 0.8):
            for mask_condition in ("magnitude", "random_mask", "other_expert_mask"):
                stats = report["pruning_by_condition"].get(f"{condition}|{mask_condition}|{sparsity:g}")
                values.append("-" if stats is None else f"{stats['mean']:.4f}")
        pruning_rows.append(
            f"| {condition} | {dense['mean']:.4f} | " + " | ".join(values) + " |"
        )

    worst = max(
        SWAP_CONDITIONS,
        key=lambda condition: report["dense"][condition]["mean"],
    )
    mildest = min(
        SWAP_CONDITIONS,
        key=lambda condition: report["dense"][condition]["mean"],
    )
    dense_figure = "![Swap dense loss](swap_dense_loss.png)" if figures_written else ""
    similarity_figure = (
        "![Swap route/mask similarity](swap_route_mask_similarity.png)" if figures_written else ""
    )
    markdown = f"""# Balanced Multi-Domain Swap Interventions

Seeds: {", ".join(map(str, report["seeds"]))}

## Key Findings

- The mildest swap was `{mildest}` at {_format_delta(report["dense"][mildest]["mean"], normal_dense)} versus normal.
- The strongest swap was `{worst}` at {_format_delta(report["dense"][worst]["mean"], normal_dense)} versus normal.
- Cyclic and global swaps test expert identity more aggressively than the original 0/1 swap.
- Layer-specific swaps isolate whether disrupting a single routing layer is enough to shift loss and mask identity.

## Dense Performance

| Condition | Intervention | Mean loss | Std | Delta vs normal |
|---|---|---:|---:|---:|
{chr(10).join(dense_rows)}

{dense_figure}

## Routing And Mask Similarity To Normal

| Condition | Routing agreement | 80% mask Jaccard |
|---|---:|---:|
{chr(10).join(similarity_rows)}

{similarity_figure}

## Direct Pruning

| Condition | Dense | 50% magnitude | 50% random | 50% other expert | 80% magnitude | 80% random | 80% other expert |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pruning_rows)}

Raw aggregate: [`swap_interventions_summary.json`](swap_interventions_summary.json)
"""
    report_path.write_text(markdown, encoding="utf-8")
    return summary_path, report_path


def run_swap_interventions(
    config_paths: list[str],
    output_dir: str,
    with_pruning: bool = True,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    statuses = []
    seed_reports = []

    for config_path in config_paths:
        base_config = load_config(config_path)
        normal_history = _normal_history(base_config)
        if not normal_history.exists():
            raise FileNotFoundError(f"Missing normal route history: {normal_history}")
        seed_status = {"seed": int(base_config["seed"]), "conditions": {}}
        for condition in SWAP_CONDITIONS:
            config = _condition_config(base_config, output_root, condition)
            run_dir = Path(config["output_dir"])
            checkpoint = _final_checkpoint(run_dir, config)
            pruning_path = run_dir / "tables" / "pruning_results.json"
            condition_status = {}
            if _is_training_complete(config):
                condition_status["training"] = "existing"
            else:
                train_from_config(config)
                condition_status["training"] = "completed"
            if with_pruning:
                if pruning_path.exists():
                    condition_status["pruning"] = "existing"
                else:
                    evaluate_pruning(config, str(checkpoint))
                    condition_status["pruning"] = "completed"
            seed_status["conditions"][condition] = condition_status
        statuses.append(seed_status)
        seed_reports.append(_analyze_seed(base_config, output_root))
        partial = _aggregate(seed_reports)
        _write_report(partial, output_root)
        (output_root / "swap_interventions_status.json").write_text(
            json.dumps({"status": statuses, "partial": True}, indent=2),
            encoding="utf-8",
        )

    report = _aggregate(seed_reports)
    summary_path, report_path = _write_report(report, output_root)
    result = {
        "status": statuses,
        "aggregate_summary": str(summary_path),
        "aggregate_report": str(report_path),
    }
    (output_root / "swap_interventions_status.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run broader balanced multi-domain swap interventions."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--without-pruning", action="store_true")
    args = parser.parse_args()
    result = run_swap_interventions(
        args.configs,
        args.output_dir,
        with_pruning=not args.without_pruning,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
