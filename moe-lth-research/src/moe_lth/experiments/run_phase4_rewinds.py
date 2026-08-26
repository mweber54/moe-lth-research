from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from moe_lth.config import load_config


REPRESENTATIVES = {
    "best_dense": {
        "label": "Best dense: 4E / top-2 / 8L",
        "dataset_dir": "wikitext103",
        "variant": "experts_4_topk_2_layers_8",
    },
    "high_capacity": {
        "label": "High capacity: 16E / top-1 / 8L",
        "dataset_dir": "wikitext103",
        "variant": "experts_16_topk_1_layers_8",
    },
    "multi_domain": {
        "label": "Balanced multi-domain: 8E / top-1 / 4L",
        "dataset_dir": "balanced_multi_domain",
        "variant": "experts_8_topk_1_layers_4",
    },
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _run_dir(phase4_root: Path, representative: str, seed: int) -> Path:
    spec = REPRESENTATIVES[representative]
    return (
        phase4_root
        / spec["dataset_dir"]
        / f"seed_{seed}"
        / spec["variant"]
    )


def _rewind_table(run_dir: Path, sparsity: float) -> Path:
    return run_dir / "tables" / f"rewind_suite_sparsity_{sparsity}.json"


def _status_key(representative: str, seed: int, sparsity: float) -> str:
    return f"{representative}|{seed}|{sparsity:g}"


def _write_status(path: Path, statuses: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(statuses.values()), indent=2), encoding="utf-8")


def run_phase4_rewinds(
    phase4_root: str,
    output_dir: str,
    representatives: list[str],
    seeds: list[int],
    sparsities: list[float],
) -> dict:
    from moe_lth.experiments.run_rewind_suite import run_rewind_suite

    source_root = Path(phase4_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "phase4_rewind_status.json"
    statuses = {
        _status_key(row["representative"], int(row["seed"]), float(row["sparsity"])): row
        for row in (_read_json(status_path) if status_path.exists() else [])
    }

    for representative in representatives:
        for seed in seeds:
            run_dir = _run_dir(source_root, representative, seed)
            resolved_config = run_dir / "resolved_config.yaml"
            summary_path = run_dir / "summary.json"
            if not resolved_config.exists() or not summary_path.exists():
                raise FileNotFoundError(f"Missing completed Phase 4 run: {run_dir}")
            config = load_config(resolved_config)
            final_checkpoint = (
                run_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"
            )
            if not final_checkpoint.exists():
                raise FileNotFoundError(f"Missing final checkpoint: {final_checkpoint}")

            for sparsity in sparsities:
                key = _status_key(representative, seed, sparsity)
                table = _rewind_table(run_dir, sparsity)
                rows = _read_json(table) if table.exists() else []
                if len(rows) == 16:
                    state = "existing"
                else:
                    state = "running"
                    statuses[key] = {
                        "representative": representative,
                        "seed": seed,
                        "sparsity": sparsity,
                        "run_dir": str(run_dir),
                        "completed_conditions": len(rows),
                        "state": state,
                    }
                    _write_status(status_path, statuses)
                    print(
                        f"[phase4-rewind] {representative}, seed {seed}, "
                        f"sparsity {sparsity:g}: resuming at {len(rows)}/16",
                        flush=True,
                    )
                    run_rewind_suite(config, str(final_checkpoint), sparsity)
                    rows = _read_json(table)
                    state = "completed"

                statuses[key] = {
                    "representative": representative,
                    "seed": seed,
                    "sparsity": sparsity,
                    "run_dir": str(run_dir),
                    "completed_conditions": len(rows),
                    "state": state,
                }
                _write_status(status_path, statuses)

    report = collect_results(
        phase4_root, representatives, seeds, sparsities, require_complete=True
    )
    return write_results(report, output_dir)


def collect_results(
    phase4_root: str,
    representatives: list[str],
    seeds: list[int],
    sparsities: list[float],
    require_complete: bool = True,
) -> dict:
    source_root = Path(phase4_root)
    cells = []
    for representative in representatives:
        dense_losses = []
        seed_rows = []
        for seed in seeds:
            run_dir = _run_dir(source_root, representative, seed)
            summary_path = run_dir / "summary.json"
            resolved_path = run_dir / "resolved_config.yaml"
            if not summary_path.exists() or not resolved_path.exists():
                if require_complete:
                    raise FileNotFoundError(f"Missing completed Phase 4 run: {run_dir}")
                continue
            summary = _read_json(summary_path)
            config = load_config(resolved_path)
            dense_loss = float(summary["final_validation_loss"])
            dense_losses.append(dense_loss)
            rows = []
            for sparsity in sparsities:
                table = _rewind_table(run_dir, sparsity)
                if not table.exists():
                    if require_complete:
                        raise FileNotFoundError(f"Missing rewind table: {table}")
                    continue
                table_rows = _read_json(table)
                if require_complete and len(table_rows) != 16:
                    raise RuntimeError(f"Incomplete rewind table ({len(table_rows)}/16): {table}")
                rows.extend(table_rows)
            seed_rows.append(
                {
                    "seed": seed,
                    "dense_loss": dense_loss,
                    "run_dir": str(run_dir),
                    "rows": rows,
                }
            )

        grouped: dict[tuple[float, str, float], list[float]] = defaultdict(list)
        for seed_result in seed_rows:
            for row in seed_result["rows"]:
                grouped[
                    (
                        float(row["sparsity"]),
                        row["condition"],
                        float(row["rewind_fraction"]),
                    )
                ].append(float(row["loss"]))
        aggregate = [
            {
                "sparsity": sparsity,
                "condition": condition,
                "rewind_fraction": rewind_fraction,
                "loss": _summary(losses),
            }
            for (sparsity, condition, rewind_fraction), losses in sorted(grouped.items())
        ]
        first_config = (
            load_config(_run_dir(source_root, representative, seeds[0]) / "resolved_config.yaml")
            if seed_rows
            else None
        )
        cells.append(
            {
                "key": representative,
                "label": REPRESENTATIVES[representative]["label"],
                "model": first_config["model"] if first_config else None,
                "seeds": [row["seed"] for row in seed_rows],
                "dense_loss": _summary(dense_losses) if dense_losses else None,
                "seed_results": seed_rows,
                "aggregate": aggregate,
            }
        )
    return {
        "phase4_root": phase4_root,
        "representatives": representatives,
        "seeds": seeds,
        "sparsities": sparsities,
        "cells": cells,
    }


def _aggregate_row(cell: dict, sparsity: float, condition: str, fraction: float) -> dict:
    return next(
        row
        for row in cell["aggregate"]
        if row["sparsity"] == sparsity
        and row["condition"] == condition
        and row["rewind_fraction"] == fraction
    )


def _ticket_summary(cell: dict, sparsity: float) -> dict:
    learned_rows = [
        row
        for row in cell["aggregate"]
        if row["sparsity"] == sparsity and row["condition"] == "learned_mask"
    ]
    best = min(learned_rows, key=lambda row: row["loss"]["mean"])
    fraction = best["rewind_fraction"]
    controls = {
        condition: _aggregate_row(cell, sparsity, condition, fraction)["loss"]["mean"]
        for condition in ("random_mask", "random_reinit", "randomized_routing")
    }
    dense = cell["dense_loss"]["mean"]
    delta = (best["loss"]["mean"] / dense - 1.0) * 100.0
    return {
        "sparsity": sparsity,
        "best_rewind_fraction": fraction,
        "mean_loss": best["loss"]["mean"],
        "std_loss": best["loss"]["std"],
        "delta_vs_dense_percent": delta,
        "controls_at_best_fraction": controls,
        "meets_full_loss_criterion": delta <= 5.0,
        "beats_all_controls": all(best["loss"]["mean"] < loss for loss in controls.values()),
    }


def write_results(report: dict, output_dir: str) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for cell in report["cells"]:
        cell["ticket_summary"] = [
            _ticket_summary(cell, sparsity) for sparsity in report["sparsities"]
        ]
    summary_path = destination / "phase4_rewind_summary.json"
    report_path = destination / "phase4_rewind_results.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figures_written = False
    try:
        _write_figure(report, destination)
        figures_written = True
    except ImportError:
        pass
    _write_report(report, report_path, figures_written)
    return {"summary": str(summary_path), "report": str(report_path)}


def _write_report(report: dict, path: Path, figures_written: bool) -> None:
    best_rows = []
    all_rows = []
    for cell in report["cells"]:
        dense = cell["dense_loss"]
        for ticket in cell["ticket_summary"]:
            best_rows.append(
                f"| {cell['label']} | {dense['mean']:.4f} +/- {dense['std']:.4f} | "
                f"{ticket['sparsity']:.0%} | {ticket['best_rewind_fraction']:.0%} | "
                f"{ticket['mean_loss']:.4f} +/- {ticket['std_loss']:.4f} | "
                f"{ticket['delta_vs_dense_percent']:+.2f}% | "
                f"{'yes' if ticket['beats_all_controls'] else 'no'} | "
                f"{'yes' if ticket['meets_full_loss_criterion'] else 'no'} |"
            )
        for row in cell["aggregate"]:
            all_rows.append(
                f"| {cell['label']} | {row['sparsity']:.0%} | {row['condition']} | "
                f"{row['rewind_fraction']:.0%} | {row['loss']['mean']:.4f} | "
                f"{row['loss']['std']:.4f} |"
            )

    figure = (
        "\n![Representative Phase 4 rewind curves](phase4_rewind_curves.png)\n"
        if figures_written
        else ""
    )
    markdown = f"""# Representative Phase 4 Rewind Results

Seeds: {", ".join(map(str, report["seeds"]))}. Each cell uses expert-local
magnitude masks at 50% and 80% sparsity, rewinds to initialization, 1%, 5%,
and 10% of training, and retrains for the original 2,500-step schedule.

## Ticket Summary

The full-loss criterion allows at most 5% degradation from the corresponding
dense baseline. A control-complete result must also beat the random mask,
random expert reinitialization, and randomized-routing controls at the same
rewind point.

| Representative | Dense loss | Sparsity | Best rewind | Ticket loss | Delta vs dense | Beats controls | Full-loss criterion |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(best_rows)}

## All Conditions

| Representative | Sparsity | Condition | Rewind fraction | Mean loss | Std |
|---|---:|---|---:|---:|---:|
{chr(10).join(all_rows)}
{figure}

Raw aggregate: [`phase4_rewind_summary.json`](phase4_rewind_summary.json)
"""
    path.write_text(markdown, encoding="utf-8")


def _write_figure(report: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(report["cells"]),
        len(report["sparsities"]),
        figsize=(6 * len(report["sparsities"]), 4 * len(report["cells"])),
        squeeze=False,
    )
    for row_index, cell in enumerate(report["cells"]):
        for column_index, sparsity in enumerate(report["sparsities"]):
            axis = axes[row_index][column_index]
            for condition in (
                "learned_mask",
                "random_mask",
                "random_reinit",
                "randomized_routing",
            ):
                selected = [
                    row
                    for row in cell["aggregate"]
                    if row["sparsity"] == sparsity and row["condition"] == condition
                ]
                selected.sort(key=lambda row: row["rewind_fraction"])
                axis.errorbar(
                    [row["rewind_fraction"] for row in selected],
                    [row["loss"]["mean"] for row in selected],
                    yerr=[row["loss"]["std"] for row in selected],
                    marker="o",
                    capsize=3,
                    label=condition.replace("_", " "),
                )
            axis.axhline(
                cell["dense_loss"]["mean"], color="black", linestyle="--", label="dense"
            )
            axis.set_title(f"{cell['label']}, {sparsity:.0%}")
            axis.set_xlabel("Rewind fraction")
            axis.set_ylabel("Validation loss")
            axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(destination / "phase4_rewind_curves.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run representative Phase 4 rewind suites."
    )
    parser.add_argument("--phase4-root", default="results/phase4_robustness")
    parser.add_argument("--output-dir", default="results/phase4_rewinds")
    parser.add_argument(
        "--representatives",
        nargs="+",
        choices=sorted(REPRESENTATIVES),
        default=list(REPRESENTATIVES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        report = collect_results(
            args.phase4_root,
            args.representatives,
            args.seeds,
            args.sparsities,
            require_complete=True,
        )
        result = write_results(report, args.output_dir)
    else:
        result = run_phase4_rewinds(
            args.phase4_root,
            args.output_dir,
            args.representatives,
            args.seeds,
            args.sparsities,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
