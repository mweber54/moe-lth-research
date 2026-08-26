from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

from moe_lth.config import load_config
from moe_lth.experiments.run_rewind_suite import run_rewind_suite


def _suite_dir(config: dict) -> Path:
    output = Path(config["output_dir"])
    return output.parent / f"{output.name}_suite"


def run_fixed_random_rewinds(
    config_paths: list[str],
    output_dir: str,
    sparsities: list[float],
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_results = []

    for config_path in config_paths:
        base_config = load_config(config_path)
        fixed_dir = _suite_dir(base_config) / "fixed_random"
        resolved_path = fixed_dir / "resolved_config.yaml"
        if not resolved_path.exists():
            raise FileNotFoundError(f"Missing fixed-random run for seed {base_config['seed']}: {fixed_dir}")
        config = load_config(resolved_path)
        final_checkpoint = fixed_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"
        if not final_checkpoint.exists():
            raise FileNotFoundError(f"Missing fixed-random checkpoint: {final_checkpoint}")

        rows = []
        for sparsity in sparsities:
            rows.extend(run_rewind_suite(config, str(final_checkpoint), sparsity))
        seed_results.append(
            {
                "seed": int(config["seed"]),
                "dense_loss": json.loads(
                    (fixed_dir / "summary.json").read_text(encoding="utf-8")
                )["final_validation_loss"],
                "rows": rows,
            }
        )

    grouped: dict[tuple[float, str, float], list[float]] = defaultdict(list)
    for seed_result in seed_results:
        for row in seed_result["rows"]:
            grouped[
                (float(row["sparsity"]), row["condition"], float(row["rewind_fraction"]))
            ].append(float(row["loss"]))

    aggregate = [
        {
            "sparsity": sparsity,
            "condition": condition,
            "rewind_fraction": fraction,
            "mean_loss": mean(losses),
            "std_loss": pstdev(losses),
            "losses": losses,
        }
        for (sparsity, condition, fraction), losses in sorted(grouped.items())
    ]
    report = {
        "seeds": [row["seed"] for row in seed_results],
        "sparsities": sparsities,
        "dense_mean_loss": mean(row["dense_loss"] for row in seed_results),
        "dense_std_loss": pstdev(row["dense_loss"] for row in seed_results),
        "seed_results": seed_results,
        "aggregate": aggregate,
    }
    (root / "fixed_random_rewind_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_report(report, root)
    return report


def _write_report(report: dict, root: Path) -> None:
    rows = [
        f"| {row['sparsity']:.0%} | {row['condition']} | {row['rewind_fraction']:.0%} | "
        f"{row['mean_loss']:.4f} | {row['std_loss']:.4f} |"
        for row in report["aggregate"]
    ]
    best_rows = []
    for sparsity in report["sparsities"]:
        selected = [
            row
            for row in report["aggregate"]
            if row["sparsity"] == sparsity and row["condition"] == "learned_mask"
        ]
        best = min(selected, key=lambda row: row["mean_loss"])
        delta = (best["mean_loss"] / report["dense_mean_loss"] - 1.0) * 100.0
        best_rows.append(
            f"| {sparsity:.0%} | {best['rewind_fraction']:.0%} | "
            f"{best['mean_loss']:.4f} | {delta:+.2f}% |"
        )

    markdown = f"""# Fixed-Random Rewind Results

Seeds: {", ".join(map(str, report["seeds"]))}

The fixed-random router projection remains frozen throughout dense training and
ticket retraining. Dense fixed-random validation loss is
`{report["dense_mean_loss"]:.4f} +/- {report["dense_std_loss"]:.4f}`.

## Best Learned-Mask Tickets

| Sparsity | Best rewind fraction | Mean loss | Difference from fixed-random dense |
|---:|---:|---:|---:|
{chr(10).join(best_rows)}

## All Rewind Conditions

| Sparsity | Condition | Rewind fraction | Mean loss | Std |
|---:|---|---:|---:|---:|
{chr(10).join(rows)}

Raw results: [`fixed_random_rewind_summary.json`](fixed_random_rewind_summary.json)

![Fixed-random rewind results](fixed_random_rewind.png)
"""
    (root / "fixed_random_rewind_results.md").write_text(markdown, encoding="utf-8")

    figure, axes = plt.subplots(
        1, len(report["sparsities"]), figsize=(6 * len(report["sparsities"]), 4), squeeze=False
    )
    for axis, sparsity in zip(axes[0], report["sparsities"], strict=True):
        for condition in ("learned_mask", "random_mask", "random_reinit", "randomized_routing"):
            selected = [
                row
                for row in report["aggregate"]
                if row["sparsity"] == sparsity and row["condition"] == condition
            ]
            selected.sort(key=lambda row: row["rewind_fraction"])
            axis.errorbar(
                [row["rewind_fraction"] for row in selected],
                [row["mean_loss"] for row in selected],
                yerr=[row["std_loss"] for row in selected],
                marker="o",
                label=condition.replace("_", " "),
            )
        axis.axhline(report["dense_mean_loss"], color="black", linestyle="--", label="dense")
        axis.set_title(f"{sparsity:.0%} sparsity")
        axis.set_xlabel("Rewind fraction")
        axis.set_ylabel("Validation loss")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "fixed_random_rewind.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and aggregate fixed-random-router rewind suites."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.8])
    args = parser.parse_args()
    report = run_fixed_random_rewinds(args.configs, args.output_dir, args.sparsities)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
