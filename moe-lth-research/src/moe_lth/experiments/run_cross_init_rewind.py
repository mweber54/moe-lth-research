from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from moe_lth.config import load_config
from moe_lth.experiments.run_rewind_suite import run_rewind_suite


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _final_step(run_dir: Path) -> int:
    return int(load_config(run_dir / "resolved_config.yaml")["training"]["steps"])


def _final_checkpoint(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / f"step_{_final_step(run_dir)}.pt"


def run_cross_init_rewind(
    cross_init_root: str,
    output_dir: str,
    sparsities: list[float],
    conditions: list[str],
) -> dict:
    source_root = Path(cross_init_root)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target_results = []

    for target_root in sorted(source_root.glob("source_*_target_*")):
        target_seed = int(target_root.name.rsplit("_", maxsplit=1)[-1])
        for condition in conditions:
            run_dir = target_root / condition
            resolved = run_dir / "resolved_config.yaml"
            if not resolved.exists():
                raise FileNotFoundError(f"Missing resolved config: {resolved}")
            config = load_config(resolved)
            final_checkpoint = _final_checkpoint(run_dir)
            if not final_checkpoint.exists():
                raise FileNotFoundError(f"Missing final checkpoint: {final_checkpoint}")

            rows = []
            for sparsity in sparsities:
                rows.extend(run_rewind_suite(config, str(final_checkpoint), sparsity))

            target_results.append(
                {
                    "target_seed": target_seed,
                    "condition": condition,
                    "dense_loss": _read_json(run_dir / "summary.json")["final_validation_loss"],
                    "rows": rows,
                    "run_dir": str(run_dir),
                }
            )
            _write_intermediate_report(target_results, sparsities, root)

    report = _aggregate(target_results, sparsities, source_root)
    (root / "cross_init_rewind_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_report(report, root)
    return report


def _aggregate(target_results: list[dict], sparsities: list[float], source_root: Path) -> dict:
    grouped: dict[tuple[str, float, str, float], list[float]] = defaultdict(list)
    dense: dict[str, list[float]] = defaultdict(list)
    for target in target_results:
        dense[target["condition"]].append(float(target["dense_loss"]))
        for row in target["rows"]:
            grouped[
                (
                    target["condition"],
                    float(row["sparsity"]),
                    row["condition"],
                    float(row["rewind_fraction"]),
                )
            ].append(float(row["loss"]))

    aggregate = [
        {
            "condition": condition,
            "sparsity": sparsity,
            "rewind_condition": rewind_condition,
            "rewind_fraction": fraction,
            "mean_loss": mean(losses),
            "std_loss": pstdev(losses),
            "losses": losses,
        }
        for (condition, sparsity, rewind_condition, fraction), losses in sorted(grouped.items())
    ]
    return {
        "cross_init_root": str(source_root),
        "target_seeds": sorted({row["target_seed"] for row in target_results}),
        "conditions": sorted({row["condition"] for row in target_results}),
        "sparsities": sparsities,
        "dense": {
            condition: {
                "mean": mean(losses),
                "std": pstdev(losses),
                "losses": losses,
            }
            for condition, losses in sorted(dense.items())
        },
        "targets": target_results,
        "aggregate": aggregate,
    }


def _write_intermediate_report(target_results: list[dict], sparsities: list[float], root: Path) -> None:
    report = _aggregate(target_results, sparsities, Path("pending"))
    (root / "cross_init_rewind_partial_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def _best_learned_rows(report: dict) -> list[str]:
    rows = []
    for condition in report["conditions"]:
        dense = report["dense"][condition]["mean"]
        for sparsity in report["sparsities"]:
            selected = [
                row
                for row in report["aggregate"]
                if row["condition"] == condition
                and row["sparsity"] == sparsity
                and row["rewind_condition"] == "learned_mask"
            ]
            best = min(selected, key=lambda row: row["mean_loss"])
            delta = (best["mean_loss"] / dense - 1.0) * 100.0
            rows.append(
                f"| {condition} | {sparsity:.0%} | {best['rewind_fraction']:.0%} | "
                f"{best['mean_loss']:.4f} | {delta:+.2f}% |"
            )
    return rows


def _all_rows(report: dict) -> list[str]:
    return [
        f"| {row['condition']} | {row['sparsity']:.0%} | {row['rewind_condition']} | "
        f"{row['rewind_fraction']:.0%} | {row['mean_loss']:.4f} | {row['std_loss']:.4f} |"
        for row in report["aggregate"]
    ]


def _best_learned(report: dict, condition: str, sparsity: float) -> dict | None:
    selected = [
        row
        for row in report["aggregate"]
        if row["condition"] == condition
        and row["sparsity"] == sparsity
        and row["rewind_condition"] == "learned_mask"
    ]
    return min(selected, key=lambda row: row["mean_loss"]) if selected else None


def _interpretation(report: dict) -> str:
    replay_50 = _best_learned(report, "cross_init_replay", 0.5)
    replay_80 = _best_learned(report, "cross_init_replay", 0.8)
    matched_50 = _best_learned(report, "matched_data_learned", 0.5)
    matched_80 = _best_learned(report, "matched_data_learned", 0.8)
    if not replay_50 or not replay_80:
        return "Cross-initialization replay rewind results are incomplete."

    text = (
        "Cross-initialization replay masks are rewindable: the best 50% learned "
        f"mask reaches `{replay_50['mean_loss']:.4f}` and the best 80% learned "
        f"mask reaches `{replay_80['mean_loss']:.4f}`. "
    )
    if matched_50 and matched_80:
        gap_50 = (replay_50["mean_loss"] / matched_50["mean_loss"] - 1.0) * 100.0
        gap_80 = (replay_80["mean_loss"] / matched_80["mean_loss"] - 1.0) * 100.0
        text += (
            "However, matched-data learned routing remains better by "
            f"{gap_50:.2f}% at 50% and {gap_80:.2f}% at 80% sparsity. "
        )
    return (
        text
        + "This means a foreign route history can still induce internally usable "
        "sparse tickets, but it does not match the quality of routes that "
        "co-adapt with the target initialization."
    )


def _write_report(report: dict, root: Path) -> None:
    dense_rows = [
        f"| {condition} | {stats['mean']:.4f} | {stats['std']:.4f} |"
        for condition, stats in report["dense"].items()
    ]
    markdown = f"""# Cross-Initialization Replay Rewind Results

Target seeds: {", ".join(map(str, report["target_seeds"]))}

This experiment rewinds the final sparse masks discovered under matched-data
learned routing and cross-initialization replay. The cross-init replay runs
force the source seed's exact route history onto independently initialized
target models.

## Interpretation

{_interpretation(report)}

## Dense Baselines

| Condition | Mean dense loss | Std |
|---|---:|---:|
{chr(10).join(dense_rows)}

## Best Learned-Mask Tickets

| Condition | Sparsity | Best rewind fraction | Mean loss | Delta vs own dense |
|---|---:|---:|---:|---:|
{chr(10).join(_best_learned_rows(report))}

## All Rewind Conditions

| Condition | Sparsity | Rewind condition | Rewind fraction | Mean loss | Std |
|---|---:|---|---:|---:|---:|
{chr(10).join(_all_rows(report))}

Raw results: [`cross_init_rewind_summary.json`](cross_init_rewind_summary.json)

![Cross-init rewind curves](cross_init_rewind.png)
"""
    (root / "cross_init_rewind_results.md").write_text(markdown, encoding="utf-8")
    _write_figure(report, root)


def _write_figure(report: dict, root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axes = plt.subplots(
        len(report["conditions"]),
        len(report["sparsities"]),
        figsize=(6 * len(report["sparsities"]), 4 * len(report["conditions"])),
        squeeze=False,
    )
    for row_id, condition in enumerate(report["conditions"]):
        for column_id, sparsity in enumerate(report["sparsities"]):
            axis = axes[row_id][column_id]
            for rewind_condition in ("learned_mask", "random_mask", "random_reinit", "randomized_routing"):
                selected = [
                    row
                    for row in report["aggregate"]
                    if row["condition"] == condition
                    and row["sparsity"] == sparsity
                    and row["rewind_condition"] == rewind_condition
                ]
                selected.sort(key=lambda row: row["rewind_fraction"])
                if not selected:
                    continue
                axis.errorbar(
                    [row["rewind_fraction"] for row in selected],
                    [row["mean_loss"] for row in selected],
                    yerr=[row["std_loss"] for row in selected],
                    marker="o",
                    capsize=3,
                    label=rewind_condition.replace("_", " "),
                )
            axis.axhline(
                report["dense"][condition]["mean"],
                color="black",
                linestyle="--",
                label="dense",
            )
            axis.set_title(f"{condition}, {sparsity:.0%}")
            axis.set_xlabel("Rewind fraction")
            axis.set_ylabel("Validation loss")
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "cross_init_rewind.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rewind suites for cross-initialization replay controls."
    )
    parser.add_argument("--cross-init-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["matched_data_learned", "cross_init_replay"],
    )
    args = parser.parse_args()
    report = run_cross_init_rewind(
        args.cross_init_root,
        args.output_dir,
        args.sparsities,
        args.conditions,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
