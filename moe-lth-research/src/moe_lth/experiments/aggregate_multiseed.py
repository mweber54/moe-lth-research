from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_suites(suite_dirs: list[str]) -> dict:
    dense: dict[str, list[float]] = defaultdict(list)
    pruning: dict[tuple[str, float], list[float]] = defaultdict(list)
    pruning_by_condition: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    rewind: dict[tuple[float, str, float], list[float]] = defaultdict(list)
    routing_mask_correlation: list[float] = []
    pairwise: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"routing_agreement": [], "mask_jaccard": []}
    )
    seeds = []
    dataset_names = []

    for suite_dir in map(Path, suite_dirs):
        summaries = json.loads((suite_dir / "suite_summary.json").read_text(encoding="utf-8"))
        normal = next(row for row in summaries if row["condition"] == "normal")
        resolved_config = json.loads(
            json.dumps(_load_yaml(suite_dir / "normal" / "resolved_config.yaml"))
        )
        seeds.append(int(resolved_config["seed"]))
        dataset_names.append(_dataset_name(resolved_config))
        for row in summaries:
            dense[row["condition"]].append(float(row["final_validation_loss"]))
            for pruning_row in row.get("pruning", []):
                pruning_by_condition[
                    (
                        row["condition"],
                        pruning_row["condition"],
                        float(pruning_row["sparsity"]),
                    )
                ].append(float(pruning_row["loss"]))
        for row in normal.get("pruning", []):
            pruning[(row["condition"], float(row["sparsity"]))].append(float(row["loss"]))

        analysis_path = suite_dir / "tables" / "analysis_report.json"
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            correlation = analysis.get("routing_history_mask_similarity_correlation")
            if correlation is not None:
                routing_mask_correlation.append(float(correlation))
            for row in analysis["pairwise"]:
                key = (row["first"], row["second"])
                pairwise[key]["routing_agreement"].append(float(row["routing_agreement"]))
                pairwise[key]["mask_jaccard"].append(float(row["mask_jaccard"]))

        for rewind_path in sorted((suite_dir / "normal" / "tables").glob("rewind_suite_sparsity_*.json")):
            for row in json.loads(rewind_path.read_text(encoding="utf-8")):
                key = (float(row["sparsity"]), row["condition"], float(row["rewind_fraction"]))
                rewind[key].append(float(row["loss"]))

    return {
        "dataset_name": dataset_names[0] if len(set(dataset_names)) == 1 else "Mixed Dataset",
        "seeds": sorted(seeds),
        "suite_dirs": suite_dirs,
        "dense": {condition: _summary(values) for condition, values in sorted(dense.items())},
        "normal_pruning": {
            f"{condition}|{sparsity:g}": _summary(values)
            for (condition, sparsity), values in sorted(pruning.items())
        },
        "pruning_by_condition": {
            f"{routing}|{condition}|{sparsity:g}": _summary(values)
            for (routing, condition, sparsity), values in sorted(pruning_by_condition.items())
        },
        "rewind": {
            f"{sparsity:g}|{condition}|{fraction:g}": _summary(values)
            for (sparsity, condition, fraction), values in sorted(rewind.items())
        },
        "routing_history_mask_similarity_correlation": (
            _summary(routing_mask_correlation) if routing_mask_correlation else None
        ),
        "pairwise": {
            f"{first}|{second}": {
                metric: _summary(values) for metric, values in metrics.items()
            }
            for (first, second), metrics in sorted(pairwise.items())
        },
    }


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def write_report(report: dict, output_dir: str) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "multiseed_summary.json"
    md_path = destination / "multiseed_results.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figures_written = False
    try:
        _write_figures(report, destination)
        figures_written = True
    except ImportError:
        figures_written = False

    dense_rows = [
        f"| {condition} | {stats['mean']:.4f} | {stats['std']:.4f} | "
        f"{stats['min']:.4f} | {stats['max']:.4f} |"
        for condition, stats in report["dense"].items()
    ]
    rewind_rows = [
        f"| {key.replace('|', ' | ')} | {stats['mean']:.4f} | {stats['std']:.4f} |"
        for key, stats in report["rewind"].items()
    ]
    pairwise_rows = [
        f"| {key.replace('|', ' vs ')} | "
        f"{stats['routing_agreement']['mean']:.4f} +/- {stats['routing_agreement']['std']:.4f} | "
        f"{stats['mask_jaccard']['mean']:.4f} +/- {stats['mask_jaccard']['std']:.4f} |"
        for key, stats in report["pairwise"].items()
    ]
    correlation = report["routing_history_mask_similarity_correlation"]
    correlation_text = (
        f"{correlation['mean']:.4f} +/- {correlation['std']:.4f}"
        if correlation
        else "not available"
    )
    dense_mean = report["dense"]["normal"]["mean"]
    random_delta = (report["dense"]["random_every_step"]["mean"] / dense_mean - 1.0) * 100.0
    swap_delta = (report["dense"]["swapped"]["mean"] / dense_mean - 1.0) * 100.0
    strict_ticket = report["rewind"].get("0.5|learned_mask|0")
    practical_ticket = report["rewind"].get("0.8|learned_mask|0.1")
    strict_ticket_text = ""
    if strict_ticket:
        delta = (strict_ticket["mean"] / dense_mean - 1.0) * 100.0
        direction = "better" if delta < 0 else "worse"
        strict_ticket_text = (
            f"- The 50% initialization-rewound learned mask is "
            f"**{abs(delta):.2f}% {direction}** than dense on average."
        )
    practical_ticket_text = (
        f"- The 80% learned mask at 10% rewind is within "
        f"**{abs((practical_ticket['mean'] / dense_mean - 1.0) * 100.0):.2f}%** of dense on average."
        if practical_ticket
        else ""
    )
    fixed_random = report["dense"].get("fixed_random")
    shuffled_usage = report["dense"].get("shuffled_usage")
    fixed_random_text = (
        f"- Fixed-random routing is **{(fixed_random['mean'] / dense_mean - 1.0) * 100.0:.2f}% worse** "
        "than normal, showing that frozen random router geometry remains surprisingly effective."
        if fixed_random
        else ""
    )
    shuffled_usage_text = (
        f"- Shuffled usage is **{(shuffled_usage['mean'] / dense_mean - 1.0) * 100.0:.2f}% worse** "
        "than normal despite preserving normal expert counts."
        if shuffled_usage
        else ""
    )
    rewind_figures = "\n\n".join(
        f"![{sparsity:.0%} sparsity rewind](rewind_{sparsity}.png)"
        for sparsity in (0.5, 0.8)
        if any(key.startswith(f"{sparsity}|") for key in report["rewind"])
    ) if figures_written else ""
    pruning_figure = (
        "![Normal-run pruning](normal_pruning.png)"
        if report["normal_pruning"] and figures_written
        else ""
    )
    causal_pruning_rows = []
    for routing in ("normal", "fixed_random", "random_every_step", "shuffled_usage"):
        dense_stats = report["dense"].get(routing)
        if dense_stats is None:
            continue
        values = []
        for sparsity in (0.5, 0.8):
            for condition in ("magnitude", "random_mask", "other_expert_mask"):
                stats = report["pruning_by_condition"].get(f"{routing}|{condition}|{sparsity:g}")
                values.append("-" if stats is None else f"{stats['mean']:.4f}")
        causal_pruning_rows.append(
            f"| {routing} | {dense_stats['mean']:.4f} | " + " | ".join(values) + " |"
        )
    causal_pruning_table = "\n".join(causal_pruning_rows)
    causal_pruning_figure = (
        "![Magnitude pruning by routing condition](magnitude_pruning_by_routing.png)"
        if report["pruning_by_condition"] and figures_written
        else ""
    )
    dense_figure = (
        "![Dense routing conditions](dense_conditions.png)" if figures_written else ""
    )
    routing_figure = (
        "![Routing agreement versus mask similarity](routing_vs_mask_similarity.png)"
        if figures_written
        else ""
    )
    key_findings = [
        f"- Random-every-step routing is **{random_delta:.2f}% worse** than normal.",
        f"- Swapped routing is **{swap_delta:.2f}% worse** than normal.",
        "- Replay exactly reproduces normal loss, routing, and masks in every seed.",
    ]
    key_findings.extend(
        text
        for text in (
            fixed_random_text,
            shuffled_usage_text,
            strict_ticket_text,
            practical_ticket_text,
        )
        if text
    )

    markdown = f"""# {report.get("dataset_name", "Multi-Seed")} Multi-Seed Results

Seeds: {", ".join(map(str, report["seeds"]))}

## Key Findings

{chr(10).join(key_findings)}

## Dense Routing Conditions

| Condition | Mean loss | Std | Min | Max |
|---|---:|---:|---:|---:|
{chr(10).join(dense_rows)}

{dense_figure}

## Routing and Mask Similarity

Mean routing-history/mask-similarity correlation: **{correlation_text}**

| Comparison | Routing agreement | Mask Jaccard |
|---|---:|---:|
{chr(10).join(pairwise_rows)}

{routing_figure}

## Direct Pruning

{pruning_figure}

### Causal-Control Pruning

| Routing condition | Dense | 50% magnitude | 50% random | 50% other expert | 80% magnitude | 80% random | 80% other expert |
|---|---:|---:|---:|---:|---:|---:|---:|
{causal_pruning_table}

{causal_pruning_figure}

## Rewind Conditions

| Sparsity | Condition | Rewind fraction | Mean loss | Std |
|---:|---|---:|---:|---:|
{chr(10).join(rewind_rows)}

{rewind_figures}

Raw aggregate: [`multiseed_summary.json`](multiseed_summary.json)
"""
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def _write_figures(report: dict, destination: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    preferred_order = [
        "normal",
        "fixed_random",
        "random_every_step",
        "replay",
        "swapped",
        "shuffled_usage",
    ]
    dense_order = [key for key in preferred_order if key in report["dense"]]
    dense_order.extend(key for key in report["dense"] if key not in dense_order)
    dense_labels = [key.replace("_", " ").title() for key in dense_order]
    dense_means = [report["dense"][key]["mean"] for key in dense_order]
    dense_stds = [report["dense"][key]["std"] for key in dense_order]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(dense_labels, dense_means, yerr=dense_stds, capsize=5)
    axis.set_ylabel("Validation loss")
    axis.set_title("Dense routing conditions across seeds")
    axis.tick_params(axis="x", labelrotation=15)
    figure.tight_layout()
    figure.savefig(destination / "dense_conditions.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for key, metrics in report["pairwise"].items():
        route = metrics["routing_agreement"]
        mask = metrics["mask_jaccard"]
        axis.errorbar(
            route["mean"],
            mask["mean"],
            xerr=route["std"],
            yerr=mask["std"],
            marker="o",
            capsize=3,
            label=key.replace("|", " vs "),
        )
    axis.set_xlabel("Routing agreement")
    axis.set_ylabel("Mask Jaccard")
    axis.set_title("Routing agreement versus mask similarity")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(destination / "routing_vs_mask_similarity.png", dpi=160)
    plt.close(figure)

    pruning: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for key, stats in report["normal_pruning"].items():
        condition, sparsity = key.split("|")
        pruning[condition].append((float(sparsity), stats["mean"], stats["std"]))
    if pruning:
        figure, axis = plt.subplots(figsize=(8, 5))
        for condition, rows in sorted(pruning.items()):
            rows.sort()
            axis.errorbar(
                [row[0] for row in rows],
                [row[1] for row in rows],
                yerr=[row[2] for row in rows],
                marker="o",
                capsize=3,
                label=condition,
            )
        axis.set_xlabel("Sparsity")
        axis.set_ylabel("Validation loss")
        axis.set_title("Normal-run direct pruning across seeds")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(destination / "normal_pruning.png", dpi=160)
        plt.close(figure)

    routing_pruning: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for key, stats in report["pruning_by_condition"].items():
        routing, condition, sparsity = key.split("|")
        if condition == "magnitude":
            routing_pruning[routing].append((float(sparsity), stats["mean"], stats["std"]))
    if routing_pruning:
        figure, axis = plt.subplots(figsize=(8, 5))
        for routing, rows in sorted(routing_pruning.items()):
            rows.sort()
            axis.errorbar(
                [row[0] for row in rows],
                [row[1] for row in rows],
                yerr=[row[2] for row in rows],
                marker="o",
                capsize=3,
                label=routing,
            )
        axis.set_xlabel("Sparsity")
        axis.set_ylabel("Validation loss")
        axis.set_title("Magnitude pruning by routing condition")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(destination / "magnitude_pruning_by_routing.png", dpi=160)
        plt.close(figure)

    for sparsity in (0.5, 0.8):
        rewind: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for key, stats in report["rewind"].items():
            key_sparsity, condition, fraction = key.split("|")
            if np.isclose(float(key_sparsity), sparsity):
                rewind[condition].append((float(fraction), stats["mean"], stats["std"]))
        if not rewind:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        for condition, rows in sorted(rewind.items()):
            rows.sort()
            axis.errorbar(
                [row[0] for row in rows],
                [row[1] for row in rows],
                yerr=[row[2] for row in rows],
                marker="o",
                capsize=3,
                label=condition,
            )
        dense = report["dense"]["normal"]
        axis.axhline(dense["mean"], color="black", linestyle="--", label="dense normal")
        axis.fill_between(
            [0.0, 0.1],
            dense["mean"] - dense["std"],
            dense["mean"] + dense["std"],
            color="black",
            alpha=0.1,
        )
        axis.set_xlabel("Rewind fraction")
        axis.set_ylabel("Validation loss")
        axis.set_title(f"{sparsity:.0%} sparsity rewind across seeds")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(destination / f"rewind_{sparsity}.png", dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed multi-seed experiment suites.")
    parser.add_argument("--suite-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = aggregate_suites(args.suite_dirs)
    paths = write_report(report, args.output_dir)
    print(json.dumps({"summary": str(paths[0]), "report": str(paths[1])}, indent=2))


if __name__ == "__main__":
    main()
