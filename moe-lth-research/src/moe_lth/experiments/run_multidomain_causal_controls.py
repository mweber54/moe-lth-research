from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

from moe_lth.config import load_config
from moe_lth.pruning.evaluate_pruning import evaluate_pruning
from moe_lth.training.train import train_from_config


CONDITIONS = {
    "normal": {"mode": "learned", "record_train_routes": True},
    "random_every_step": {"mode": "random_every_step", "record_train_routes": False},
    "fixed_random": {"mode": "fixed_random", "record_train_routes": False},
    "shuffled_usage": {"mode": "shuffled_usage", "record_train_routes": False},
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


def _condition_config(
    base_config: dict,
    suite_dir: Path,
    condition: str,
    baseline_history: Path | None,
) -> dict:
    config = deepcopy(base_config)
    condition_settings = CONDITIONS[condition]
    config["routing"]["mode"] = condition_settings["mode"]
    config["output_dir"] = str(suite_dir / condition)
    config["training"]["record_train_routes"] = bool(condition_settings["record_train_routes"])
    config["training"]["save_optimizer"] = False
    if condition == "shuffled_usage":
        if baseline_history is None:
            raise ValueError("shuffled_usage requires a baseline route history")
        config["routing"]["replay_path"] = str(baseline_history)
    else:
        config["routing"]["replay_path"] = None
    return config


def _is_training_complete(config: dict, condition: str) -> bool:
    run_dir = Path(config["output_dir"])
    complete = (run_dir / "summary.json").exists() and _final_checkpoint(run_dir, config).exists()
    if condition == "normal":
        complete = complete and (run_dir / "logs" / "train_route_history.npz").exists()
    return complete


def _upsert_summary(suite_summary_path: Path, summary: dict) -> None:
    summaries = _read_json(suite_summary_path) if suite_summary_path.exists() else []
    summaries = [row for row in summaries if row.get("condition") != summary["condition"]]
    summaries.append(summary)
    summaries.sort(key=lambda row: list(CONDITIONS).index(row.get("condition", "normal")))
    suite_summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def _read_usage(path: Path) -> dict[tuple[int, int, int], tuple[int, float]]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (int(record["step"]), int(record["layer_id"]), int(record["expert_id"]))
        records[key] = (int(record["token_count"]), float(record["usage_fraction"]))
    return records


def _usage_match(normal_dir: Path, shuffled_dir: Path) -> dict:
    normal_path = normal_dir / "logs" / "expert_usage.jsonl"
    shuffled_path = shuffled_dir / "logs" / "expert_usage.jsonl"
    if not normal_path.exists() or not shuffled_path.exists():
        return {"available": False}
    normal = _read_usage(normal_path)
    shuffled = _read_usage(shuffled_path)
    shared = sorted(set(normal) & set(shuffled))
    mismatches = [
        key
        for key in shared
        if normal[key][0] != shuffled[key][0] or abs(normal[key][1] - shuffled[key][1]) > 1e-12
    ]
    return {
        "available": True,
        "normal_records": len(normal),
        "shuffled_records": len(shuffled),
        "shared_records": len(shared),
        "missing_from_shuffled": len(set(normal) - set(shuffled)),
        "extra_in_shuffled": len(set(shuffled) - set(normal)),
        "mismatches": len(mismatches),
        "exact_match": len(mismatches) == 0
        and len(set(normal) - set(shuffled)) == 0
        and len(set(shuffled) - set(normal)) == 0,
    }


def _aggregate(suite_dirs: list[Path]) -> dict:
    dense: dict[str, list[float]] = defaultdict(list)
    pruning: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    pairwise: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"routing_agreement": [], "mask_jaccard": []}
    )
    correlations = []
    usage_matches = []
    seeds = []

    for suite_dir in suite_dirs:
        summaries = _read_json(suite_dir / "suite_summary.json")
        normal_config = load_config(suite_dir / "normal" / "resolved_config.yaml")
        seeds.append(int(normal_config["seed"]))
        for row in summaries:
            condition = row["condition"]
            dense[condition].append(float(row["final_validation_loss"]))
            for pruning_row in row.get("pruning", []):
                pruning[
                    (
                        condition,
                        pruning_row["condition"],
                        float(pruning_row["sparsity"]),
                    )
                ].append(float(pruning_row["loss"]))

        analysis_path = suite_dir / "tables" / "analysis_report.json"
        if analysis_path.exists():
            analysis = _read_json(analysis_path)
            correlation = analysis.get("routing_history_mask_similarity_correlation")
            if correlation is not None:
                correlations.append(float(correlation))
            for row in analysis.get("pairwise", []):
                key = (row["first"], row["second"])
                pairwise[key]["routing_agreement"].append(float(row["routing_agreement"]))
                pairwise[key]["mask_jaccard"].append(float(row["mask_jaccard"]))

        usage_matches.append(
            {
                "seed": int(normal_config["seed"]),
                **_usage_match(suite_dir / "normal", suite_dir / "shuffled_usage"),
            }
        )

    return {
        "dataset_name": "Balanced Multi-Domain",
        "seeds": sorted(seeds),
        "suite_dirs": [str(path) for path in suite_dirs],
        "dense": {condition: _summary(values) for condition, values in sorted(dense.items())},
        "pruning_by_condition": {
            f"{condition}|{mask_condition}|{sparsity:g}": _summary(values)
            for (condition, mask_condition, sparsity), values in sorted(pruning.items())
        },
        "routing_history_mask_similarity_correlation": (
            _summary(correlations) if correlations else None
        ),
        "pairwise": {
            f"{first}|{second}": {
                metric: _summary(values) for metric, values in metrics.items()
            }
            for (first, second), metrics in sorted(pairwise.items())
        },
        "usage_match": usage_matches,
    }


def _write_figures(report: dict, output_dir: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    order = [condition for condition in CONDITIONS if condition in report["dense"]]
    labels = [condition.replace("_", " ").title() for condition in order]
    means = [report["dense"][condition]["mean"] for condition in order]
    stds = [report["dense"][condition]["std"] for condition in order]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, means, yerr=stds, capsize=5)
    axis.set_ylabel("Validation loss")
    axis.set_title("Balanced multi-domain causal controls")
    axis.tick_params(axis="x", labelrotation=15)
    figure.tight_layout()
    figure.savefig(output_dir / "dense_conditions.png", dpi=160)
    plt.close(figure)

    if report["pairwise"]:
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
        figure.savefig(output_dir / "routing_vs_mask_similarity.png", dpi=160)
        plt.close(figure)

    for sparsity in (0.5, 0.8):
        rows = []
        for condition in order:
            key = f"{condition}|magnitude|{sparsity:g}"
            stats = report["pruning_by_condition"].get(key)
            if stats:
                rows.append((condition, stats["mean"], stats["std"]))
        if rows:
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.bar(
                [row[0].replace("_", " ").title() for row in rows],
                [row[1] for row in rows],
                yerr=[row[2] for row in rows],
                capsize=5,
            )
            axis.set_ylabel("Validation loss")
            axis.set_title(f"{sparsity:.0%} magnitude pruning by routing condition")
            axis.tick_params(axis="x", labelrotation=15)
            figure.tight_layout()
            figure.savefig(output_dir / f"magnitude_pruning_{sparsity:g}.png", dpi=160)
            plt.close(figure)
    return True


def _format_delta(value: float, baseline: float) -> str:
    return f"{(value / baseline - 1.0) * 100.0:+.2f}%"


def _write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "multidomain_causal_summary.json"
    report_path = output_dir / "multidomain_causal_results.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figures_written = _write_figures(report, output_dir)

    dense_rows = []
    normal_mean = report["dense"]["normal"]["mean"]
    for condition, stats in report["dense"].items():
        dense_rows.append(
            f"| {condition} | {stats['mean']:.4f} | {stats['std']:.4f} | "
            f"{_format_delta(stats['mean'], normal_mean) if condition != 'normal' else '-'} |"
        )

    pruning_rows = []
    for condition in CONDITIONS:
        if condition not in report["dense"]:
            continue
        values = []
        for sparsity in (0.5, 0.8):
            for mask_condition in ("magnitude", "random_mask", "other_expert_mask"):
                stats = report["pruning_by_condition"].get(f"{condition}|{mask_condition}|{sparsity:g}")
                values.append("-" if stats is None else f"{stats['mean']:.4f}")
        pruning_rows.append(
            f"| {condition} | {report['dense'][condition]['mean']:.4f} | "
            + " | ".join(values)
            + " |"
        )

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
    exact_matches = sum(1 for row in report["usage_match"] if row.get("exact_match"))
    usage_rows = [
        f"| {row['seed']} | {row.get('normal_records', '-')} | {row.get('shuffled_records', '-')} | "
        f"{row.get('mismatches', '-')} | {row.get('exact_match', False)} |"
        for row in report["usage_match"]
    ]
    dense_figure = "![Dense causal controls](dense_conditions.png)" if figures_written else ""
    routing_figure = (
        "![Routing agreement versus mask similarity](routing_vs_mask_similarity.png)"
        if figures_written and report["pairwise"]
        else ""
    )
    pruning_figures = (
        "\n\n".join(
            f"![{sparsity:.0%} magnitude pruning](magnitude_pruning_{sparsity:g}.png)"
            for sparsity in (0.5, 0.8)
            if figures_written
        )
    )
    shuffled = report["dense"].get("shuffled_usage")
    fixed = report["dense"].get("fixed_random")
    random = report["dense"].get("random_every_step")
    key_findings = [
        f"- Fixed-random routing is {_format_delta(fixed['mean'], normal_mean)} versus normal."
        if fixed
        else "",
        f"- Random-every-step routing is {_format_delta(random['mean'], normal_mean)} versus normal."
        if random
        else "",
        f"- Shuffled usage is {_format_delta(shuffled['mean'], normal_mean)} versus normal."
        if shuffled
        else "",
        f"- Shuffled usage exactly matched normal expert-count logs for {exact_matches}/{len(report['usage_match'])} seeds.",
        f"- Routing-history/mask-similarity correlation: {correlation_text}.",
    ]
    markdown = f"""# Balanced Multi-Domain Causal Controls

Seeds: {", ".join(map(str, report["seeds"]))}

## Key Findings

{chr(10).join(row for row in key_findings if row)}

## Dense Routing Conditions

| Condition | Mean loss | Std | Delta vs normal |
|---|---:|---:|---:|
{chr(10).join(dense_rows)}

{dense_figure}

## Shuffled-Usage Count Check

| Seed | Normal records | Shuffled records | Mismatches | Exact match |
|---:|---:|---:|---:|---:|
{chr(10).join(usage_rows)}

## Direct Pruning

| Routing condition | Dense | 50% magnitude | 50% random | 50% other expert | 80% magnitude | 80% random | 80% other expert |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pruning_rows)}

{pruning_figures}

## Routing and Mask Similarity

Mean routing-history/mask-similarity correlation: **{correlation_text}**

| Comparison | Routing agreement | Mask Jaccard |
|---|---:|---:|
{chr(10).join(pairwise_rows)}

{routing_figure}

Raw aggregate: [`multidomain_causal_summary.json`](multidomain_causal_summary.json)
"""
    report_path.write_text(markdown, encoding="utf-8")
    return summary_path, report_path


def run_multidomain_causal_controls(
    config_paths: list[str],
    output_dir: str,
    with_pruning: bool = True,
) -> dict:
    suite_dirs: list[Path] = []
    statuses = []

    for config_path in config_paths:
        base_config = load_config(config_path)
        suite_dir = _suite_dir(base_config)
        suite_dir.mkdir(parents=True, exist_ok=True)
        suite_dirs.append(suite_dir)
        suite_summary_path = suite_dir / "suite_summary.json"
        seed_status = {"seed": int(base_config["seed"]), "suite_dir": str(suite_dir), "conditions": {}}

        baseline_history = suite_dir / "normal" / "logs" / "train_route_history.npz"
        for condition in CONDITIONS:
            config = _condition_config(base_config, suite_dir, condition, baseline_history)
            run_dir = Path(config["output_dir"])
            final_checkpoint = _final_checkpoint(run_dir, config)
            pruning_path = run_dir / "tables" / "pruning_results.json"
            condition_status = {}

            if condition == "shuffled_usage" and not baseline_history.exists():
                raise FileNotFoundError(
                    f"Missing normal route history for shuffled usage: {baseline_history}"
                )

            if _is_training_complete(config, condition):
                summary = _read_json(run_dir / "summary.json")
                condition_status["training"] = "existing"
            else:
                summary = train_from_config(config)
                condition_status["training"] = "completed"

            if with_pruning:
                if pruning_path.exists():
                    summary["pruning"] = _read_json(pruning_path)
                    condition_status["pruning"] = "existing"
                else:
                    summary["pruning"] = evaluate_pruning(config, str(final_checkpoint))
                    condition_status["pruning"] = "completed"

            summary["condition"] = condition
            _upsert_summary(suite_summary_path, summary)
            seed_status["conditions"][condition] = condition_status

        try:
            from moe_lth.experiments.analyze import analyze_suite

            analyze_suite(str(suite_dir))
            seed_status["analysis"] = "completed"
        except ImportError as error:
            seed_status["analysis"] = f"skipped: {error}"
        statuses.append(seed_status)

        partial = _aggregate(suite_dirs)
        _write_report(partial, Path(output_dir))
        status_path = Path(output_dir) / "multidomain_causal_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"status": statuses, "partial": True}, indent=2),
            encoding="utf-8",
        )

    report = _aggregate(suite_dirs)
    summary_path, report_path = _write_report(report, Path(output_dir))
    result = {
        "status": statuses,
        "aggregate_summary": str(summary_path),
        "aggregate_report": str(report_path),
    }
    (Path(output_dir) / "multidomain_causal_status.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run balanced multi-domain fixed-random and shuffled-usage causal controls."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--without-pruning", action="store_true")
    args = parser.parse_args()
    result = run_multidomain_causal_controls(
        args.configs,
        args.output_dir,
        with_pruning=not args.without_pruning,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
