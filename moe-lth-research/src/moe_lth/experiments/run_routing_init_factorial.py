from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def _coerce_float(value):
    return float(value)


def summarize_factorial(matrix: dict, metric: str = "cross_init_replay_loss") -> dict:
    """Summarize a route × initialization matrix as a two-way variance decomposition.

    The decomposition is intentionally simple and report-friendly: it measures the
    variance accounted for by the route main effect, the initialization main effect,
    and their interaction. This matches the revision plan's emphasis on separate
    route and init contributions without requiring a full mixed-effects model.
    """
    if "results" not in matrix:
        raise KeyError("Expected a factorial matrix with a 'results' list.")
    routes = list(matrix.get("source_seeds") or sorted({row["source_seed"] for row in matrix["results"]}))
    inits = list(matrix.get("target_seeds") or sorted({row["target_seed"] for row in matrix["results"]}))

    cells: dict[tuple[int, int], float] = {}
    for row in matrix["results"]:
        source_seed = int(row["source_seed"])
        target_seed = int(row["target_seed"])
        cells[(source_seed, target_seed)] = _coerce_float(row[metric])

    missing = []
    for route_seed in routes:
        for init_seed in inits:
            if (route_seed, init_seed) not in cells:
                missing.append((route_seed, init_seed))
    if missing:
        raise ValueError(f"Missing matrix entries for the requested metric: {missing[:5]}")

    grand_mean = mean(cells.values())
    route_means = {
        route_seed: mean(cells[(route_seed, init_seed)] for init_seed in inits)
        for route_seed in routes
    }
    init_means = {
        init_seed: mean(cells[(route_seed, init_seed)] for route_seed in routes)
        for init_seed in inits
    }

    route_effects = {route_seed: route_means[route_seed] - grand_mean for route_seed in routes}
    init_effects = {init_seed: init_means[init_seed] - grand_mean for init_seed in inits}
    route_effect_variance = mean(value * value for value in route_effects.values())
    init_effect_variance = mean(value * value for value in init_effects.values())

    interaction_values = []
    for route_seed in routes:
        for init_seed in inits:
            cell = cells[(route_seed, init_seed)]
            interaction = cell - route_means[route_seed] - init_means[init_seed] + grand_mean
            interaction_values.append(interaction)
    interaction_variance = mean(value * value for value in interaction_values)
    total_variance = mean((value - grand_mean) ** 2 for value in cells.values())

    report = {
        "metric": metric,
        "routes": routes,
        "initializations": inits,
        "grand_mean": grand_mean,
        "route_means": {str(seed): route_means[seed] for seed in routes},
        "init_means": {str(seed): init_means[seed] for seed in inits},
        "route_effects": {str(seed): route_effects[seed] for seed in routes},
        "init_effects": {str(seed): init_effects[seed] for seed in inits},
        "route_effect_variance": route_effect_variance,
        "init_effect_variance": init_effect_variance,
        "interaction_variance": interaction_variance,
        "total_variance": total_variance,
        "route_fraction_of_variance": route_effect_variance / total_variance if total_variance else 0.0,
        "init_fraction_of_variance": init_effect_variance / total_variance if total_variance else 0.0,
        "interaction_fraction_of_variance": interaction_variance / total_variance if total_variance else 0.0,
        "matrix": {
            str(route_seed): {str(init_seed): cells[(route_seed, init_seed)] for init_seed in inits}
            for route_seed in routes
        },
    }
    return report


def _write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "factorial_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    route_rows = [
        f"| {route_seed} | {report['route_means'][str(route_seed)]:.4f} | {report['route_effects'][str(route_seed)]:+.4f} |"
        for route_seed in report["routes"]
    ]
    init_rows = [
        f"| {init_seed} | {report['init_means'][str(init_seed)]:.4f} | {report['init_effects'][str(init_seed)]:+.4f} |"
        for init_seed in report["initializations"]
    ]
    matrix_rows = []
    for route_seed in report["routes"]:
        values = " | ".join(
            f"{report['matrix'][str(route_seed)][str(init_seed)]:.4f}"
            for init_seed in report["initializations"]
        )
        matrix_rows.append(f"| {route_seed} | {values} |")

    markdown = f'''# Routing × Initialization Factorial Summary

Metric: {report['metric']}\n
Grand mean: **{report['grand_mean']:.4f}**\n
- Route main-effect variance: **{report['route_effect_variance']:.4f}**\n- Initialization main-effect variance: **{report['init_effect_variance']:.4f}**\n- Route × initialization interaction variance: **{report['interaction_variance']:.4f}**\n- Total variance: **{report['total_variance']:.4f}**\n
## Route means

| Route seed | Mean metric | Effect vs. grand mean |
|---:|---:|---:|
{chr(10).join(route_rows)}

## Initialization means

| Target seed | Mean metric | Effect vs. grand mean |
|---:|---:|---:|
{chr(10).join(init_rows)}

## Cell matrix

| Route seed | {' | '.join(str(seed) for seed in report['initializations'])} |
|---:|{''.join(' ---:|' for _ in report['initializations'])}
{chr(10).join(matrix_rows)}

Raw results: [`factorial_summary.json`](factorial_summary.json)
'''
    (output_dir / "factorial_analysis_results.md").write_text(markdown, encoding="utf-8")


def run_routing_init_factorial(summary_path: str | Path, output_dir: str | Path, metric: str = "cross_init_replay_loss") -> dict:
    """Analyze a route × initialization matrix summary JSON and write a report."""
    source = Path(summary_path)
    matrix = json.loads(source.read_text(encoding="utf-8"))
    report = summarize_factorial(matrix, metric=metric)
    destination = Path(output_dir)
    _write_report(report, destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Variance-decompose a route × initialization factorial matrix.")
    parser.add_argument("--summary-json", required=True, help="A matrix summary JSON file with source_seeds/target_seeds/results entries.")
    parser.add_argument("--output-dir", required=True, help="Directory for the factorial summary and markdown report.")
    parser.add_argument("--metric", default="cross_init_replay_loss", help="Metric to decompose within each cell.")
    args = parser.parse_args()
    report = run_routing_init_factorial(args.summary_json, args.output_dir, metric=args.metric)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
