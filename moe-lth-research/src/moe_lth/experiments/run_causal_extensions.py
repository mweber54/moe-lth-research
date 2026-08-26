from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from moe_lth.config import load_config
from moe_lth.experiments.aggregate_multiseed import aggregate_suites, write_report
from moe_lth.pruning.evaluate_pruning import evaluate_pruning
from moe_lth.training.train import train_from_config


CAUSAL_EXTENSIONS = {
    "fixed_random": {"mode": "fixed_random"},
    "shuffled_usage": {"mode": "shuffled_usage"},
    "deconfounded_shuffle": {"mode": "deconfounded_shuffle"},
    "graded_corruption_0.25": {"mode": "graded_corruption", "corruption_fraction": 0.25},
}


def _suite_dir(config: dict) -> Path:
    output = Path(config["output_dir"])
    return output.parent / f"{output.name}_suite"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _upsert_summary(path: Path, summary: dict) -> None:
    summaries = _read_json(path) if path.exists() else []
    summaries = [row for row in summaries if row.get("condition") != summary["condition"]]
    summaries.append(summary)
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def run_causal_extensions(
    config_paths: list[str],
    output_dir: str,
    with_pruning: bool = True,
) -> dict:
    suite_dirs = []
    statuses = []
    for config_path in config_paths:
        base_config = load_config(config_path)
        suite_dir = _suite_dir(base_config)
        suite_dirs.append(str(suite_dir))
        baseline_history = suite_dir / "normal" / "logs" / "rich_train_route_history.npz"
        if not baseline_history.exists():
            baseline_history = suite_dir / "normal" / "logs" / "train_route_history.npz"
        if not baseline_history.exists():
            raise FileNotFoundError(
                f"Missing normal route history for seed {base_config['seed']}: {baseline_history}"
            )
        suite_summary = suite_dir / "suite_summary.json"
        seed_status = {"seed": int(base_config["seed"]), "suite_dir": str(suite_dir), "conditions": {}}

        for condition, routing in CAUSAL_EXTENSIONS.items():
            config = deepcopy(base_config)
            config["routing"].update(routing)
            config["output_dir"] = str(suite_dir / condition)
            config["training"]["record_train_routes"] = False
            config["training"]["record_rich_routes"] = condition in {"deconfounded_shuffle", "graded_corruption_0.25"}
            if condition in {"shuffled_usage", "deconfounded_shuffle", "graded_corruption_0.25"}:
                config["routing"]["replay_path"] = str(baseline_history)
            else:
                config["routing"]["replay_path"] = None

            run_dir = Path(config["output_dir"])
            summary_path = run_dir / "summary.json"
            final_checkpoint = run_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"
            pruning_path = run_dir / "tables" / "pruning_results.json"
            condition_status = {}

            if summary_path.exists() and final_checkpoint.exists():
                summary = _read_json(summary_path)
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
            _upsert_summary(suite_summary, summary)
            seed_status["conditions"][condition] = condition_status

        try:
            from moe_lth.experiments.analyze import analyze_suite

            analyze_suite(str(suite_dir))
            seed_status["analysis"] = "completed"
        except ImportError as error:
            seed_status["analysis"] = f"skipped: {error}"
        statuses.append(seed_status)

    report = aggregate_suites(suite_dirs)
    summary_path, report_path = write_report(report, output_dir)
    result = {
        "status": statuses,
        "aggregate_summary": str(summary_path),
        "aggregate_report": str(report_path),
    }
    status_path = Path(output_dir) / "causal_extensions_status.json"
    status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append fixed-random and shuffled-usage controls to completed suites."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--without-pruning", action="store_true")
    args = parser.parse_args()
    result = run_causal_extensions(
        args.configs,
        args.output_dir,
        with_pruning=not args.without_pruning,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
