from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_lth.config import load_config
from moe_lth.experiments.aggregate_multiseed import aggregate_suites, write_report
from moe_lth.experiments.analyze import analyze_suite
from moe_lth.experiments.analyze_checkpoint import analyze_checkpoint
from moe_lth.experiments.run_rewind_suite import run_rewind_suite
from moe_lth.experiments.run_suite import run_suite


def _suite_dir(config: dict) -> Path:
    output = Path(config["output_dir"])
    return output.parent / f"{output.name}_suite"


def run_multiseed(
    config_paths: list[str],
    sparsities: list[float],
    output_dir: str,
    with_pruning: bool = True,
) -> dict:
    suite_dirs = []
    status = []
    for config_path in config_paths:
        config = load_config(config_path)
        suite_dir = _suite_dir(config)
        suite_dirs.append(str(suite_dir))
        seed_status = {"seed": int(config["seed"]), "suite_dir": str(suite_dir), "stages": {}}

        suite_summary = suite_dir / "suite_summary.json"
        if suite_summary.exists():
            seed_status["stages"]["suite"] = "existing"
        else:
            run_suite(config, with_pruning=with_pruning)
            seed_status["stages"]["suite"] = "completed"

        analysis_path = suite_dir / "tables" / "analysis_report.json"
        if analysis_path.exists():
            seed_status["stages"]["analysis"] = "existing"
        else:
            analyze_suite(str(suite_dir))
            seed_status["stages"]["analysis"] = "completed"

        normal_dir = suite_dir / "normal"
        final_checkpoint = normal_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"
        checkpoint_analysis = normal_dir / "tables" / "checkpoint_analysis.json"
        if checkpoint_analysis.exists():
            seed_status["stages"]["checkpoint_analysis"] = "existing"
        else:
            normal_config = load_config(normal_dir / "resolved_config.yaml")
            analyze_checkpoint(normal_config, str(final_checkpoint))
            seed_status["stages"]["checkpoint_analysis"] = "completed"

        for sparsity in sparsities:
            table = normal_dir / "tables" / f"rewind_suite_sparsity_{sparsity}.json"
            stage = f"rewind_{sparsity:g}"
            if table.exists():
                seed_status["stages"][stage] = "existing"
            else:
                run_rewind_suite(config, str(final_checkpoint), sparsity)
                seed_status["stages"][stage] = "completed"
        status.append(seed_status)

    report = aggregate_suites(suite_dirs)
    summary_path, report_path = write_report(report, output_dir)
    result = {
        "status": status,
        "aggregate_summary": str(summary_path),
        "aggregate_report": str(report_path),
    }
    status_path = Path(output_dir) / "run_status.json"
    status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and aggregate independent experiment seeds.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--without-pruning", action="store_true")
    args = parser.parse_args()
    result = run_multiseed(
        args.configs,
        args.sparsities,
        args.output_dir,
        with_pruning=not args.without_pruning,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
