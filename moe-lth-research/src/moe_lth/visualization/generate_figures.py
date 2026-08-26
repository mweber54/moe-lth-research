from __future__ import annotations

import argparse
from pathlib import Path

from .plot_pruning import plot_pruning_curves
from .plot_results import plot_rewind_curves


def generate_run_figures(run_dir: str) -> list[str]:
    root = Path(run_dir)
    generated = []
    pruning_results = root / "tables" / "pruning_results.json"
    if pruning_results.exists():
        destination = root / "figures" / "pruning_curves.png"
        plot_pruning_curves(str(pruning_results), str(destination))
        generated.append(str(destination))
    for rewind_results in (root / "tables").glob("rewind_suite_sparsity_*.json"):
        sparsity = rewind_results.stem.removeprefix("rewind_suite_sparsity_")
        destination = root / "figures" / f"rewind_curves_sparsity_{sparsity}.png"
        plot_rewind_curves(str(rewind_results), str(destination))
        generated.append(str(destination))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    for path in generate_run_figures(args.run_dir):
        print(path)


if __name__ == "__main__":
    main()
