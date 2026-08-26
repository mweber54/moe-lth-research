from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_pruning_curves(results_path: str, output_path: str) -> None:
    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        series[row["condition"]].append((row["sparsity"], row["loss"]))
    figure, axis = plt.subplots(figsize=(7, 4))
    for condition, points in series.items():
        points.sort()
        axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=condition)
    axis.set(xlabel="sparsity", ylabel="validation loss")
    axis.legend()
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
