from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moe_lth.utils import read_jsonl


def plot_expert_usage(usage_log: str, output_path: str) -> None:
    records = read_jsonl(usage_log)
    series: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        series[(record["layer_id"], record["expert_id"])].append((record["step"], record["usage_fraction"]))
    figure, axes = plt.subplots(
        max(1, len({layer for layer, _ in series})),
        1,
        figsize=(8, max(3, 2.5 * len({layer for layer, _ in series}))),
        squeeze=False,
    )
    for (layer_id, expert_id), points in series.items():
        steps, usage = zip(*points)
        axes[layer_id][0].plot(steps, usage, label=f"expert {expert_id}")
        axes[layer_id][0].set_title(f"Layer {layer_id} expert usage")
        axes[layer_id][0].set_ylabel("usage fraction")
    axes[-1][0].set_xlabel("training step")
    axes[0][0].legend(ncol=4, fontsize=7)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_routing_stability(stability: list[dict], output_path: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    by_condition: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in stability:
        by_condition[row["condition"]].append((row["checkpoint"], row["agreement"]))
    for condition, points in by_condition.items():
        points.sort()
        axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=condition)
    axis.set(xlabel="checkpoint", ylabel="routing agreement", ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
