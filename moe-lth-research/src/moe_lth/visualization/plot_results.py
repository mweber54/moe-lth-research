from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_conceptual_diagram(output_path: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 3))
    axis.axis("off")
    boxes = [("Tokens", 0.08), ("Router", 0.30), ("Experts", 0.53), ("Sparse masks", 0.78)]
    for label, x in boxes:
        axis.text(x, 0.55, label, ha="center", va="center", bbox={"boxstyle": "round", "fc": "white"})
    for (_, left), (_, right) in zip(boxes, boxes[1:]):
        axis.annotate("", xy=(right - 0.08, 0.55), xytext=(left + 0.08, 0.55), arrowprops={"arrowstyle": "->"})
    axis.text(0.53, 0.15, r"Routing trajectory $H_e$ causally shapes mask $M_e$?", ha="center")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_rewind_curves(results_path: str, output_path: str) -> None:
    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        series[row["condition"]].append((row["rewind_fraction"], row["loss"]))
    figure, axis = plt.subplots(figsize=(7, 4))
    for condition, points in series.items():
        points.sort()
        axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=condition)
    axis.set(xlabel="rewind fraction", ylabel="validation loss")
    axis.legend()
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_route_mask_scatter(rows: list[dict], output_path: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    for row in rows:
        axis.scatter(row["routing_agreement"], row["mask_jaccard"])
        axis.annotate(f"{row['first']} / {row['second']}", (row["routing_agreement"], row["mask_jaccard"]), fontsize=7)
    axis.set(xlabel="routing-history similarity", ylabel="mask Jaccard similarity", xlim=(0, 1), ylim=(0, 1))
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_expert_specificity_heatmap(matrix: list[list[float]], output_path: str, title: str) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, cmap="magma")
    axis.set(xlabel="substitute expert", ylabel="routed-token source expert", title=title)
    figure.colorbar(image, ax=axis, label="loss")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
