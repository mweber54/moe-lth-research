"""Generate the publication figures used by ``paper.tex``.

The experiment runners intentionally keep their own diagnostic plots.  This
script is a read-only presentation layer over the recorded JSON artifacts: it
validates the expected experiment grid, computes paired derived quantities
from per-seed records where possible, and writes vector PDF plus 300-dpi PNG
previews.  It never modifies anything below ``results/``.

Usage::

    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --output-dir figures/paper --dpi 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[1]

MULTISEED = Path("results/wikitext103_gpu_multiseed/multiseed_summary.json")
CROSS_INIT = Path(
    "results/wikitext103_cross_init_replay/cross_init_replay_summary.json"
)
PHASE4 = Path("results/phase4_robustness/phase4_summary.json")
PHASE4_REWINDS = Path("results/phase4_rewinds/phase4_rewind_summary.json")

# Okabe--Ito colors, supplemented only with neutral grays.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
BLACK = "#222222"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D7D7D7"

ROUTING_ORDER = [
    "normal",
    "replay",
    "swapped",
    "fixed_random",
    "random_every_step",
    "shuffled_usage",
]
ROUTING_LABEL = {
    "normal": "Learned routing",
    "replay": "Route replay",
    "swapped": "Expert swap",
    "fixed_random": "Fixed random",
    "random_every_step": "Random / step",
    "shuffled_usage": "Shuffled usage",
}
ROUTING_COLOR = {
    "normal": BLUE,
    "replay": GREEN,
    "swapped": PURPLE,
    "fixed_random": SKY,
    "random_every_step": VERMILION,
    "shuffled_usage": ORANGE,
}
ROUTING_MARKER = {
    "normal": "o",
    "replay": "s",
    "swapped": "D",
    "fixed_random": "P",
    "random_every_step": "X",
    "shuffled_usage": "^",
}

MASK_STYLE = {
    "magnitude": ("Magnitude", BLUE, "o"),
    "other_expert_mask": ("0→1 transfer", GREEN, "D"),
    "random_mask": ("Random mask", ORANGE, "s"),
    "magnitude_mask_random_reinit": ("Random reinit", VERMILION, "X"),
}
REWIND_STYLE = {
    "learned_mask": ("Learned mask", BLUE, "o"),
    "random_mask": ("Random mask", ORANGE, "s"),
    "random_reinit": ("Random reinit", VERMILION, "X"),
    "randomized_routing": ("Random routing", PURPLE, "D"),
}

_READ_SOURCES: set[Path] = set()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "errorbar.capsize": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _repo_path(path: str | Path) -> Path:
    normalized = str(path).replace("\\", "/")
    candidate = Path(normalized)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_json(path: str | Path) -> Any:
    resolved = _repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required figure source is missing: {resolved}")
    _READ_SOURCES.add(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml(path: str | Path) -> Any:
    resolved = _repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required figure source is missing: {resolved}")
    _READ_SOURCES.add(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Figure-source validation failed: {message}")


def _summary(values: Sequence[float], *, population: bool = False) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    _require(array.size > 0, "cannot summarize an empty sequence")
    ddof = 0 if population or array.size == 1 else 1
    return float(array.mean()), float(array.std(ddof=ddof))


def _style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(length=2.5, width=0.6, color="#555555")
    ax.grid(axis=grid_axis, color="#E7E7E7", linewidth=0.55)
    ax.set_axisbelow(True)


def _set_log2_ticks(ax: plt.Axes, values: Sequence[float]) -> None:
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(FixedLocator(values))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}×"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda _value, _pos: ""))


def _save(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    relative_files: list[str] = []
    for suffix in ("pdf", "png"):
        destination = output_dir / f"{stem}.{suffix}"
        metadata: dict[str, Any] = {
            "Creator": "scripts/generate_paper_figures.py",
        }
        if suffix == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        fig.savefig(
            destination,
            dpi=dpi if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
            metadata=metadata,
        )
        relative_files.append(destination.relative_to(ROOT).as_posix())
    plt.close(fig)
    return relative_files


def _pruning_lookup(run: dict[str, Any], condition: str, sparsity: float) -> float:
    matches = [
        row["loss"]
        for row in run["pruning"]
        if row["condition"] == condition
        and np.isclose(float(row["sparsity"]), sparsity)
    ]
    _require(
        len(matches) == 1,
        f"expected one {condition}@{sparsity} row for {run.get('condition')}",
    )
    return float(matches[0])


def _load_wiki_runs(multiseed: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = multiseed.get("seeds")
    suite_dirs = multiseed.get("suite_dirs")
    _require(seeds == [7, 17, 29], f"unexpected WikiText seed set: {seeds}")
    _require(len(suite_dirs) == len(seeds), "suite directory/seed count mismatch")

    loaded: list[dict[str, Any]] = []
    for seed, suite_dir in zip(seeds, suite_dirs):
        suite_path = _repo_path(suite_dir)
        records = _read_json(suite_path / "suite_summary.json")
        resolved_config = _read_yaml(suite_path / "normal" / "resolved_config.yaml")
        _require(
            int(resolved_config["seed"]) == seed,
            f"suite {suite_dir} records seed {resolved_config['seed']}, expected {seed}",
        )
        by_condition = {row["condition"]: row for row in records}
        _require(
            set(ROUTING_ORDER).issubset(by_condition),
            f"seed {seed} is missing routing conditions",
        )
        loaded.append(
            {
                "seed": seed,
                "suite_dir": suite_path,
                "conditions": by_condition,
            }
        )
    return loaded


def _wiki_dense(run: dict[str, Any], routing: str) -> float:
    return _pruning_lookup(run["conditions"][routing], "dense", 0.0)


def _validate_sources(
    multiseed: dict[str, Any],
    cross_init: dict[str, Any],
    phase4: dict[str, Any],
    phase4_rewinds: dict[str, Any],
) -> None:
    _require(multiseed.get("seeds") == [7, 17, 29], "WikiText seeds changed")
    _require(
        set(multiseed.get("dense", {})) == set(ROUTING_ORDER),
        "WikiText routing grid changed",
    )
    _require(
        cross_init.get("source_seed") == 7
        and cross_init.get("target_seeds") == [17, 29],
        "cross-initialization seed design changed",
    )
    _require(
        cross_init.get("sparsities") == [0.5, 0.8]
        and len(cross_init.get("results", [])) == 2,
        "cross-initialization grid changed",
    )
    _require(
        len(phase4.get("architecture", [])) == 12
        and len(phase4.get("datasets", [])) == 3,
        "Phase-4 robustness grid changed",
    )
    _require(
        phase4_rewinds.get("seeds") == [7, 17, 29]
        and phase4_rewinds.get("sparsities") == [0.5, 0.8]
        and len(phase4_rewinds.get("cells", [])) == 3,
        "Phase-4 rewind grid changed",
    )
    _require(
        all(
            {int(result["seed"]) for result in cell.get("seed_results", [])}
            == {7, 17, 29}
            for cell in phase4_rewinds["cells"]
        ),
        "Phase-4 rewind cells do not contain the expected seed set",
    )


def figure_direct_pruning(
    runs: list[dict[str, Any]], output_dir: Path, dpi: int
) -> list[str]:
    fig, ax = plt.subplots(figsize=(3.25, 2.28))
    sparsities = [0.5, 0.8]
    methods = [
        "magnitude",
        "other_expert_mask",
        "random_mask",
        "magnitude_mask_random_reinit",
    ]
    offsets = np.linspace(-0.24, 0.24, len(methods))
    seed_jitter = np.asarray([-0.017, 0.0, 0.017])

    for method, offset in zip(methods, offsets):
        label, color, marker = MASK_STYLE[method]
        for position, sparsity in enumerate(sparsities):
            values = [
                _pruning_lookup(run["conditions"]["normal"], method, sparsity)
                / _wiki_dense(run, "normal")
                for run in runs
            ]
            mean, std = _summary(values)
            ax.scatter(
                position + offset + seed_jitter,
                values,
                s=8,
                color=color,
                alpha=0.28,
                linewidths=0,
                zorder=2,
            )
            ax.errorbar(
                position + offset,
                mean,
                yerr=std,
                color=color,
                marker=marker,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linewidth=1.0,
                zorder=3,
                label=label if position == 0 else None,
            )

    ax.axhline(1.0, color=BLACK, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(range(len(sparsities)), ["50%", "80%"])
    ax.set_xlabel("Expert-weight sparsity")
    ax.set_ylabel("Validation loss / dense loss")
    _set_log2_ticks(ax, [1, 2, 4, 8])
    ax.set_ylim(0.88, 12.5)
    ax.set_xlim(-0.48, 1.48)
    _style_axis(ax, grid_axis="y")
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.15,
    )
    fig.tight_layout(pad=0.35)
    return _save(fig, output_dir, "fig2a_direct_pruning", dpi)


def _load_wiki_rewinds(
    runs: list[dict[str, Any]], sparsity: float
) -> list[dict[str, Any]]:
    per_seed: list[dict[str, Any]] = []
    for run in runs:
        path = (
            run["suite_dir"]
            / "normal"
            / "tables"
            / f"rewind_suite_sparsity_{sparsity}.json"
        )
        rows = _read_json(path)
        per_seed.append(
            {
                "seed": run["seed"],
                "dense": _wiki_dense(run, "normal"),
                "rows": rows,
            }
        )
    return per_seed


def _find_rewind_loss(
    rows: Iterable[dict[str, Any]], condition: str, fraction: float, sparsity: float
) -> float:
    matches = [
        float(row["loss"])
        for row in rows
        if row["condition"] == condition
        and np.isclose(float(row["rewind_fraction"]), fraction)
        and np.isclose(float(row["sparsity"]), sparsity)
    ]
    _require(
        len(matches) == 1,
        f"expected one rewind row for {condition}@{fraction}/{sparsity}",
    )
    return matches[0]


def figure_wiki_rewinds(
    runs: list[dict[str, Any]], output_dir: Path, dpi: int
) -> list[str]:
    sparsity = 0.8
    per_seed = _load_wiki_rewinds(runs, sparsity)
    fractions = [0.0, 0.01, 0.05, 0.1]
    x = np.asarray([0, 1, 5, 10], dtype=float)

    fig, ax = plt.subplots(figsize=(3.25, 2.28))
    for condition, (label, color, marker) in REWIND_STYLE.items():
        means: list[float] = []
        stds: list[float] = []
        for fraction in fractions:
            values = [
                100.0
                * (
                    _find_rewind_loss(
                        seed_run["rows"], condition, fraction, sparsity
                    )
                    / seed_run["dense"]
                    - 1.0
                )
                for seed_run in per_seed
            ]
            mean, std = _summary(values)
            means.append(mean)
            stds.append(std)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            color=color,
            marker=marker,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=label,
        )

    ax.axhline(0.0, color=BLACK, linewidth=0.8, linestyle="--")
    ax.axhline(5.0, color=MID_GRAY, linewidth=0.75, linestyle=":")
    ax.text(10.35, 5.0, "5%", va="center", ha="left", color=MID_GRAY, fontsize=6.2)
    ax.set_xticks(x, ["0", "1", "5", "10"])
    ax.set_xlabel("Rewind point (% of dense training)")
    ax.set_ylabel("Loss gap vs. paired dense (%)")
    ax.set_xlim(-0.55, 11.25)
    _style_axis(ax, grid_axis="y")
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        columnspacing=0.75,
        handletextpad=0.35,
        borderaxespad=0.1,
    )
    fig.tight_layout(pad=0.35)
    return _save(fig, output_dir, "fig2b_rewind_80", dpi)


def figure_routing_mask_advantage(
    runs: list[dict[str, Any]], output_dir: Path, dpi: int
) -> list[str]:
    fig, ax = plt.subplots(figsize=(2.22, 2.32))
    sparsities = [(0.5, BLUE, "o", "50%"), (0.8, ORANGE, "s", "80%")]
    y = np.arange(len(ROUTING_ORDER))[::-1]

    for sparsity, color, marker, label in sparsities:
        offset = 0.12 if np.isclose(sparsity, 0.5) else -0.12
        for row_index, routing in enumerate(ROUTING_ORDER):
            values = []
            for run in runs:
                condition_run = run["conditions"][routing]
                magnitude = _pruning_lookup(condition_run, "magnitude", sparsity)
                random = _pruning_lookup(condition_run, "random_mask", sparsity)
                values.append(100.0 * (random - magnitude) / random)
            mean, std = _summary(values)
            ax.errorbar(
                mean,
                y[row_index] + offset,
                xerr=std,
                color=color,
                marker=marker,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linestyle="none",
                label=label if row_index == 0 else None,
            )

    ax.axvline(0.0, color=BLACK, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, [ROUTING_LABEL[key] for key in ROUTING_ORDER])
    ax.set_xlabel("Mask advantage (%)")
    _style_axis(ax, grid_axis="x")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3)
    fig.tight_layout(pad=0.25)
    return _save(fig, output_dir, "fig3a_routing_mask_advantage", dpi)


def figure_route_mask_association(
    multiseed: dict[str, Any], output_dir: Path, dpi: int
) -> list[str]:
    fig, ax = plt.subplots(figsize=(2.22, 2.32))
    pairwise = multiseed["pairwise"]

    # Plot the ten non-baseline pairs quietly so the complete 15-pair analysis
    # remains visible without a 15-entry legend.
    for key, value in pairwise.items():
        if "normal" in key.split("|"):
            continue
        ax.errorbar(
            value["routing_agreement"]["mean"],
            value["mask_jaccard"]["mean"],
            xerr=value["routing_agreement"]["std"],
            yerr=value["mask_jaccard"]["std"],
            fmt="o",
            markersize=2.7,
            color="#B7B7B7",
            ecolor="#D7D7D7",
            elinewidth=0.55,
            alpha=0.8,
            zorder=1,
        )

    annotations = {
        "replay": ("Replay (+)", (-46, -1)),
        "swapped": ("Swap", (-10, 11)),
        "fixed_random": ("Fixed", (4, -10)),
        "shuffled_usage": ("Shuffled", (4, 7)),
        "random_every_step": ("Random", (4, -3)),
    }
    for key, value in pairwise.items():
        parts = key.split("|")
        if "normal" not in parts:
            continue
        other = parts[0] if parts[1] == "normal" else parts[1]
        x = value["routing_agreement"]["mean"]
        y = value["mask_jaccard"]["mean"]
        color = ROUTING_COLOR[other]
        marker = ROUTING_MARKER[other]
        ax.errorbar(
            x,
            y,
            xerr=value["routing_agreement"]["std"],
            yerr=value["mask_jaccard"]["std"],
            fmt=marker,
            markersize=4.1,
            markeredgecolor="white",
            markeredgewidth=0.4,
            color=color,
            ecolor=color,
            elinewidth=0.7,
            zorder=3,
        )
        text, offset = annotations[other]
        ax.annotate(
            text,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.8,
            color=color if other != "replay" else BLACK,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.45},
        )

    correlation = multiseed["routing_history_mask_similarity_correlation"]
    ax.text(
        0.03,
        0.94,
        f"r = {correlation['mean']:.3f} ± {correlation['std']:.3f}\n15 pairs / seed",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color=BLACK,
    )
    ax.set_xlabel("Primary-route agreement")
    ax.set_ylabel("80%-mask Jaccard")
    ax.set_xlim(0.08, 1.055)
    ax.set_ylim(0.425, 1.045)
    _style_axis(ax, grid_axis="both")
    fig.tight_layout(pad=0.25)
    return _save(fig, output_dir, "fig3b_route_mask_association", dpi)


def figure_cross_init(
    cross_init: dict[str, Any], output_dir: Path, dpi: int
) -> list[str]:
    comparisons = [
        ("Source ↔ matched", "source_vs_matched_data_learned"),
        ("Source ↔ replay", "source_vs_cross_init_replay"),
        ("Matched ↔ replay", "matched_data_learned_vs_cross_init_replay"),
    ]
    sparsities = [("0.5", BLUE, "o", "50%"), ("0.8", ORANGE, "s", "80%")]
    y = np.arange(len(comparisons))[::-1]

    fig, ax = plt.subplots(figsize=(2.22, 2.32))
    for sparsity, color, marker, label in sparsities:
        offset = 0.11 if sparsity == "0.5" else -0.11
        for row_index, (_row_label, field) in enumerate(comparisons):
            values = [
                float(result["mask_similarity"][sparsity][field])
                for result in cross_init["results"]
            ]
            mean = float(np.mean(values))
            center = y[row_index] + offset
            target_offsets = np.linspace(-0.035, 0.035, len(values))
            ax.scatter(
                values,
                center + target_offsets,
                marker="|",
                s=38,
                color=color,
                linewidths=0.9,
                alpha=0.7,
                zorder=2,
            )
            ax.scatter(
                mean,
                center,
                marker=marker,
                s=24,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
                label=label if row_index == 0 else None,
            )

    ax.set_yticks(y, [label for label, _field in comparisons])
    ax.set_xlabel("Expert-mask Jaccard")
    ax.set_xlim(0.15, 0.64)
    _style_axis(ax, grid_axis="x")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.3)
    fig.tight_layout(pad=0.25)
    return _save(fig, output_dir, "fig3c_cross_init", dpi)


def _phase4_architecture_rows(phase4: dict[str, Any]) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in phase4["rows"]:
        if row["scope"] != "architecture":
            continue
        key = (int(row["num_experts"]), int(row["top_k"]), int(row["num_layers"]))
        grouped.setdefault(key, []).append(row)
    _require(
        len(grouped) == 12 and all(len(rows) == 3 for rows in grouped.values()),
        "expected 12 architecture cells with three seeds each",
    )
    _require(
        all({int(row["seed"]) for row in rows} == {7, 17, 29} for rows in grouped.values()),
        "architecture cells do not contain the expected seed set",
    )
    return grouped


def _pruning_value(row: dict[str, Any], field: str) -> float:
    value = row["pruning"][field]
    return float(value["loss"] if isinstance(value, dict) else value)


def figure_architecture_robustness(
    phase4: dict[str, Any], output_dir: Path, dpi: int
) -> list[str]:
    grouped = _phase4_architecture_rows(phase4)
    experts = [4, 8, 16]
    columns = [(1, 4), (1, 8), (2, 4), (2, 8)]
    matrix = np.zeros((len(experts), len(columns)), dtype=float)
    for row_index, expert_count in enumerate(experts):
        for column_index, (top_k, layers) in enumerate(columns):
            values = []
            for row in grouped[(expert_count, top_k, layers)]:
                magnitude = _pruning_value(row, "magnitude_0.8")
                random = _pruning_value(row, "random_mask_0.8")
                values.append(100.0 * (random - magnitude) / random)
            matrix[row_index, column_index] = np.mean(values)

    fig, ax = plt.subplots(figsize=(2.22, 2.28))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=60.0, aspect="auto")
    ax.set_xticks(
        range(len(columns)),
        [f"k={top_k}\n{layers}L" for top_k, layers in columns],
    )
    ax.set_yticks(range(len(experts)), [str(value) for value in experts])
    ax.set_ylabel("Experts per layer")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if value >= 33 else BLACK,
            )
    # Reference architecture: 8 experts, top-1, four layers.
    ax.add_patch(
        Rectangle(
            (-0.5, 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor=VERMILION,
            linewidth=1.2,
        )
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.035)
    colorbar.set_label("Random-mask loss reduction (%)", fontsize=6.4)
    colorbar.ax.tick_params(labelsize=6.0, length=2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.25)
    return _save(fig, output_dir, "fig4a_architecture_robustness", dpi)


def _phase4_dataset_rows(phase4: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "TinyStories": [],
        "WikiText-103": [],
        "Balanced Multi-Domain": [],
    }
    for row in phase4["rows"]:
        if row["scope"] == "dataset":
            grouped[row["dataset"]].append(row)
        elif (
            row["scope"] == "architecture"
            and row["dataset"] == "WikiText-103"
            and int(row["num_experts"]) == 8
            and int(row["top_k"]) == 1
            and int(row["num_layers"]) == 4
        ):
            grouped["WikiText-103"].append(row)
    _require(
        all(len(rows) == 3 for rows in grouped.values()),
        "expected three paired seeds for every dataset",
    )
    _require(
        all({int(row["seed"]) for row in rows} == {7, 17, 29} for rows in grouped.values()),
        "dataset cells do not contain the expected seed set",
    )
    return grouped


def figure_dataset_robustness(
    phase4: dict[str, Any], output_dir: Path, dpi: int
) -> list[str]:
    grouped = _phase4_dataset_rows(phase4)
    datasets = ["TinyStories", "WikiText-103", "Balanced Multi-Domain"]
    labels = ["TinyStories", "WikiText", "Balanced mix"]
    y = np.arange(len(datasets))[::-1]
    sparsities = [(0.5, BLUE, "o", "50%"), (0.8, ORANGE, "s", "80%")]

    fig, ax = plt.subplots(figsize=(2.22, 2.28))
    for sparsity, color, marker, label in sparsities:
        offset = 0.11 if np.isclose(sparsity, 0.5) else -0.11
        field_suffix = str(sparsity)
        for row_index, dataset in enumerate(datasets):
            values = []
            for row in grouped[dataset]:
                magnitude = _pruning_value(row, f"magnitude_{field_suffix}")
                random = _pruning_value(row, f"random_mask_{field_suffix}")
                values.append(100.0 * (random - magnitude) / random)
            mean, std = _summary(values)
            ax.errorbar(
                mean,
                y[row_index] + offset,
                xerr=std,
                color=color,
                marker=marker,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linestyle="none",
                label=label if row_index == 0 else None,
            )
    ax.axvline(0.0, color=BLACK, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mask advantage (%)")
    ax.set_xlim(-3.0, 75.0)
    _style_axis(ax, grid_axis="x")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3)
    fig.tight_layout(pad=0.25)
    return _save(fig, output_dir, "fig4b_dataset_robustness", dpi)


def figure_phase4_rewind_summary(
    phase4_rewinds: dict[str, Any], output_dir: Path, dpi: int
) -> list[str]:
    conditions = ["learned_mask", "random_mask", "random_reinit", "randomized_routing"]
    short_labels = {
        "best_dense": "4E/k2/8L",
        "high_capacity": "16E/k1/8L",
        "multi_domain": "Mix 8E/k1/4L",
    }
    legend_labels = {
        "learned_mask": "Learned",
        "random_mask": "Random mask",
        "random_reinit": "Reinit",
        "randomized_routing": "Random route",
    }
    y = np.arange(len(phase4_rewinds["cells"]))[::-1]
    offsets = np.linspace(0.21, -0.21, len(conditions))

    fig, ax = plt.subplots(figsize=(2.22, 2.28))
    for condition, offset in zip(conditions, offsets):
        _label, color, marker = REWIND_STYLE[condition]
        for row_index, cell in enumerate(phase4_rewinds["cells"]):
            values = []
            for seed_result in cell["seed_results"]:
                matches = [
                    row
                    for row in seed_result["rows"]
                    if row["condition"] == condition
                    and np.isclose(float(row["sparsity"]), 0.8)
                    and np.isclose(float(row["rewind_fraction"]), 0.1)
                ]
                _require(
                    len(matches) == 1,
                    f"missing Phase-4 80%@10% row for {cell['key']}/{condition}",
                )
                values.append(
                    100.0
                    * (float(matches[0]["loss"]) / float(seed_result["dense_loss"]) - 1.0)
                )
            mean, std = _summary(values, population=True)
            ax.errorbar(
                mean,
                y[row_index] + offset,
                xerr=std,
                color=color,
                marker=marker,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linestyle="none",
                label=legend_labels[condition] if row_index == 0 else None,
            )

    ax.axvline(0.0, color=BLACK, linewidth=0.8, linestyle="--")
    ax.axvline(5.0, color=MID_GRAY, linewidth=0.75, linestyle=":")
    ax.set_yticks(y, [short_labels[cell["key"]] for cell in phase4_rewinds["cells"]])
    ax.set_xlabel("Loss gap vs. dense (%)")
    _style_axis(ax, grid_axis="x")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.18),
        ncol=2,
        frameon=False,
        columnspacing=0.6,
        handletextpad=0.25,
    )
    fig.tight_layout(pad=0.25)
    return _save(fig, output_dir, "fig4c_phase4_rewinds", dpi)


def figure_dense_routing(
    multiseed: dict[str, Any], output_dir: Path, dpi: int
) -> list[str]:
    y = np.arange(len(ROUTING_ORDER))[::-1]
    baseline = float(multiseed["dense"]["normal"]["mean"])
    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    for row_index, routing in enumerate(ROUTING_ORDER):
        stats = multiseed["dense"][routing]
        ax.errorbar(
            stats["mean"],
            y[row_index],
            xerr=stats["std"],
            color=ROUTING_COLOR[routing],
            marker=ROUTING_MARKER[routing],
            markeredgecolor="white",
            markeredgewidth=0.45,
            linestyle="none",
        )
    ax.axvline(baseline, color=BLACK, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, [ROUTING_LABEL[key] for key in ROUTING_ORDER])
    ax.set_xlabel("Dense validation loss")
    _style_axis(ax, grid_axis="x")
    fig.tight_layout(pad=0.3)
    return _save(fig, output_dir, "figA1_dense_routing", dpi)


def figure_routing_pruning_curves(
    runs: list[dict[str, Any]], output_dir: Path, dpi: int
) -> list[str]:
    sparsities = [0.5, 0.7, 0.8, 0.9, 0.95]
    x = np.asarray([50, 70, 80, 90, 95])
    fig, ax = plt.subplots(figsize=(3.25, 2.35))

    for routing in ROUTING_ORDER:
        means: list[float] = []
        stds: list[float] = []
        for sparsity in sparsities:
            values = [
                _pruning_lookup(run["conditions"][routing], "magnitude", sparsity)
                / _wiki_dense(run, routing)
                for run in runs
            ]
            mean, std = _summary(values)
            means.append(mean)
            stds.append(std)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            color=ROUTING_COLOR[routing],
            marker=ROUTING_MARKER[routing],
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=ROUTING_LABEL[routing],
        )

    ax.axhline(1.0, color=BLACK, linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xlabel("Expert-weight sparsity (%)")
    ax.set_ylabel("Magnitude-pruned loss / dense loss")
    _set_log2_ticks(ax, [1, 2, 4, 8])
    ax.set_ylim(0.88, 10.5)
    _style_axis(ax, grid_axis="y")
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        columnspacing=0.65,
        handletextpad=0.3,
        borderaxespad=0.1,
    )
    fig.tight_layout(pad=0.3)
    return _save(fig, output_dir, "figA2_routing_pruning_curves", dpi)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(
    output_dir: Path, generated: dict[str, list[str]], dpi: int
) -> Path:
    sources = []
    for path in sorted(_READ_SOURCES):
        sources.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
        )
    outputs = []
    for relative_path in sorted(path for files in generated.values() for path in files):
        path = _repo_path(relative_path).resolve()
        outputs.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "generator": "scripts/generate_paper_figures.py",
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "formats": ["pdf", "png"],
        "png_dpi": dpi,
        "figures": generated,
        "outputs": outputs,
        "sources": sources,
        "uncertainty": {
            "wiki_multiseed_and_phase4_robustness": "sample SD over seeds 7, 17, and 29",
            "phase4_rewinds": "population SD over seeds 7, 17, and 29, matching the archived follow-up",
            "cross_initialization": "individual target seeds 17 and 29; no error-bar estimate",
        },
        "notes": [
            "All loss ratios and control advantages are paired within seed before aggregation.",
            "The route-mask correlation includes all 15 unordered routing-condition pairs, including the exact learned-routing/replay positive control.",
            "The transfer condition is the implemented expert-0-to-expert-1 replacement, not an all-expert cyclic transfer.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "figure_manifest.json"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def generate(output_dir: Path, dpi: int) -> dict[str, list[str]]:
    _configure_style()
    multiseed = _read_json(MULTISEED)
    cross_init = _read_json(CROSS_INIT)
    phase4 = _read_json(PHASE4)
    phase4_rewinds = _read_json(PHASE4_REWINDS)
    _validate_sources(multiseed, cross_init, phase4, phase4_rewinds)
    runs = _load_wiki_runs(multiseed)

    generated = {
        "fig2a_direct_pruning": figure_direct_pruning(runs, output_dir, dpi),
        "fig2b_rewind_80": figure_wiki_rewinds(runs, output_dir, dpi),
        "fig3a_routing_mask_advantage": figure_routing_mask_advantage(
            runs, output_dir, dpi
        ),
        "fig3b_route_mask_association": figure_route_mask_association(
            multiseed, output_dir, dpi
        ),
        "fig3c_cross_init": figure_cross_init(cross_init, output_dir, dpi),
        "fig4a_architecture_robustness": figure_architecture_robustness(
            phase4, output_dir, dpi
        ),
        "fig4b_dataset_robustness": figure_dataset_robustness(
            phase4, output_dir, dpi
        ),
        "fig4c_phase4_rewinds": figure_phase4_rewind_summary(
            phase4_rewinds, output_dir, dpi
        ),
        "figA1_dense_routing": figure_dense_routing(multiseed, output_dir, dpi),
        "figA2_routing_pruning_curves": figure_routing_pruning_curves(
            runs, output_dir, dpi
        ),
    }
    _write_manifest(output_dir, generated, dpi)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/paper"),
        help="Output directory, relative to the repository root by default.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require(args.dpi >= 150, "PNG dpi must be at least 150")
    output_dir = _repo_path(args.output_dir)
    generated = generate(output_dir, args.dpi)
    print(f"Generated {len(generated)} paper panels in {output_dir.relative_to(ROOT)}")
    print(f"Manifest: {(output_dir / 'figure_manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
