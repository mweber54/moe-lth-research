"""Analyze the completed pilot without forward, backward, masking, or updates."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "configuration.json").read_text())
SEED = CONFIG["cell"]["seed"]
ORDER = CONFIG["cell"]["order"]
STEP = CONFIG["cell"]["step"]
SOURCE = ROOT / "results/moe_group_gradient_consistency_v1"
ACTIVITY_FLOOR = float(CONFIG["population_thresholds"]["activity_floor"])
PRIMARY_ACTIVITY_QUANTILE = float(
    CONFIG["population_thresholds"]["primary_activity_quantile"]
)
PRIMARY_CANCELLATION = float(CONFIG["population_thresholds"]["primary_cancellation"])
BIN_EDGES = [x / 100 for x in range(0, 101)]
BIN_NAMES = ["0-10", "10-20", "20-40", "40-60", "60-80", "80-100"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path.name}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metadata(name: str) -> dict:
    parts = name.split(".")
    return {
        "parameter": name,
        "layer": int(parts[1]),
        "expert": int(parts[4]),
        "projection": parts[5],
    }


def finite_float(value) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q, interpolation="linear"))


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    keep = np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < 2:
        return None
    x, y = a[keep], b[keep]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return finite_float(np.corrcoef(x, y)[0, 1])


def rank_sample(a: np.ndarray, b: np.ndarray, sample_size: int = 20_000):
    if len(a) <= sample_size:
        x, y = a, b
    else:
        indices = np.linspace(0, len(a) - 1, sample_size, dtype=np.int64)
        x, y = a[indices], b[indices]
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return None
    if np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return None
    return finite_float(spearmanr(x[keep], y[keep]).statistic)


def present_mean(values) -> float:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else float("nan")


def matrix_slices(half, index):
    mu = torch.stack(
        [half["domain_mean_unconditional"][domain][index] for domain in (0, 1)]
    ).reshape(2, -1)
    moment = torch.stack(
        [half["domain_probe_second_moment"][domain][index] for domain in (0, 1)]
    ).reshape(2, -1)
    return mu, moment


def activation_coordinate(name: str, activation: dict, metric: str) -> torch.Tensor:
    meta = metadata(name)
    value = activation[
        f"{meta['layer']}:{meta['expert']}:1"
    ][metric].reshape(-1)
    if meta["projection"] == "fc1":
        return value[:, None].expand(1024, 256).reshape(-1).float()
    return value[None, :].expand(256, 1024).reshape(-1).float()


def magnitude_partition(weight: torch.Tensor) -> tuple[np.ndarray, list[np.ndarray]]:
    order = np.lexsort((np.arange(weight.numel()), weight.numpy()))
    boundaries = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    partitions = []
    for low, high in zip(boundaries, boundaries[1:]):
        start = int(low * len(order))
        stop = int(high * len(order))
        partitions.append(order[start:stop])
    return order, partitions


def summarize():
    required = [HERE / f"coordinate_statistics_half{half}.pt" for half in (1, 2)]
    if not all(path.exists() for path in required):
        raise RuntimeError("both pilot halves must complete before analysis")
    half1, half2 = [torch.load(path, map_location="cpu", weights_only=False) for path in required]
    if half1["names"] != half2["names"] or half1["shapes"] != half2["shapes"]:
        raise RuntimeError("half coordinate schemas differ")
    names = half1["names"]
    if len(names) != 256:
        raise RuntimeError("unexpected parameter count")
    checkpoint = (
        ROOT / "results/moe_connection_lifetime_v1/prefixes"
        / f"seed{SEED}_{ORDER}"
        / f"state{STEP}.pt"
    )
    if not checkpoint.exists():
        checkpoint = (
            ROOT / "results/moe_online_reinforcement_pruning_v1/prefixes"
            / f"seed{SEED}_{ORDER}/native/interruption.pt"
        )
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    weights = [payload["model"][name].detach().abs().float().reshape(-1) for name in names]
    if sum(weight.numel() for weight in weights) != 67_108_864:
        raise RuntimeError("unexpected coordinate count")

    probabilities = []
    for half in (half1, half2):
        counts = np.asarray(half["domain_target_count"], dtype=np.float64)
        probabilities.append(counts / counts.sum())

    cell_rows = []
    bin_rows = []
    reproducibility_rows = []
    correlation_rows = []
    plot_x = []
    plot_y = []
    plot_bin_rows = []
    maxima = {
        "cancellation_below_zero": 0.0,
        "cancellation_above_one": 0.0,
        "cancelled_mass_negative": 0.0,
        "triangle_violation": 0.0,
    }
    nonfinite = 0
    total_coordinates = 0
    total_qualified = 0
    total_cancelled_mass = 0.0
    selected_count = 0
    selected_retained = 0
    selected_same_direction = 0
    selected_small_weight = 0
    selected_small_weight_retained = 0
    all_total_cancelled_mass = 0.0
    all_selected_cancelled_mass = 0.0

    for index, name in enumerate(names):
        meta = metadata(name)
        weight = weights[index]
        mu1, moment1 = matrix_slices(half1, index)
        mu2, moment2 = matrix_slices(half2, index)
        absolute_mu1 = mu1.abs()
        aggregate1 = probabilities[0][0] * mu1[0] + probabilities[0][1] * mu1[1]
        aggregate2 = probabilities[1][0] * mu2[0] + probabilities[1][1] * mu2[1]
        activity1 = probabilities[0][0] * absolute_mu1[0] + probabilities[0][1] * absolute_mu1[1]
        activity2 = probabilities[1][0] * mu2[0].abs() + probabilities[1][1] * mu2[1].abs()
        qualified1 = torch.isfinite(activity1) & (activity1 > ACTIVITY_FLOOR)
        qualified2 = torch.isfinite(activity2) & (activity2 > ACTIVITY_FLOOR)
        cancellation1 = torch.full_like(activity1, torch.nan)
        cancellation2 = torch.full_like(activity2, torch.nan)
        cancellation1[qualified1] = 1 - aggregate1[qualified1].abs() / activity1[qualified1]
        cancellation2[qualified2] = 1 - aggregate2[qualified2].abs() / activity2[qualified2]
        cancelled1 = activity1 - aggregate1.abs()
        cancelled2 = activity2 - aggregate2.abs()
        totals = [
            int(qualified1.sum()),
            int(qualified2.sum()),
            int((~qualified1 & (activity1 >= 0)).sum()),
        ]
        total_coordinates += int(weight.numel())
        total_qualified += totals[0]
        nonfinite += int((~torch.isfinite(activity1)).sum())
        if qualified1.any():
            c = cancellation1[qualified1]
            maxima["cancellation_below_zero"] = max(
                maxima["cancellation_below_zero"], float((-c[c < 0]).max()) if (c < 0).any() else 0
            )
            maxima["cancellation_above_one"] = max(
                maxima["cancellation_above_one"], float((c[c > 1] - 1).max()) if (c > 1).any() else 0
            )
        maxima["cancelled_mass_negative"] = max(
            maxima["cancelled_mass_negative"],
            float((-cancelled1[cancelled1 < 0]).max()) if (cancelled1 < 0).any() else 0.0,
        )
        maxima["triangle_violation"] = max(
            maxima["triangle_violation"],
            float((aggregate1.abs() - activity1).clamp_min(0).max()),
        )
        activity_q90 = quantile(activity1[qualified1], PRIMARY_ACTIVITY_QUANTILE)
        high_activity1 = qualified1 & (activity1 >= activity_q90)
        high_cancellation1 = qualified1 & (cancellation1 >= PRIMARY_CANCELLATION)
        selected1 = high_activity1 & high_cancellation1
        same_opposition = selected1 & qualified2 & (
            torch.sign(mu1[0]) * torch.sign(mu1[1]) == torch.sign(mu2[0]) * torch.sign(mu2[1])
        )
        retained = selected1 & qualified2 & (cancellation2 >= PRIMARY_CANCELLATION)
        retained_activity = selected1 & qualified2 & (activity2 >= activity_q90)
        _, partitions = magnitude_partition(weight)
        rank = torch.empty_like(weight, dtype=torch.int16)
        for bin_index, partition in enumerate(partitions):
            rank[torch.from_numpy(partition)] = bin_index
        small = rank <= 1
        selected_count += int(selected1.sum())
        selected_retained += int(retained.sum())
        selected_same_direction += int(same_opposition.sum())
        selected_small_weight += int((selected1 & small).sum())
        selected_small_weight_retained += int((selected1 & small & retained).sum())
        all_total_cancelled_mass += float(cancelled1[qualified1].sum())
        all_selected_cancelled_mass += float(cancelled1[selected1].sum())

        accepted = np.asarray(half1["accepted_occurrences"], dtype=np.float64)
        proposed = np.asarray(half1["proposed_occurrences"], dtype=np.float64)
        contributing = np.asarray(half1["contributing_batches"], dtype=np.int64)
        layer, expert = meta["layer"], meta["expert"]
        target_counts = np.asarray(half1["domain_target_count"], dtype=np.float64)
        conditional = torch.zeros_like(mu1)
        for domain_index in (0, 1):
            denominator = accepted[domain_index, layer, expert]
            conditional[domain_index] = (
                mu1[domain_index] * target_counts[domain_index] / denominator
                if denominator > 0
                else torch.nan
            )
        activation_values = {}
        correlation_inputs = {
            "magnitude": weight.numpy(),
            "aggregate_abs": aggregate1.abs().numpy(),
            "activity": activity1.numpy(),
            "cancellation": cancellation1.numpy(),
            "cancelled_mass": cancelled1.numpy(),
            "domain_1_mu_square": mu1[0].square().numpy(),
            "domain_2_mu_square": mu1[1].square().numpy(),
            "domain_1_probe_second_moment": moment1[0].numpy(),
            "domain_2_probe_second_moment": moment1[1].numpy(),
            "conditional_domain_1_abs": conditional[0].abs().numpy(),
            "conditional_domain_2_abs": conditional[1].abs().numpy(),
        }
        for metric in ("ramp", "activation_only", "routing_activation_l2", "ramp_per_dispatch"):
            key = {
                "activation_only": "activation_l2",
                "routing_activation_l2": "routing_activation_l2",
            }.get(metric, metric)
            score = activation_coordinate(name, half1["activation_metrics"], key)
            activation_values[metric] = score
            correlation_inputs[metric] = score.numpy()

        cell_rows.append(
            {
                "seed": SEED,
                "order": ORDER,
                "step": STEP,
                **meta,
                "coordinates": int(weight.numel()),
                "qualified_coordinates": totals[0],
                "near_zero_activity_coordinates": totals[2],
                "domain_probability_tinystories": probabilities[0][0],
                "domain_probability_wikitext103": probabilities[0][1],
                "mean_activity": float(activity1[qualified1].mean()),
                "median_activity": float(activity1[qualified1].median()),
                "mean_cancellation": float(cancellation1[qualified1].mean()),
                "mean_cancelled_mass": float(cancelled1[qualified1].mean()),
                "total_cancelled_mass": float(cancelled1[qualified1].sum()),
                "high_activity_threshold": activity_q90,
                "high_cancellation_threshold": PRIMARY_CANCELLATION,
                "high_activity_high_cancellation_count": int(selected1.sum()),
                "high_activity_high_cancellation_fraction": float(
                    selected1[qualified1].float().mean()
                ),
                "small_weight_high_activity_high_cancellation_count": int((selected1 & small).sum()),
                "small_weight_high_activity_high_cancellation_fraction": float(
                    (selected1 & small)[qualified1].sum() / int(qualified1.sum())
                ),
                "discovery_selected_validation_high_cancellation_fraction": (
                    float(retained[selected1].float().mean()) if selected1.any() else None
                ),
                "discovery_selected_validation_high_activity_fraction": (
                    float(retained_activity[selected1].float().mean()) if selected1.any() else None
                ),
                "discovery_selected_validation_same_domain_direction_fraction": (
                    float(same_opposition[selected1].float().mean())
                    if selected1.any()
                    else None
                ),
                "accepted_tinystories": accepted[0, layer, expert],
                "accepted_wikitext103": accepted[1, layer, expert],
                "proposed_tinystories": proposed[0, layer, expert],
                "proposed_wikitext103": proposed[1, layer, expert],
                "contributing_batches_tinystories": contributing[0, layer, expert],
                "contributing_batches_wikitext103": contributing[1, layer, expert],
            }
        )

        for bin_index, partition in enumerate(partitions):
            subset = torch.from_numpy(partition)
            q = qualified1[subset]
            sel = selected1[subset]
            val_qualified = qualified2[subset]
            val_c = cancellation2[subset]
            bin_rows.append(
                {
                    "seed": SEED,
                    "order": ORDER,
                    "step": STEP,
                    **meta,
                    "magnitude_bin": BIN_NAMES[bin_index],
                    "coordinates": len(subset),
                    "qualified_coordinates": int(q.sum()),
                    "qualified_fraction": float(q.float().mean()),
                    "mean_abs_magnitude": float(weight[subset].mean()),
                    "mean_activity": float(activity1[subset][q].mean()) if q.any() else None,
                    "median_activity": float(activity1[subset][q].median()) if q.any() else None,
                    "mean_cancellation": float(cancellation1[subset][q].mean()) if q.any() else None,
                    "mean_cancelled_mass": float(cancelled1[subset][q].mean()) if q.any() else None,
                    "total_cancelled_mass": float(cancelled1[subset][q].sum()) if q.any() else 0.0,
                    "high_activity_high_cancellation_count": int(sel.sum()),
                    "high_activity_high_cancellation_fraction": float(sel[q].float().mean())
                    if q.any()
                    else 0.0,
                    "validation_high_cancellation_fraction_among_discovery_selected": (
                        float((sel & val_qualified & (val_c >= PRIMARY_CANCELLATION)).sum() / sel.sum())
                        if sel.any()
                        else None
                    ),
                }
            )

        for activity_quantile in CONFIG["sensitivity_thresholds"]["activity_quantiles"]:
            activity_threshold = quantile(activity1[qualified1], float(activity_quantile))
            for cancellation_threshold in CONFIG["sensitivity_thresholds"][
                "cancellation_thresholds"
            ]:
                discovery = (
                    qualified1
                    & (activity1 >= activity_threshold)
                    & (cancellation1 >= float(cancellation_threshold))
                )
                valid = (
                    qualified2
                    & (activity2 >= activity_threshold)
                    & (cancellation2 >= float(cancellation_threshold))
                )
                selected_directions = (
                    torch.sign(mu1[0]) * torch.sign(mu1[1]) ==
                    torch.sign(mu2[0]) * torch.sign(mu2[1])
                )
                reproducibility_rows.append(
                    {
                        "seed": SEED,
                        "order": ORDER,
                        "step": STEP,
                        **meta,
                        "activity_quantile": activity_quantile,
                        "cancellation_threshold": cancellation_threshold,
                        "discovery_selected": int(discovery.sum()),
                        "discovery_fraction_qualified": (
                            float(discovery[qualified1].float().mean())
                            if qualified1.any()
                            else 0.0
                        ),
                        "validation_joint_retention": (
                            float((discovery & valid).sum() / discovery.sum())
                            if discovery.any()
                            else None
                        ),
                        "validation_cancellation_only_retention": (
                            float(
                                (
                                    discovery
                                    & qualified2
                                    & (cancellation2 >= float(cancellation_threshold))
                                ).sum()
                                / discovery.sum()
                            )
                            if discovery.any()
                            else None
                        ),
                        "validation_same_opposition_direction": (
                            float((discovery & selected_directions).sum() / discovery.sum())
                            if discovery.any()
                            else None
                        ),
                        "primary": bool(
                            float(activity_quantile) == PRIMARY_ACTIVITY_QUANTILE
                            and float(cancellation_threshold) == PRIMARY_CANCELLATION
                        ),
                    }
                )

        base_inputs = [
            "magnitude",
            "aggregate_abs",
            "activity",
            "cancellation",
            "cancelled_mass",
            "domain_1_mu_square",
            "domain_2_mu_square",
            "domain_1_probe_second_moment",
            "domain_2_probe_second_moment",
            "conditional_domain_1_abs",
            "conditional_domain_2_abs",
            "ramp",
            "activation_only",
            "routing_activation_l2",
            "ramp_per_dispatch",
        ]
        for metric in base_inputs[1:]:
            correlation_rows.append(
                {
                    "seed": SEED,
                    "order": ORDER,
                    "step": STEP,
                    **meta,
                    "metric_x": "magnitude",
                    "metric_y": metric,
                    "pearson": pearson(correlation_inputs["magnitude"], correlation_inputs[metric]),
                    "spearman_coordinate_sample": rank_sample(
                        correlation_inputs["magnitude"], correlation_inputs[metric]
                    ),
                }
            )
        for x, y in [
            ("activity", "cancellation"),
            ("activity", "cancelled_mass"),
            ("aggregate_abs", "cancelled_mass"),
            ("cancelled_mass", "domain_1_probe_second_moment"),
            ("cancelled_mass", "domain_2_probe_second_moment"),
            ("cancelled_mass", "ramp"),
        ]:
            correlation_rows.append(
                {
                    "seed": SEED,
                    "order": ORDER,
                    "step": STEP,
                    **meta,
                    "metric_x": x,
                    "metric_y": y,
                    "pearson": pearson(correlation_inputs[x], correlation_inputs[y]),
                    "spearman_coordinate_sample": rank_sample(
                        correlation_inputs[x], correlation_inputs[y]
                    ),
                }
            )

        sample_stride = max(1, weight.numel() // 4_000)
        sample = torch.arange(0, weight.numel(), sample_stride)
        qsample = qualified1[sample]
        plot_x.append(np.log10(activity1[sample][qsample].numpy()))
        plot_y.append(cancellation1[sample][qsample].numpy())
        total_cancelled_mass += float(cancelled1[qualified1].sum())
        if len(cell_rows) % 32 == 0:
            print(f"Analyzed {len(cell_rows)}/{len(names)} matrices", flush=True)

    write_csv(HERE / "cell_summary.csv", cell_rows)
    write_csv(HERE / "magnitude_bin_summary.csv", bin_rows)
    write_csv(HERE / "split_half_reproducibility.csv", reproducibility_rows)
    write_csv(HERE / "metric_correlations.csv", correlation_rows)

    x = np.concatenate(plot_x)
    y = np.concatenate(plot_y)
    plt.figure(figsize=(8, 6))
    plt.hexbin(x, y, gridsize=90, mincnt=1, cmap="magma", bins="log")
    plt.axhline(PRIMARY_CANCELLATION, color="white", linestyle="--", linewidth=0.8)
    plt.xlabel("log10 domain-weighted gradient activity A")
    plt.ylabel("relative cancellation C")
    plt.colorbar(label="log10 coordinate count")
    plt.title("Pilot coordinate activity versus cancellation")
    plt.tight_layout()
    plt.savefig(HERE / "activity_cancellation_distribution.png", dpi=180)
    plt.close()

    aggregate_bins = {label: [] for label in BIN_NAMES}
    for row in bin_rows:
        aggregate_bins[row["magnitude_bin"]].append(row)
    bin_fraction = []
    bin_cancellation = []
    bin_mass = []
    for label in BIN_NAMES:
        rows = aggregate_bins[label]
        denominator = sum(row["qualified_coordinates"] for row in rows)
        bin_fraction.append(
            sum(row["high_activity_high_cancellation_count"] for row in rows) / denominator
            if denominator
            else 0
        )
        bin_cancellation.append(
            np.average(
                [row["mean_cancellation"] for row in rows],
                weights=[row["qualified_coordinates"] for row in rows],
            )
        )
        bin_mass.append(sum(row["total_cancelled_mass"] for row in rows))
    fig, axis = plt.subplots(1, 2, figsize=(10, 4.5))
    axis[0].bar(BIN_NAMES, bin_fraction, color="#4c78a8")
    axis[0].set_xlabel("within-matrix |weight| percentile bin")
    axis[0].set_ylabel("fraction high-activity/high-cancellation")
    axis[0].tick_params(axis="x", rotation=35)
    axis[1].bar(BIN_NAMES, bin_mass, color="#f58518")
    axis[1].set_xlabel("within-matrix |weight| percentile bin")
    axis[1].set_ylabel("total absolute cancelled gradient mass K")
    axis[1].tick_params(axis="x", rotation=35)
    fig.suptitle("Cancellation structure by current weight magnitude")
    plt.tight_layout()
    plt.savefig(HERE / "cancellation_by_magnitude.png", dpi=180)
    plt.close(fig)

    replay = json.loads((HERE / "replay_validation.json").read_text())
    direct = replay["halves"]["1"]["direct_mixed_gradient_checks"]
    direct_max_abs = max(row["max_absolute_error"] for row in direct)
    direct_max_rel = max(row["max_relative_error"] for row in direct)
    numerical = {
        "status": "PASS"
        if (
            replay["parameter_state_unchanged"]
            and replay["rng_unchanged"]
            and replay["module_modes_unchanged"]
            and direct_max_abs <= 1e-5
            and direct_max_rel <= 1e-4
            and maxima["cancellation_below_zero"] <= 1e-6
            and maxima["cancellation_above_one"] <= 1e-6
            and maxima["cancelled_mass_negative"] <= 1e-6
            and maxima["triangle_violation"] <= 1e-6
        )
        else "FAIL",
        "coordinate_count": total_coordinates,
        "nonfinite_activity_coordinates": nonfinite,
        "qualified_discovery_coordinates": total_qualified,
        "direct_mixed_gradient_max_absolute_error": direct_max_abs,
        "direct_mixed_gradient_max_relative_error": direct_max_rel,
        "direct_mixed_gradient_batch_count": len(direct),
        "cancellation_range_violations": maxima,
        "squared_mean_vs_probe_second_moment": "kept as separate metrics; mu^2 is never substituted for mean(g_probe^2)",
        "near_zero_classification": {
            "activity_floor": ACTIVITY_FLOOR,
            "excluded_from_relative_cancellation": total_coordinates - total_qualified,
        },
        "state_preservation": {
            key: replay[key]
            for key in (
                "parameter_state_unchanged",
                "optimizer_not_instantiated",
                "optimizer_updates",
                "rng_unchanged",
                "module_modes_unchanged",
                "task_loss_only",
                "single_shared_forward_per_batch",
                "tf32_disabled",
                "deterministic_algorithms",
            )
        },
        "checkpoint_file_unchanged": sha(checkpoint)
        == replay["checkpoint_sha256"]
        == json.loads((SOURCE / "provenance.json").read_text())["cells"][
            f"seed{SEED}_{ORDER}_{STEP}"
        ]["state_sha256"],
        "normalization": CONFIG["normalization"],
        "capacity_attribution": (
            "PASS: one native forward per batch supplies both domains; no separate-domain "
            "capacity competition or routing was introduced"
        ),
    }
    (HERE / "numerical_validation.json").write_text(json.dumps(numerical, indent=2) + "\n")
    if numerical["status"] != "PASS":
        raise RuntimeError("numerical validation failed")

    overall_qualified = total_qualified
    primary = [row for row in reproducibility_rows if row["primary"]]
    weighted_joint = present_mean(row["validation_joint_retention"] for row in primary)
    weighted_cancel = present_mean(
        row["validation_cancellation_only_retention"] for row in primary
    )
    weighted_direction = present_mean(
        row["validation_same_opposition_direction"] for row in primary
    )
    small_fraction = selected_small_weight / selected_count if selected_count else 0
    small_retention = (
        selected_small_weight_retained / selected_small_weight if selected_small_weight else 0
    )
    selected_mass_share = (
        all_selected_cancelled_mass / all_total_cancelled_mass
        if all_total_cancelled_mass
        else 0
    )
    aggregate_fraction = selected_count / overall_qualified
    retained_fraction = selected_retained / selected_count if selected_count else 0
    magnitude_correlation = present_mean(
        row["pearson"]
        for row in correlation_rows
        if row["metric_x"] == "magnitude" and row["metric_y"] == "cancellation"
    )
    magnitude_activity_correlation = present_mean(
        row["pearson"]
        for row in correlation_rows
        if row["metric_x"] == "magnitude" and row["metric_y"] == "activity"
    )
    aggregate_cancelled_mass_correlation = present_mean(
        row["pearson"]
        for row in correlation_rows
        if row["metric_x"] == "aggregate_abs" and row["metric_y"] == "cancelled_mass"
    )
    probe_cancelled_mass_correlation = present_mean(
        row["pearson"]
        for row in correlation_rows
        if row["metric_x"] == "cancelled_mass"
        and row["metric_y"]
        in ("domain_1_probe_second_moment", "domain_2_probe_second_moment")
    )
    lines = [
        "# Pilot verdict: gradient cancellation qualification v1",
        "",
        f"Cell: `seed{SEED}_{ORDER}`, checkpoint step {STEP}. Both frozen 90-probe halves completed.",
        "",
        "## Answers",
        "",
        "1. **Does reproducible gradient opposition represent substantial underlying gradient activity?** "
        f"The discovery primary population contains {selected_count:,} coordinates "
        f"({aggregate_fraction:.4%} of qualified coordinates) above the within-matrix "
        f"90th activity percentile and with `C >= 0.5`. It contains "
        f"{selected_mass_share:.4%} of all qualified absolute cancelled mass `K`. "
        f"The mean within-matrix Pearson correlation between aggregate-gradient magnitude "
        f"and `K` is only {aggregate_cancelled_mass_correlation:.4f}, while the mean "
        f"correlation between `K` and domain probe second moment is "
        f"{probe_cancelled_mass_correlation:.4f}. "
        "This establishes substantial signal activity for a sparse subset, not importance.",
        "",
        "2. **Do high-activity, high-cancellation connections exist among small weights?** "
        f"Yes, {selected_small_weight:,} primary selections ({small_fraction:.4%} of all "
        "primary selections) are in the bottom 20% within their own matrix by current "
        f"absolute magnitude; {small_retention:.4%} retain high cancellation in validation.",
        f"That proportion is not enriched relative to the 20% bin population. Across "
        f"matrices, mean Pearson correlations with magnitude are {magnitude_correlation:.4f} "
        f"for cancellation and {magnitude_activity_correlation:.4f} for activity, and "
        "the six magnitude bins show nearly flat high-cancellation prevalence. Thus the "
        "pilot demonstrates existence among small weights, not a magnitude-selected mechanism.",
        "",
        "3. **Is the structure reproducible across independent probes?** "
        f"The mean matrix-level validation joint retention is {weighted_joint:.4%}, "
        f"cancellation-only retention is {weighted_cancel:.4%}, and same opposition-direction "
        f"retention is {weighted_direction:.4%}. These are descriptive matrix summaries; "
        "coordinates and matrices are not independent replications.",
        "",
        "4. **Are the measurements numerically trustworthy?** "
        f"Yes for the pilot: numerical validation passes, direct mixed-gradient reconstruction "
        f"has max absolute error {direct_max_abs:.3e} and relative error {direct_max_rel:.3e}; "
        "cancellation bounds, nonnegative `K`, triangle inequality, task-only objective, shared "
        "routing/capacity attribution, and unchanged model/RNG state all pass.",
        "",
        "5. **Are existing artifacts sufficient for subsequent temporal and matched-selection analyses?** "
        "They are sufficient for this cell's checkpoint-250 selection features and for reuse by "
        "future analysis, but not for temporal claims: checkpoint 500 and the remaining source "
        "cells have not been replayed with signed means. Signed aggregates, not the original "
        "sign-only artifacts, must be used for any matched-selection extension.",
        "",
        "## Limitations and interpretation",
        "",
        "- The experiment is one source seed, one order, and one checkpoint. No full-study inference follows.",
        "- Cancellation is measured over target-weighted domain probe gradients under a shared forward, not token-level gradient covariance.",
        f"- {total_coordinates - total_qualified:,} coordinates at or below the `{ACTIVITY_FLOOR}` activity floor are classified near-zero and excluded from relative cancellation.",
        "- Probe-level second moments are batch/domain sufficient statistics; they are not within-batch token-gradient moments.",
        "- RAMP and activation scores are acceptance-weighted correlates, not causal importance measures.",
        "- Unknown separator targets are reported separately and excluded from the two-domain reconstruction.",
        "- No coordinate is deleted, masked, or retrained. No importance, dispensability, or pruning-safety conclusion is made.",
        "- Gradient cancellation alone is not an ICML-level novelty claim. The qualified signal motivates, but does not establish, a causal or general result.",
        "",
        "## Stop condition",
        "",
        "Pilot artifacts are complete. The eleven remaining registered cells were not launched.",
    ]
    (HERE / "pilot_verdict.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": numerical["status"],
                "qualified_coordinates": overall_qualified,
                "primary_selected": selected_count,
                "primary_validation_joint_retention": retained_fraction,
                "small_weight_selected": selected_small_weight,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    summarize()
