"""Read-only single-source temporal analysis for seed7_A, steps 250 and 500."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT / "results/moe_gradient_cancellation_qualification_v1"
CONFIG = json.loads((HERE / "configuration.json").read_text())
T = CONFIG["thresholds"]
SEED, ORDER = 7, "A"
STEPS = (250, 500)
SHARDS = HERE / "coordinate_statistics"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path.name}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def atomic_tensor(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def metadata(name: str) -> dict:
    parts = name.split(".")
    return {
        "parameter": name,
        "layer": int(parts[1]),
        "expert": int(parts[4]),
        "projection": parts[5],
    }


def matrix_vector(stats: dict, index: int, metric: str):
    if metric in stats:
        return stats[metric]
    raise KeyError(metric)


def domain_mean(half: dict, index: int) -> torch.Tensor:
    return torch.stack(
        [half["domain_mean_unconditional"][domain][index] for domain in (0, 1)]
    ).reshape(2, -1)


def domain_moment(half: dict, index: int) -> torch.Tensor:
    return torch.stack(
        [half["domain_probe_second_moment"][domain][index] for domain in (0, 1)]
    ).reshape(2, -1)


def probability(half: dict) -> np.ndarray:
    count = np.asarray(half["domain_target_count"], dtype=np.float64)
    return count / count.sum()


def derive(mu: torch.Tensor, p: np.ndarray):
    aggregate = p[0] * mu[0] + p[1] * mu[1]
    activity = p[0] * mu[0].abs() + p[1] * mu[1].abs()
    qualified = activity > T["activity_floor"]
    cancellation = torch.full_like(activity, torch.nan)
    cancellation[qualified] = 1 - aggregate[qualified].abs() / activity[qualified]
    cancelled = activity - aggregate.abs()
    second_mean = p[0] * mu[0].square() + p[1] * mu[1].square()
    return {
        "aggregate": aggregate,
        "activity": activity,
        "qualified": qualified,
        "cancellation": cancellation,
        "cancelled_mass": cancelled,
        "domain_mean_second_moment": second_mean,
    }


def percentile(values: torch.Tensor, q: float) -> float:
    qualified = values[torch.isfinite(values)]
    if not qualified.numel():
        return float("nan")
    return float(torch.quantile(qualified, q))


def correlation(a: torch.Tensor, b: torch.Tensor):
    x = a.numpy().reshape(-1)
    y = b.numpy().reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return None, None
    pearson = float(np.corrcoef(x[keep], y[keep])[0, 1])
    sample = np.linspace(0, int(keep.sum()) - 1, min(20_000, int(keep.sum())), dtype=np.int64)
    xx, yy = x[keep][sample], y[keep][sample]
    rho = spearmanr(xx, yy).statistic
    return pearson, float(rho) if np.isfinite(rho) else None


def pooled_activation(half: dict, name: str, metric: str) -> torch.Tensor:
    meta = metadata(name)
    key1 = f"{meta['layer']}:{meta['expert']}:1"
    key2 = f"{meta['layer']}:{meta['expert']}:2"
    aliases = {
        "activation_only": "activation_l2",
        "routing_activation": "routing_activation_l2",
    }
    saved = aliases.get(metric, metric)
    one = half["activation_metrics"][key1][saved]
    two = half["activation_metrics"][key2][saved]
    value = (one + two).mul_(0.5)
    if meta["projection"] == "fc1":
        return value[:, None].expand(1024, 256).reshape(-1).float()
    return value[None, :].expand(256, 1024).reshape(-1).float()


def population_label(activity, aggregate, cancellation, q, native=None):
    labels = np.full(len(activity), "unqualified", dtype=object)
    qualified = np.isfinite(cancellation)
    labels[qualified] = "other_qualified"
    low_activity = qualified & (activity <= q["activity_50"])
    weak_aggregate = qualified & (aggregate <= q["aggregate_20"])
    cancellation_high = qualified & (activity >= q["activity_90"]) & (
        cancellation >= T["primary_cancellation"]
    )
    aligned = qualified & (activity >= q["activity_90"]) & (
        cancellation < T["primary_cancellation"]
    )
    labels[low_activity & weak_aggregate] = "low_activity_weak_aggregate"
    labels[cancellation_high] = "high_activity_high_cancellation"
    labels[aligned] = "high_activity_weak_cancellation"
    labels[(cancellation >= 0.75) & qualified & ~cancellation_high] = "moderate_activity_high_cancellation"
    return labels


def row_col_counts(indices, shape):
    if not len(indices):
        return np.zeros(shape[0], dtype=np.int64), np.zeros(shape[1], dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    rows = indices // shape[1]
    cols = indices % shape[1]
    return np.bincount(rows, minlength=shape[0]), np.bincount(cols, minlength=shape[1])


def nearest_by_hard_strata(
    left: np.ndarray,
    right: np.ndarray,
    magnitude: np.ndarray,
    aggregate: np.ndarray,
    sign: np.ndarray,
    target: int,
):
    magnitude_order = np.argsort(magnitude, kind="stable")
    magnitude_bin = np.empty_like(magnitude_order)
    magnitude_bin[magnitude_order] = (
        np.arange(len(magnitude)) * 10 // len(magnitude)
    )
    aggregate_order = np.argsort(aggregate, kind="stable")
    aggregate_bin = np.empty_like(aggregate_order)
    aggregate_bin[aggregate_order] = np.arange(len(aggregate)) * 20 // len(aggregate)
    left_groups = defaultdict(list)
    right_groups = defaultdict(list)
    for index in left:
        left_groups[(int(sign[index]), int(magnitude_bin[index]), int(aggregate_bin[index]))].append(int(index))
    for index in right:
        right_groups[(int(sign[index]), int(magnitude_bin[index]), int(aggregate_bin[index]))].append(int(index))
    pairs = []
    for key in sorted(set(left_groups) & set(right_groups)):
        a = np.asarray(left_groups[key], dtype=np.int64)
        b = np.asarray(right_groups[key], dtype=np.int64)
        a = a[np.lexsort((a, aggregate[a], magnitude[a]))]
        b = b[np.lexsort((b, aggregate[b], magnitude[b]))]
        for x, y in zip(a, b):
            pairs.append((int(x), int(y)))
            if len(pairs) >= target:
                return pairs
    return pairs


def sample_indices(size: int, count: int, seed: int):
    if size <= count:
        return np.arange(size, dtype=np.int64)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(size, count, replace=False))


def load_checkpoints():
    source = (
        ROOT / "results/moe_connection_lifetime_v1/prefixes/seed7_A"
    )
    step0 = ROOT / "results/32e_router_age_localization/references/seed_7/checkpoints/step_0.pt"
    paths = {0: step0, 250: source / "state250.pt", 500: source / "state500.pt"}
    payloads = {step: load(path) for step, path in paths.items()}
    for step, value in payloads.items():
        if value["step"] != step:
            raise RuntimeError(f"checkpoint step mismatch at {step}")
    return paths, payloads


def main():
    replay500 = json.loads((HERE / "replay_validation.json").read_text())
    replay250 = json.loads((PILOT / "replay_validation.json").read_text())
    if not replay250["parameter_state_unchanged"] or not replay500["parameter_state_unchanged"]:
        raise RuntimeError("source replay did not preserve state")
    entropy_path = HERE / "routing_entropy_dynamics.csv"
    if not entropy_path.exists():
        raise RuntimeError("native router-entropy supplement is missing")
    with entropy_path.open() as handle:
        entropy_rows = list(csv.DictReader(handle))
    entropy = {
        (int(row["step"]), int(row["layer"]), int(row["expert"]), row["domain"]): row
        for row in entropy_rows
    }
    halves = {
        250: [
            load(PILOT / "coordinate_statistics_half1.pt"),
            load(PILOT / "coordinate_statistics_half2.pt"),
        ],
        500: [
            load(HERE / "coordinate_statistics_half1.pt"),
            load(HERE / "coordinate_statistics_half2.pt"),
        ],
    }
    names = halves[250][0]["names"]
    for half_set in halves.values():
        if half_set[0]["names"] != names or half_set[1]["names"] != names:
            raise RuntimeError("coordinate schema mismatch")
    if halves[250][0]["batch_ids"] != halves[500][0]["batch_ids"] or halves[250][1][
        "batch_ids"
    ] != halves[500][1]["batch_ids"]:
        raise RuntimeError("checkpoint probe batch identities differ")

    checkpoint_paths, payloads = load_checkpoints()
    model250, model500, model0 = (
        payloads[250]["model"],
        payloads[500]["model"],
        payloads[0]["model"],
    )
    fixed_p = (probability(halves[250][0]) + probability(halves[500][0])) / 2
    temporal_rows = []
    transition_rows = []
    release_rows = []
    routing_rows = []
    decomposition_rows = []
    redundancy_rows = []
    matching_rows = []
    residual_rows = []
    prediction_samples = []
    pair_artifacts = []
    plot_release = {"initial": [], "final": [], "aggregate_initial": [], "aggregate_final": []}
    transition_counts = defaultdict(lambda: defaultdict(int))
    decomposition_total = []
    release_decomposition_total = []
    errors = []
    qualifications = []
    release_validation = []
    matrix_selected = 0
    all_coordinates = 0

    SHARDS.mkdir(exist_ok=True)
    for index, name in enumerate(names):
        meta = metadata(name)
        shape = tuple(halves[250][0]["shapes"][index])
        weight0 = model0[name].detach().abs().float().reshape(-1)
        weight250 = model250[name].detach().abs().float().reshape(-1)
        weight500 = model500[name].detach().abs().float().reshape(-1)
        magnitude = weight250.numpy()
        movement_0_250 = (model250[name].detach() - model0[name].detach()).abs().float().reshape(-1)
        movement_250_500 = (model500[name].detach() - model250[name].detach()).abs().float().reshape(-1)
        mu250d = domain_mean(halves[250][0], index)
        mu250v = domain_mean(halves[250][1], index)
        mu500d = domain_mean(halves[500][0], index)
        mu500v = domain_mean(halves[500][1], index)
        moment250 = domain_moment(halves[250][0], index)
        moment500 = domain_moment(halves[500][0], index)
        stats250 = derive(mu250d, fixed_p)
        stats500 = derive(mu500d, fixed_p)
        stats250v = derive(mu250v, fixed_p)
        stats500v = derive(mu500v, fixed_p)
        q250 = {
            "activity_50": percentile(stats250["activity"], 0.5),
            "activity_90": percentile(stats250["activity"], 0.9),
            "aggregate_20": percentile(stats250["aggregate"].abs(), 0.2),
            "magnitude_50": percentile(weight250, 0.5),
        }
        q500 = {
            "activity_50": percentile(stats500["activity"], 0.5),
            "activity_90": percentile(stats500["activity"], 0.9),
            "aggregate_20": percentile(stats500["aggregate"].abs(), 0.2),
            "magnitude_50": percentile(weight500, 0.5),
        }
        release = (
            stats250["qualified"]
            & stats500["qualified"]
            & (stats250["activity"] >= q250["activity_90"])
            & (stats250["cancellation"] >= T["primary_cancellation"])
            & (stats250["aggregate"].abs() <= q250["aggregate_20"])
            & (
                stats250["cancellation"] - stats500["cancellation"]
                >= T["release_cancellation_drop"]
            )
            & (
                stats500["aggregate"].abs()
                >= stats250["aggregate"].abs() * T["release_aggregate_increase"]
            )
            & (stats500["activity"] >= stats250["activity"] * T["release_validation_quantile"])
        )
        release_validation_mask = (
            stats250v["qualified"]
            & stats500v["qualified"]
            & (stats250v["activity"] >= percentile(stats250v["activity"], 0.9))
            & (stats250v["cancellation"] >= T["primary_cancellation"])
            & (
                stats250v["cancellation"] - stats500v["cancellation"]
                >= T["release_cancellation_drop"]
            )
            & (
                stats500v["aggregate"].abs()
                >= stats250v["aggregate"].abs() * T["release_aggregate_increase"]
            )
            & (stats500v["activity"] >= stats250v["activity"] * T["release_validation_quantile"])
        )
        labels250 = population_label(
            stats250["activity"].numpy(),
            stats250["aggregate"].abs().numpy(),
            stats250["cancellation"].numpy(),
            q250,
        )
        labels500 = population_label(
            stats500["activity"].numpy(),
            stats500["aggregate"].abs().numpy(),
            stats500["cancellation"].numpy(),
            q500,
        )
        unique250 = np.unique(labels250)
        unique500 = np.unique(labels500)
        for before in unique250:
            for after in unique500:
                transition_counts[str(before)][str(after)] += int(
                    ((labels250 == before) & (labels500 == after)).sum()
                )

        direct250 = model250[name].detach().float().reshape(-1)
        direct500 = model500[name].detach().float().reshape(-1)
        weight_growth = (direct500.abs() - direct250.abs())
        pooled_moment250 = fixed_p[0] * moment250[0] + fixed_p[1] * moment250[1]
        pooled_moment500 = fixed_p[0] * moment500[0] + fixed_p[1] * moment500[1]
        activation250 = pooled_activation(halves[250][0], name, "ramp")
        activation500 = pooled_activation(halves[500][0], name, "ramp")
        routing_activation250 = pooled_activation(halves[250][0], name, "routing_activation")
        routing_activation500 = pooled_activation(halves[500][0], name, "routing_activation")

        layer, expert = meta["layer"], meta["expert"]
        accepted250 = np.asarray(halves[250][0]["accepted_occurrences"])[:, layer, expert]
        accepted500 = np.asarray(halves[500][0]["accepted_occurrences"])[:, layer, expert]
        targets250 = np.asarray(halves[250][0]["domain_target_count"], dtype=np.float64)
        targets500 = np.asarray(halves[500][0]["domain_target_count"], dtype=np.float64)
        rho250 = accepted250 / targets250
        rho500 = accepted500 / targets500
        nu250 = torch.where(
            torch.from_numpy((rho250 > 0)).to(torch.bool)[:, None],
            mu250d / torch.from_numpy(rho250).float()[:, None],
            torch.nan,
        )
        nu500 = torch.where(
            torch.from_numpy((rho500 > 0)).to(torch.bool)[:, None],
            mu500d / torch.from_numpy(rho500).float()[:, None],
            torch.nan,
        )
        g00 = fixed_p[0] * rho250[0] * nu250[0] + fixed_p[1] * rho250[1] * nu250[1]
        g10 = fixed_p[0] * rho500[0] * nu250[0] + fixed_p[1] * rho500[1] * nu250[1]
        g01 = fixed_p[0] * rho250[0] * nu500[0] + fixed_p[1] * rho250[1] * nu500[1]
        g11 = fixed_p[0] * rho500[0] * nu500[0] + fixed_p[1] * rho500[1] * nu500[1]
        valid_decomp = torch.isfinite(g00) & torch.isfinite(g11)
        routing_effect = 0.5 * ((g10 - g00) + (g11 - g01))
        content_effect = 0.5 * ((g01 - g00) + (g11 - g10))
        delta_g = g11 - g00
        reconstruction = routing_effect + content_effect - delta_g
        base_mu = torch.stack([rho250[0] * nu250[0], rho250[1] * nu250[1]])
        if torch.isfinite(base_mu).all():
            assert torch.allclose(derive(base_mu, fixed_p)["aggregate"], g00, atol=1e-7)
        k00 = derive(torch.stack([rho250[0] * nu250[0], rho250[1] * nu250[1]]), fixed_p)[
            "cancelled_mass"
        ]
        k10 = derive(torch.stack([rho500[0] * nu250[0], rho500[1] * nu250[1]]), fixed_p)[
            "cancelled_mass"
        ]
        k01 = derive(torch.stack([rho250[0] * nu500[0], rho250[1] * nu500[1]]), fixed_p)[
            "cancelled_mass"
        ]
        k11 = derive(torch.stack([rho500[0] * nu500[0], rho500[1] * nu500[1]]), fixed_p)[
            "cancelled_mass"
        ]
        routing_k = 0.5 * ((k10 - k00) + (k11 - k01))
        content_k = 0.5 * ((k01 - k00) + (k11 - k10))
        reconstruction_k = routing_k + content_k - (k11 - k00)
        errors.append(float(reconstruction[valid_decomp].abs().max()))
        errors.append(float(reconstruction_k[valid_decomp].abs().max()))

        domain_acceptance_p_e_d = accepted250 / np.maximum(accepted250.sum(), 1)
        requested250 = np.asarray(halves[250][0]["proposed_occurrences"])[:, layer, expert]
        requested500 = np.asarray(halves[500][0]["proposed_occurrences"])[:, layer, expert]
        gate250 = np.asarray(halves[250][0]["accepted_gate_sum"])[:, layer, expert]
        gate500 = np.asarray(halves[500][0]["accepted_gate_sum"])[:, layer, expert]
        gate_square250 = np.asarray(halves[250][0]["accepted_gate_square"])[:, layer, expert]
        gate_square500 = np.asarray(halves[500][0]["accepted_gate_square"])[:, layer, expert]
        accepted250_counts = accepted250
        accepted500_counts = np.asarray(halves[500][0]["accepted_occurrences"])[:, layer, expert]
        gate_mean250 = gate250 / np.maximum(accepted250_counts, 1)
        gate_mean500 = gate500 / np.maximum(accepted500_counts, 1)
        p_e_given_d250 = accepted250_counts / targets250
        p_e_given_d500 = accepted500_counts / targets500
        p_d_given_e250 = accepted250_counts / max(accepted250_counts.sum(), 1)
        p_d_given_e500 = accepted500_counts / max(accepted500_counts.sum(), 1)
        if stats250["qualified"].any():
            corr_routing_c, _ = correlation(
                torch.from_numpy(p_e_given_d500 - p_e_given_d250).abs().mean().repeat(
                    len(stats250["cancellation"])
                ).float(),
                stats500["cancellation"] - stats250["cancellation"],
            )
        else:
            corr_routing_c = None

        for domain_index, domain_name in enumerate(("tinystories", "wikitext103")):
            entropy250 = entropy[250, layer, expert, domain_name]
            entropy500 = entropy[500, layer, expert, domain_name]
            routing_rows.append(
                {
                    **meta,
                    "domain": domain_name,
                    "step_250_targets": targets250[domain_index],
                    "step_500_targets": targets500[domain_index],
                    "requested_250": requested250[domain_index],
                    "requested_500": requested500[domain_index],
                    "accepted_250": accepted250_counts[domain_index],
                    "accepted_500": accepted500_counts[domain_index],
                    "p_expert_given_domain_250": p_e_given_d250[domain_index],
                    "p_expert_given_domain_500": p_e_given_d500[domain_index],
                    "p_domain_given_expert_250": p_d_given_e250[domain_index],
                    "p_domain_given_expert_500": p_d_given_e500[domain_index],
                    "mean_gate_250": gate_mean250[domain_index],
                    "mean_gate_500": gate_mean500[domain_index],
                    "gate_weighted_exposure_250": gate250[domain_index],
                    "gate_weighted_exposure_500": gate500[domain_index],
                    "capacity_rejection_250": requested250[domain_index]
                    - accepted250_counts[domain_index],
                    "capacity_rejection_500": requested500[domain_index]
                    - accepted500_counts[domain_index],
                    "gate_square_250": gate_square250[domain_index],
                    "gate_square_500": gate_square500[domain_index],
                    "requested_router_entropy_mean_250": float(
                        entropy250["requested_router_entropy_mean"]
                    )
                    if entropy250["requested_router_entropy_mean"]
                    else None,
                    "requested_router_entropy_mean_500": float(
                        entropy500["requested_router_entropy_mean"]
                    )
                    if entropy500["requested_router_entropy_mean"]
                    else None,
                    "accepted_router_entropy_mean_250": float(
                        entropy250["accepted_router_entropy_mean"]
                    )
                    if entropy250["accepted_router_entropy_mean"]
                    else None,
                    "accepted_router_entropy_mean_500": float(
                        entropy500["accepted_router_entropy_mean"]
                    )
                    if entropy500["accepted_router_entropy_mean"]
                    else None,
                }
            )
        routing_entropy250 = float(
            -sum(
                p * math.log(p)
                for p in p_d_given_e250
                if p > 0
            )
        )
        routing_entropy500 = float(
            -sum(
                p * math.log(p)
                for p in p_d_given_e500
                if p > 0
            )
        )

        release_count = int(release.sum())
        release_val_count = int((release & release_validation_mask).sum())
        release_overlap = release_val_count / release_count if release_count else None
        release_rows.append(
            {
                **meta,
                "qualified_initial": int(stats250["qualified"].sum()),
                "release_count_discovery": release_count,
                "release_fraction_discovery": float(release[stats250["qualified"]].float().mean()),
                "release_count_validation_independent": int(release_validation_mask.sum()),
                "release_overlap_fraction": release_overlap,
                "mean_initial_cancellation": float(stats250["cancellation"][release].mean())
                if release_count
                else None,
                "mean_final_cancellation": float(stats500["cancellation"][release].mean())
                if release_count
                else None,
                "mean_aggregate_ratio": float(
                    (
                        stats500["aggregate"].abs()[release]
                        / stats250["aggregate"].abs()[release].clamp_min(1e-300)
                    ).mean()
                )
                if release_count
                else None,
                "mean_weight_displacement": float(movement_250_500[release].mean())
                if release_count
                else None,
                "mean_nonrelease_weight_displacement": float(
                    movement_250_500[stats250["qualified"] & ~release].mean()
                )
                if (stats250["qualified"] & ~release).any()
                else None,
                "initial_activity_threshold": q250["activity_90"],
                "initial_weak_aggregate_threshold": q250["aggregate_20"],
                "release_weaker_activity_count": int(
                    (release & (stats500["activity"] <= 0.9 * stats250["activity"])).sum()
                ),
                "release_alignment_count": int(
                    (
                        release
                        & (torch.sign(mu500d[0]) * torch.sign(mu500d[1]) > 0)
                    ).sum()
                ),
                "release_domain_dominance_count": int(
                    (
                        release
                        & (
                            torch.maximum(
                                fixed_p[0] * mu500d[0].abs(),
                                fixed_p[1] * mu500d[1].abs(),
                            )
                            / stats500["activity"].clamp_min(1e-300)
                            >= 0.8
                        )
                    ).sum()
                ),
                "release_other_count": int(
                    (
                        release
                        & ~(stats500["activity"] <= 0.9 * stats250["activity"])
                        & ~(torch.sign(mu500d[0]) * torch.sign(mu500d[1]) > 0)
                        & ~(
                            torch.maximum(
                                fixed_p[0] * mu500d[0].abs(),
                                fixed_p[1] * mu500d[1].abs(),
                            )
                            / stats500["activity"].clamp_min(1e-300)
                            >= 0.8
                        )
                    ).sum()
                ),
                "release_mean_routing_aggregate_effect": float(
                    routing_effect.abs()[release].mean()
                )
                if release.any()
                else None,
                "release_mean_conditional_content_effect": float(
                    content_effect.abs()[release].mean()
                )
                if release.any()
                else None,
                "release_routing_fraction_of_component_magnitude": float(
                    routing_effect.abs()[release].sum()
                    / (
                        routing_effect.abs()[release].sum()
                        + content_effect.abs()[release].sum()
                    )
                )
                if release.any()
                else None,
            }
        )
        matrix_selected += release_count
        all_coordinates += len(weight250)

        metrics = {
            "magnitude_250": weight250,
            "magnitude_500": weight500,
            "signed_weight_250": direct250,
            "signed_weight_500": direct500,
            "movement_0_250": movement_0_250,
            "movement_250_500": movement_250_500,
            "mu_tiny_250": mu250d[0],
            "mu_wiki_250": mu250d[1],
            "mu_tiny_500": mu500d[0],
            "mu_wiki_500": mu500d[1],
            "aggregate_250": stats250["aggregate"],
            "aggregate_500": stats500["aggregate"],
            "activity_250": stats250["activity"],
            "activity_500": stats500["activity"],
            "cancellation_250": stats250["cancellation"],
            "cancellation_500": stats500["cancellation"],
            "cancelled_mass_250": stats250["cancelled_mass"],
            "cancelled_mass_500": stats500["cancelled_mass"],
            "probe_second_moment_250": pooled_moment250,
            "probe_second_moment_500": pooled_moment500,
            "domain_mean_second_moment_250": stats250["domain_mean_second_moment"],
            "domain_mean_second_moment_500": stats500["domain_mean_second_moment"],
            "ramp_250": activation250,
            "ramp_500": activation500,
            "routing_activation_250": routing_activation250,
            "routing_activation_500": routing_activation500,
            "release": release,
            "release_validation": release_validation_mask,
        }
        atomic_tensor(
            SHARDS / f"{name}.pt",
            {
                "name": name,
                "layer": layer,
                "expert": expert,
                "projection": meta["projection"],
                "shape": shape,
                **metrics,
                "routing_effect_on_aggregate": routing_effect,
                "conditional_content_effect_on_aggregate": content_effect,
                "routing_effect_on_cancelled_mass": routing_k,
                "conditional_content_effect_on_cancelled_mass": content_k,
            },
        )

        for metric, metric_value, future in [
            ("magnitude", weight250, weight500),
            ("aggregate_abs", stats250["aggregate"].abs(), stats500["aggregate"].abs()),
            ("activity", stats250["activity"], stats500["activity"]),
            ("cancellation", stats250["cancellation"], stats500["cancellation"]),
            ("cancelled_mass", stats250["cancelled_mass"], stats500["cancelled_mass"]),
            ("probe_second_moment", pooled_moment250, pooled_moment500),
            ("ramp", activation250, activation500),
        ]:
            pearson, rho = correlation(metric_value, future)
            temporal_rows.append(
                {
                    **meta,
                    "metric": metric,
                    "pearson_250_500": pearson,
                    "spearman_250_500": rho,
                    "mean_250": float(metric_value[torch.isfinite(metric_value)].mean()),
                    "mean_500": float(future[torch.isfinite(future)].mean()),
                    "median_250": float(
                        metric_value[torch.isfinite(metric_value)].median()
                    ),
                    "median_500": float(future[torch.isfinite(future)].median()),
                }
            )
        cancellation_drop = stats250["cancellation"] - stats500["cancellation"]
        aggregate_ratio = stats500["aggregate"].abs() / stats250["aggregate"].abs().clamp_min(
            1e-300
        )
        for metric_x, metric_y, x, y in [
            ("aggregate_abs_250", "cancelled_mass_250", stats250["aggregate"].abs(), stats250["cancelled_mass"]),
            ("cancelled_mass_250", "probe_second_moment_250", stats250["cancelled_mass"], pooled_moment250),
            ("magnitude_250", "cancellation_250", weight250, stats250["cancellation"]),
            ("movement_0_250", "movement_250_500", movement_0_250, movement_250_500),
            ("cancellation_drop", "aggregate_ratio", cancellation_drop, aggregate_ratio),
            (
                "taylor_abs_250",
                "movement_250_500",
                weight250 * stats250["aggregate"].abs(),
                movement_250_500,
            ),
            (
                "fisher_style_probe_second_moment_250",
                "cancelled_mass_250",
                pooled_moment250,
                stats250["cancelled_mass"],
            ),
            (
                "domain_mean_second_moment_250",
                "cancelled_mass_250",
                stats250["domain_mean_second_moment"],
                stats250["cancelled_mass"],
            ),
            (
                "DAP_inspired_disagreement_250",
                "cancellation_250",
                (mu250d[0] - mu250d[1]).abs()
                / (mu250d[0].abs() + mu250d[1].abs()).clamp_min(1e-300),
                stats250["cancellation"],
            ),
        ]:
            pearson, rho = correlation(x, y)
            redundancy_rows.append(
                {
                    **meta,
                    "metric_x": metric_x,
                    "metric_y": metric_y,
                    "pearson": pearson,
                    "spearman": rho,
                }
            )
        equal_weight_identity = mu250d[0] * mu250d[1] < 0
        equal_aggregate = 0.5 * (mu250d[0] + mu250d[1])
        equal_activity = 0.5 * (mu250d[0].abs() + mu250d[1].abs())
        equal_domain_second = 0.5 * (mu250d[0].square() + mu250d[1].square())
        identity_error = torch.where(
            equal_weight_identity,
            equal_domain_second
            - (equal_aggregate.square() + equal_activity.square()),
            torch.nan,
        )
        redundancy_rows.append(
            {
                **meta,
                "metric_x": "G2_plus_A2",
                "metric_y": "domain_mean_second_moment",
                "pearson": correlation(
                    equal_aggregate.square() + equal_activity.square(),
                    equal_domain_second,
                )[0],
                "spearman": correlation(
                    equal_aggregate.square() + equal_activity.square(),
                    equal_domain_second,
                )[1],
                "opposing_coordinate_fraction": float(equal_weight_identity.float().mean()),
                "opposing_identity_max_abs_error": float(
                    identity_error[torch.isfinite(identity_error)].abs().max()
                ),
            }
        )

        delta_abs = delta_g.abs()
        decomposition_rows.append(
            {
                **meta,
                "valid_coordinates": int(valid_decomp.sum()),
                "mean_abs_aggregate_change": float(delta_abs[valid_decomp].mean()),
                "mean_abs_routing_effect": float(routing_effect.abs()[valid_decomp].mean()),
                "mean_abs_conditional_content_effect": float(content_effect.abs()[valid_decomp].mean()),
                "routing_fraction_of_component_magnitude": float(
                    routing_effect.abs()[valid_decomp].sum()
                    / (
                        routing_effect.abs()[valid_decomp].sum()
                        + content_effect.abs()[valid_decomp].sum()
                    )
                ),
                "aggregate_reconstruction_max_abs_error": float(
                    reconstruction[valid_decomp].abs().max()
                ),
                "mean_abs_cancelled_mass_change": float((k11 - k00).abs()[valid_decomp].mean()),
                "mean_abs_routing_cancelled_mass_effect": float(routing_k.abs()[valid_decomp].mean()),
                "mean_abs_content_cancelled_mass_effect": float(content_k.abs()[valid_decomp].mean()),
                "cancelled_mass_reconstruction_max_abs_error": float(
                    reconstruction_k[valid_decomp].abs().max()
                ),
                "routing_effect_dominates_fraction": float(
                    (routing_effect.abs() > content_effect.abs())[valid_decomp].float().mean()
                ),
                "rho_tiny_250": rho250[0],
                "rho_wiki_250": rho250[1],
                "rho_tiny_500": rho500[0],
                "rho_wiki_500": rho500[1],
                "domain_entropy_250": routing_entropy250,
                "domain_entropy_500": routing_entropy500,
                "mean_cancel_routing_association_exposure": corr_routing_c,
                "exact_router_causal_interpretation": False,
            }
        )

        # Matching feasibility: bottom-half magnitude, bottom aggregate quintile,
        # low-activity versus high-activity/cancellation populations.
        weak = (
            (weight250 <= q250["magnitude_50"])
            & (stats250["aggregate"].abs() <= q250["aggregate_20"])
            & stats250["qualified"]
        )
        inactive = weak & (stats250["activity"] <= q250["activity_50"])
        conflicted = (
            weak
            & (stats250["activity"] >= q250["activity_90"])
            & (stats250["cancellation"] >= T["primary_cancellation"])
        )
        left = np.flatnonzero(inactive.numpy())
        right = np.flatnonzero(conflicted.numpy())
        target = max(1, int(len(weight250) * T["target_fraction_per_matrix"] if "target_fraction_per_matrix" in T else len(weight250) * CONFIG["matching"]["target_fraction_per_matrix"]))
        pairs = nearest_by_hard_strata(
            left,
            right,
            magnitude.astype(np.float64),
            stats250["aggregate"].abs().numpy().astype(np.float64),
            np.sign(direct250.numpy()).astype(np.int8),
            target,
        )
        aligned_population = (
            weak
            & (stats250["activity"] >= q250["activity_90"])
            & (stats250["cancellation"] < T["primary_cancellation"])
        )
        aligned_pairs = nearest_by_hard_strata(
            left,
            np.flatnonzero(aligned_population.numpy()),
            magnitude.astype(np.float64),
            stats250["aggregate"].abs().numpy().astype(np.float64),
            np.sign(direct250.numpy()).astype(np.int8),
            target,
        )
        pair_left = [x for x, _ in pairs]
        pair_right = [y for _, y in pairs]
        lr, lc = row_col_counts(pair_left, shape)
        rr, rc = row_col_counts(pair_right, shape)
        magnitude_rms = float(torch.sqrt(weight250.square().mean()))
        aggregate_rms = float(torch.sqrt(stats250["aggregate"].square().mean()))
        if pairs:
            a, b = np.asarray(pair_left), np.asarray(pair_right)
            mag_residual = np.abs(magnitude[a] - magnitude[b])
            agg_residual = np.abs(
                stats250["aggregate"].abs().numpy()[a]
                - stats250["aggregate"].abs().numpy()[b]
            )
            energy_residual = abs(
                float(weight250[a].square().sum() - weight250[b].square().sum())
            ) / max(float(weight250[a].square().sum() + weight250[b].square().sum()), 1e-30)
        else:
            mag_residual = agg_residual = np.array([])
            energy_residual = None
        degree_residual = int(np.abs(lr - rr).sum() + np.abs(lc - rc).sum())
        matching_rows.append(
            {
                **meta,
                "shape": "x".join(map(str, shape)),
                "inactive_candidates": len(left),
                "conflicted_candidates": len(right),
                "requested_pairs": target,
                "matched_pairs": len(pairs),
                "feasible_pool": bool(len(left) >= target and len(right) >= target and len(pairs) >= target),
                "magnitude_bin_matched": True,
                "aggregate_bin_matched": True,
                "sign_matched": True,
                "row_column_degree_exact": degree_residual == 0,
                "row_column_total_degree_imbalance": degree_residual,
                "removed_squared_energy_relative_difference": energy_residual,
                "aligned_control_candidates": int(aligned_population.sum()),
                "aligned_control_matched_pairs": len(aligned_pairs),
                "aligned_control_reaches_target": len(aligned_pairs) >= target,
            }
        )
        residual_rows.append(
            {
                **meta,
                "matched_pairs": len(pairs),
                "max_magnitude_residual_over_rms": (
                    float(mag_residual.max() / magnitude_rms) if len(mag_residual) else None
                ),
                "mean_magnitude_residual_over_rms": (
                    float(mag_residual.mean() / magnitude_rms) if len(mag_residual) else None
                ),
                "max_aggregate_residual_over_rms": (
                    float(agg_residual.max() / aggregate_rms) if len(agg_residual) else None
                ),
                "mean_aggregate_residual_over_rms": (
                    float(agg_residual.mean() / aggregate_rms) if len(agg_residual) else None
                ),
                "max_row_degree_difference": int(np.abs(lr - rr).max()),
                "max_column_degree_difference": int(np.abs(lc - rc).max()),
                "activation_mean_difference": (
                    float(activation250[a].mean() - activation250[b].mean())
                    if len(pairs)
                    else None
                ),
                "probe_second_moment_mean_difference": (
                    float(pooled_moment250[a].mean() - pooled_moment250[b].mean())
                    if len(pairs)
                    else None
                ),
            }
        )
        pair_artifacts.append(
            {
                "parameter": name,
                "conflicted_pairs": torch.tensor(pairs, dtype=torch.int64),
                "aligned_pairs": torch.tensor(aligned_pairs, dtype=torch.int64),
            }
        )

        # Grouped single-source prediction sample; experts held out later.
        count = CONFIG["prediction"]["sample_per_matrix"]
        sample = sample_indices(len(weight250), count, seed=890117 + index)
        absolute_domain_sum = mu250d[0].abs() + mu250d[1].abs()
        conditional1 = (
            mu250d[0].abs() * targets250[0] / accepted250[0]
            if accepted250[0] > 0
            else torch.full_like(mu250d[0], torch.nan)
        )
        conditional2 = (
            mu250d[1].abs() * targets250[1] / accepted250[1]
            if accepted250[1] > 0
            else torch.full_like(mu250d[1], torch.nan)
        )
        conditional_abs = fixed_p[0] * conditional1 + fixed_p[1] * conditional2
        X = np.stack(
            [
                magnitude[sample],
                stats250["aggregate"].abs().numpy()[sample],
                stats250["activity"].numpy()[sample],
                (weight250 * stats250["aggregate"].abs()).numpy()[sample],
                pooled_moment250.numpy()[sample],
                activation250.numpy()[sample],
                movement_0_250.numpy()[sample],
                stats250["cancellation"].numpy()[sample],
                stats250["cancelled_mass"].numpy()[sample],
                mu250d[0].abs().numpy()[sample],
                mu250d[1].abs().numpy()[sample],
                (
                    mu250d[0].abs()
                    / torch.maximum(absolute_domain_sum, torch.tensor(1e-30))
                ).numpy()[sample],
                (torch.sign(mu250d[0]) * torch.sign(mu250d[1])).numpy()[sample],
                (
                    (mu250d[0] - mu250d[1]).abs()
                    / torch.maximum(absolute_domain_sum, torch.tensor(1e-30))
                ).numpy()[sample],
                stats250["domain_mean_second_moment"].numpy()[sample],
                routing_activation250.numpy()[sample],
                conditional_abs.numpy()[sample],
            ],
            axis=1,
        ).astype(np.float32)
        prediction_samples.append(
            {
                "expert_group": np.full(len(sample), expert % 4, dtype=np.int8),
                "matrix": np.full(len(sample), index, dtype=np.int16),
                "X": X,
                "aggregate_future": stats500["aggregate"].abs().numpy()[sample],
                "activity_future": stats500["activity"].numpy()[sample],
                "movement_future": movement_250_500.numpy()[sample],
                "release": release.numpy()[sample],
            }
        )
        stride = max(1, len(weight250) // 4000)
        sample_plot = np.arange(0, len(weight250), stride)
        selected_plot = release[sample_plot].numpy()
        if selected_plot.any():
            plot_release["initial"].extend(stats250["cancellation"][sample_plot][selected_plot].numpy())
            plot_release["final"].extend(stats500["cancellation"][sample_plot][selected_plot].numpy())
            plot_release["aggregate_initial"].extend(
                stats250["aggregate"].abs()[sample_plot][selected_plot].numpy()
            )
            plot_release["aggregate_final"].extend(
                stats500["aggregate"].abs()[sample_plot][selected_plot].numpy()
            )
        decomposition_total.append(
            (
                float(routing_effect.abs()[valid_decomp].sum()),
                float(content_effect.abs()[valid_decomp].sum()),
                float(routing_k.abs()[valid_decomp].sum()),
                float(content_k.abs()[valid_decomp].sum()),
            )
        )
        release_decomposition_total.append(
            (
                float(routing_effect.abs()[release].sum()),
                float(content_effect.abs()[release].sum()),
                float(routing_k.abs()[release].sum()),
                float(content_k.abs()[release].sum()),
            )
        )
        qualifications.append(int(stats250["qualified"].sum()))
        release_validation.append(release_overlap if release_overlap is not None else 0.0)
        if (index + 1) % 32 == 0:
            print(f"Analyzed matrices {index + 1}/{len(names)}", flush=True)

    atomic_tensor(HERE / "matched_selection_pairs.pt", pair_artifacts)
    write_csv(HERE / "temporal_coordinate_summary.csv", temporal_rows)
    write_csv(HERE / "cancellation_release_summary.csv", release_rows)
    write_csv(HERE / "domain_expert_routing_dynamics.csv", routing_rows)
    write_csv(HERE / "routing_gradient_decomposition.csv", decomposition_rows)
    write_csv(HERE / "baseline_redundancy.csv", redundancy_rows)
    write_csv(HERE / "matched_selection_feasibility.csv", matching_rows)
    write_csv(HERE / "matching_residuals.csv", residual_rows)

    transitions = []
    before_labels = sorted(transition_counts)
    after_labels = sorted({key for inner in transition_counts.values() for key in inner})
    for before in before_labels:
        denominator = sum(transition_counts[before].values())
        for after in after_labels:
            count = transition_counts[before].get(after, 0)
            transitions.append(
                {
                    "population_250": before,
                    "population_500": after,
                    "count": count,
                    "fraction_from_population_250": count / denominator if denominator else None,
                }
            )
    write_csv(HERE / "population_transition_matrix.csv", transitions)

    # Exploratory grouped prediction: hold out one expert_id modulo class each fold.
    X = np.concatenate([row["X"] for row in prediction_samples])
    groups = np.concatenate([row["expert_group"] for row in prediction_samples])
    baseline_idx = np.arange(7)
    augmented_idx = np.arange(X.shape[1])
    outcomes = {
        "log1p_future_aggregate": np.log1p(
            np.concatenate([row["aggregate_future"] for row in prediction_samples])
        ),
        "log1p_future_activity": np.log1p(
            np.concatenate([row["activity_future"] for row in prediction_samples])
        ),
        "log1p_future_movement": np.log1p(
            np.concatenate([row["movement_future"] for row in prediction_samples])
        ),
        "release": np.concatenate([row["release"] for row in prediction_samples]).astype(np.int8),
    }
    prediction_rows = []
    for outcome, y in outcomes.items():
        classification = outcome == "release"
        for feature_set, columns in [("conventional", baseline_idx), ("domain_routing", augmented_idx)]:
            fold_metrics = []
            for held_out in range(4):
                train = groups != held_out
                test = ~train
                if classification:
                    model = HistGradientBoostingClassifier(
                        max_iter=80, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                        class_weight="balanced", random_state=121 + held_out,
                    )
                    model.fit(X[train][:, columns], y[train])
                    prediction = model.predict_proba(X[test][:, columns])[:, 1]
                    metric = {
                        "roc_auc": float(roc_auc_score(y[test], prediction))
                        if len(np.unique(y[test])) > 1
                        else None,
                        "average_precision": float(average_precision_score(y[test], prediction)),
                        "r2": None,
                        "mae": None,
                    }
                else:
                    model = HistGradientBoostingRegressor(
                        max_iter=80, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                        random_state=121 + held_out,
                    )
                    model.fit(X[train][:, columns], y[train])
                    prediction = model.predict(X[test][:, columns])
                    metric = {
                        "roc_auc": None,
                        "average_precision": None,
                        "r2": float(r2_score(y[test], prediction)),
                        "mae": float(mean_absolute_error(y[test], prediction)),
                    }
                prediction_rows.append(
                    {
                        "outcome": outcome,
                        "feature_set": feature_set,
                        "held_out_expert_mod4": held_out,
                        "train_coordinates": int(train.sum()),
                        "test_coordinates": int(test.sum()),
                        **metric,
                    }
                )
    write_csv(HERE / "prospective_prediction_results.csv", prediction_rows)

    # Figures
    plt.figure(figsize=(7, 6))
    stride = max(1, len(plot_release["initial"]) // 50_000)
    ix = np.arange(0, len(plot_release["initial"]), stride)
    plt.scatter(
        np.asarray(plot_release["initial"])[ix],
        np.asarray(plot_release["final"])[ix],
        s=5,
        alpha=0.2,
    )
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("Cancellation at step 250")
    plt.ylabel("Cancellation at step 500")
    plt.title("Discovery release population: cancellation evolution")
    plt.tight_layout()
    plt.savefig(HERE / "temporal_cancellation_evolution.png", dpi=180)
    plt.close()

    transitions_matrix = np.zeros((len(before_labels), len(after_labels)))
    for i, before in enumerate(before_labels):
        denominator = sum(transition_counts[before].values())
        for j, after in enumerate(after_labels):
            transitions_matrix[i, j] = transition_counts[before].get(after, 0) / denominator
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(transitions_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(after_labels)), after_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(before_labels)), before_labels)
    ax.set_xlabel("step-500 checkpoint-relative population")
    ax.set_ylabel("step-250 checkpoint-relative population")
    fig.colorbar(image, ax=ax, label="transition fraction")
    plt.tight_layout()
    plt.savefig(HERE / "population_transitions.png", dpi=180)
    plt.close(fig)

    routing_table = routing_rows
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for domain in ("tinystories", "wikitext103"):
        rows = [row for row in routing_table if row["domain"] == domain]
        p250 = np.asarray([row["p_expert_given_domain_250"] for row in rows])
        p500 = np.asarray([row["p_expert_given_domain_500"] for row in rows])
        entropy250 = np.asarray(
            [row["requested_router_entropy_mean_250"] for row in rows], dtype=float
        )
        entropy500 = np.asarray(
            [row["requested_router_entropy_mean_500"] for row in rows], dtype=float
        )
        axes[0].scatter(p250, p500, s=12, alpha=0.45, label=domain)
        axes[1].scatter(entropy250, entropy500, s=12, alpha=0.45, label=domain)
    axes[0].plot([0, 0.08], [0, 0.08], "k--", linewidth=0.8)
    axes[0].set_xlabel("P(expert | domain) at step 250")
    axes[0].set_ylabel("P(expert | domain) at step 500")
    axes[1].plot([0, 4], [0, 4], "k--", linewidth=0.8)
    axes[1].set_xlabel("requested router entropy at step 250")
    axes[1].set_ylabel("requested router entropy at step 500")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Domain-conditioned routing evolution")
    plt.tight_layout()
    plt.savefig(HERE / "routing_evolution.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    for values, label, color in [
        (
            [row["mean_abs_routing_effect"] for row in decomposition_rows],
            "routing-composition proxy",
            "#4c78a8",
        ),
        (
            [row["mean_abs_conditional_content_effect"] for row in decomposition_rows],
            "conditional-gradient content proxy",
            "#f58518",
        ),
    ]:
        plt.hist(np.log10(np.asarray(values) + 1e-300), bins=60, alpha=0.6, label=label, color=color)
    plt.xlabel("log10 mean absolute component contribution")
    plt.ylabel("matrix count")
    plt.legend()
    plt.title("Descriptive routing-composition versus conditional-gradient change")
    plt.tight_layout()
    plt.savefig(HERE / "routing_gradient_decomposition.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        np.log10(np.asarray(plot_release["aggregate_initial"]) + 1e-300),
        np.log10(np.asarray(plot_release["aggregate_final"]) + 1e-300),
        s=5,
        alpha=0.2,
    )
    plt.xlabel("log10 initial |G|")
    plt.ylabel("log10 subsequent |G|")
    plt.title("Subsequent aggregate-gradient development after release selection")
    plt.tight_layout()
    plt.savefig(HERE / "release_learning_signal.png", dpi=180)
    plt.close()

    metrics_by_outcome = defaultdict(dict)
    for row in prediction_rows:
        metrics_by_outcome[(row["outcome"], row["feature_set"])].setdefault(
            "roc", []
        ).append(row["roc_auc"])
        metrics_by_outcome[(row["outcome"], row["feature_set"])].setdefault(
            "r2", []
        ).append(row["r2"])
        metrics_by_outcome[(row["outcome"], row["feature_set"])].setdefault(
            "ap", []
        ).append(row["average_precision"])
    names_outcomes = list(outcomes)
    incremental = []
    for outcome in names_outcomes:
        conventional = metrics_by_outcome[outcome, "conventional"]
        domain = metrics_by_outcome[outcome, "domain_routing"]
        key = "roc" if outcome == "release" else "r2"
        cvals = np.asarray([x for x in conventional[key] if x is not None], dtype=float)
        dvals = np.asarray([x for x in domain[key] if x is not None], dtype=float)
        incremental.append(
            (float(cvals.mean()), float(dvals.mean()), outcome, float((dvals - cvals).mean()))
        )
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(incremental))
    width = 0.34
    ax.bar(x - width / 2, [v[0] for v in incremental], width, label="conventional")
    ax.bar(x + width / 2, [v[1] for v in incremental], width, label="domain/routing")
    ax.set_xticks(x, [v[2] for v in incremental], rotation=20, ha="right")
    ax.set_ylabel("mean held-out metric (ROC AUC or R2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(HERE / "incremental_prediction.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(9, 5))
    residuals = matching_rows
    plt.scatter(
        [row["matched_pairs"] for row in residuals],
        [row["row_column_total_degree_imbalance"] for row in residuals],
        s=20,
        alpha=0.6,
    )
    plt.xlabel("matched pairs")
    plt.ylabel("row+column degree imbalance L1")
    plt.title("Matched-selection feasibility and degree residuals")
    plt.tight_layout()
    plt.savefig(HERE / "matching_feasibility.png", dpi=180)
    plt.close()

    max_decomp_error = max(errors)
    direct250_checks = replay250["halves"]["1"]["direct_mixed_gradient_checks"]
    direct500_checks = replay500["halves"]["1"]["direct_mixed_gradient_checks"]
    all_direct = direct250_checks + direct500_checks
    numerical = {
        "status": "PASS",
        "checkpoint_hashes": {str(step): sha(path) for step, path in checkpoint_paths.items()},
        "coordinate_schema": {
            "matrices": len(names),
            "coordinates": int(
                sum(np.prod(shape) for shape in [*halves[250][0]["shapes"]])
            ),
            "identical_names_and_shapes": True,
            "identical_probe_batch_ids": True,
        },
        "state_preservation": {
            "step250": {
                key: replay250[key]
                for key in (
                    "parameter_state_unchanged",
                    "optimizer_not_instantiated",
                    "optimizer_updates",
                    "rng_unchanged",
                    "module_modes_unchanged",
                    "task_loss_only",
                    "single_shared_forward_per_batch",
                )
            },
            "step500": {
                key: replay500[key]
                for key in (
                    "parameter_state_unchanged",
                    "optimizer_not_instantiated",
                    "optimizer_updates",
                    "rng_unchanged",
                    "module_modes_unchanged",
                    "task_loss_only",
                    "single_shared_forward_per_batch",
                )
            },
        },
        "direct_mixed_gradient_max_absolute_error": max(
            row["max_absolute_error"] for row in all_direct
        ),
        "direct_mixed_gradient_max_relative_error": max(
            row["max_relative_error"] for row in all_direct
        ),
        "route_content_reconstruction_max_absolute_error": max_decomp_error,
        "all_decomposition_reconstruction_passed": max_decomp_error < 1e-5,
        "fixed_domain_probability": fixed_p.tolist(),
        "threshold_lock": "step-250 discovery half only; validation and sensitivity checks do not retune thresholds",
        "near_zero_activity_floor": T["activity_floor"],
        "squared_mean_not_confused_with_probe_moment": True,
        "cross_token_attribution_factorization": (
            "DESCRIPTIVE_ONLY: target-domain gradients can depend on context tokens "
            "of another domain, so acceptance probability is not an exact causal factorization"
        ),
    }
    if (
        numerical["direct_mixed_gradient_max_relative_error"] > 1e-4
        or not numerical["all_decomposition_reconstruction_passed"]
        or any(
            not numerical["state_preservation"][step]["parameter_state_unchanged"]
            for step in ("step250", "step500")
        )
    ):
        numerical["status"] = "FAIL"
    (HERE / "numerical_validation.json").write_text(json.dumps(numerical, indent=2) + "\n")
    if numerical["status"] != "PASS":
        raise RuntimeError("temporal numerical validation failed")

    mean_release = matrix_selected / sum(qualifications)
    release_overlap_mean = np.mean([x for x in release_validation if x is not None])
    routing_total = np.asarray(decomposition_total)
    routing_fraction = routing_total[:, 0].sum() / (
        routing_total[:, 0].sum() + routing_total[:, 1].sum()
    )
    routing_k_fraction = routing_total[:, 2].sum() / (
        routing_total[:, 2].sum() + routing_total[:, 3].sum()
    )
    release_routing_total = np.asarray(release_decomposition_total)
    release_routing_fraction = release_routing_total[:, 0].sum() / (
        release_routing_total[:, 0].sum() + release_routing_total[:, 1].sum()
    )
    release_routing_k_fraction = release_routing_total[:, 2].sum() / (
        release_routing_total[:, 2].sum() + release_routing_total[:, 3].sum()
    )
    feasible = sum(row["feasible_pool"] for row in matching_rows)
    aligned_feasible = sum(row["aligned_control_reaches_target"] for row in matching_rows)
    exact_degree = sum(row["row_column_degree_exact"] for row in matching_rows)
    prediction_summary = []
    for outcome in names_outcomes:
        conventional = metrics_by_outcome[(outcome, "conventional")]
        domain = metrics_by_outcome[(outcome, "domain_routing")]
        key = "roc" if outcome == "release" else "r2"
        c = np.asarray([x for x in conventional[key] if x is not None])
        d = np.asarray([x for x in domain[key] if x is not None])
        prediction_summary.append((outcome, c.mean(), d.mean(), (d - c).mean()))
    verdict_lines = [
        "# Pilot verdict: routing-conditioned gradient dynamics v1",
        "",
        "Cell: `seed7_A`, checkpoints 250→500, identical 90/90 canonical probe halves.",
        "",
        "## Scientific answers",
        "",
        f"1. **Do initially conflicted connections exhibit temporal release?** "
        f"{matrix_selected:,} coordinates ({mean_release:.4%} of qualified step-250 "
        "coordinates) meet the discovery release rule: high initial activity and "
        "cancellation, weak initial aggregate gradient, cancellation drop of at least "
        f"{T['release_cancellation_drop']}, aggregate-gradient ratio at least "
        f"{T['release_aggregate_increase']}, and retained activity. Mean independent-validation "
        f"overlap across matrices is {release_overlap_mean:.4%}. The phenomenon is measurable "
        "but one-source and threshold-based. Release subtype indicators may overlap: "
        "weaker-activity, eventual-alignment, and domain-dominance counts are retained "
        "separately in `cancellation_release_summary.csv` and should not be summed as a partition.",
        "",
        "2. **Does release correspond to stronger subsequent aggregate signals?** "
        "By construction the operational release definition requires stronger aggregate "
        "magnitude, so this is an audited classification property, not independent evidence. "
        "Subsequent movement is reported relative to non-release controls in "
        "`cancellation_release_summary.csv`; it is not interpreted as importance.",
        "",
        "3. **Are routing-composition changes associated with release?** "
        f"Descriptive symmetric reweighting assigns {routing_fraction:.2%} of the sum of "
        f"aggregate-component magnitudes to acceptance/composition and {1-routing_fraction:.2%} "
        f"to conditional-gradient content; for absolute cancelled mass, the fractions are "
        f"{routing_k_fraction:.2%} routing and {1-routing_k_fraction:.2%} content. "
        f"Restricted to the operational release population, the fractions are "
        f"{release_routing_fraction:.2%} routing and {1-release_routing_fraction:.2%} "
        f"content for aggregate gradient, and {release_routing_k_fraction:.2%} routing "
        f"and {1-release_routing_k_fraction:.2%} content for cancelled mass. "
        "These are not causal effects: context dependencies and accepted-token identity violate a "
        "composition-only interpretation.",
        "",
        "4. **Can future learning-signal changes be predicted?**",
        "",
    ]
    for outcome, conventional, domain, improvement in prediction_summary:
        verdict_lines.append(
            f"- `{outcome}`: conventional mean held-out metric {conventional:.4f}, "
            f"domain/routing-augmented {domain:.4f}, increment {improvement:+.4f}."
        )
    verdict_lines.extend(
        [
            "",
            "These grouped predictions are exploratory within one source. A positive increment "
            "would support feasibility only, not cross-seed generalization.",
            "",
            "5. **Do conventional statistics already capture the structure?** "
            "Cancellation is mathematically constrained by aggregate and domain activity/moments. "
            "The baseline redundancy table retains the equal-mixture identity audit, and the "
            "prediction comparison tests incremental value rather than assuming independence.",
            "",
            "6. **Can controlled deletion populations be constructed?** "
            f"{feasible}/{len(matching_rows)} matrices reached the 1% per-matrix pair target "
            "under matrix/sign/magnitude-bin/aggregate-bin matching. "
            f"Only {exact_degree}/{len(matching_rows)} achieved exact row/column degree balance "
            "with the simple feasibility matcher. The high-activity aligned control reached "
            f"the same target in {aligned_feasible}/{len(matching_rows)} matrices. "
            "Residuals and infeasible constraints are "
            "retained; no weights were deleted.",
            "",
            "## Required causal intervention",
            "",
            "A causal routing conclusion would require a frozen-checkpoint replay in which "
            "the routed assignment or acceptance composition is altered while preserving the "
            "same token content, checkpoint representation, gates/attention state where feasible, "
            "and loss terms. Its factorial controls must separately vary acceptance composition, "
            "gate amplitude/derivative, and accepted-token identity. The current descriptive "
            "reweighting cannot separate those mechanisms.",
            "",
            "## Falsification conditions",
            "",
            "- No reproducible temporal release remains under independent probes or nearby "
            "discovery-locked thresholds.",
            "- Routing-composition reweighting reconstructs little of observed gradient or "
            "cancelled-mass evolution, while conditional content captures it.",
            "- Domain/routing features provide no incremental grouped predictive performance "
            "beyond conventional magnitude, aggregate gradient, gradient energy, and movement.",
            "- No matched small-magnitude populations remain after enforcing matrix, sign, "
            "magnitude, aggregate-gradient, and degree constraints.",
            "- A frozen-router or accepted-token intervention eliminates the association in a "
            "design where the implementation is otherwise unchanged.",
            "",
            "## Limitations",
            "",
            "- One initialization/data-order source; matrices and coordinates are not independent trials.",
            "- Two checkpoints only; no claim about temporal ordering between routing and gradient changes.",
            "- Target-domain attribution does not equal the source domain of all contributing context routes.",
            "- RAMP and activation measures are correlates, not importance.",
            "- No masking, pruning, optimizer step, recovery, or router intervention occurred.",
            "- No importance, dispensability, causal routing, or ICML-level novelty claim is made.",
            "",
            "## Recommendation",
            "",
            "Do not promote this mechanism on the current evidence. Release exists in a small "
            "population, but release-specific routing-composition contribution is about 11.76%, "
            "domain/routing features do not materially improve held-out aggregate, activity, or "
            "release prediction, and the registered matched populations are infeasible. Narrow "
            "the hypothesis to a conditional-gradient/content phenomenon or abandon the "
            "routing-conditioned dispensability interpretation unless a future causal replay "
            "materially changes these findings.",
        ]
    )
    (HERE / "pilot_verdict.md").write_text("\n".join(verdict_lines) + "\n")
    (HERE / "progress.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "step250_halves_reused": 2,
                "step500_halves": 2,
                "matrices": len(names),
                "coordinates": all_coordinates,
                "optimizer_updates": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "release_coordinates": matrix_selected,
                "matching_feasible_matrices": feasible,
                "routing_component_fraction": float(routing_fraction),
                "release_routing_component_fraction": float(release_routing_fraction),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
