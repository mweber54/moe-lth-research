"""Validate completed branches and produce seed-level adaptation analyses."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

from common import (
    CONDITIONS,
    GLOBAL_STEPS,
    MODEL_TYPES,
    OFFSETS,
    OUT,
    SEEDS,
    TRANSITION,
    csv_save,
    json_read,
    json_save,
    sha256,
)


T_CRITICAL_95_DF4 = 2.7764451051977987
COLORS = {
    "full": "#222222",
    "router_frozen": "#1f77b4",
    "experts_frozen": "#d95f02",
    "router_experts_frozen": "#7b3294",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def late_dense_audit() -> dict:
    rows = read_csv(TRANSITION / "trajectory_metrics.csv")
    by_key = {
        (int(row["seed"]), row["condition"], int(row["step"])): row
        for row in rows
    }
    dense_rows = []
    sparse_rows = []
    for seed in SEEDS:
        start = float(by_key[(seed, "dense", 4500)]["ce"])
        end = float(by_key[(seed, "dense", 5000)]["ce"])
        dense_rows.append(
            {
                "seed": seed,
                "step_4500_ce": start,
                "step_5000_ce": end,
                "ce_change_4500_to_5000": end - start,
                "deteriorated": end > start,
                "target_tokens": int(by_key[(seed, "dense", 5000)]["target_tokens"]),
                "batches": int(by_key[(seed, "dense", 5000)]["batches"]),
                "precision": by_key[(seed, "dense", 5000)]["precision"],
            }
        )
        for condition in ("abrupt", "fixed", "gradual"):
            sparse_4500 = float(by_key[(seed, condition, 4500)]["ce"])
            sparse_5000 = float(by_key[(seed, condition, 5000)]["ce"])
            sparse_rows.append(
                {
                    "seed": seed,
                    "sparse_condition": condition,
                    "sparse_step_4500_ce": sparse_4500,
                    "dense_step_4500_ce": start,
                    "sparse_minus_dense_step_4500": sparse_4500 - start,
                    "sparse_step_5000_ce": sparse_5000,
                    "dense_step_5000_ce": end,
                    "sparse_minus_dense_step_5000": sparse_5000 - end,
                    "sparse_ce_change_4500_to_5000": sparse_5000 - sparse_4500,
                    "sparse_improved": sparse_5000 < sparse_4500,
                    "same_validation_target_tokens": (
                        int(by_key[(seed, condition, 4500)]["target_tokens"])
                        == int(by_key[(seed, "dense", 4500)]["target_tokens"])
                        == int(by_key[(seed, condition, 5000)]["target_tokens"])
                        == int(by_key[(seed, "dense", 5000)]["target_tokens"])
                    ),
                }
            )
    dense_path = OUT / "late_dense_audit.csv"
    sparse_path = OUT / "late_sparse_comparisons.csv"
    csv_save(dense_path, dense_rows)
    csv_save(sparse_path, sparse_rows)
    worsened = [row for row in dense_rows if row["deteriorated"]]
    sparse_nonpositive = [
        row for row in sparse_rows if row["sparse_minus_dense_step_5000"] <= 0
    ]
    sparse_improved = [row for row in sparse_rows if row["sparse_improved"]]
    report = f"""# Separate late-dense trajectory audit

This is an analysis-only audit of already observed 4500–5000 trajectories; it
does not extend training or enter the primary router × expert factorial.
Validation is the common target-token-weighted FP32/TF32-off readout.

* Dense validation CE increased for {len(worsened)}/{len(dense_rows)} seeds:
  {", ".join(str(row["seed"]) for row in worsened) if worsened else "none"}.
* Sparse-minus-dense CE was positive at step 4500 for
  {sum(row["sparse_minus_dense_step_4500"] > 0 for row in sparse_rows)}/{len(sparse_rows)}
  seed-condition comparisons.
* Sparse-minus-dense CE was nonpositive at step 5000 for
  {len(sparse_nonpositive)}/{len(sparse_rows)} comparisons.
* Sparse CE decreased from step 4500 to 5000 for
  {len(sparse_improved)}/{len(sparse_rows)} comparisons.

These observations do not establish overfitting, a pruning advantage, or a
causal explanation. The full per-seed table is in `late_dense_audit.csv`; the
15 seed-condition contrasts are in `late_sparse_comparisons.csv`.
"""
    (OUT / "late_dense_audit.md").write_text(report, encoding="utf-8")
    return {
        "dense_seeds_increased": [row["seed"] for row in worsened],
        "dense_seed_count": len(dense_rows),
        "sparse_comparisons": len(sparse_rows),
        "sparse_nonpositive_at_5000": len(sparse_nonpositive),
        "sparse_improved_4500_to_5000": len(sparse_improved),
    }


def _validate_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    registration = json_read(OUT / "registration.json")
    pilot_path = OUT / "pilot_validation.json"
    if not pilot_path.is_file() or json_read(pilot_path).get("status") != "PASS":
        raise RuntimeError("The registered seed-17 pilot has not passed.")
    input_hashes = json_read(OUT / "input_code_configuration_hashes.json")
    probe_path = OUT / "diagnostic_probe_manifest.json"
    if not probe_path.is_file():
        raise FileNotFoundError("Sealed diagnostic probe manifest is missing.")
    if sha256(probe_path) != registration.get("diagnostic_probe_manifest_sha256"):
        raise RuntimeError("Sealed probe manifest hash differs from registration.")
    for path_string, expected in input_hashes.items():
        path = Path(path_string)
        if not path.is_absolute():
            continue
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Registered source/input hash mismatch: {path}")

    trajectory_rows = []
    update_rows = []
    branch_starts = []
    for seed in SEEDS:
        for model_type in MODEL_TYPES:
            for condition in CONDITIONS:
                job_id = f"seed{seed}_{model_type}_{condition}"
                status_path = OUT / "status" / f"{job_id}.json"
                if not status_path.is_file():
                    raise RuntimeError(f"Missing completion status for {job_id}.")
                status = json_read(status_path)
                if status.get("status") != "COMPLETE":
                    raise RuntimeError(f"Job is not complete: {job_id} ({status.get('status')}).")
                result_path = Path(status["result"])
                result = json_read(result_path)
                checkpoint_path = Path(result["final_checkpoint"])
                if sha256(checkpoint_path) != result["final_checkpoint_sha256"]:
                    raise RuntimeError(f"Final checkpoint hash mismatch for {job_id}.")
                if int(result["attempted_updates"]) != 500:
                    raise RuntimeError(f"Wrong attempted update count for {job_id}.")
                if int(result["applied_updates"]) + int(result["amp_skips"]) != 500:
                    raise RuntimeError(f"Update accounting does not sum to 500 for {job_id}.")
                attempt = Path(result["attempt"])
                start = json_read(attempt / "branch_start.json")
                branch_starts.append(start)
                current = read_csv(attempt / "trajectory_metrics.csv")
                if sorted(int(row["global_step"]) for row in current) != list(GLOBAL_STEPS):
                    raise RuntimeError(f"Validation grid incomplete for {job_id}.")
                if any(row["precision"] != "FP32_TF32off" for row in current):
                    raise RuntimeError(f"Validation precision mismatch for {job_id}.")
                current_updates = read_csv(attempt / "optimizer_and_updates.csv")
                if len(current_updates) != 1500:
                    raise RuntimeError(f"Update diagnostic table incomplete for {job_id}.")
                if any(row["frozen_optimizer_state_unchanged"] not in ("True", "1") for row in current_updates):
                    raise RuntimeError(f"Frozen optimizer state check failed for {job_id}.")
                diagnostic_manifest = json_read(attempt / "diagnostic_manifest.json")
                if diagnostic_manifest.get("probe_identity") != registration["probe_identities"][str(seed)]:
                    raise RuntimeError(f"Diagnostic probe identity mismatch for {job_id}.")
                if not diagnostic_manifest.get("diagnostic_preservation_verified_at_all_offsets"):
                    raise RuntimeError(f"Diagnostic preservation was not verified for {job_id}.")
                for filename, rows_per_offset in (
                    ("routing_and_gate_diagnostics.csv", 384),
                    ("expert_function_diagnostics.csv", 256),
                    ("shared_representation_diagnostics.csv", 9),
                ):
                    diagnostics = read_csv(attempt / filename)
                    counts = {}
                    for row in diagnostics:
                        offset = int(row["offset"])
                        counts[offset] = counts.get(offset, 0) + 1
                    if counts != {offset: rows_per_offset for offset in OFFSETS}:
                        raise RuntimeError(
                            f"Diagnostic grid/count mismatch for {job_id}/{filename}: {counts}"
                        )
                trajectory_rows.extend(current)
                update_rows.extend(current_updates)

    grouped = {}
    for row in branch_starts:
        grouped.setdefault((int(row["seed"]), row["model_type"]), []).append(row)
    for key, starts in grouped.items():
        hashes = {row["state_sha256"] for row in starts}
        losses = [float(row["baseline_ce"]) for row in starts]
        if len(hashes) != 1 or max(losses) - min(losses) > 1e-6:
            raise RuntimeError(f"Branch-start equality check failed for {key}.")
    if len(trajectory_rows) != 200:
        raise RuntimeError(f"Expected 200 trajectory points, found {len(trajectory_rows)}.")
    return trajectory_rows, update_rows, branch_starts


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _summary_rows(values_by_contrast: dict[str, dict[int, float]]) -> list[dict]:
    rows = []
    for name, by_seed in sorted(values_by_contrast.items()):
        values = [float(by_seed[seed]) for seed in SEEDS]
        mean = _mean(values)
        sd = _sample_sd(values)
        half_width = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
        rows.append(
            {
                "contrast": name,
                "n_seeds": len(values),
                "mean": mean,
                "median": sorted(values)[len(values) // 2],
                "minimum": min(values),
                "maximum": max(values),
                "range": max(values) - min(values),
                "sample_sd": sd,
                "descriptive_t95_low": mean - half_width,
                "descriptive_t95_high": mean + half_width,
                "interpretation": "descriptive interval over five seed-level values; not token-level uncertainty",
            }
        )
    return rows


def _svg_trajectories(trajectories: dict) -> str:
    width, height = 1100, 600
    margin_left, margin_right, margin_top, margin_bottom = 72, 30, 48, 70
    panel_gap = 70
    panel_width = (width - margin_left - margin_right - panel_gap) / 2
    panel_height = height - margin_top - margin_bottom
    all_values = list(trajectories.values())
    low = min(all_values)
    high = max(all_values)
    padding = max((high - low) * 0.06, 0.02)
    low -= padding
    high += padding
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="27" text-anchor="middle" font-family="sans-serif" font-size="20">Validation CE across the registered adaptation grid</text>',
    ]
    for panel, model_type in enumerate(MODEL_TYPES):
        x0 = margin_left + panel * (panel_width + panel_gap)
        y0 = margin_top
        pieces.append(
            f'<text x="{x0 + panel_width/2}" y="{y0-12}" text-anchor="middle" font-family="sans-serif" font-size="16">{html.escape(model_type)}</text>'
        )
        pieces.append(f'<line x1="{x0}" y1="{y0+panel_height}" x2="{x0+panel_width}" y2="{y0+panel_height}" stroke="#222"/>')
        pieces.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+panel_height}" stroke="#222"/>')
        for tick in range(5):
            y = y0 + panel_height * tick / 4
            value = high - (high - low) * tick / 4
            pieces.append(f'<line x1="{x0}" y1="{y}" x2="{x0+panel_width}" y2="{y}" stroke="#ddd"/>')
            pieces.append(f'<text x="{x0-9}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3f}</text>')
        for offset in OFFSETS:
            x = x0 + panel_width * offset / 500
            pieces.append(f'<line x1="{x}" y1="{y0+panel_height}" x2="{x}" y2="{y0+panel_height+4}" stroke="#222"/>')
            pieces.append(f'<text x="{x}" y="{y0+panel_height+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{offset}</text>')
        pieces.append(f'<text x="{x0+panel_width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13">Updates after global step 1000</text>')
        for condition in CONDITIONS:
            by_seed = {
                seed: [
                    trajectories[(seed, model_type, condition, step)]
                    for step in GLOBAL_STEPS
                ]
                for seed in SEEDS
            }
            for values in by_seed.values():
                coordinates = []
                for step, value in zip(OFFSETS, values, strict=True):
                    x = x0 + panel_width * step / 500
                    y = y0 + panel_height * (high - value) / (high - low)
                    coordinates.append(f"{x:.2f},{y:.2f}")
                pieces.append(
                    f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{COLORS[condition]}" stroke-width="1" opacity="0.17"/>'
                )
            means = [
                _mean([by_seed[seed][index] for seed in SEEDS])
                for index in range(len(GLOBAL_STEPS))
            ]
            coordinates = []
            for offset, value in zip(OFFSETS, means, strict=True):
                x = x0 + panel_width * offset / 500
                y = y0 + panel_height * (high - value) / (high - low)
                coordinates.append(f"{x:.2f},{y:.2f}")
            pieces.append(
                f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{COLORS[condition]}" stroke-width="3"/>'
            )
        pieces.append(f'<text x="{x0+panel_width/2}" y="{y0+panel_height+44}" text-anchor="middle" font-family="sans-serif" font-size="12">thin = individual seeds; thick = seed mean</text>')
    legend_x = 850
    for index, condition in enumerate(CONDITIONS):
        y = 52 + index * 21
        pieces.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" stroke="{COLORS[condition]}" stroke-width="3"/>')
        pieces.append(f'<text x="{legend_x+30}" y="{y+4}" font-family="sans-serif" font-size="11">{html.escape(condition)}</text>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def analyze() -> dict:
    trajectory_rows, update_rows, branch_starts = _validate_inputs()
    trajectory = {
        (int(row["seed"]), row["model_type"], row["condition"], int(row["global_step"])): float(row["ce"])
        for row in trajectory_rows
    }
    trajectory_long = []
    for row in trajectory_rows:
        trajectory_long.append(
            {
                "seed": int(row["seed"]),
                "model_type": row["model_type"],
                "condition": row["condition"],
                "global_step": int(row["global_step"]),
                "offset": int(row["offset"]),
                "ce": float(row["ce"]),
                "loss_sum": float(row["loss_sum"]),
                "target_tokens": int(row["target_tokens"]),
                "batches": int(row["batches"]),
                "precision": row["precision"],
                "applied_updates": int(row["applied_updates"]),
                "amp_skips": int(row["amp_skips"]),
                "source": row["source"],
            }
        )
    csv_save(OUT / "factorial_trajectory_metrics.csv", trajectory_long)

    penalties: dict[str, dict[int, float]] = {}
    contrasts = []
    for seed in SEEDS:
        row = {"seed": seed}
        for model_type in MODEL_TYPES:
            values = {
                condition: trajectory[(seed, model_type, condition, 1500)]
                for condition in CONDITIONS
            }
            p_router = values["router_frozen"] - values["full"]
            p_experts = values["experts_frozen"] - values["full"]
            p_both = values["router_experts_frozen"] - values["full"]
            interaction = (
                values["full"]
                - values["experts_frozen"]
                - values["router_frozen"]
                + values["router_experts_frozen"]
            )
            row[f"{model_type}_P_R"] = p_router
            row[f"{model_type}_P_E"] = p_experts
            row[f"{model_type}_P_RE"] = p_both
            row[f"{model_type}_I"] = interaction
            for condition in CONDITIONS:
                baseline = trajectory[(seed, model_type, condition, 1000)]
                endpoint = values[condition]
                row[f"{model_type}_{condition}_absolute_CE_reduction"] = baseline - endpoint
                penalties[f"{model_type}:{condition}:absolute_CE_reduction"] = (
                    penalties.get(f"{model_type}:{condition}:absolute_CE_reduction", {})
                )
                penalties[f"{model_type}:{condition}:absolute_CE_reduction"][seed] = baseline - endpoint
        row["E_R_sparse_minus_unpruned"] = row["sparse_P_R"] - row["unpruned_P_R"]
        row["E_E_sparse_minus_unpruned"] = row["sparse_P_E"] - row["unpruned_P_E"]
        row["I_sparse_minus_unpruned"] = row["sparse_I"] - row["unpruned_I"]
        contrasts.append(row)
        for name in ("P_R", "P_E", "P_RE", "I"):
            for model_type in MODEL_TYPES:
                key = f"{model_type}:{name}"
                penalties.setdefault(key, {})[seed] = row[f"{model_type}_{name}"]
        for name in ("E_R_sparse_minus_unpruned", "E_E_sparse_minus_unpruned", "I_sparse_minus_unpruned"):
            penalties.setdefault(name, {})[seed] = row[name]
    csv_save(OUT / "seed_level_factorial_contrasts.csv", contrasts)
    summary = _summary_rows(penalties)
    csv_save(OUT / "seed_level_contrast_summary.csv", summary)

    integrated_rows = []
    integrated_by_contrast = {}
    for seed in SEEDS:
        for model_type in MODEL_TYPES:
            full = [trajectory[(seed, model_type, "full", step)] for step in GLOBAL_STEPS]
            for condition in CONDITIONS:
                values = [
                    trajectory[(seed, model_type, condition, step)] - full[index]
                    for index, step in enumerate(GLOBAL_STEPS)
                ]
                area = sum(
                    0.5 * (values[index] + values[index + 1]) * (OFFSETS[index + 1] - OFFSETS[index])
                    for index in range(len(OFFSETS) - 1)
                )
                normalized = area / 500
                key = f"{model_type}:{condition}:integrated_excess_CE"
                integrated_by_contrast.setdefault(key, {})[seed] = area
                integrated_by_contrast.setdefault(
                    f"{model_type}:{condition}:normalized_average_excess_CE", {}
                )[seed] = normalized
                integrated_rows.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "condition": condition,
                        "integrated_excess_ce_updates": area,
                        "normalized_average_excess_ce": normalized,
                        "grid_offsets": ",".join(map(str, OFFSETS)),
                        "integration": "linear trapezoid over observed grid only",
                    }
                )
    csv_save(OUT / "trajectory_integrated_excess.csv", integrated_rows)
    csv_save(OUT / "trajectory_integrated_excess_summary.csv", _summary_rows(integrated_by_contrast))

    update_path = OUT / "optimizer_compute_accounting.csv"
    csv_save(update_path, update_rows)
    _write_figures(trajectory, contrasts)
    late = late_dense_audit()
    report = {
        "status": "COMPLETE",
        "independent_unit": "seed",
        "n_seeds": len(SEEDS),
        "branches": len(trajectory_rows) // len(GLOBAL_STEPS),
        "trajectory_points": len(trajectory_rows),
        "per_group_update_rows": len(update_rows),
        "late_dense_audit": late,
        "outputs": [
            "factorial_trajectory_metrics.csv",
            "seed_level_factorial_contrasts.csv",
            "seed_level_contrast_summary.csv",
            "trajectory_integrated_excess.csv",
            "trajectory_integrated_excess_summary.csv",
            "optimizer_compute_accounting.csv",
            "trajectory_ce.svg",
            "endpoint_penalties.svg",
            "late_dense_audit.csv",
            "late_sparse_comparisons.csv",
        ],
    }
    json_save(OUT / "analysis_status.json", report)
    _write_findings(contrasts, summary, integrated_by_contrast, late)
    return report


def _write_figures(trajectory: dict, contrasts: list[dict]) -> None:
    svg = _svg_trajectories(trajectory)
    (OUT / "trajectory_ce.svg").write_text(svg, encoding="utf-8")
    # One point set per model type; condition is encoded by contrast, so seed
    # colors are kept neutral to avoid implying a categorical seed effect.
    width, height = 900, 520
    left, right, top, bottom = 85, 30, 60, 65
    panel_gap = 85
    panel_width = (width - left - right - panel_gap) / 2
    labels = ["P_R", "P_E", "P_RE"]
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="19">Seed-level endpoint freezing penalties</text>',
    ]
    for panel, model_type in enumerate(MODEL_TYPES):
        x0 = left + panel * (panel_width + panel_gap)
        y0 = top
        panel_rows = [row for row in contrasts if row["seed"] in SEEDS]
        panel_values = [float(row[f"{model_type}_{label}"]) for row in panel_rows for label in labels]
        panel_low = min(panel_values) - max((max(panel_values) - min(panel_values)) * 0.1, 0.01)
        panel_high = max(panel_values) + max((max(panel_values) - min(panel_values)) * 0.1, 0.01)
        plot_height = height - top - bottom
        pieces.append(f'<text x="{x0+panel_width/2}" y="{y0-15}" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(model_type)}</text>')
        pieces.append(f'<line x1="{x0}" y1="{y0+plot_height}" x2="{x0+panel_width}" y2="{y0+plot_height}" stroke="#222"/>')
        pieces.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+plot_height}" stroke="#222"/>')
        for label_index, label in enumerate(labels):
            x = x0 + panel_width * (label_index + 0.5) / 3
            pieces.append(f'<text x="{x}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="13">{label}</text>')
            for seed_index, row in enumerate(sorted(contrasts, key=lambda r: int(r["seed"]))):
                value = float(row[f"{model_type}_{label}"])
                y = y0 + plot_height * (panel_high - value) / (panel_high - panel_low)
                pieces.append(f'<circle cx="{x+(seed_index-2)*4}" cy="{y}" r="4" fill="#426788" opacity="0.8"><title>seed {row["seed"]}: {value:.6f}</title></circle>')
        pieces.append(f'<text transform="translate({x0-52},{y0+plot_height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="12">CE penalty relative to full adaptation</text>')
    pieces.append("</svg>")
    (OUT / "endpoint_penalties.svg").write_text("\n".join(pieces), encoding="utf-8")


def _write_findings(contrasts: list[dict], summary: list[dict], integrated: dict, late: dict) -> None:
    summary_lookup = {row["contrast"]: row for row in summary}
    lines = [
        "# Post-pruning adaptation factorial findings",
        "",
        "## Status and interpretation",
        "",
        "This analysis is complete only after every registered branch, endpoint hash,",
        "diagnostic grid, and update accounting check passes. The independent unit is",
        "the source seed (n=5). The 1000–1500 window was selected after prior",
        "trajectories were observed; this is not an overall preregistration.",
        "",
        "Freezing contrasts estimate the effect of restricting adaptation under this",
        "specific training policy. They do not identify a mechanism or establish",
        "universality, a pruning algorithm, a routing mechanism, or pre-pruning",
        "recoverability prediction.",
        "",
        "## Seed-level endpoint contrasts",
        "",
        "Individual values are in `seed_level_factorial_contrasts.csv`; descriptive",
        "seed-level summaries (including sample SD and t intervals) are in",
        "`seed_level_contrast_summary.csv`. Intervals are not token-level uncertainty.",
        "",
        "| Contrast | Mean | Median | Range | Sample SD | Descriptive 95% t interval |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['contrast']} | {row['mean']:.6f} | {row['median']:.6f} | "
            f"{row['range']:.6f} | {row['sample_sd']:.6f} | "
            f"[{row['descriptive_t95_low']:.6f}, {row['descriptive_t95_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "The factorial interaction is `L11 - L10 - L01 + L00`, where 1 denotes",
            "training and 0 denotes freezing for router and experts, respectively.",
            "The row `P_RE` is the joint-freezing penalty. Sparse-excess penalties are",
            "reported as sparse minus unpruned.",
            "",
            "## Trajectory and function diagnostics",
            "",
            "Validation trajectories are in `factorial_trajectory_metrics.csv` and",
            "`trajectory_ce.svg`. Trapezoidal integrated excess CE is calculated only",
            "over the observed offsets 0, 25, 100, 250, and 500; see",
            "`trajectory_integrated_excess.csv` and its seed-level summary. Update",
            "norms, clipping, AMP skips, and path-length/net-displacement accounting",
            "are in `optimizer_compute_accounting.csv`. Routing, expert-function,",
            "shared-representation, and diagnostic-overhead tables remain in each",
            "attempt directory.",
            "",
            "Movement and routing summaries are descriptive diagnostics; they are not",
            "causal decompositions of validation CE.",
            "",
            "## Separate late dense audit",
            "",
            f"Dense CE rose for {len(late['dense_seeds_increased'])}/5 seeds in the",
            "historical 4500–5000 interval. The complete audit is in",
            "`late_dense_audit.md`; it is separate from the factorial and does not",
            "establish overfitting.",
            "",
            "## Bounded follow-up",
            "",
            "A single falsifiable follow-up would be a separately authorized",
            "router-only adaptation intervention with the same step-1000 state and",
            "batch stream, to test whether the interaction pattern persists when",
            "expert parameters are held fixed. It is not launched or included here.",
        ]
    )
    (OUT / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--late-audit", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.late_audit:
        print(json.dumps(late_dense_audit(), indent=2))
    else:
        print(json.dumps(analyze(), indent=2))
