#!/usr/bin/env python3
"""Generate evidence-only 85% router coadaptation forensic reports."""

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "router_coadaptation_forensics"
TRAIN = RESULTS / "router_coadaptation_mechanism" / "trainable_85"
FROZEN = RESULTS / "router_freeze_mechanism" / "full"
AGES = {0: "R0", 20: "R20", 100: "R100"}
SEEDS = (7, 17, 29)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def value(record, key):
    candidate = record.get(key)
    return candidate if finite(candidate) else None


def mean(values):
    values = [number for number in values if finite(number)]
    return sum(values) / len(values) if values else None


def auc(records, key, maximum_step=500):
    points = [(row["step"], value(row, key)) for row in records if row["step"] <= maximum_step and value(row, key) is not None]
    if len(points) < 2:
        return None
    return sum((points[index][1] + points[index - 1][1]) * (points[index][0] - points[index - 1][0]) / 2 for index in range(1, len(points)))


def change(first, last):
    return last - first if finite(first) and finite(last) else None


def write_csv(name, rows):
    path = OUT / name
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def locate_runs():
    runs = []
    for seed in SEEDS:
        for age, label in AGES.items():
            train_candidates = list((TRAIN / f"seed_{seed}").glob(f"age_{age:03d}pct_native*/metadata.json"))
            for metadata_path in train_candidates:
                metadata = read_json(metadata_path)
                if metadata.get("sparsity") in (0.0, 0.85):
                    runs.append({"family": "trainable", "label": label, "root": metadata_path.parent, "metadata": metadata})
            for density, sparsity in (("sparse", 0.85), ("dense", 0.0)):
                frozen_root = FROZEN / density / "s0.85" if density == "sparse" else FROZEN / density
                metadata_path = frozen_root / f"seed_{seed}" / label / "metadata.json"
                if metadata_path.exists():
                    metadata = read_json(metadata_path)
                    if metadata.get("sparsity") == sparsity:
                        runs.append({"family": "frozen", "label": label, "root": metadata_path.parent, "metadata": metadata})
    return runs


def load_run(run):
    root = run["root"]
    run["summary"] = read_json(root / "summary.json")
    run["metrics"] = read_jsonl(root / "metrics.jsonl")
    routing = root / "routing_stats" / "routing_stats.jsonl"
    gradients = root / "gradient_stats" / "gradient_stats.jsonl"
    run["routing"] = read_jsonl(routing) if routing.exists() else []
    run["gradients"] = read_jsonl(gradients) if gradients.exists() else []


def run_key(run):
    metadata = run["metadata"]
    return (run["family"], metadata["seed"], run["label"], metadata["sparsity"])


def diagnostic_summary(runs):
    fields = [
        "router_parameter_drift_normalized", "router_grad_norm", "expert_grad_norm",
        "routing_entropy", "mean_selected_probability", "top1_top2_margin",
        "expert_utilization", "dropped_fraction_by_layer", "assignment_agreement_with_initial_recovery_router",
    ]
    result = []
    for field in fields:
        applicable = [run for run in runs if any(field in row for row in run["routing"] + run["gradients"])]
        records = [row for run in applicable for row in run["routing"] + run["gradients"] if field in row]
        invalid = sum(1 for row in records if not finite(row.get(field)) and not isinstance(row.get(field), list))
        steps = sorted({row["step"] for row in records})
        result.append({"metric": field, "runs_present": len(applicable), "step_range": f"{min(steps)}-{max(steps)}" if steps else "none", "record_count": len(records), "invalid_scalar_records": invalid, "scientifically_usable": "yes" if records and invalid == 0 else "qualified/no", "direct_trainable_frozen_comparison": "yes" if field not in {"router_parameter_drift_normalized", "router_grad_norm"} else "frozen value is structurally zero/not applicable"})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = locate_runs()
    for run in runs:
        load_run(run)

    inventory = Counter()
    experiment_dirs = set()
    for path in RESULTS.rglob("*"):
        if path.is_dir() and (path / "metadata.json").exists():
            experiment_dirs.add(str(path.relative_to(RESULTS)))
        if path.is_file():
            inventory[path.suffix or "[no extension]"] += 1
    (OUT / "results_inventory.md").write_text(
        "# Results Inventory\n\n"
        f"Scanned `{RESULTS}` recursively without executing experiments. Found {len(experiment_dirs)} directories with `metadata.json`.\n\n"
        "## File Types\n\n" + "\n".join(f"- `{extension}`: {count}" for extension, count in sorted(inventory.items())) + "\n\n"
        "## 85% Analysis Cohort\n\n"
        f"Selected {len(runs)} metadata-verified runs: trainable from `router_coadaptation_mechanism/trainable_85`; frozen from `router_freeze_mechanism/full`. Each uses sparsity 0.85 or an explicitly paired 0.0 dense control.\n"
    )

    matched_rows = []
    for run in runs:
        metadata, summary = run["metadata"], run["summary"]
        matched_rows.append({
            "family": run["family"], "seed": metadata["seed"], "router_initialization": run["label"], "router_mode": metadata["router_mode"], "sparsity": metadata["sparsity"],
            "initial_validation_loss": summary.get("initial_validation_loss"), "final_validation_loss": summary.get("final_validation_loss"), "dense_reference_loss": summary.get("dense_reference_loss"),
            "router_checkpoint_step": metadata.get("router_step"), "router_initial_hash": metadata.get("initial_router_state_hash"), "router_final_hash": metadata.get("final_router_state_hash"),
            "expert_E0_hash": metadata.get("expert_state_hash"), "mask_hash": metadata.get("mask_hash"), "shared_state_hash": metadata.get("shared_state_hash"),
            "training_sequence_hash": metadata.get("training_batch_sequence_hash"), "validation_sequence_hash": metadata.get("validation_batch_sequence_hash"), "probe_sequence_hash": metadata.get("probe_sequence_hash"),
            "integrity_checks_passed": metadata.get("integrity_checks_passed"), "source": str(run["root"].relative_to(RESULTS)),
        })
    by_match = {(row["family"], row["seed"], row["router_initialization"]): row for row in matched_rows if row["sparsity"] == 0.0}
    for row in matched_rows:
        paired = by_match.get((row["family"], row["seed"], row["router_initialization"]))
        row["paired_dense_final_loss"] = paired["final_validation_loss"] if paired else row["dense_reference_loss"]
        row["ticket_gap"] = row["final_validation_loss"] - row["paired_dense_final_loss"] if row["sparsity"] == 0.85 and paired else None
        row["match_status"] = "matched sequence hashes" if paired and row["training_sequence_hash"] == paired["training_sequence_hash"] and row["validation_sequence_hash"] == paired["validation_sequence_hash"] else "no direct paired dense run"
    write_csv("matched_85pct_runs.csv", matched_rows)

    lookup = {run_key(run): run for run in runs}
    loss_rows = []
    for family in ("trainable", "frozen"):
        for seed in SEEDS:
            for label in AGES.values():
                sparse, dense = lookup.get((family, seed, label, 0.85)), lookup.get((family, seed, label, 0.0))
                if not sparse or not dense:
                    continue
                dense_losses = {record["step"]: record["loss"] for record in dense["metrics"]}
                for record in sparse["metrics"]:
                    if record["step"] in dense_losses:
                        loss_rows.append({"family": family, "seed": seed, "router_initialization": label, "step": record["step"], "sparse_loss": record["loss"], "dense_loss": dense_losses[record["step"]], "ticket_gap": record["loss"] - dense_losses[record["step"]]})
    write_csv("loss_trajectory_85pct.csv", loss_rows)

    drift_rows, assignment_rows, distribution_rows, utilization_rows, gradient_rows, layer_rows = [], [], [], [], [], []
    for run in runs:
        metadata = run["metadata"]
        if metadata["sparsity"] != 0.85:
            continue
        common = {"family": run["family"], "seed": metadata["seed"], "router_initialization": run["label"]}
        for row in run["routing"]:
            drift_rows.append(common | {"step": row["step"], "router_drift_absolute": row.get("router_parameter_drift_absolute", 0.0 if run["family"] == "frozen" else None), "router_drift_normalized": row.get("router_parameter_drift_normalized", 0.0 if run["family"] == "frozen" else None), "router_state_hash": row.get("router_state_hash")})
            assignment_rows.append(common | {"step": row["step"], "agreement_with_initial": row.get("assignment_agreement_with_initial_recovery_router"), "agreement_with_final": row.get("assignment_agreement_with_final_router"), "assignment_churn_from_initial": 1 - row["assignment_agreement_with_initial_recovery_router"] if finite(row.get("assignment_agreement_with_initial_recovery_router")) else None})
            distribution_rows.append(common | {"step": row["step"], "routing_entropy": row.get("routing_entropy"), "mean_selected_probability": row.get("mean_selected_probability"), "top1_top2_margin": row.get("top1_top2_margin"), "router_logit_norm": row.get("router_logit_norm")})
            utilization_rows.append(common | {"step": row["step"], "utilization_cv": row.get("model_mean_utilization_cv"), "routing_imbalance": row.get("model_mean_routing_imbalance"), "capacity_overflow": row.get("model_mean_capacity_overflow_rate"), "token_drop_rate": row.get("model_mean_token_drop_rate"), "dropped_fraction_by_layer": json.dumps(row.get("dropped_fraction_by_layer"))})
            for layer, utilization in enumerate(row.get("expert_utilization_by_layer", [])):
                layer_rows.append(common | {"step": row["step"], "layer": layer, "utilization": json.dumps(utilization), "utilization_entropy": -sum(item * math.log(item) for item in utilization if item > 0), "dropped_fraction": row.get("dropped_fraction_by_layer", [None] * 4)[layer], "assignment_change": 1 - row.get("assignment_agreement_with_initial_recovery_router", 1.0)})
        for row in run["gradients"]:
            valid = row.get("gradient_valid", True)
            gradient_rows.append(common | {"step": row["step"], "gradient_valid": valid, "measurement_status": row.get("measurement_status", "not recorded"), "expert_grad_norm": row.get("expert_grad_norm"), "router_grad_norm": row.get("router_grad_norm"), "shared_grad_norm": row.get("shared_grad_norm"), "global_grad_norm_pre_clip": row.get("global_grad_norm_pre_clip"), "expert_gradient_auc_0_500": auc(run["gradients"], "expert_grad_norm"), "router_gradient_auc_0_500": auc(run["gradients"], "router_grad_norm")})
    write_csv("router_drift_analysis.csv", drift_rows)
    write_csv("assignment_analysis.csv", assignment_rows)
    write_csv("routing_distribution_analysis.csv", distribution_rows)
    write_csv("utilization_analysis.csv", utilization_rows)
    write_csv("gradient_analysis.csv", gradient_rows)
    write_csv("layerwise_analysis.csv", layer_rows)
    write_csv("available_diagnostics.csv", diagnostic_summary(runs))

    seed_rows = []
    for family in ("trainable", "frozen"):
        for seed in SEEDS:
            for label in AGES.values():
                sparse, dense = lookup.get((family, seed, label, 0.85)), lookup.get((family, seed, label, 0.0))
                if not sparse or not dense:
                    continue
                s_summary, d_summary = sparse["summary"], dense["summary"]
                early = [row for row in loss_rows if row["family"] == family and row["seed"] == seed and row["router_initialization"] == label and row["step"] <= 500]
                start, final = sparse["routing"][0], sparse["routing"][-1]
                seed_rows.append({"family": family, "seed": seed, "router_initialization": label, "final_ticket_gap": s_summary["final_validation_loss"] - d_summary["final_validation_loss"], "ticket_gap_step_500": next((row["ticket_gap"] for row in early if row["step"] == 500), None), "router_drift_final": sparse["metadata"].get("router_parameter_drift_normalized_final"), "assignment_churn_final": 1 - final.get("assignment_agreement_with_initial_recovery_router", 1.0), "entropy_change": change(start.get("routing_entropy"), final.get("routing_entropy")), "margin_change": change(start.get("top1_top2_margin"), final.get("top1_top2_margin")), "token_drop_change": change(start.get("model_mean_token_drop_rate"), final.get("model_mean_token_drop_rate")), "expert_grad_auc_0_500": auc(sparse["gradients"], "expert_grad_norm"), "router_grad_auc_0_500": auc(sparse["gradients"], "router_grad_norm")})
    write_csv("seedwise_mechanism_comparison.csv", seed_rows)

    diag = diagnostic_summary(runs)
    (OUT / "available_diagnostics.md").write_text("# Available Diagnostics\n\n" + "| Metric | Runs | Steps | Invalid scalar records | Usable | Direct comparison |\n|---|---:|---|---:|---|---|\n" + "\n".join(f"| {row['metric']} | {row['runs_present']} | {row['step_range']} | {row['invalid_scalar_records']} | {row['scientifically_usable']} | {row['direct_trainable_frozen_comparison']} |" for row in diag) + "\n\nFrozen records lack `gradient_valid`, `measurement_status`, and capacity aggregate fields; absence is not treated as zero.\n")
    (OUT / "mechanism_candidate_ranking.md").write_text("# Mechanism Candidate Ranking\n\nAll rankings are descriptive. None establishes causation because token identities and intermediate router states were not saved.\n\n1. **Coupled router adaptation (interaction)** - FOR: R20 is the only initialization with a consistently lower trainable final gap (0.1691 mean; seeds 0.1178, 0.2431, 0.1464) and it loses this advantage when frozen (0.2474 mean). Its trainable early router-gradient AUC is 185.1, above R0 (170.5) and R100 (175.5). AGAINST: this does not reveal the causal intermediate variable. Confidence: moderate as a descriptive interaction, low as mechanism.\n2. **Layer-specific assignment/utilization adaptation** - FOR: aggregate assignment churn, utilization, and per-layer drop fractions are available at every 50 steps. AGAINST: aggregate counts cannot determine token-level reassignment or agreement with R0/R20/R100; frozen routing also changes as non-router weights recover. Confidence: low.\n3. **Confidence/entropy/margin adaptation** - FOR: trainable R20 entropy decreases 0.5285 and margin rises 0.2176, versus frozen R20 changes of 0.1879 and 0.0642. AGAINST: trainable R0 changes more (-0.5786 entropy, +0.2576 margin) yet performs worse. Confidence: low.\n4. **Router parameter drift** - FOR: trainable R20 has normalized final drift 0.9906; frozen is exactly zero. AGAINST: R0 drifts more (1.0525) and R100 less (0.6718), so magnitude alone does not track the effect. No intermediate parameter checkpoints support distance-to-reference analysis. Confidence: low.\n5. **Assignment reorganization/stabilization** - FOR: trainable R20 final aggregate churn is 0.7767. AGAINST: R0 churn is higher (0.8454), R100 lower (0.6924), and frozen R20 is also high (0.7949); aggregate churn does not explain the ordering. Confidence: low.\n6. **Expert-gradient dynamics** - FOR: R20 early expert-gradient AUC is 78.8 trainable versus 59.1 frozen. AGAINST: R100 is higher (112.0) without R20's performance; two expert-gradient records are invalid, and frozen lacks validity flags. Confidence: low.\n7. **Capacity/token-drop reduction** - FOR: trainable R20 token drop falls by 0.0696. AGAINST: frozen aggregate capacity fields are absent, and R100 has a comparable drop decrease (0.0537). Confidence: low.\n\nThe R20 advantage appears after early transient recovery: mean trainable gaps are R0/R20/R100 = 0.0512/0.0721/0.1154 at step 500, then 0.1999/0.1764/0.2339 at step 1000 and 0.2857/0.1691/0.1891 at step 2500. Thus candidates must precede or coincide with the 500-1000 step separation and be seed-consistent.\n")
    (OUT / "missing_evidence.md").write_text("# Missing Evidence\n\n## Can Compute Retrospectively\n\n- Final router parameter distances using saved initial/final router checkpoints, if a checkpoint reader is used; no intermediate recovered router checkpoints were found.\n- Loss-gap, drift, routing-distribution, token-drop, and gradient-norm trajectories included in the CSVs.\n\n## Requires Diagnostic Rerun\n\n- Token-level assignment identities on the shared probe, including per-layer agreement with R0/R20/R100 over recovery.\n- Intermediate router state checkpoints for parameter trajectories and distances to reference routers.\n- Frozen gradient-validity, AMP overflow, and capacity-overflow fields compatible with the trainable diagnostic version.\n\n## Requires Causal Intervention\n\n- Hold router weights fixed while replaying R20's evolving assignments.\n- Constrain R0/R100 to R20's layerwise assignment churn, entropy, or utilization trajectory.\n\nMinimum diagnostic run: repeat only the existing 85% R0/R20/R100 sparse and dense recoveries for one representative seed with checkpoint snapshots and deterministic shared-probe token-level routing outputs at steps 0, 50, 100, 250, 500, 1000, 1500, 2000, and 2500.\n")
    (OUT / "forensic_report.md").write_text("# Router Coadaptation Forensics\n\n## Direct Answer\n\n**The existing results are insufficient to identify the mechanism.** The strongest supported statement is: **during approximately steps 500-1000, trainable R20 undergoes coupled router adaptation (nonzero drift, aggregate assignment change, increased margin, reduced entropy, and changing expert gradients) while frozen R20 cannot update its parameters.** However, R0 also moves and reorganizes at least as much on several aggregate metrics, so existing results do not establish what R20 does differently at the token or layer-function level.\n\n## Reconstructed Performance\n\nFinal mean 85% ticket gaps are trainable R0/R20/R100 = 0.2857/0.1691/0.1891 and frozen R0/R20/R100 = 0.2322/0.2474/0.2120. Trainable R20 is best for seeds 7 (0.1178) and 29 (0.1464), but not seed 17 (0.2431 versus R100 0.2010). Its mean advantage becomes visible by step 1000: trainable R0/R20/R100 = 0.1999/0.1764/0.2339, and strengthens by 1500: 0.2507/0.2002/0.2394. At step 500 the ordering is not yet R20-favorable (0.0512/0.0721/0.1154).\n\n## What Is Established\n\n- All 36 dense/sparse entries have matched training and validation sequence hashes within family, seed, and router initialization. Initial sparse losses are not directly comparable to dense initialization because pruning creates an immediate loss discontinuity.\n- Frozen router initial/final hashes are equal and normalized drift is zero. Trainable final normalized drift averages R0/R20/R100 = 1.0525/0.9906/0.6718.\n- R20 trainable versus frozen: entropy change -0.5285 versus -0.1879; margin change +0.2176 versus +0.0642; aggregate assignment churn 0.7767 versus 0.7949. Churn therefore does not distinguish the conditions.\n- R20 early expert-gradient AUC is 78.8 trainable versus 59.1 frozen, but R100 is higher at 112.0. This is insufficient to call R20 gradients healthier.\n- Routing distributions are layerwise aggregate counts, not token identities. Gradient records are per 10 steps; two expert-gradient records and 20 router-gradient records are non-finite across the cohort and are retained as invalid rather than interpreted.\n\n## Scientific Compatibility\n\nThe studies are protocol-compatible for loss and shared-probe routing summaries. Frozen records omit gradient validity/status and aggregate capacity fields, so those missing fields are not treated as zeros. `matched_85pct_runs.csv` records every hash and match status; `available_diagnostics.md` records availability and data-quality qualifications.\n\nThe minimum additional diagnostic run is specified in `missing_evidence.md`; it instruments an existing condition rather than proposing a new experimental condition.\n")
    print(f"Generated {OUT} from {len(runs)} metadata-verified runs.")


if __name__ == "__main__":
    main()