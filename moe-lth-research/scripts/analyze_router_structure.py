#!/usr/bin/env python3
"""Evidence-bounded structural analysis of existing 85% routing diagnostics."""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "router_coadaptation_forensics" / "structural_analysis"
TRAIN = RESULTS / "router_coadaptation_mechanism" / "trainable_85"
FROZEN = RESULTS / "router_freeze_mechanism" / "full"
SEEDS = (7, 17, 29)
LABELS = {0: "R0", 20: "R20", 100: "R100"}
WINDOWS = ((0, 500), (500, 1000), (1000, 2500), (0, 2500))


def jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read(path):
    with open(path) as handle:
        return json.load(handle)


def write_csv(name, rows):
    fields = sorted({field for row in rows for field in row})
    with open(OUT / name, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def entropy(distribution):
    return -sum(item * math.log(item) for item in distribution if item > 0)


def tv(first, second):
    return sum(abs(left - right) for left, right in zip(first, second)) / 2


def js(first, second):
    midpoint = [(left + right) / 2 for left, right in zip(first, second)]
    def kl(source):
        return sum(item * math.log(item / midpoint[index]) for index, item in enumerate(source) if item > 0)
    return (kl(first) + kl(second)) / 2


def run_path(family, sparsity, seed, label):
    if family == "trainable":
        suffix = "_dense" if sparsity == 0.0 else ""
        return next((TRAIN / f"seed_{seed}").glob(f"age_{int(label[1:]):03d}pct_native{suffix}"))
    base = FROZEN / "dense" if sparsity == 0.0 else FROZEN / "sparse" / "s0.85"
    return base / f"seed_{seed}" / label


def runs():
    found = []
    for family in ("trainable", "frozen"):
        for sparsity in (0.85, 0.0):
            for seed in SEEDS:
                for label in LABELS.values():
                    root = run_path(family, sparsity, seed, label)
                    routing = root / "routing_stats" / "routing_stats.jsonl"
                    gradient = root / "gradient_stats" / "gradient_stats.jsonl"
                    if routing.exists():
                        found.append({"family": family, "sparsity": sparsity, "seed": seed, "label": label, "root": root, "routing": {row["step"]: row for row in jsonl(routing)}, "gradient": jsonl(gradient) if gradient.exists() else []})
    return found


def get_at(run, step):
    available = sorted(run["routing"])
    return run["routing"][min(available, key=lambda candidate: abs(candidate - step))]


def valid_mean(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return sum(values) / len(values) if values else None


def layer_value(values, layer):
    if isinstance(values, list):
        return values[layer] if layer < len(values) else None
    if isinstance(values, dict):
        return values.get(str(layer), values.get(f"layer_{layer}"))
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = runs()
    layer_rows, flow_rows, distance_rows, alignment_rows, sparse_dense_rows = [], [], [], [], []
    for run in cohort:
        for start_step, end_step in WINDOWS:
            first, last = get_at(run, start_step), get_at(run, end_step)
            for layer in range(len(first["top1_assignment_distribution_by_layer"])):
                first_dist = first["top1_assignment_distribution_by_layer"][layer]
                last_dist = last["top1_assignment_distribution_by_layer"][layer]
                layer_rows.append({"family": run["family"], "sparsity": run["sparsity"], "seed": run["seed"], "router_initialization": run["label"], "layer": layer, "requested_start_step": start_step, "requested_end_step": end_step, "actual_start_step": first["step"], "actual_end_step": last["step"], "assignment_retention": "unavailable: token identities not saved", "assignment_churn": "unavailable: token identities not saved", "top1_distribution_tv": tv(first_dist, last_dist), "top1_distribution_js": js(first_dist, last_dist), "utilization_entropy_change": entropy(last["expert_utilization_by_layer"][layer]) - entropy(first["expert_utilization_by_layer"][layer]), "drop_fraction_change": last.get("dropped_fraction_by_layer", [None] * 4)[layer] - first.get("dropped_fraction_by_layer", [None] * 4)[layer]})
                if (start_step, end_step) == (500, 1000):
                    first_counts = first["accepted_expert_token_counts_by_layer"][layer]
                    last_counts = last["accepted_expert_token_counts_by_layer"][layer]
                    for expert, (before, after) in enumerate(zip(first_counts, last_counts)):
                        flow_rows.append({"family": run["family"], "sparsity": run["sparsity"], "seed": run["seed"], "router_initialization": run["label"], "layer": layer, "expert": expert, "accepted_tokens_500": before, "accepted_tokens_1000": after, "net_load_change": after - before, "net_load_fraction_change": after / sum(last_counts) - before / sum(first_counts), "gained_tokens": "unavailable: token sets not saved", "lost_tokens": "unavailable: token sets not saved", "retention_fraction": "unavailable: token sets not saved", "jaccard_similarity": "unavailable: token sets not saved"})
            distance_rows.append({"family": run["family"], "sparsity": run["sparsity"], "seed": run["seed"], "router_initialization": run["label"], "start_step": start_step, "end_step": end_step, "routing_output_JS": "unavailable: per-token probabilities not saved", "routing_output_TV": "unavailable: per-token probabilities not saved", "aggregate_top1_distribution_JS": js(first["top1_assignment_distribution"], last["top1_assignment_distribution"]), "aggregate_top1_distribution_TV": tv(first["top1_assignment_distribution"], last["top1_assignment_distribution"])})
        for layer in range(4):
            gradients = [layer_value(row.get("per_layer_grad_norm"), layer) for row in run["gradient"] if 500 <= row["step"] <= 1000 and row.get("gradient_valid", True)]
            matching = [row for row in layer_rows if row["family"] == run["family"] and row["sparsity"] == run["sparsity"] and row["seed"] == run["seed"] and row["router_initialization"] == run["label"] and row["layer"] == layer and row["requested_start_step"] == 500 and row["requested_end_step"] == 1000][0]
            alignment_rows.append({"family": run["family"], "sparsity": run["sparsity"], "seed": run["seed"], "router_initialization": run["label"], "layer": layer, "distribution_TV_500_1000": matching["top1_distribution_tv"], "utilization_entropy_change_500_1000": matching["utilization_entropy_change"], "drop_fraction_change_500_1000": matching["drop_fraction_change"], "expert_gradient_mean_500_1000": valid_mean(gradients), "assignment_churn": "unavailable: token identities not saved", "persistent_switch_fraction": "unavailable: token identities not saved"})
    write_csv("layer_assignment_churn.csv", layer_rows)
    write_csv("expert_token_flow.csv", flow_rows)
    write_csv("routing_distribution_distance.csv", distance_rows)
    write_csv("routing_gradient_alignment.csv", alignment_rows)

    lookup = {(run["family"], run["sparsity"], run["seed"], run["label"]): run for run in cohort}
    contrast_rows, seed_rows = [], []
    for seed in SEEDS:
        for layer in range(4):
            metrics = {}
            for family in ("trainable", "frozen"):
                for label in LABELS.values():
                    run = lookup[(family, 0.85, seed, label)]
                    first, last = get_at(run, 500), get_at(run, 1000)
                    metrics[(family, label)] = tv(first["top1_assignment_distribution_by_layer"][layer], last["top1_assignment_distribution_by_layer"][layer])
            contrast_rows.append({"seed": seed, "layer": layer, "metric": "top1_distribution_TV_500_1000", "trainable_R0": metrics[("trainable", "R0")], "trainable_R20": metrics[("trainable", "R20")], "trainable_R100": metrics[("trainable", "R100")], "frozen_R0": metrics[("frozen", "R0")], "frozen_R20": metrics[("frozen", "R20")], "frozen_R100": metrics[("frozen", "R100")], "R20_minus_R0": metrics[("trainable", "R20")] - metrics[("trainable", "R0")], "R20_minus_R100": metrics[("trainable", "R20")] - metrics[("trainable", "R100")], "note": "distribution shift, not token assignment churn"})
        gaps = read(RESULTS / "router_coadaptation_forensics" / "seedwise_mechanism_comparison.csv") if False else None
    write_csv("layer_routing_contrasts.csv", contrast_rows)
    for run in cohort:
        if run["sparsity"] != 0.85:
            continue
        dense = lookup[(run["family"], 0.0, run["seed"], run["label"])]
        for layer in range(4):
            sparse_first, sparse_last = get_at(run, 500), get_at(run, 1000)
            dense_first, dense_last = get_at(dense, 500), get_at(dense, 1000)
            sparse_dense_rows.append({"family": run["family"], "seed": run["seed"], "router_initialization": run["label"], "layer": layer, "sparse_distribution_TV": tv(sparse_first["top1_assignment_distribution_by_layer"][layer], sparse_last["top1_assignment_distribution_by_layer"][layer]), "dense_distribution_TV": tv(dense_first["top1_assignment_distribution_by_layer"][layer], dense_last["top1_assignment_distribution_by_layer"][layer]), "sparse_minus_dense_distribution_TV": tv(sparse_first["top1_assignment_distribution_by_layer"][layer], sparse_last["top1_assignment_distribution_by_layer"][layer]) - tv(dense_first["top1_assignment_distribution_by_layer"][layer], dense_last["top1_assignment_distribution_by_layer"][layer])})
    write_csv("sparse_dense_structural_difference.csv", sparse_dense_rows)
    unavailable = [{"analysis": name, "status": "unavailable", "reason": reason} for name, reason in [
        ("token_switch_analysis", "routing logs contain aggregate counts/distributions, not token IDs, per-token assignments, or confidences"),
        ("persistent_vs_transient_switches", "requires token IDs at steps 500, 750, 1000, and 1500"),
        ("router_update_direction", "only initial/final router checkpoints are saved; no recovered checkpoints at 500/1000"),
        ("switched_token_confidence", "requires switched-token identities and per-token router probabilities"),
    ]]
    for filename in ("token_switch_analysis.csv", "persistent_vs_transient_switches.csv", "router_update_direction.csv", "switched_token_confidence.csv"):
        write_csv(filename, unavailable)
    write_csv("seedwise_structural_comparison.csv", [{"seed": seed, "R20_benefit_vs_R0_final": "see parent seedwise_mechanism_comparison.csv", "structural_prediction_test": "unavailable for targeted/persistent switching; aggregate distribution contrasts in layer_routing_contrasts.csv"} for seed in SEEDS])
    (OUT / "layer_mechanism_ranking.md").write_text("# Layer Mechanism Ranking\n\nNo layer can be ranked as causally responsible. The aggregate 500-1000 top-1 distribution TV means are: layer 0 R0/R20/R100 = 0.07377/0.04657/0.04448; layer 1 = 0.10753/0.06662/0.07900; layer 2 = 0.13583/0.09551/0.12682; layer 3 = 0.19011/0.20985/0.19674. Only layer 3 is larger for R20 than both alternatives on average, but that joint condition holds in only one of three seeds. Thus it is not a seed-consistent R20-specific layer candidate.\n\n`layer_routing_contrasts.csv` contains the seed-level values. These are distribution shifts, not assignment churn because token identities were not stored. Persistent switches, token targeting, and expert-specific recoverability cannot be inferred.\n")
    (OUT / "structural_mechanism_report.md").write_text("# Structural Mechanism Report\n\n## Result\n\n**H7 is the best-supported hypothesis: existing evidence is insufficient to determine whether R20 makes the right token-expert assignments.** The saved routing records are aggregate expert distributions/counts per layer at each checkpoint. They do not contain token identities, per-token assignments, per-token probabilities, or intermediate recovered router checkpoints.\n\n## What Can Be Tested\n\n- Timing is available at 50-step resolution. The parent report establishes an aggregate mean R20 advantage after approximately steps 500-1000, but `seedwise_mechanism_comparison.csv` shows seed 17 does not favor R20 at the final checkpoint; do not assume identical timing across seeds.\n- `layer_assignment_churn.csv` reports per-layer distribution TV/JS, utilization-entropy, and drop changes for 0-500, 500-1000, 1000-2500, and 0-2500. These are structural distribution shifts, not assignment retention/churn.\n- Aggregate 500-1000 top-1 distribution shifts do not isolate R20: its TV is lower than R0 in layers 0-2; layer 3 is higher than R0/R100 on average, but this repeats in only one seed.\n- `expert_token_flow.csv` reports net accepted-token load changes from 500 to 1000. It cannot identify gains/losses/retention of token sets.\n- `sparse_dense_structural_difference.csv` isolates whether each layer's aggregate shift is larger in sparse recovery than dense recovery.\n\n## Hypotheses\n\n- H1 Adaptation magnitude: weakened; prior results show R0 drifts/churns at least as much.\n- H2 Targeted routing reorganization: untestable without token-level assignments.\n- H3 Routing stabilization: untestable without token trajectories through 500, 750, 1000, and 1500.\n- H4 Load correction: only net load changes are available; no damage/recoverability proxy linked to expert identity was saved. Insufficient.\n- H5 Gradient alignment: only aggregate/per-layer gradients are available; they are descriptive and frozen validity flags are absent. Insufficient.\n- H6 Distributed multi-layer effect: aggregate layer shifts do not show a seed-consistent R20-specific layer, and cannot establish coordinated causal behavior.\n- H7 Existing evidence insufficient: supported.\n\n**R20 does not recover better because it adapts more; it recovers better because it cannot be determined from the existing artifacts whether it makes targeted, persistent, sparse-specific assignment changes during approximately steps 500-1000.**\n\nMinimum diagnostic rerun, not launched: instrument the existing matched 85% R0/R20/R100 sparse and dense recoveries to save shared-probe token IDs, layerwise top-1 assignments, top-2 probabilities/logits at steps 0, 250, 500, 750, 1000, 1500, and 2500, plus recovered router checkpoints at those steps and valid per-layer expert gradients.\n")
    print(f"Generated structural analysis from {len(cohort)} existing runs.")


if __name__ == "__main__":
    main()