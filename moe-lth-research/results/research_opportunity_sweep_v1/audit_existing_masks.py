import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "verified_observations.csv"
LOWER = ROOT / "results/dataset_b_lower_sparsity"
DATASET_A = ROOT / "results/final_dataset_a"
LOWER_A = ROOT / "results/dataset_a_full_lower_sparsity/analysis_final"

FIELDS = [
    "family",
    "metric",
    "comparison",
    "num_experts",
    "seed",
    "sparsity",
    "router_age",
    "seed_value",
    "mean_value",
    "control_value",
    "paired_difference",
    "dense_reference",
    "seed_count",
    "strict_wins",
    "seed_values_json",
    "seed_winners_json",
    "condition_ids",
    "evidence_path",
    "audit_status",
]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_result(row):
    attempt = Path(row["artifact_path"])
    result = json.loads((attempt / "validated_result.json").read_text())
    summary = json.loads((attempt / "summary.json").read_text())
    with (attempt / "metrics.jsonl").open() as stream:
        metrics = [json.loads(line) for line in stream if line.strip()]
    expected_steps = list(range(0, 2501, 50))
    assert result["integrity_checks_passed"] is True, row["condition_id"]
    assert result["optimizer_step_attempts"] == 2500, row["condition_id"]
    assert result["final_checkpoint_sha256"] == row["artifact_sha256"], row["condition_id"]
    assert [point["step"] for point in metrics] == expected_steps, row["condition_id"]
    assert metrics[-1]["loss"] == summary["final_validation_loss"], row["condition_id"]
    assert summary["early_auc_window_steps"] == 1250, row["condition_id"]
    return result, summary


def add_record(records, **values):
    record = {field: "" for field in FIELDS}
    record.update(values)
    records.append(record)


def main():
    manifest = read_csv(LOWER / "manifests/run_manifest.csv")
    assert len(manifest) == 115
    assert sum(row["status"] == "COMPLETE" for row in manifest) == 100
    assert sum(row["status"] == "REUSED" for row in manifest) == 15
    assert all(row["status"] in {"COMPLETE", "REUSED"} for row in manifest)

    by_id = {row["condition_id"]: row for row in manifest}
    results = {row["condition_id"]: load_result(row) for row in manifest}
    pairs = []
    for learned_row in manifest:
        if learned_row["job_type"] != "sparse" or learned_row["mask_type"] != "learned":
            continue
        age = int(float(learned_row["router_age"]))
        if age not in {0, 100}:
            continue
        seed = int(learned_row["seed"])
        sparsity = float(learned_row["sparsity"])
        random_id = (
            f"A_e32_seed{seed}_R{age}_sparse_random_s{sparsity:.2f}"
        )
        random_row = by_id[random_id]
        dense_row = by_id[learned_row["dense_condition_id"]]
        learned_result, learned_summary = results[learned_row["condition_id"]]
        random_result, random_summary = results[random_id]
        dense_result, dense_summary = results[dense_row["condition_id"]]

        assert dense_row["job_type"] == "dense"
        assert dense_row["status"] == "REUSED"
        assert learned_row["dense_condition_id"] == random_row["dense_condition_id"]
        assert learned_row["source_id"] == random_row["source_id"]
        assert learned_row["router_checkpoint_step"] == random_row["router_checkpoint_step"]
        assert learned_result["initial_router_state_hash"] == random_result["initial_router_state_hash"]
        assert learned_result["training_batch_sequence_hash"] == random_result["training_batch_sequence_hash"]
        assert learned_result["validation_batch_sequence_hash"] == random_result["validation_batch_sequence_hash"]
        assert learned_result["mask_hash"] != random_result["mask_hash"]
        assert dense_result["validation_batch_sequence_hash"] == learned_result["validation_batch_sequence_hash"]
        assert learned_summary["final_validation_loss"] == learned_result["final_validation_loss"]
        assert random_summary["final_validation_loss"] == random_result["final_validation_loss"]

        learned_loss = learned_summary["final_validation_loss"]
        random_loss = random_summary["final_validation_loss"]
        dense_loss = dense_summary["final_validation_loss"]
        pairs.append(
            {
                "seed": seed,
                "sparsity": sparsity,
                "age": age,
                "learned_loss": learned_loss,
                "random_loss": random_loss,
                "dense_loss": dense_loss,
                "difference": learned_loss - random_loss,
                "learned_gap": learned_loss - dense_loss,
                "random_gap": random_loss - dense_loss,
                "learned_auc": learned_summary["early_auc"],
                "random_auc": random_summary["early_auc"],
                "learned_updates": learned_result["optimizer_steps_applied"],
                "random_updates": random_result["optimizer_steps_applied"],
                "learned_id": learned_row["condition_id"],
                "random_id": random_id,
                "dense_id": dense_row["condition_id"],
            }
        )

    assert len(pairs) == 40
    assert len({(p["seed"], p["sparsity"], p["age"]) for p in pairs}) == 40
    records = []
    for pair in pairs:
        add_record(
            records,
            family="Dataset B lower-sparsity, E32",
            metric="final validation cross-entropy (nats/token)",
            comparison="learned mask minus per-tensor-count-matched random mask",
            num_experts=32,
            seed=pair["seed"],
            sparsity=pair["sparsity"],
            router_age=pair["age"],
            seed_value=pair["learned_loss"],
            control_value=pair["random_loss"],
            paired_difference=pair["difference"],
            dense_reference=pair["dense_loss"],
            seed_count=1,
            strict_wins=int(pair["difference"] < 0),
            condition_ids=";".join((pair["learned_id"], pair["random_id"], pair["dense_id"])),
            evidence_path="results/dataset_b_lower_sparsity/recoveries/",
            audit_status="validated_result hashes, status, 2500-attempt curves, and paired data hashes checked",
        )

    for sparsity in (0.50, 0.55, 0.60, 0.65):
        for age in (0, 100):
            group = [p for p in pairs if p["sparsity"] == sparsity and p["age"] == age]
            differences = [p["difference"] for p in group]
            add_record(
                records,
                family="Dataset B lower-sparsity, E32",
                metric="final validation cross-entropy difference (nats/token)",
                comparison="mean learned minus random, paired within seed; descriptive",
                num_experts=32,
                sparsity=sparsity,
                router_age=age,
                mean_value=statistics.mean(differences),
                control_value=statistics.mean(p["random_loss"] for p in group),
                paired_difference=statistics.mean(differences),
                dense_reference=statistics.mean(p["dense_loss"] for p in group),
                seed_count=5,
                strict_wins=sum(value < 0 for value in differences),
                seed_values_json=json.dumps(
                    {str(p["seed"]): p["difference"] for p in group}, sort_keys=True
                ),
                condition_ids=";".join(
                    condition_id
                    for p in group
                    for condition_id in (p["learned_id"], p["random_id"], p["dense_id"])
                ),
                evidence_path="results/dataset_b_lower_sparsity/recoveries/",
                audit_status="all five paired source seeds retained; repeated strata are not independent",
            )

    master = read_csv(DATASET_A / "summaries/dataset_a_master_results.csv")
    assert len(master) == 30
    assert all(row["complete_five_seed_curve"] == "True" for row in master)
    for row in master:
        add_record(
            records,
            family="Dataset A router-age by sparsity",
            metric="mean winning router age and seed-wise agreement",
            comparison="mean-curve winner versus seed-specific winners",
            num_experts=int(row["num_experts"]),
            sparsity=float(row["sparsity"]),
            mean_value=row["mean_minimizing_router_age"],
            control_value=row["seeds_agreeing_with_mean_winner"],
            seed_count=5,
            strict_wins=row["seeds_agreeing_with_mean_winner"],
            seed_values_json=row["seed_specific_winning_ages"],
            seed_winners_json=row["mean_winning_ages"],
            evidence_path="results/final_dataset_a/summaries/dataset_a_master_results.csv",
            audit_status="complete five-seed Dataset A condition; descriptive selection result",
        )

    lower_rows = read_csv(LOWER_A / "setting_summary.csv")
    assert len(lower_rows) == 220
    seed_columns = [f"gap_seed_{seed}" for seed in (7, 17, 29, 43, 61)]
    for row in lower_rows:
        assert int(row["n"]) == 5
        for seed, column in zip((7, 17, 29, 43, 61), seed_columns):
            add_record(
                records,
                family="Dataset A lower-sparsity, 50-65%",
                metric="sparse minus matched dense endpoint CE gap (nats/token)",
                comparison="individual source-seed value; age and sparsity retained",
                num_experts=int(row["num_experts"]),
                seed=seed,
                sparsity=float(row["sparsity"]),
                router_age=int(row["router_age"]),
                seed_value=float(row[column]),
                mean_value=float(row["mean_gap"]),
                seed_count=5,
                strict_wins=int(float(row[column]) <= 0),
                evidence_path="results/dataset_a_full_lower_sparsity/analysis_final/setting_summary.csv",
                audit_status="published audit table; five seed values independently retained",
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(
        json.dumps(
            {
                "status": "complete",
                "dataset_b_lower_sparsity_manifest_rows": len(manifest),
                "validated_sparse_rows": sum(r["job_type"] == "sparse" for r in manifest),
                "validated_dense_rows": sum(r["job_type"] == "dense" for r in manifest),
                "learned_random_pairs": len(pairs),
                "learned_wins_by_sparsity_age": {
                    f"{sparsity:.2f}_R{age}": sum(
                        p["difference"] < 0
                        for p in pairs
                        if p["sparsity"] == sparsity and p["age"] == age
                    )
                    for sparsity in (0.50, 0.55, 0.60, 0.65)
                    for age in (0, 100)
                },
                "dataset_a_complete_age_curves": len(master),
                "dataset_a_lower_grid_seed_cells": len(lower_rows) * 5,
                "verified_observation_rows": len(records),
                "no_model_weights_loaded_or_forward_passes_run": True,
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
