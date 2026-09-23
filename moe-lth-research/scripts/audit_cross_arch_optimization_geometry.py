#!/usr/bin/env python3
"""Create evidence-availability deliverables; this script never trains or loads models."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cross_arch_optimization_geometry"
E8 = ROOT / "results" / "expert_count_replication" / "e8"
E12 = ROOT / "results" / "router_development_12expert_retry4_20260902"
SEEDS = (7, 17, 29)
CONDITIONS = {8: ("R0", "R20", "R100"), 12: ("R20", "R60", "R100")}
STEPS = (0, 100, 250, 500, 750, 1000, 1500, 2000, 2500)


def write_csv(name, rows):
    fields = sorted({key for row in rows for key in row})
    with open(OUT / name, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metadata_path(experts, sparse, seed, router):
    root = E8 if experts == 8 else (E12 if router == "R60" else ROOT / "results" / "expert_count_replication" / "e12")
    if sparse:
        return root / "sparse" / "s0.85" / f"seed_{seed}" / router / "metadata.json"
    return root / "dense" / f"seed_{seed}" / router / "metadata.json"


def main():
    if OUT.exists() and any(path.is_file() for path in OUT.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for experts, routers in CONDITIONS.items():
        for seed in SEEDS:
            for router in routers:
                sparse_metadata = metadata_path(experts, True, seed, router)
                dense_metadata = metadata_path(experts, False, seed, router)
                sparse = json.loads(sparse_metadata.read_text())
                dense = json.loads(dense_metadata.read_text())
                compatible = all(sparse[key] == dense[key] for key in ("seed", "router_age_percent", "router_step", "shared_state_hash", "training_batch_sequence_hash", "validation_batch_sequence_hash", "recovery_steps", "optimizer", "scheduler", "precision"))
                for step in STEPS:
                    sparse_checkpoint = sparse_metadata.parent / "checkpoints" / "final_recovered.pt"
                    dense_checkpoint = dense_metadata.parent / "checkpoints" / "final_recovered.pt"
                    endpoint = step == 2500
                    manifest.append({"expert_count": experts, "seed": seed, "router_initialization": router, "sparsity": .85, "recovery_step": step, "sparse_full_checkpoint_available": endpoint and sparse_checkpoint.exists(), "dense_full_checkpoint_available": endpoint and dense_checkpoint.exists(), "optimizer_state_available": False, "pairing_metadata_compatible": compatible, "gradient_norm_log_available": (sparse_metadata.parent / "gradient_stats" / "gradient_stats.jsonl").exists() and (dense_metadata.parent / "gradient_stats" / "gradient_stats.jsonl").exists(), "primary_alignment_recoverable": endpoint, "reason": "Only final recovered checkpoints saved; no intermediate model/optimizer states." if not endpoint else "Endpoint gradients can be recomputed on a new fixed batch, but no temporal update alignment is recoverable."})
    write_csv("condition_manifest.csv", manifest)
    unavailable = [
        ("gradient_alignment_native", "unavailable at requested intermediate steps", "requires matched sparse/dense model checkpoints at each diagnostic recovery step"),
        ("gradient_alignment_forced_sparse_routes", "unavailable at requested intermediate steps", "requires model checkpoints plus non-mutating forced-routing gradient probe"),
        ("gradient_alignment_forced_dense_routes", "unavailable at requested intermediate steps", "requires model checkpoints plus non-mutating forced-routing gradient probe"),
        ("update_alignment", "unavailable", "requires sparse/dense expert state snapshots at consecutive diagnostic steps"),
        ("parameter_trajectory_distance", "unavailable at requested intermediate steps", "requires sparse/dense expert state snapshots"),
        ("functional_expert_similarity", "unavailable at requested intermediate steps", "requires sparse/dense checkpoints and fixed forced-common inputs"),
        ("sparse_dense_routing_overlap", "unavailable at requested intermediate steps", "requires sparse/dense checkpoints on identical probe tokens"),
        ("gradient_coherence", "unavailable", "requires per-microbatch diagnostic backward passes from saved intermediate states"),
    ]
    for filename in ("gradient_alignment_native.csv", "gradient_alignment_forced_sparse_routes.csv", "gradient_alignment_forced_dense_routes.csv", "gradient_norms.csv", "update_alignment.csv", "parameter_trajectory_distance.csv", "functional_expert_similarity.csv", "sparse_dense_routing_overlap.csv", "gradient_coherence.csv", "expert_level_summary.csv", "layer_level_summary.csv", "temporal_alignment_summary.csv", "seed_specific_mechanism.csv", "optimization_geometry_vs_ticket_gap.csv", "cross_architecture_mechanism_summary.csv"):
        write_csv(filename, [{"analysis": name, "status": status, "reason": reason} for name, status, reason in unavailable])
    (OUT / "diagnostic_probe_manifest.json").write_text(json.dumps({"status": "not created", "reason": "No diagnostic rerun launched. Existing probe hashes prove schedule equality but do not retain a standalone fixed diagnostic batch for gradient replay.", "required_identity": "sample_id, batch_row, position, token_id"}, indent=2) + "\n")
    (OUT / "preflight_audit.md").write_text("# Optimization Geometry Preflight\n\n## Existing Pairing\n\nAll 18 required 85% sparse/dense pairs are present: e8 R0/R20/R100 and e12 R20/R60/R100 across seeds 7, 17, and 29. Metadata confirms matched seed, router age/step, shared-state hash, training/validation sequence hashes, recovery budget, optimizer, scheduler, and precision. Sparse masks are expert-local magnitude masks and existing integrity checks passed.\n\n## Falsified Retrospective Path\n\nOnly `final_recovered.pt` (step 2500) is saved for recovery runs. There are no full sparse/dense recovery checkpoints at 0, 100, 250, 500, 750, 1000, 1500, or 2000, and no optimizer states. Historical `gradient_stats.jsonl` provides norms only, not gradient vectors. Therefore the pre-specified primary metrics cannot be reconstructed across time; norms must not be substituted for cosines.\n")
    (OUT / "optimization_geometry_report.md").write_text("# Cross-Architecture Sparse-Expert Optimization Geometry\n\n## Finding\n\nThe historical results are protocol-compatible but insufficient for the pre-specified optimization-geometry test. The required gradient-vector and expert-update comparisons need matched sparse/dense recovery states at intermediate steps. Existing outputs retain only final recovery states and scalar gradient norms.\n\n## Consequence\n\nNo conclusion is drawn about native versus forced-common-routing gradient alignment, update-direction alignment, trajectory divergence, expert functional alignment, routing-set overlap, gradient coherence, temporal ordering, or seed-specific optimization geometry. A final-only gradient probe would not answer whether alignment precedes the ticket-gap divergence.\n\n## Minimum Diagnostic Rerun\n\nRun exactly 18 sparse/dense paired recoveries at 85% with the existing scientific protocol: e8 R0/R20/R100 and e12 R20/R60/R100, each for seeds 7/17/29. Save full model or expert-only state snapshots at steps 0, 100, 250, 500, 750, 1000, 1500, 2000, and 2500. At each snapshot, run non-mutating fp32, AMP-unscaled diagnostic backwards on a persisted fixed batch under native routing and separately forced sparse-router assignments (and dense-router assignments if cost permits). Compare dense gradients and updates only on sparse mask-surviving coordinates. This is diagnostic instrumentation, not a new recovery condition.\n\n## Verdict\n\nINSUFFICIENT CHECKPOINTS — MINIMAL DIAGNOSTIC RERUN REQUIRED\n\n**Most important implication for the paper:** The existing performance results show architecture-specific favorable router ages, but they cannot establish a common sparse-expert optimization geometry. Scalar gradient norms and final checkpoints are not substitutes for time-resolved gradient/update direction data.\n\n**Most important next experiment:** Run the 18-condition, 85%-only paired sparse/dense diagnostic rerun described above, with intermediate state snapshots and forced-common-routing gradient vectors.\n")


if __name__ == "__main__":
    main()