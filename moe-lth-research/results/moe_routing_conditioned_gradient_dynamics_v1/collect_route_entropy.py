"""Add native router-entropy statistics for the identical temporal probes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "results/moe_gradient_cancellation_qualification_v1/run_pilot.py"

spec = importlib.util.spec_from_file_location("qualification_collector", SOURCE)
collector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(collector)


def run_step(step: int, checkpoint_path: Path, provenance, indices, halves, data, labels):
    payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    if payload["step"] != step:
        raise RuntimeError("checkpoint step mismatch")
    model, names, parameters, _ = collector.prepare_model(payload)
    before = collector.state_hash(model.state_dict())
    before_rng = collector.rng_snapshot()
    output = {
        (layer, expert, domain): {
            "targets": 0,
            "requested": 0,
            "accepted": 0,
            "entropy_sum": 0.0,
            "margin_sum": 0.0,
            "gate_sum": 0.0,
            "accepted_entropy_sum": 0.0,
        }
        for layer in range(4)
        for expert in range(32)
        for domain in (1, 2)
    }
    total_by_domain = np.zeros(2, dtype=np.int64)
    with torch.inference_mode():
        for half in halves:
            for batch_id in half:
                offsets = indices[batch_id].numpy()[:, None] * 128 + np.arange(128)
                x = torch.tensor(data[offsets].astype(np.int64), device=collector.DEVICE)
                domain = torch.tensor(labels[offsets + 1].astype(np.int64), device=collector.DEVICE)
                result = model(x)
                flat_domain = domain.reshape(-1)
                for layer, trace in enumerate(result.routes):
                    selected = trace.selected_expert_indices.reshape(-1)
                    accepted = trace.accepted_mask.reshape(-1)
                    entropy = trace.entropy.reshape(-1)
                    margin = trace.margin.reshape(-1)
                    gates = trace.selected_probabilities.reshape(-1)
                    for d in (1, 2):
                        keep_domain = flat_domain == d
                        total_by_domain[d - 1] += int(keep_domain.sum())
                        for expert in range(32):
                            selected_expert = selected == expert
                            requested = keep_domain & selected_expert
                            accepted_expert = requested & accepted
                            requested_count = int(requested.sum())
                            accepted_count = int(accepted_expert.sum())
                            current = output[layer, expert, d]
                            current["targets"] += int(keep_domain.sum())
                            current["requested"] += requested_count
                            current["accepted"] += accepted_count
                            if requested_count:
                                current["entropy_sum"] += float(entropy[requested].sum())
                                current["margin_sum"] += float(margin[requested].sum())
                                current["gate_sum"] += float(gates[requested].sum())
                                current["accepted_entropy_sum"] += float(
                                    entropy[accepted_expert].sum()
                                )
                del result, x, domain
    after = collector.state_hash(model.state_dict())
    if before != after or not collector.rng_equal(before_rng, collector.rng_snapshot()):
        raise RuntimeError("route-stat replay mutated model or RNG")
    rows = []
    for (layer, expert, domain), value in sorted(output.items()):
        requested = value["requested"]
        accepted = value["accepted"]
        rows.append(
            {
                "step": step,
                "layer": layer,
                "expert": expert,
                "domain": "tinystories" if domain == 1 else "wikitext103",
                "domain_targets": int(total_by_domain[domain - 1]),
                "requested": requested,
                "accepted": accepted,
                "requested_router_entropy_mean": (
                    value["entropy_sum"] / requested if requested else None
                ),
                "accepted_router_entropy_mean": (
                    value["accepted_entropy_sum"] / accepted if accepted else None
                ),
                "requested_margin_mean": (
                    value["margin_sum"] / requested if requested else None
                ),
                "requested_gate_mean": (
                    value["gate_sum"] / requested if requested else None
                ),
            }
        )
    return rows


def main():
    provenance, indices, halves, data, labels = collector.select_inputs()
    native = json.loads(
        (ROOT / "results/moe_connection_lifetime_v1/native_provenance.json").read_text()
    )
    collector.model_sources(native)
    prefix = ROOT / "results/moe_connection_lifetime_v1/prefixes/seed7_A"
    rows = []
    for step in (250, 500):
        checkpoint = prefix / f"state{step}.pt"
        expected = provenance["cells"][f"seed7_A_{step}"]["state_sha256"]
        if collector.sha(checkpoint) != expected:
            raise RuntimeError(f"checkpoint hash mismatch at step {step}")
        rows.extend(
            run_step(step, checkpoint, provenance, indices, halves, data, labels)
        )
        print(f"route entropy step {step} complete", flush=True)
    collector.write = None
    import csv

    keys = list(rows[0])
    with (HERE / "routing_entropy_dynamics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "routing_entropy_validation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "steps": [250, 500],
                "batches_per_step": 180,
                "shared_forward_per_batch": True,
                "optimizer_updates": 0,
                "state_and_rng_unchanged": True,
                "unknown_targets_excluded": True,
                "elapsed_generated": time.time(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
