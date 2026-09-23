#!/usr/bin/env python3
"""Analyze the seed-7 high-resolution routing diagnostic without training."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "router_coadaptation_highres"
RUNS = ("R0", "R20", "R100")
STEPS = (0, 100, 250, 400, 500, 600, 700, 800, 900, 1000, 1100, 1250, 1500, 2000, 2500)

def csv_out(name, rows):
    fields = sorted({key for row in rows for key in row})
    with open(OUT / name, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def js(first, second):
    midpoint = [(a + b) / 2 for a, b in zip(first, second)]
    return sum(a * math.log(a / midpoint[i]) for i, a in enumerate(first) if a > 0) / 2 + sum(b * math.log(b / midpoint[i]) for i, b in enumerate(second) if b > 0) / 2

def main():
    rows = pq.read_table(OUT / "token_routing_trajectory.parquet").to_pylist()
    states = {(r["run"], r["step"], r["layer"], r["stable_token_id"]): r for r in rows}
    switch_rows, overlap_rows, distance_rows, transitions, expert_rows = [], [], [], [], []
    for run in RUNS:
        for layer in range(max(r["layer"] for r in rows) + 1):
            for start, end in zip(STEPS, STEPS[1:]):
                pairs = [(states[(run, start, layer, token)], states[(run, end, layer, token)]) for token in {r["stable_token_id"] for r in rows if r["run"] == run and r["step"] == start and r["layer"] == layer}]
                switched = [(a, b) for a, b in pairs if a["top1_expert"] != b["top1_expert"]]
                for before, after in switched:
                    later = [states.get((run, checkpoint, layer, before["stable_token_id"])) for checkpoint in STEPS if checkpoint > end]
                    later = [item for item in later if item is not None]
                    persistent_two = len(later) >= 2 and all(item["top1_expert"] == after["top1_expert"] for item in later[:2])
                    switch_rows.append({"run": run, "layer": layer, "start_step": start, "end_step": end, "stable_token_id": before["stable_token_id"], "source_expert": before["top1_expert"], "destination_expert": after["top1_expert"], "confidence_before": before["top1_probability"], "confidence_after": after["top1_probability"], "margin_before": before["margin"], "margin_after": after["margin"], "entropy_before": before["entropy"], "entropy_after": after["entropy"], "persistent_next_two": persistent_two})
                transitions.append({"run": run, "layer": layer, "start_step": start, "end_step": end, "total_tokens": len(pairs), "switches": len(switched), "switch_rate": len(switched) / max(1, len(pairs)), "persistent_next_two": sum(row["persistent_next_two"] for row in switch_rows if row["run"] == run and row["layer"] == layer and row["start_step"] == start and row["end_step"] == end)})
                for expert in range(len(pairs[0][0]["probabilities"])):
                    first = {a["stable_token_id"] for a, _ in pairs if a["top1_expert"] == expert}; second = {b["stable_token_id"] for _, b in pairs if b["top1_expert"] == expert}
                    overlap_rows.append({"run": run, "layer": layer, "expert": expert, "start_step": start, "end_step": end, "jaccard": len(first & second) / max(1, len(first | second)), "first_count": len(first), "second_count": len(second)})
                distance_rows.append({"run": run, "layer": layer, "start_step": start, "end_step": end, "mean_JS": sum(js(a["probabilities"], b["probabilities"]) for a, b in pairs) / len(pairs), "mean_TV": sum(sum(abs(x-y) for x,y in zip(a["probabilities"], b["probabilities"])) / 2 for a,b in pairs) / len(pairs), "assignment_disagreement": len(switched) / len(pairs)})
            for step in STEPS:
                members = [r for r in rows if r["run"] == run and r["layer"] == layer and r["step"] == step]
                for expert in range(len(members[0]["probabilities"])):
                    selected = [r for r in members if r["top1_expert"] == expert]
                    expert_rows.append({"run": run, "step": step, "layer": layer, "expert": expert, "probe_token_count": len(selected), "utilization": len(selected) / len(members), "accepted_count": sum(r["accepted_by_capacity"] for r in selected), "dropped_count": sum(r["dropped_or_overflowed"] for r in selected), "mean_assigned_confidence": sum(r["top1_probability"] for r in selected) / len(selected) if selected else None})
    pq.write_table(pa.Table.from_pylist(expert_rows), OUT / "expert_trajectory.parquet")
    csv_out("token_switch_analysis.csv", switch_rows)
    csv_out("token_set_overlap.csv", overlap_rows)
    csv_out("functional_router_distance.csv", distance_rows)
    transition_dir = OUT / "expert_transition_matrices"; transition_dir.mkdir(exist_ok=True)
    for run in RUNS:
        for layer in range(4):
            subset = [r for r in switch_rows if r["run"] == run and r["layer"] == layer and r["start_step"] == 500 and r["end_step"] == 1000]
            matrix = [[sum(r["source_expert"] == source and r["destination_expert"] == dest for r in subset) for dest in range(8)] for source in range(8)]
            with open(transition_dir / f"{run}_layer{layer}_500_1000.csv", "w", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(["source\\destination", *range(8)]); writer.writerows([[index, *line] for index, line in enumerate(matrix)])
    csv_out("persistent_switch_analysis.csv", transitions)
    manifest = list(csv.DictReader(open(OUT / "router_checkpoint_manifest.csv")))
    directions = []
    intervals = list(zip((0,400,500,600,700,800,900,1000,1100,1500),(400,500,600,700,800,900,1000,1100,1500,2500)))
    for start,end in intervals:
        updates = {}
        for run in RUNS:
            paths = {int(r["step"]): OUT / r["path"] for r in manifest if r["run"] == run}
            a, b = torch.load(paths[start], map_location="cpu", weights_only=True), torch.load(paths[end], map_location="cpu", weights_only=True)
            updates[run] = torch.cat([(b[key].float() - a[key].float()).flatten() for key in a])
        for left,right in (("R0","R20"),("R20","R100"),("R0","R100")):
            a,b=updates[left],updates[right]; directions.append({"start_step":start,"end_step":end,"left":left,"right":right,"left_norm":float(a.norm()),"right_norm":float(b.norm()),"cosine_similarity":float(torch.dot(a,b)/(a.norm()*b.norm()).item())})
    csv_out("router_update_direction.csv", directions)
    gradients=[]
    for run in RUNS:
        for line in open(OUT / "runs" / run / "gradient_stats" / "gradient_stats.jsonl"):
            r=json.loads(line)
            if 400 <= r["step"] <= 1100:
                gradients.append({"run":run,"step":r["step"],"gradient_valid":r.get("gradient_valid"),"router_grad_norm":r.get("router_grad_norm"),"expert_grad_norm":r.get("expert_grad_norm"),"per_layer_grad_norm":json.dumps(r.get("per_layer_grad_norm")),"per_expert_grad_norm":json.dumps(r.get("per_expert_grad_norm"))})
    csv_out("gradient_trajectory.csv", gradients)
    coupling=[]
    for run in RUNS:
        for layer in range(4):
            d=next(r for r in distance_rows if r["run"]==run and r["layer"]==layer and r["start_step"]==500 and r["end_step"]==1000)
            g=[r["expert_grad_norm"] for r in gradients if r["run"]==run and r["gradient_valid"] and isinstance(r["expert_grad_norm"],(int,float))]
            coupling.append({"run":run,"layer":layer,"assignment_disagreement_500_1000":d["assignment_disagreement"],"functional_JS_500_1000":d["mean_JS"],"expert_gradient_mean_400_1100":sum(g)/len(g) if g else None,"note":"descriptive alignment; no mediation claim"})
    csv_out("routing_gradient_coupling.csv", coupling)
    (OUT / "seed7_mechanism_report.md").write_text("# Seed-7 Mechanism Diagnostic\n\nPending evidence summary: inspect the generated CSVs before assigning a verdict. This analysis preserves all token-level switches, persistence flags, transition matrices, functional distances, and update directions. No seeds 17/29 were launched.\n")

if __name__ == "__main__": main()