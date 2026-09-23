#!/usr/bin/env python3
"""Aggregate completed expert-count replication records without running experiments."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "expert_count_replication"
AGES = (0, 20, 100)
SPARSITIES = (0.75, 0.85, 0.95)


def read(name):
    with open(OUT / name) as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    fields = sorted({key for row in rows for key in row})
    with open(OUT / name, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    sparse = read("sparse_results.csv")
    dense = read("dense_baselines.csv")
    seed_rows = []
    for row in sparse:
        seed_rows.append({"expert_count": int(row["expert_count"]), "seed": int(row["seed"]), "sparsity": float(row["sparsity"]), "router_age": int(row["router_age_percent"]), "sparse_final_loss": float(row["final_validation_loss"]), "dense_final_loss": float(row["matched_dense_final_loss"]), "ticket_gap": float(row["ticket_gap"]), "mask_hash": row["mask_hash"], "initial_router_hash": row["initial_router_state_hash"], "final_router_hash": row["final_router_state_hash"], "training_sequence_hash": row["training_batch_sequence_hash"], "validation_sequence_hash": row["validation_batch_sequence_hash"], "integrity_checks_passed": row["integrity_checks_passed"]})
    write("ticket_gap_seed_level.csv", seed_rows)
    grouped = defaultdict(list)
    for row in seed_rows: grouped[(row["expert_count"], row["sparsity"], row["router_age"])].append(row)
    aggregates = []
    for key, rows in sorted(grouped.items()):
        gaps = [row["ticket_gap"] for row in rows]
        aggregates.append({"expert_count": key[0], "sparsity": key[1], "router_age": key[2], "mean_ticket_gap": mean(gaps), "sd_ticket_gap": stdev(gaps) if len(gaps)>1 else 0.0, "num_seeds": len(gaps), "seed_values": json.dumps({str(row["seed"]): row["ticket_gap"] for row in rows})})
    write("ticket_gap_aggregate.csv", aggregates)
    contrasts = []
    for experts in (8, 12):
        for sparsity in SPARSITIES:
            by_age = {age: {row["seed"]: row["ticket_gap"] for row in seed_rows if row["expert_count"] == experts and row["sparsity"] == sparsity and row["router_age"] == age} for age in AGES}
            for seed in sorted(by_age[0]):
                contrasts.append({"expert_count": experts,"sparsity": sparsity,"seed": seed,"R0_gap":by_age[0][seed],"R20_gap":by_age[20][seed],"R100_gap":by_age[100][seed],"R0_to_R20_improvement":by_age[0][seed]-by_age[20][seed],"R100_to_R20_improvement":by_age[100][seed]-by_age[20][seed]})
    write("router_age_contrasts.csv", contrasts)
    summary = []
    classifications = {}
    for experts in (8,12):
        for sparsity in SPARSITIES:
            rows=[row for row in contrasts if row["expert_count"]==experts and row["sparsity"]==sparsity]
            summary.append({"expert_count":experts,"sparsity":sparsity,**{f"R{age}_gap":mean(row[f"R{age}_gap"] for row in rows) for age in AGES},"R0_to_R20_improvement":mean(row["R0_to_R20_improvement"] for row in rows),"R100_to_R20_improvement":mean(row["R100_to_R20_improvement"] for row in rows)})
        at85=[row for row in contrasts if row["expert_count"]==experts and row["sparsity"]==0.85]
        r0=sum(row["R0_to_R20_improvement"]>0 for row in at85); r100=sum(row["R100_to_R20_improvement"]>=0 for row in at85)
        classifications[experts] = "STRONG REPLICATION" if r0 >= 2 and r100 >= 2 else "PARTIAL REPLICATION" if r0 >= 2 else "NO REPLICATION"
    write("cross_architecture_summary.csv", summary)
    figures=OUT/"figures"; figures.mkdir(exist_ok=True)
    for experts in (8,12):
        rows=[r for r in summary if r["expert_count"]==experts and r["sparsity"]==0.85]
        plt.figure(); plt.plot(["R0","R20","R100"],[rows[0]["R0_gap"],rows[0]["R20_gap"],rows[0]["R100_gap"]],marker="o"); plt.ylabel("Mean ticket gap"); plt.title(f"e{experts}: 85% sparsity"); plt.savefig(figures/f"ticket_gap_router_age_e{experts}_85pct.png",dpi=160,bbox_inches="tight"); plt.close()
    plt.figure()
    for experts in (8,12):
        for age in AGES: plt.plot(SPARSITIES,[next(r for r in summary if r["expert_count"]==experts and r["sparsity"]==s)[f"R{age}_gap"] for s in SPARSITIES],marker="o",label=f"e{experts} R{age}")
    plt.legend(); plt.xlabel("Expert sparsity"); plt.ylabel("Mean ticket gap"); plt.savefig(figures/"ticket_gap_by_sparsity.png",dpi=160,bbox_inches="tight"); plt.close()
    plt.figure()
    for experts in (8,12): plt.plot(SPARSITIES,[next(r for r in summary if r["expert_count"]==experts and r["sparsity"]==s)["R0_to_R20_improvement"] for s in SPARSITIES],marker="o",label=f"e{experts}")
    plt.axhline(0,color="black",lw=.8); plt.legend(); plt.savefig(figures/"r0_to_r20_improvement.png",dpi=160,bbox_inches="tight"); plt.close()
    plt.figure()
    for experts in (8,12):
        rows=[r for r in contrasts if r["expert_count"]==experts and r["sparsity"]==.85]
        plt.scatter([experts-.12]*len(rows),[r["R0_gap"] for r in rows],label=f"e{experts} R0"); plt.scatter([experts+.12]*len(rows),[r["R20_gap"] for r in rows],marker="x",label=f"e{experts} R20")
    plt.legend(); plt.savefig(figures/"seed_paired_r0_r20_85pct.png",dpi=160,bbox_inches="tight"); plt.close()
    plt.figure(); labels=[f"e{r['expert_count']} s{r['sparsity']:.2f}" for r in summary]; plt.bar(range(len(labels)),[r["R0_to_R20_improvement"] for r in summary]); plt.xticks(range(len(labels)),labels,rotation=45,ha="right"); plt.axhline(0,color="black",lw=.8); plt.savefig(figures/"cross_architecture_summary.png",dpi=160,bbox_inches="tight"); plt.close()
    (OUT / "replication_report.md").write_text("# Expert-Count Replication Report\n\nThis report is generated only after all 72 new recoveries complete with matching dense baselines. Architecture-specific classifications: " + ", ".join(f"e{k}: **{v}**" for k,v in classifications.items()) + ".\n\nAt 85%, inspect seed-level paired contrasts in `router_age_contrasts.csv`; positive values mean R20 improves on the stated comparator. This is cross-architecture replication evidence only, not universal generalization. The original e8 study remains historical context and is not altered or mixed into the new statistics.\n\n## Overall Verdict\n\n`" + ("CROSS-ARCHITECTURE REPLICATION SUPPORTED" if all(v=="STRONG REPLICATION" for v in classifications.values()) else "PARTIAL CROSS-ARCHITECTURE REPLICATION" if any(v in {"STRONG REPLICATION","PARTIAL REPLICATION"} for v in classifications.values()) else "NO CROSS-ARCHITECTURE REPLICATION") + "`\n")

if __name__ == "__main__": main()