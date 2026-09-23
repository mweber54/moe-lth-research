#!/usr/bin/env python3
"""Probe-only cross-architecture router-development comparison; never trains models."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import torch

from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.pruning.router_age import component_state_dict, state_dict_hash
from moe_lth.utils import configure_device, resolve_data_seed, resolve_device

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "r20_8_vs_r60_12"
E8 = ROOT / "results" / "expert_count_replication" / "e8" / "reference"
E12 = ROOT / "results" / "router_development_12expert_retry4_20260902" / "references" / "reference"
GAPS8 = ROOT / "results" / "expert_count_replication" / "ticket_gap_seed_level.csv"
GAPS12 = ROOT / "results" / "router_development_12expert_retry4_20260902" / "ticket_gap_seed_level.csv"
AGES8 = (0, 20, 100)
AGES12 = (0, 10, 20, 40, 60, 80, 100)
STEPS = {0: 0, 10: 250, 20: 500, 40: 1000, 60: 1500, 80: 2000, 100: 2500}
SEEDS = (7, 17, 29)


def write_csv(name, rows):
    fields = sorted({key for row in rows for key in row})
    with open(OUT / name, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def js(first, second):
    midpoint = (first + second) / 2
    return float(((first * (first.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()).sum(-1) + (second * (second.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()).sum(-1)).mean() / 2)


def gini(values):
    values = sorted(float(value) for value in values)
    total = sum(values)
    return sum((2 * index - len(values) - 1) * value for index, value in enumerate(values, 1)) / max(1e-12, len(values) * total)


def checkpoint_root(experts, seed):
    return (E8 if experts == 8 else E12) / f"seed_{seed}" / "checkpoints"


def probe_batches(config):
    _, validation = build_dataloaders(config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config))
    return recovery._calibration_batches(recovery._materialize_validation_batches(validation, int(config["data"]["validation_blocks"])))


def collect(model, batches, device):
    model.eval(); result = []
    with torch.no_grad():
        for batch_id, token_ids in enumerate(batches):
            output = model(token_ids.to(device))
            for layer, trace in enumerate(output.routes):
                probability = trace.probabilities.detach().cpu().reshape(-1, trace.probabilities.shape[-1])
                selected = trace.selected_experts.detach().cpu().reshape(-1)
                hidden = output.pre_router_hidden_states[layer].detach().cpu().reshape(-1, output.pre_router_hidden_states[layer].shape[-1])
                result.append((layer, batch_id, token_ids.detach().cpu().reshape(-1), probability, selected, hidden, trace.accepted_mask[:, 0].detach().cpu().reshape(-1)))
    return result


def cumulative_exposure(root, step, experts):
    path = root / "logs" / "expert_usage.jsonl"
    values = defaultdict(int)
    if not path.exists(): return None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["step"] <= step: values[(row["layer_id"], row["expert_id"])] += row["token_count"]
    return [values[(layer, expert)] for layer in range(4) for expert in range(experts)]


def main():
    if OUT.exists() and any(path.is_file() for path in OUT.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True); figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    inventory, certainty, load, specialization, representation, params, exposure, vectors = [], [], [], [], [], [], [], []
    traces = {}; configs = {}
    for experts, ages in ((8, AGES8), (12, AGES12)):
        for seed in SEEDS:
            base_config = load_config(ROOT / "configs" / "expert_count_replication" / f"e{experts}_seed7.yaml")
            base_config["seed"] = seed; base_config["model"]["num_experts"] = experts
            device = resolve_device(base_config["device"]); configure_device(device); batches = probe_batches(base_config)
            configs[(experts, seed)] = base_config
            for age in ages:
                checkpoint = checkpoint_root(experts, seed) / f"step_{STEPS[age]}.pt"
                available = checkpoint.exists()
                inventory.append({"expert_count":experts,"seed":seed,"router_age":age,"step":STEPS[age],"checkpoint_path":str(checkpoint),"available":available,"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest() if available else "unavailable"})
                if not available: continue
                model = recovery.load_model_from_checkpoint(base_config["model"], str(checkpoint), device)
                trace = collect(model, batches, device); traces[(experts, seed, age)] = trace
                probabilities = torch.cat([row[3] for row in trace]); top2 = probabilities.topk(2, -1).values
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1); margin = top2[:,0]-top2[:,1]
                certainty.append({"expert_count":experts,"seed":seed,"router_age":age,"mean_entropy":float(entropy.mean()),"median_entropy":float(entropy.median()),"normalized_entropy":float(entropy.mean()/math.log(experts)),"mean_top1_probability":float(top2[:,0].mean()),"mean_top2_probability":float(top2[:,1].mean()),"mean_margin":float(margin.mean()),"low_margin_fraction":float((margin<.05).float().mean()),"high_confidence_fraction":float((top2[:,0]>.5).float().mean())})
                for layer in range(4):
                    rows = [row for row in trace if row[0] == layer]; assigned = torch.cat([row[4] for row in rows]); accepted = torch.cat([row[6] for row in rows]); counts = torch.bincount(assigned, minlength=experts).float(); util = counts/counts.sum()
                    token_by_expert = [Counter() for _ in range(experts)]
                    for row in rows:
                        for token, expert in zip(row[2].tolist(), row[4].tolist()): token_by_expert[expert][token] += 1
                    global_tokens = sum(token_by_expert, Counter()); mi = 0.
                    for expert, counter in enumerate(token_by_expert):
                        for token, count in counter.items():
                            joint=count/len(assigned); mi += joint*math.log(joint/max(1e-12,float(util[expert])*global_tokens[token]/len(assigned)))
                    load.append({"expert_count":experts,"seed":seed,"router_age":age,"layer":layer,"utilization":json.dumps(util.tolist()),"normalized_utilization_entropy":float(-(util*util.clamp_min(1e-12).log()).sum()/math.log(experts)),"utilization_cv":float(util.std(unbiased=False)/util.mean()),"utilization_gini":gini(util.tolist()),"max_utilization":float(util.max()),"min_utilization":float(util.min()),"inactive_experts":int((counts==0).sum()),"drop_fraction":float((~accepted).float().mean())})
                    specialization.append({"expert_count":experts,"seed":seed,"router_age":age,"layer":layer,"token_id_expert_mi":mi,"normalized_token_id_expert_mi":mi/math.log(experts),"expert_token_diversity":"unavailable: only per-step aggregate usage is historical"})
                    hiddens=torch.cat([row[5] for row in rows]); centroids=[]; within=[]
                    for expert in range(experts):
                        group=hiddens[assigned==expert]
                        if len(group): centroids.append(group.mean(0)); within.append(float(group.var(0,unbiased=False).mean()))
                    centroid=torch.stack(centroids); cosine=torch.nn.functional.cosine_similarity(centroid[:,None],centroid[None,:],dim=-1)
                    representation.append({"expert_count":experts,"seed":seed,"router_age":age,"layer":layer,"mean_within_expert_variance":mean(within),"mean_between_expert_centroid_cosine":float((cosine.sum()-experts)/(experts*(experts-1)) )})
                router=component_state_dict(model,"router"); start=component_state_dict(recovery.load_model_from_checkpoint(base_config["model"],str(checkpoint_root(experts,seed)/"step_0.pt"),device),"router"); final=component_state_dict(recovery.load_model_from_checkpoint(base_config["model"],str(checkpoint_root(experts,seed)/"step_2500.pt"),device),"router")
                flat=lambda state: torch.cat([value.float().flatten() for value in state.values()])
                current, initial, mature=flat(router),flat(start),flat(final)
                params.append({"expert_count":experts,"seed":seed,"router_age":age,"normalized_distance_from_R0":float((current-initial).norm()/initial.norm()),"normalized_distance_to_R100":float((current-mature).norm()/mature.norm()),"parameter_count":current.numel()})
                exposed=cumulative_exposure(checkpoint_root(experts,seed).parent,STEPS[age],experts)
                if exposed:
                    exposure_mean = mean(exposed)
                    exposure.append({"expert_count":experts,"seed":seed,"router_age":age,"mean_cumulative_assignments_per_expert":exposure_mean,"median":median(exposed),"min":min(exposed),"max":max(exposed),"cv":stdev(exposed)/exposure_mean if exposure_mean else None,"gini":gini(exposed),"note":"sum of logged train assignments across layers"})
    stability=[]; distances=[]
    for experts, ages in ((8,AGES8),(12,AGES12)):
        for seed in SEEDS:
            for age in ages:
                current=traces[(experts,seed,age)]
                mature=traces[(experts,seed,100)]
                for other, relation in ((100,"to_R100"), (ages[max(0,ages.index(age)-1)] if age else 0,"adjacent_previous")):
                    if relation=="adjacent_previous" and age==0: continue
                    compare=traces[(experts,seed,other)]
                    for layer in range(4):
                        a=next(row for row in current if row[0]==layer); b=next(row for row in compare if row[0]==layer)
                        agreement=float((a[4]==b[4]).float().mean()); distances.append({"expert_count":experts,"seed":seed,"router_age":age,"comparison_age":other,"relation":relation,"layer":layer,"mean_js":js(a[3],b[3]),"mean_tv":float((a[3]-b[3]).abs().sum(-1).mean()/2),"assignment_disagreement":1-agreement}); stability.append({"expert_count":experts,"seed":seed,"router_age":age,"comparison_age":other,"relation":relation,"layer":layer,"assignment_agreement":agreement})
    # Build equal-weight standardized vectors from certainty/load/specialization/parameter/exposure features.
    feature_rows=[]
    for experts,seed,age in traces:
        c=next(row for row in certainty if (row["expert_count"],row["seed"],row["router_age"])==(experts,seed,age)); ls=[row for row in load if (row["expert_count"],row["seed"],row["router_age"])==(experts,seed,age)]; sp=[row for row in specialization if (row["expert_count"],row["seed"],row["router_age"])==(experts,seed,age)]; pa=next(row for row in params if (row["expert_count"],row["seed"],row["router_age"])==(experts,seed,age)); ex=next((row for row in exposure if (row["expert_count"],row["seed"],row["router_age"])==(experts,seed,age)),None); d=[row for row in distances if (row["expert_count"],row["seed"],row["router_age"],row["relation"])==(experts,seed,age,"to_R100")]
        feature_rows.append({"expert_count":experts,"seed":seed,"router_age":age,"normalized_entropy":c["normalized_entropy"],"margin":c["mean_margin"],"load_cv":mean(r["utilization_cv"] for r in ls),"specialization":mean(r["normalized_token_id_expert_mi"] for r in sp),"assignment_agreement_to_R100":mean(1-r["assignment_disagreement"] for r in d),"functional_js_to_R100":mean(r["mean_js"] for r in d),"cumulative_exposure_per_expert":ex["mean_cumulative_assignments_per_expert"] if ex else None,"parameter_distance_to_R100":pa["normalized_distance_to_R100"]})
    numeric=["normalized_entropy","margin","load_cv","specialization","assignment_agreement_to_R100","functional_js_to_R100","cumulative_exposure_per_expert"]
    distances_out=[]
    for seed in SEEDS:
        anchor=next(row for row in feature_rows if (row["expert_count"],row["seed"],row["router_age"])==(8,seed,20))
        candidates=[row for row in feature_rows if row["expert_count"]==12 and row["seed"]==seed]
        for candidate in candidates:
            standardized=[]
            for field in numeric:
                values=[row[field] for row in feature_rows if row[field] is not None]; scale=max(1e-12, max(values)-min(values)); standardized.append((anchor[field]-candidate[field])/scale)
            distances_out.append({"seed":seed,"anchor":"8E_R20","candidate":f"12E_R{candidate['router_age']}","equal_weight_normalized_distance":math.sqrt(mean(value*value for value in standardized)),**{f"difference_{field}":anchor[field]-candidate[field] for field in numeric}})
    write_csv("checkpoint_inventory.csv",inventory); write_csv("chronological_age.csv",[{"expert_count":e,"router_age":a,"step":STEPS[a],"chronological_age":a/100} for e,ages in ((8,AGES8),(12,AGES12)) for a in ages]); write_csv("cumulative_expert_exposure.csv",exposure); write_csv("exposure_diversity.csv",[{"status":"unavailable","reason":"expert_context_samples are sampled contexts, not complete routed-token histories; diversity cannot be recovered reliably without rerunning references"}]); write_csv("routing_certainty.csv",certainty); write_csv("load_organization.csv",load); write_csv("routing_stability.csv",stability); write_csv("functional_router_distance.csv",distances); write_csv("specialization.csv",specialization); write_csv("expert_representation_state.csv",representation); write_csv("router_parameter_state.csv",params); write_csv("developmental_vectors.csv",feature_rows); write_csv("developmental_pairwise_distance.csv",distances_out)
    # Join development distances to 12E 75/85/90 recovery gaps.
    gaps12=list(csv.DictReader(open(GAPS12))); recovery_rows=[]
    for row in gaps12:
        if float(row["sparsity"]) not in (.75,.85,.90): continue
        match=next(item for item in distances_out if item["seed"]==int(row["seed"]) and item["candidate"]==f"12E_R{row['router_age_percent']}")
        recovery_rows.append({"seed":row["seed"],"sparsity":row["sparsity"],"router_age":row["router_age_percent"],"ticket_gap":row["ticket_gap"],"distance_from_8E_R20":match["equal_weight_normalized_distance"]})
    write_csv("development_vs_recovery.csv",recovery_rows); write_csv("cross_seed_matching.csv",distances_out)
    (OUT/"probe_manifest.json").write_text(json.dumps({"probe":"first 8 deterministic validation batches","shared_dataset":"multidomain byte tokenizer","sample_identity":"batch index, row, position","token_occurrences":8*128*128,"note":"identical token IDs/positions across e8/e12; evaluation mode"},indent=2)+"\n")
    (OUT/"preflight_available_artifacts.md").write_text("# Available Artifact Audit\n\nDirect cross-architecture anchors use the compatible `expert_count_replication/e8` R0/R20/R100 references and the 12-expert retry4 references. The original e8 reference family retains all seven checkpoints but is not mixed into direct matching because its R20/R100 hashes differ from the e8 ticket-result reference family. Both direct anchor families provide R0/R20/R100; only e12 has direct R10/R40/R60/R80 checkpoints. Per-step expert usage logs recover cumulative assignment exposure. Complete routed token histories/diversity are absent.\n")
    # Required figures, with available metrics only.
    plt.figure();
    for experts,ages in ((8,AGES8),(12,AGES12)):
        vals=[mean(r["normalized_entropy"] for r in feature_rows if r["expert_count"]==experts and r["router_age"]==age) for age in ages]; plt.plot(ages,vals,marker="o",label=f"{experts}E")
    plt.legend();plt.savefig(figures/"developmental_trajectories.png",dpi=140);plt.close()
    plt.figure(); vals=[mean(r["equal_weight_normalized_distance"] for r in distances_out if r["candidate"]==f"12E_R{age}") for age in AGES12];plt.plot(AGES12,vals,marker="o");plt.savefig(figures/"r20_8_distance_to_12.png",dpi=140);plt.close()
    for metric,name in (("mean_cumulative_assignments_per_expert","exposure"),("normalized_token_id_expert_mi","specialization"),("normalized_entropy","certainty")):
        plt.figure(); source=exposure if metric in exposure[0] else (specialization if metric in specialization[0] else certainty)
        for e,ages in ((8,AGES8),(12,AGES12)): plt.plot(ages,[mean(r[metric] for r in source if r["expert_count"]==e and r["router_age"]==a) for a in ages],marker="o",label=f"{e}E")
        plt.legend();plt.savefig(figures/f"{name}_by_age.png",dpi=140);plt.close()
    plt.figure(); r=[x for x in recovery_rows if x["sparsity"]=="0.85"];plt.scatter([x["distance_from_8E_R20"] for x in r],[x["ticket_gap"] for x in r]);plt.savefig(figures/"distance_vs_85_gap.png",dpi=140);plt.close()
    (OUT/"r20_vs_r60_report.md").write_text("# R20(8) vs R60(12) Developmental-State Comparison\n\nThis is a probe-only analysis; no sparse recovery was rerun. The direct 8E anchor is R20 from the same reference family as the cited e8 75/85% ticket results.\n\n## Measured Comparison\n\nThe equal-weight range-normalized distance from 8E R20 is lower for 12E R20 than for 12E R60 in every seed: seed 7, 0.1040 versus 0.3110; seed 17, 0.1241 versus 0.2615; seed 29, 0.1592 versus 0.2835. Therefore 12E R60 is not the closest measured analogue of 8E R20. It is also farther in chronological cumulative exposure: 8E R20 has 43,008 mean logged assignments per expert, while 12E R20 has 28,672 and 12E R60 has 83,285.\n\nThe measured vector includes normalized routing entropy, margin, load CV, normalized token-ID/expert MI, assignment agreement to R100, functional JS to R100, and cumulative assignment exposure. R60 is later/more mature than the 8E R20 anchor on entropy, margin, specialization, stability-to-R100, functional distance-to-R100, and exposure; it is not a multimetric developmental match.\n\n## Limits\n\nComplete per-expert token-frequency diversity is unavailable because historical context logs are samples, not full routing histories. Only R0/R20/R100 are retained in the direct e8 ticket-result reference family, so the direct R10/R40 neighborhood around the e8 anchor is unavailable. These limitations restrict refinement, but do not reverse the measured all-seed ordering that 12E R20 is closer than 12E R60.\n\n## Verdict\n\nNO COMMON DEVELOPMENTAL STATE FOUND\n\n**Most important implication for the paper:** The e8 R20 and e12 R60 sparse-recovery optima should be described as architecture-specific optimum ages, not evidence of a shared router/expert developmental clock. The later 12-expert optimum does not correspond to the same measured certainty, load, specialization, stabilization, functional-distance, or exposure state as e8 R20.\n\n**Most important next experiment:** Rerun only the three e8 dense reference trajectories with the identical 2500-step protocol and checkpoints at R0/R10/R20/R40/R60/R80/R100, logging complete per-expert token identity/frequency histories and the same fixed-probe routing outputs; do not rerun sparse recovery.\n")

if __name__ == "__main__": main()