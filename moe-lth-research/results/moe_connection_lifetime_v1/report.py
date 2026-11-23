"""Validate study-local endpoints and write paired scientific tables."""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from functools import lru_cache
from scipy.stats import t as student

from stage import HERE,FRESH,ORDERS,branch,check_seal,read

assert Path(__file__).resolve().parent == HERE


def table(name, rows):
    if not rows:
        return
    path=HERE/(name+".csv")
    temp=path.with_suffix(".csv.tmp")
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with temp.open("w") as file:
        writer=csv.DictWriter(file,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp,path)


@lru_cache
def parameter_names():
    config=read(HERE/"native_provenance.json")["config"]["model"]
    with torch.device("meta"):
        model=branch.TinyMoELanguageModel(config)
    return [n for n,_ in model.named_parameters()]


def validate_saved(folder,rec):
    ck=torch.load(folder/"checkpoint.pt",map_location="cpu",mmap=True,weights_only=False)
    ident=rec["identity"]
    assert ck["identity"]==ident and ck["global_step"]==ck["next_batch_index"]==2500
    assert ck["scheduler"] is None and ck["losses"]==rec["measurements"]
    assert ck["applied_since_branch"]+len(ck["overflow_attempts"])==2500-ident["t"]
    assert ck["live_parameter_tokens"]==rec["live_parameter_tokens"]
    assert len(ck["cuda_rng"])==1 and ck["scaler"]["scale"]>0
    names=parameter_names()
    ids=ck["optimizer"]["param_groups"][0]["params"]
    assert len(names)==len(ids)
    states={n:ck["optimizer"]["state"][i] for n,i in zip(names,ids) if i in ck["optimizer"]["state"]}
    assert int(states["token_embedding.weight"]["step"])==rec["successful_updates"]
    assert all(0<=int(s["step"])<=rec["successful_updates"] for s in states.values())
    if ident["arm"]!="dense":
        if folder.parent.name=="random_control":
            path=HERE/"control_groups/sets.pt"
        else:
            path=HERE/"groups"/f"seed{ident['seed']}_{ident['order']}"/"sets.pt"
        assert branch.sha(path)==ident["sets_sha256"]
        import hashlib
        digest=hashlib.sha256()
        for n,g in sorted(torch.load(path,weights_only=True).items()):
            idx=g["M"]
            w=ck["model"][n].flatten()
            assert torch.count_nonzero(w[idx])==0
            for k in ("exp_avg","exp_avg_sq","max_exp_avg_sq"):
                if k in states[n]:
                    assert torch.count_nonzero(states[n][k].flatten()[idx])==0
            mask=torch.ones(w.numel(),dtype=torch.bool)
            mask[idx]=False
            digest.update(n.encode()+b"\0")
            digest.update(mask.numpy().tobytes())
        assert digest.hexdigest()==ident["final_mask_sha256"]


def endpoint(folder):
    p=folder/"verified.json"
    if not p.is_file():
        return None
    record=read(p)
    assert record["checkpoint_sha256"]==branch.sha(folder/"checkpoint.pt")
    assert record["attempts"]==2500
    assert record["identity"]["arm"] in ("H","R","0","dense")
    assert len(record["measurements"]["2500"]["sequence_loss_sums"])>0
    validate_saved(folder,record)
    return record


def interval(values,confidence=.95):
    values=np.asarray(values,dtype=float)
    return float(student.ppf((1+confidence)/2,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))


def development_tables():
    from stage import DEV
    cells=[]
    controls=[]
    for seed in DEV:
        for order in ORDERS:
            for t in (250,500):
                seal=HERE/"seals"/f"seed{seed}_{order}_{t}.json"
                if not seal.exists():
                    continue
                receipt=check_seal(seed,order,t)
                data=read(Path(receipt["forecast"]))
                batches=data["batches"]
                rows=dict(seed=seed,order=order,t=t,update_contexts=len({b["update_batch"] for b in batches}),
                          readout_contexts=len({b["readout_batch"] for b in batches}),scoring_seconds=data["elapsed_s"])
                for key,metrics in data["effects"].items():
                    rows.update({key+"_"+k:v for k,v in metrics.items()})
                for arm in ("H","R","0"):
                    rows["skips_"+arm]=sum(not b["attempt"][arm]["applied"] for b in batches)
                rows["duplicate_replay_floor"]=batches[0]["duplicate_floor"]
                cells.append(rows)
                b=batches[0]
                for arm,c in b["controls"].items():
                    controls.append(dict(seed=seed,order=order,t=t,arm=arm,update_batch=b["update_batch"],
                        readout_batch=b["readout_batch"],ordinary_partner_ce=b["partner"][arm]["ce"],
                        blocked_partner_ce=c["block"][1]["ce"],shuffled_partner_ce=c["shuffle"][1]["ce"],
                        ordinary_whole_ce=b["whole"][arm]["ce"],blocked_whole_ce=c["block"][2]["ce"],
                        shuffled_whole_ce=c["shuffle"][2]["ce"],ordinary_applied=b["attempt"][arm]["applied"],
                        blocked_applied=c["block"][0]["applied"],shuffled_applied=c["shuffle"][0]["applied"],**c["audit"]))
    table("development_forecasts",cells)
    table("development_feedback_controls",controls)


def analyze():
    development_tables()
    rows=[]
    trajectory=[]
    runrows=[]
    missing=[]
    domain_rows=[]
    seedrows=[]
    summary=[]
    comparisons=[]
    coverage=[]
    timings=[]
    counts=dict(held_out=0,dense=0,random_control=0)
    control={arm:endpoint(HERE/"random_control"/arm) for arm in ("H","R")}
    counts["random_control"]=sum(z is not None for z in control.values())
    control_gap=None
    if all(control.values()):
        control_gap=control["R"]["measurements"]["2500"]["ce"]-control["H"]["measurements"]["2500"]["ce"]
    floor=read(HERE/"freeze.json")["replay_floor"] if (HERE/"freeze.json").exists() else None
    for seed in FRESH:
        for o in ORDERS:
            cell=f"seed{seed}_{o}"
            root=HERE/"branches"/cell
            dense=endpoint(root/"250"/"dense")
            counts["dense"]+=int(dense is not None)
            if dense is None:
                missing.append(f"{cell}/dense")
            for t in (250,500):
                arms={}
                for a in ("H","R","0"):
                    rec=endpoint(root/str(t)/a)
                    if rec is None:
                        missing.append(f"{cell}/{t}/{a}")
                        continue
                    arms[a]=rec
                    counts["held_out"]+=1
                    ident=rec["identity"]
                    assert (ident["seed"],ident["order"],ident["arm"],ident["t"])==(seed,o,a,t)
                    runrows.append(dict(cell=cell,t=t,arm=a,ce=rec["measurements"]["2500"]["ce"],
                       applied=rec["successful_updates"],overflows=len(rec["overflow_attempts"]),
                       elapsed_s=rec["elapsed_seconds"],live_parameter_tokens=rec["live_parameter_tokens"],
                       mask=ident["final_mask_sha256"],checkpoint=rec["checkpoint_sha256"]))
                if len(arms)<3:
                    continue
                assert len({rec["identity"]["final_mask_sha256"] for rec in arms.values()})==1
                assert len({rec["identity"]["train_hash"] for rec in arms.values()})==1
                assert len({rec["identity"]["validation_hash"] for rec in arms.values()})==1
                assert len({rec["measurements"]["pre_first"]["ce"] for rec in arms.values()})==1
                assert arms["H"]["live_parameter_tokens"]==arms["R"]["live_parameter_tokens"]
                seal=check_seal(seed,o,t)
                z=read(Path(seal["forecast"]))
                m={a:r["measurements"] for a,r in arms.items()}
                keys=m["H"].keys()
                for k in keys:
                    rec=dict(seed=seed,order=o,t=t,at=k)
                    for a in m:
                        rec[a]=m[a][k]["ce"]
                    if dense:
                        dense_key={"pre_first":"pre_first" if t==250 else "500",
                                   "post_first":"post_first" if t==250 else "500",
                                   "pre_second":str(t+250),"post_second":str(t+250)}.get(k,k)
                        rec["dense"]=dense["measurements"][dense_key]["ce"]
                        for a in m:
                            rec["excess_"+a]=rec[a]-rec["dense"]
                        if k=="pre_first":
                            assert rec["dense"]==rec["H"]
                    rec["Y"]=rec["R"]-rec["H"]
                    trajectory.append(rec)
                y=m["R"]["2500"]["ce"]-m["H"]["2500"]["ce"]
                p=z["effects"]["partner"]["P"]
                deltas=(np.asarray(m["R"]["2500"]["sequence_loss_sums"])-m["H"]["2500"]["sequence_loss_sums"])/128
                assert abs(deltas.mean()-y)<1e-12
                labels=np.array(read(HERE/"validation_domains.json")["target_counts_by_sequence"])
                assert len(labels)==len(deltas)
                for tag,selected in (("TinyStories",labels[:,0]==128),("WikiText103",labels[:,1]==128),("mixed",labels.max(axis=1)<128)):
                    v=deltas[selected]
                    domain_rows.append(dict(seed=seed,order=o,t=t,domain=tag,sequences=len(v),Y=v.mean() if len(v) else None,
                              paired_se=v.std(ddof=1)/math.sqrt(len(v)) if len(v)>1 else None))
                jump=lambda a:m[a]["post_second"]["ce"]-m[a]["pre_second"]["ce"]
                row=dict(seed=seed,order=o,t=t,P=p,Y=y,second_jump_R_minus_H=jump("R")-jump("H"),
                         sequence_paired_se=deltas.std(ddof=1)/np.sqrt(len(deltas)),
                         sequence_paired_95_half_width=interval(deltas),scoring_s=z["elapsed_s"],
                         I_H=z["effects"]["partner"]["I_H"],I_R=z["effects"]["partner"]["I_R"])
                for a in ("H","R"):
                    row["cost_"+a]=m["0"]["2500"]["ce"]-m[a]["2500"]["ce"]
                    seq_cost=(np.asarray(m["0"]["2500"]["sequence_loss_sums"])-m[a]["2500"]["sequence_loss_sums"])/128
                    row["cost_"+a+"_sequence_se"]=seq_cost.std(ddof=1)/math.sqrt(len(seq_cost))
                if dense:
                    for a in m:
                        row["excess_"+a]=m[a]["2500"]["ce"]-dense["measurements"]["2500"]["ce"]
                for rule,value in dict(partner=p,**z["baselines"]).items():
                    threshold=floor if rule in ("partner","whole","damage") else 0.
                    vote=0 if abs(value)<=threshold else int(np.sign(value))
                    row["score_"+rule]=value
                    row["vote_"+rule]=vote
                    row["gain_"+rule]=vote*y
                    if rule=="partner":
                        row["abstain"]=int(vote==0)
                        row["agreement"]=None if not vote or y==0 else int(vote==np.sign(y))
                rows.append(row)
    for t in (250,500):
        for seed in FRESH:
            pair=[r for r in rows if r["seed"]==seed and r["t"]==t]
            if len(pair)==2:
                metrics=["I_H","I_R","P","Y","cost_H","cost_R","second_jump_R_minus_H"]+[k for k in pair[0] if k.startswith("gain_")]
                seedrows.append(dict(t=t,seed=seed,**{k:np.mean([r[k] for r in pair]) for k in metrics}))
        entries=[r for r in seedrows if r["t"]==t]
        if len(entries)==3:
            rules=[k.removeprefix("gain_") for k in entries[0] if k.startswith("gain_")]
            eligible=[r for r in rows if r["t"]==t]
            for rule in rules:
                votes=[r["vote_"+rule] for r in eligible]
                coverage.append(dict(t=t,rule=rule,covered=sum(bool(x) for x in votes),
                      abstained=sum(x==0 for x in votes),total=len(votes),
                      direction_matches=sum(int(v!=0 and v==np.sign(r["Y"])) for v,r in zip(votes,eligible))))
                if rule!="partner":
                    diff=[r["gain_partner"]-r["gain_"+rule] for r in entries]
                    comparisons.append(dict(t=t,rule=rule,mean_increment=np.mean(diff),
                        seed_95_half_width=interval(diff),identical_votes=all(r["vote_partner"]==r["vote_"+rule] for r in eligible)))
            for k in entries[0]:
                if k in ("seed","t"):
                    continue
                vals=[r[k] for r in entries]
                row=dict(t=t,metric=k,mean=np.mean(vals),seed_95_half_width=interval(vals))
                if k.startswith("cost_"):
                    row["pointwise_upper_95"]=row["mean"]+student.ppf(.95,2)*np.std(vals,ddof=1)/math.sqrt(3)
                    row["within_margin"]=bool(row["pointwise_upper_95"]<.001)
                summary.append(row)
    for path in sorted((HERE/"timing").glob("*.json")):
        timings.append(read(path))
    for name,items in [("job_timing",timings),("coverage",coverage),("incremental_gains",comparisons),("domains",domain_rows),("runs",runrows),("paired",rows),("trajectories",trajectory),("seed_means",seedrows),("summary",summary)]:
        table(name,items)
    branch.save_json(HERE/"status.json",dict(counts=counts,missing=missing,control_endpoint_gap=control_gap))
    if not missing and counts["random_control"]==2:
        text=["# Connection lifetime pilot","",
              "All 36 pruned endpoints, six dense references and two random controls verified.",
              f"The 7/A random-control endpoint contrast was {control_gap:+.9g} nats/token.",""]
        for t in (250,500):
            subset=[r for r in rows if r["t"]==t]
            by_seed=[r for r in seedrows if r["t"]==t]
            rules=[k.removeprefix("gain_") for k in by_seed[0] if k.startswith("gain_")]
            gains={k:np.mean([r["gain_"+k] for r in by_seed]) for k in rules}
            best=max((k for k in rules if k not in ("partner","whole")),key=lambda k:gains[k])
            hit=sum(r["agreement"] or 0 for r in subset)
            coverage=sum(1-r["abstain"] for r in subset)
            text.extend([f"## Decision at attempt {t}","",
                f"The partner forecast selected an arm in {coverage}/6 cells and agreed with {hit} endpoint directions. "
                f"Its mean gain over the opposite order was {gains['partner']:+.9g} nats/token. "
                f"The highest observed simple-comparator gain was {best} at {gains[best]:+.9g}.",
                "The A/B-averaged partner gains for seeds 37, 43 and 53 were "+", ".join(f"{r['gain_partner']:+.9g}" for r in by_seed)+".",""])
            identical=[r["rule"] for r in comparisons if r["t"]==t and r["identical_votes"]]
            if identical:
                text.append("Identical votes across the six cells leave incremental value over "+", ".join(identical)+" unresolved.")
            for a in ("H","R"):
                row=next(r for r in summary if r["t"]==t and r["metric"]=="cost_"+a)
                conclusion="below" if row["within_margin"] else "at or above"
                text.append(f"For {a}-only, earlier-cut cost averaged {row['mean']:+.9g}. Its pointwise one-sided 95% upper bound of {row['pointwise_upper_95']:+.9g} was {conclusion} the provisional 0.001-nat/token margin.")
            text.append("")
        text.extend(["The experimental unit was a matched scalar-connection set and its recipient covered essentially the full final-layer input-weight block. The seed count for each timing estimate was three. See paired.csv for arm-specific local effects, seed_means.csv for seed heterogeneity, domains.csv for sequence coverage and runs.csv for elapsed continuation time and parameter-token counts."])
        (HERE/"verdict.md").write_text("\n".join(text)+"\n")
    print(json.dumps(counts),flush=True)


if __name__=="__main__":
    analyze()
