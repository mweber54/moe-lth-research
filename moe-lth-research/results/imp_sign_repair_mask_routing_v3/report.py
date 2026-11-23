"""Report verified early replays and the causal repair interaction."""
import csv
from collections import defaultdict

from audit import OUT, EARLY, p


def table(name,rows):
    if not rows:
        return
    path=OUT/name
    temp=path.with_suffix(".tmp")
    with temp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    manifest=p.read(OUT/"audit_manifest.json")
    rows,readouts,interactions=[],[],[]
    conditions=defaultdict(dict)
    for source in manifest["jobs"]:
        job=source["job"]
        name=p.job_id(job)
        for mode in ("free","repair"):
            root=OUT/mode/name
            if not (root/"result.json").is_file():
                continue
            result=p.read(root/"result.json")
            assert result["job"]==job and result["mode"]==mode
            assert result["attempts"]==(250 if mode=="free" else 2500)
            for step in EARLY:
                item=p.read(root/"attempt"/f"readout_{step:04d}.json")
                readouts.append(dict(**job,mode=mode,**item,stream_hash=result["readout_stream_hash"]))
                if job["draw"]==1:
                    conditions[job["seed"],job["order"],"readout",step][mode,job["arm"]]=item["ce"]
            records=source["records"] if mode=="free" else [
                dict(step=step,cross_entropy=p.read(root/"attempt"/f"full_{step:04d}.json")["ce"])
                for step in p.STEPS]
            for rec in records:
                row=dict(**job,mode=mode,step=rec["step"],ce=rec["cross_entropy"],attempts=2500,
                         applied=len(source["applied_steps"]) if mode=="free" else result["applied"],
                         replay_attempts=result["attempts"] if mode=="free" else 0,
                         replay_applied=result["applied"] if mode=="free" else 0,ce_source="archived_full_validation" if mode=="free" else "new_full_validation")
                rows.append(row)
                if job["draw"]==1:
                    conditions[job["seed"],job["order"],"full",row["step"]][mode,job["arm"]]=row["ce"]
    for (seed,order,readout,step),arms in sorted(conditions.items()):
        if len(arms)!=4:
            continue
        high=arms["repair","high"]-arms["free","high"]
        random=arms["repair","random"]-arms["free","random"]
        interactions.append(dict(seed=seed,order=order,readout=readout,step=step,high_constraint_penalty=high,
                                 random_constraint_penalty=random,repair_interaction=high-random))
    table("recovery_table.csv",rows)
    table("fixed_readouts.csv",readouts)
    table("repair_interaction.csv",interactions)
    print("Verified",len({(r["seed"],r["order"],r["draw"],r["arm"],r["mode"]) for r in rows}),"recovery traces",flush=True)
