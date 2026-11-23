"""Publish the archived step-250 sign observations while early replay runs."""
from audit import OUT,SOURCE,p
from common_v2 import load
from trace import vector

table=load("v3_retained_report",OUT/"report.py").table
torch=p.torch


def main():
    manifest=p.read(OUT/"audit_manifest.json")
    rows=[]
    jobs={(s["job"]["seed"],s["job"]["order"],s["job"]["draw"],s["job"]["arm"]):s for s in manifest["jobs"]}
    for entry in manifest["inputs"]:
        seed,order=entry["seed"],entry["order"]
        start,_=p.baseline_state(entry)
        untouched=p.N._read_model_payload(p.Path(jobs[seed,order,0,"untouched"]["paths"]["250"]))["model"]
        for draw in (1,2,3):
            flip=torch.load(SOURCE/f"flips_{seed}_{order}_imp_{draw}.pt",weights_only=True)
            for arm in ("high","random"):
                other=flip["random" if arm=="high" else "high"]
                picked=flip[arm]
                shared=torch.cat([torch.isin(index,other[name]) for name,index in picked.items()])
                model=p.N._read_model_payload(p.Path(jobs[seed,order,draw,arm]["paths"]["250"]))["model"]
                original,baseline,observed=(vector(state,picked) for state in (start,untouched,model))
                for label,group in (("shared",shared),("exclusive",~shared)):
                    valid=(baseline!=0)&(observed!=0)
                    changed=(baseline!=original)&(baseline!=0)&(original!=0)
                    rows.append(dict(seed=seed,order=order,draw=draw,arm=arm,coordinates=label,step=250,
                        count=int(group.sum()),restored=int((group&(original!=0)&(observed==original)).sum()),
                        near_zero=int((group&(observed==0)).sum()),
                        untouched_agreement=int((group&valid&(observed==baseline)).sum()),
                        untouched_near_zero=int((group&(baseline==0)).sum()),
                        untouched_changed_from_start=int((group&changed).sum())))
                del model
        table("retained_step250_signs.csv",rows)
        print("Archived step 250",seed,order,flush=True)


if __name__=="__main__":
    main()
