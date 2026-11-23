"""Report paired cross-mask differences and fixed-flip routing interactions."""
from statistics import mean

from audit import OUT, EARLY, p
from later import jobs
from common_v2 import load

table = load("v3_study_report",OUT/"report.py").table
trace = load("v3_sign_trace",OUT/"trace.py")
positions, vector = trace.positions, trace.vector

torch=p.torch


def main(stage):
    root=OUT/stage
    manifest=p.read(root/"manifest.json")
    digest=p.digest(root/"manifest.json")
    rows=[]
    signs=[]
    data={}
    paths={}
    for entry,source,mode in jobs(stage,manifest):
        job,name=source["job"],source["name"]
        folder=root/"jobs"/name
        if not (folder/"result.json").exists():
            continue
        result=p.read(folder/"result.json")
        assert result["selection_manifest"]==digest and result["job"]==job
        assert result["attempts"]==2500 and result["applied"]<=2500
        condition="scaled" if source.get("scaled") else "native"
        key=(job["seed"],job["order"],job["mask"],condition,job["arm"])
        paths[key]=folder/"attempt",source,entry
        for readout,steps in (("full",p.STEPS),("readout",EARLY)):
            for step in steps:
                value=p.read(folder/"attempt"/f"{readout}_{step:04d}.json")["ce"]
                data[*key,readout,step]=value
                rows.append(dict(**job,recovery=condition,readout=readout,step=step,ce=value,
                                 attempts=2500,applied=result["applied"]))
    if stage=="connectivity" and manifest["target"]==.34207:
        audit=p.read(OUT/"audit_manifest.json")
        for item in audit["jobs"]:
            job=item["job"]
            if job["draw"] not in (0,1):
                continue
            key=job["seed"],job["order"],"imp","native",job["arm"]
            for record in item["records"]:
                step=record["step"]
                data[*key,"full",step]=record["cross_entropy"]
            for step in EARLY:
                path=OUT/"free"/p.job_id(job)/"attempt"/f"readout_{step:04d}.json"
                if path.is_file():
                    data[*key,"readout",step]=p.read(path)["ce"]
    table(f"{stage}_per_run.csv",rows)
    pairs={}
    for key,ce in sorted(data.items()):
        seed,order,source,condition,arm,readout,step=key
        if arm!="high" or (seed,order,source,condition,"random",readout,step) not in data:
            continue
        random=data[seed,order,source,condition,"random",readout,step]
        baseline_source=source if stage=="connectivity" else condition
        baseline=data.get((seed,order,baseline_source,condition,"untouched",readout,step))
        pairs[seed,order,source,condition,readout,step]=dict(seed=seed,order=order,source=source,
            recovery=condition,readout=readout,step=step,high_ce=ce,random_ce=random,baseline_ce=baseline,
            high_minus_random=ce-random,high_minus_baseline=None if baseline is None else ce-baseline,
            random_minus_baseline=None if baseline is None else random-baseline)
    table(f"{stage}_pairs.csv",list(pairs.values()))
    contrasts=[]
    for (seed,order,source,condition,readout,step),row in pairs.items():
        def contrast(s,c):
            value=pairs.get((seed,order,s,c,readout,step))
            return value["high_minus_random"] if value else None
        if stage=="connectivity" and source=="imp":
            a,b=(contrast(f"degree{i}","native") for i in (1,2))
            if None not in (a,b):
                contrasts.append(dict(seed=seed,order=order,step=step,readout=readout,
                    imp=row["high_minus_random"],degree1=a,degree2=b,
                    interaction=row["high_minus_random"]-mean((a,b))))
        if stage=="routing" and source==condition=="native":
            nn,ns,sn,ss=(contrast(s,c) for s,c in (("native","native"),("native","scaled"),
                                                  ("scaled","native"),("scaled","scaled")))
            if None not in (nn,ns,sn,ss):
                contrasts.append(dict(seed=seed,order=order,step=step,readout=readout,
                    native_native=nn,native_scaled=ns,scaled_native=sn,scaled_scaled=ss,
                    interaction=(nn-ns)-(sn-ss)))
    table(f"{stage}_interactions.csv",contrasts)
    print(stage,len(paths),"verified recoveries",flush=True)


def sign_agreement(stage):
    root=OUT/stage
    manifest=p.read(root/"manifest.json")
    if stage!="connectivity":
        return
    complete={}
    for entry,source,_ in jobs(stage,manifest):
        job=source["job"]
        folder=root/"jobs"/source["name"]
        if (folder/"result.json").is_file():
            complete[job["seed"],job["order"],job["mask"],job["arm"]]=folder/"attempt",source,entry
    rows=[]
    for (seed,order,label,arm),(folder,source,entry) in complete.items():
        if arm=="untouched" or (seed,order,label,"untouched") not in complete:
            continue
        base=complete[seed,order,label,"untouched"][0]
        picks=torch.load(source["flip_paths"][0],weights_only=True)[arm]
        full=torch.load(folder/"coordinates.pt",weights_only=True)
        base_full=torch.load(base/"coordinates.pt",weights_only=True)
        left,right=positions(full,picks),positions(base_full,picks)
        state,_=p.baseline_state(entry)
        original=vector(state,picks)
        for step in EARLY+p.STEPS[2:]:
            if step in EARLY:
                a=torch.load(folder/f"signs_{step:04d}.pt",weights_only=True)[left]
                b=torch.load(base/f"signs_{step:04d}.pt",weights_only=True)[right]
            else:
                a,b=(vector(p.N._read_model_payload(d/f"snapshots/step_{step:04d}.pt")["model"],picks)
                      for d in (folder,base))
            rows.append(dict(seed=seed,order=order,mask=label,arm=arm,step=step,count=len(original),
                restored=int(((a==original)&(original!=0)).sum()),near_zero=int((a==0).sum()),
                agreement=int(((a==b)&(b!=0)).sum()),untouched_near_zero=int((b==0).sum()),
                untouched_changed_from_start=int(((b!=original)&(b!=0)&(original!=0)).sum())))
    table("connectivity_signs.csv",rows)



def repair_controls():
    root=OUT/"connectivity"
    manifest=p.read(root/"manifest.json")
    free={}
    for _,source,_ in jobs("connectivity",manifest):
        job=source["job"]
        free[job["seed"],job["order"],job["mask"],job["arm"]]=root/"jobs"/source["name"]
    out={}
    for _,source,_ in jobs("connectivity",manifest,repair=True):
        job=source["job"]
        seed,order,label,arm=(job[key] for key in ("seed","order","mask","arm"))
        constrained=root/"jobs"/source["name"]
        ordinary=free[seed,order,label,arm]
        if not all((folder/"result.json").is_file() for folder in (constrained,ordinary)):
            continue
        for folder in (constrained,ordinary):
            assert p.read(folder/"result.json")["selection_manifest"]==p.digest(root/"manifest.json")
        for step in p.STEPS:
            ce=[p.read(folder/"attempt"/f"full_{step:04d}.json")["ce"] for folder in (constrained,ordinary)]
            out[seed,order,label,arm,step]=ce[0]-ce[1]
    rows=[]
    for (seed,order,label,arm,step),high in sorted(out.items()):
        if arm!="high" or (seed,order,label,"random",step) not in out:
            continue
        random=out[seed,order,label,"random",step]
        rows.append(dict(seed=seed,order=order,mask=label,step=step,high_penalty=high,
                         random_penalty=random,repair_interaction=high-random))
    table("connectivity_repair_interactions.csv",rows)
