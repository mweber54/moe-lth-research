"""Combine replayed early signs and archived later signs on the same coordinates."""
import csv
import itertools

from audit import OUT, SOURCE, EARLY, EPSILON, p
from common_v2 import load

table = load("v3_study_report",OUT/"report.py").table
from replay import classify

torch=p.torch
STEPS=EARLY+p.STEPS[2:]


def vector(state,coordinates):
    return torch.cat([classify(state[name].flatten()[index]) for name,index in coordinates.items()])


def initial(entry):
    state,_=p.baseline_state(entry)
    return state


def traces(source,coordinates,positions):
    root=OUT/"free"/p.job_id(source["job"])/"attempt"
    early={step:torch.load(root/f"signs_{step:04d}.pt",weights_only=True)[positions] for step in EARLY}
    for step in p.STEPS[2:]:
        model=p.N._read_model_payload(p.Path(source["paths"][str(step)]))["model"]
        early[step]=vector(model,coordinates)
    return early


def positions(full,part):
    offset=0
    result=[]
    for name,indices in full.items():
        lookup=torch.searchsorted(indices,part[name])
        assert torch.equal(indices[lookup],part[name])
        result.append(lookup+offset)
        offset+=len(indices)
    return torch.cat(result)


def main():
    manifest=p.read(OUT/"audit_manifest.json")
    source={(j["job"]["seed"],j["job"]["order"],j["job"]["draw"],j["job"]["arm"]):j for j in manifest["jobs"]}
    rows=[]
    for entry in manifest["inputs"]:
        seed,order=entry["seed"],entry["order"]
        untouched=source[seed,order,0,"untouched"]
        inputs=initial(entry)
        base_root=OUT/"free"/p.job_id(untouched["job"])
        if not (base_root/"result.json").is_file():
            continue
        base_coords=torch.load(base_root/"attempt/coordinates.pt",weights_only=True)
        for draw in (1,2,3):
            arms={arm:source[seed,order,draw,arm] for arm in ("high","random")}
            roots={arm:OUT/"free"/p.job_id(s["job"]) for arm,s in arms.items()}
            if not all((root/"result.json").is_file() for root in roots.values()):
                continue
            picks=torch.load(SOURCE/f"flips_{seed}_{order}_imp_{draw}.pt",weights_only=True)
            coords=torch.load(roots["high"]/"attempt/coordinates.pt",weights_only=True)
            sets={arm:torch.cat([torch.isin(index,picks[arm][name]) for name,index in coords.items()])
                  for arm in arms}
            slices={"shared":sets["high"]&sets["random"],"high_only":sets["high"]&~sets["random"],
                    "random_only":sets["random"]&~sets["high"]}
            start=vector(inputs,coords)
            baseline=traces(untouched,coords,positions(base_coords,coords))
            for arm,s in arms.items():
                signs=traces(s,coords,torch.arange(len(start)))
                tracking=torch.load(roots[arm]/"attempt/tracking_0250.pt",weights_only=True)
                first,last_other=tracking["first"],tracking["last_other"]
                observed=torch.stack([signs[step] for step in STEPS])
                matches=(observed==start)&(start!=0)
                baseline_observed=torch.stack([baseline[step] for step in STEPS])
                valid_untouched=(baseline_observed!=0)&(start!=0)
                crosses=(valid_untouched&(baseline_observed!=start)).to(torch.int8).cumsum(dim=0)>0
                recorded_first=torch.full_like(first,-1)
                recorded_first[first>0]=first[first>0]
                for n,step in enumerate(STEPS):
                    if step>250:
                        recorded_first[(recorded_first<0)&matches[n]]=step
                    for label,group in slices.items():
                        count=int(group.sum())
                        if not count:
                            continue
                        credible=(signs[step]!=0)&(baseline[step]!=0)
                        rows.append(dict(seed=seed,order=order,draw=draw,arm=arm,coordinates=label,step=step,
                            count=count,restored=int((matches[n]&group).sum()),
                            near_zero=int(((signs[step]==0)&group).sum()),
                            untouched_near_zero=int(((baseline[step]==0)&group).sum()),
                            untouched_agreement=int((credible&(signs[step]==baseline[step])&group).sum()),
                            untouched_ever_crossed=int((crosses[n]&group).sum()),
                            agreement_when_untouched_ever_crossed=int((credible&(signs[step]==baseline[step])&crosses[n]&group).sum())))
                stable=(first>0)&(last_other<first)&matches[STEPS.index(250):].all(dim=0)
                p.save_tensor(OUT/f"return_{seed}_{order}_{draw}_{arm}.pt",dict(
                    exact_first_through_250=first,last_nonrestored_through_250=last_other,
                    first_observed_return=recorded_first,stable_at_observed_later_steps=stable,
                    signs=observed,untouched_signs=baseline_observed,steps=STEPS))
            print("Tracked",seed,order,draw,flush=True)
        del inputs
    table("restoration.csv",rows)
    if rows:
        plots(rows)
    paired_ce()


def paired_ce():
    data={}
    for file,readout in (("fixed_readouts.csv","fixed_readout"),("recovery_table.csv","full")):
        path=OUT/file
        if not path.is_file():
            continue
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["mode"]=="free":
                    key=(int(row["seed"]),row["order"],int(row["draw"]),row["arm"],int(row["step"]),readout)
                    data[key]=float(row["ce"])
    rows=[]
    for (seed,order,draw,arm,step,readout),high in sorted(data.items()):
        random=data.get((seed,order,draw,"random",step,readout))
        if arm!="high" or random is None:
            continue
        untouched=data[seed,order,0,"untouched",step,readout]
        rows.append(dict(seed=seed,order=order,draw=draw,step=step,readout=readout,high_ce=high,
                         random_ce=random,untouched_ce=untouched,high_minus_random=high-random,
                         high_minus_untouched=high-untouched,random_minus_untouched=random-untouched))
    table("free_paired_ce.csv",rows)


def plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax,(order,seed) in zip(axes.flat,itertools.product(p.ORDERS,p.SEEDS)):
        for arm in ("high","random"):
            for group,style in ((arm+"_only","-"),("shared","--")):
                means=[]
                for step in STEPS:
                    chosen=[r for r in rows if (r["seed"],r["order"],r["arm"],r["coordinates"],r["step"])==
                            (seed,order,arm,group,step)]
                    if chosen:
                        means.append((step,sum(r["restored"] for r in chosen)/sum(r["count"] for r in chosen)))
                if means:
                    ax.plot(*zip(*means),style,label=f"{arm} {group}",color={"high":"tab:red","random":"tab:blue"}[arm])
        ax.set(title=f"Seed {seed} order {order}",xscale="symlog",xlabel="Attempted recovery step",ylabel="Starting signs restored")
    handles, labels=axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,loc="lower center",ncol=4)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(OUT/"restoration.pdf")
    plt.close(fig)


if __name__=="__main__":
    main()
