"""Publish seed-level contrasts and plots from verified stage tables."""
import csv
from collections import defaultdict
from statistics import mean

from audit import OUT,p
from common_v2 import load

table = load("v3_study_report",OUT/"report.py").table

FIELDS={"repair":"repair_interaction","connectivity":"interaction","routing":"interaction"}


def main(stage):
    origin=OUT/("repair_interaction.csv" if stage=="repair" else f"{stage}_interactions.csv")
    with origin.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    field=FIELDS[stage]
    grouped=defaultdict(list)
    for row in rows:
        grouped[int(row["seed"]),row.get("readout","full"),int(row["step"])].append(float(row[field]))
    results=[dict(seed=seed,readout=readout,step=step,orders=len(values),interaction=mean(values))
             for (seed,readout,step),values in sorted(grouped.items())]
    table(f"{stage}_seed_means.csv",results)
    endpoint=[r["interaction"] for r in results if r["step"]==2500 and r["readout"]=="full"]
    early=[r["interaction"] for r in results if r["step"]==250 and r["readout"]=="full"]
    assert len(endpoint)==len(early)==3
    p.write(OUT/f"{stage}_outcome.json",dict(stage=stage,seed_means_step250=early,
                    seed_means_step2500=endpoint,positive_seeds_step250=sum(x>0 for x in early),
                    positive_seeds_step2500=sum(x>0 for x in endpoint)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True)
    for ax,seed in zip(axes,p.SEEDS):
        for order in p.ORDERS:
            for readout in ("full","readout"):
                points=sorted((int(r["step"]),float(r[field])) for r in rows if
                   (int(r["seed"]),r["order"],r.get("readout","full"))==(seed,order,readout))
                if points:
                    ax.plot(*zip(*points),marker=".",linestyle="-" if readout=="full" else "--",
                            color={"A":"tab:blue","B":"tab:orange"}[order],label=f"{order} {readout}")
        ax.axhline(0,color="gray",linewidth=.8)
        ax.set(title=f"Seed {seed}",xlabel="Attempted recovery step",ylabel="CE interaction",xscale="symlog")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT/f"{stage}_interaction.pdf")
    plt.close(fig)
    loss_table=OUT/("recovery_table.csv" if stage=="repair" else f"{stage}_per_run.csv")
    with loss_table.open(newline="") as stream:
        losses=list(csv.DictReader(stream))
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for row,order in enumerate(p.ORDERS):
        for column,seed in enumerate(p.SEEDS):
            axes_cell=axes[row,column]
            groups=defaultdict(list)
            for cell in losses:
                if int(cell["seed"])!=seed or cell["order"]!=order or cell.get("readout","full")!="full":
                    continue
                if stage=="repair" and int(cell["draw"]) not in (0,1):
                    continue
                label=" ".join((cell["mask"],cell.get("mode",cell.get("recovery","")),cell["arm"]))
                groups[label].append((int(cell["step"]),float(cell["ce"])))
            for label,points in sorted(groups.items()):
                axes_cell.plot(*zip(*sorted(points)),marker=".",label=label)
            axes_cell.set(title=f"Seed {seed}, order {order}",xlabel="Attempted recovery step",
                          ylabel="Validation CE",xscale="symlog",yscale="log")
    handles,labels=axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,loc="lower center",ncol=4)
    fig.tight_layout(rect=(0,.12,1,1))
    fig.savefig(OUT/f"{stage}_ce_trajectories.pdf")
    plt.close(fig)


def repair_projections():
    rows,steps=[],[]
    for result in sorted((OUT/"repair").glob("*/result.json")):
        job=p.read(result)["job"]
        summary=p.torch.load(result.parent/"attempt/tracking_0250.pt",weights_only=True)
        for step,(events,l1,l2_squared) in summary["bindings"]:
            steps.append(dict(**job,step=step,binding_events=int(events),projection_l1=l1,
                              projection_l2_squared=l2_squared))
        for name,hits in summary["hits"].items():
            rows.append(dict(**job,tensor=name,eligible=len(hits),affected=int((hits>0).sum()),
                binding_events=int(hits.sum()),maximum_events_per_weight=int(hits.max()) if len(hits) else 0))
    table("repair_projection_distribution.csv",rows)
    table("repair_projection_steps.csv",steps)


if __name__=="__main__":
    import sys
    stage=sys.argv[1]
    main(stage)
    if stage=="repair":
        repair_projections()
