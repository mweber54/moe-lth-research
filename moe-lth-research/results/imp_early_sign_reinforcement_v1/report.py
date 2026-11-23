"""Report verified recoveries and contrasts at each validation checkpoint."""
from collections import defaultdict
import csv
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run import OUT, SEEDS, ORDERS, MASKS, ARMS, STEPS, digest, job_id, read, validate_losses


def table(name, data):
    if not data:
        return
    target = OUT/name
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    temporary.replace(target)


def mean(values):
    return sum(values)/len(values)


def retention_table():
    path = OUT/"retention_strata.json"
    if (OUT/"matched_retention.csv").exists():
        return
    if not path.exists():
        return
    cells = defaultdict(dict)
    for row in read(path):
        key = (row["seed"],row["order"],row["tensor"],row["stratum"])
        assert row["mask"] not in cells[key]
        cells[key][row["mask"]] = (row["survivors"],row["high"])
    summaries = defaultdict(list)
    for (seed,order,_,_), masks in cells.items():
        if set(masks) == set(MASKS):
            weight = min(n for n,_ in masks.values())
            summaries[seed,order].append((weight,{m:h/n for m,(n,h) in masks.items()}))
    result = []
    for (seed,order), items in sorted(summaries.items()):
        total = sum(w for w,_ in items)
        rates = {m:sum(w*p[m] for w,p in items)/total for m in MASKS}
        result.append(dict(seed=seed,order=order,matched_weight=total,**rates,
                           imp_minus_random_mean=rates["imp"]-mean([rates["random1"],rates["random2"]])))
    table("matched_retention.csv",result)


def main():
    manifest = read(OUT/"inputs.json")
    entries = {(x["seed"],x["order"],x["mask"]):x for x in manifest["inputs"]}
    baseline, data, contrasts, results = {}, {}, [], []
    for key,entry in entries.items():
        seed,order,mask = key
        path = OUT/f"baseline_{seed}_{order}_{mask}.json"
        if path.exists():
            records = read(path)["records"]
            validate_losses(records,entry)
            for row in records:
                baseline[*key,row["step"]] = row["cross_entropy"]
                results.append(dict(seed=seed,order=order,mask=mask,draw="",arm="untouched",
                                    step=row["step"],ce=row["cross_entropy"],delta_from_untouched=0.,
                                    attempts="",applied_updates=""))
    for job in manifest["jobs"]:
        path = OUT/"jobs"/job_id(job)/"result.json"
        if not path.exists():
            continue
        result = read(path)
        assert result["job"] == job and result["seal"] == digest(OUT/"matching_seal.json")
        key = (job["seed"],job["order"],job["mask"])
        records = result["records"]
        validate_losses(records,entries[key])
        assert result["attempts"] == 2500 and 0 < result["applied_updates"] <= 2500
        for row in records:
            step,ce = row["step"],row["cross_entropy"]
            data[*key,job["draw"],job["arm"],step] = ce
            results.append(dict(**job,step=step,ce=ce,delta_from_untouched=ce-baseline[*key,step],
                                attempts=result["attempts"],applied_updates=result["applied_updates"]))
    table("per_run.csv",results)
    for seed,order,mask,draw,step in itertools.product(SEEDS,ORDERS,MASKS,(1,2,3),STEPS):
        values = {a:data.get((seed,order,mask,draw,a,step)) for a in ARMS}
        if any(v is None for v in values.values()):
            continue
        primary = values["high"]-values["random"]
        interaction = None
        if mask == "imp":
            other = [data.get((seed,order,m,draw,a,step))
                     for m in ("random1","random2") for a in ("high","random")]
            if all(v is not None for v in other):
                interaction = primary-mean([other[0]-other[1],other[2]-other[3]])
        base = baseline[seed,order,mask,step]
        contrasts.append(dict(seed=seed,order=order,mask=mask,draw=draw,step=step,
                              high_minus_random=primary,high_minus_weak=values["high"]-values["weak"],
                              mask_interaction=interaction,baseline_ce=base,
                              **{a+"_minus_untouched":x-base for a,x in values.items()}))
    table("paired_results.csv",contrasts)
    retention_table()
    endpoints = [r for r in contrasts if r["step"]==2500]
    table("paired_endpoints.csv",endpoints)
    groups = defaultdict(list)
    fields = ("high_minus_random","high_minus_weak","mask_interaction",
              "high_minus_untouched","weak_minus_untouched","random_minus_untouched")
    for row in endpoints:
        for order in (row["order"],"mean"):
            groups[row["seed"],row["mask"],order].append(row)
    summaries = []
    for (seed,mask,order),items in sorted(groups.items()):
        means = {k:mean(v) if (v:=[r[k] for r in items if r[k] is not None]) else None for k in fields}
        summaries.append(dict(seed=seed,mask=mask,order=order,n_pairs=len(items),**means))
    table("seed_endpoints.csv",summaries)
    if len(endpoints) != 54:
        print(f"Verified {len(data)//len(STEPS)} / 162 recoveries and {len(baseline)//len(STEPS)} / 18 baselines")
        return
    for label in ("loss","delta"):
        fig, axes = plt.subplots(3,3,figsize=(13,10),sharex=True)
        for row,seed in zip(axes,SEEDS):
            for ax,mask in zip(row,MASKS):
                for order in ORDERS:
                    for arm in (*ARMS,"untouched"):
                        y = [baseline[seed,order,mask,t] if arm=="untouched" else
                             mean([data[seed,order,mask,d,arm,t] for d in (1,2,3)]) for t in STEPS]
                        if label == "delta":
                            y = [v-baseline[seed,order,mask,t] for t,v in zip(STEPS,y)]
                        ax.plot(STEPS,y,label=f"{arm}, {order}",
                                color=dict(high="tab:red",weak="tab:blue",random="tab:green",
                                           untouched="black")[arm],ls="-" if order=="A" else "--")
                ax.set_title(f"Seed {seed}, {mask}")
                ax.set_xlabel("Recovery step")
                ax.set_ylabel("Validation CE" if label=="loss" else "CE minus untouched")
                if label=="loss":
                    ax.set_yscale("log")
        handles,labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=4)
        fig.tight_layout(rect=(0,.06,1,1))
        fig.savefig(OUT/f"recovery_{label}.pdf")
        plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(11,4),sharex=True)
    for ax,key in zip(axes,fields[:3]):
        for i,seed in enumerate(SEEDS):
            items = [r for r in endpoints if r["seed"]==seed and r["mask"]=="imp"]
            for order,marker,offset in (("A","o",-.09),("B","s",.09)):
                ax.scatter([i+offset]*3,[r[key] for r in items if r["order"]==order],
                           marker=marker,label=order if i==0 else None)
            ax.scatter(i,mean([r[key] for r in items]),marker="_",c="black",s=200)
        ax.axhline(0,color="gray",lw=.8)
        ax.set_xticks(range(3),SEEDS)
        ax.set_xlabel("Seed")
        ax.set_title(key.replace("_"," "))
    axes[0].set_ylabel("Validation CE contrast")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT/"endpoint_contrasts.pdf")
    plt.close(fig)
    seed_means = [r for r in summaries if r["mask"]=="imp" and r["order"]=="mean"]
    headings = ["# Early sign reinforcement and E32 recovery", "",
                "Final validation cross-entropy differences are paired within seed, mask, order, and draw.",
                "Each seed contributes one mean across its six paired differences.", "",
                "| Seed | High − random in IMP | High − weak in IMP | IMP minus random-mask interaction |",
                "| --- | ---: | ---: | ---: |"]
    for r in seed_means:
        headings.append(f"| {r['seed']} | {r['high_minus_random']:+.6f} | "
                        f"{r['high_minus_weak']:+.6f} | {r['mask_interaction']:+.6f} |")
    if all(r["high_minus_random"] > 0 and r["high_minus_weak"] > 0 and r["mask_interaction"] > 0
           for r in seed_means):
        verdict = ("Flipping high-reinforcement signs hurt more than both matched controls in all three "
                   "IMP seed means. The high-versus-random penalty was larger in IMP than in the "
                   "per-matrix sparsity-matched random masks for all three seeds.")
    elif all(r["high_minus_random"] > 0 for r in seed_means):
        verdict = ("The high-reinforcement arm had higher final CE than matched random flips in all "
                   "three IMP seed means. The secondary contrast and mask interaction show how "
                   "consistently the effect distinguishes IMP from the controls.")
    else:
        verdict = ("The seed means do not show a consistent positive high-versus-random penalty in IMP. "
                   "The paired estimates give the direction and magnitude in each seed.")
    notes = []
    summary_file = OUT/"matching_summary.json"
    if summary_file.exists():
        matching = read(summary_file)
        n = matching["matrix_triplets_with_neuron_concentration_gap_ge_10pp"]
        total = matching["matrix_triplets"]
        notes = ["", f"The common normalized L2 target was {matching['target_distance']:.6f}. In "
                 f"{n:,} of {total:,} matrix triplets, the arm range in the share of flips on the "
                 "most-targeted decile of rows or columns reached at least ten percentage points. "
                 "Differences in neuron concentration can contribute to the observed flip penalties."]
    (OUT/"findings.md").write_text("\n".join(headings + ["",verdict] + notes + ["",
        "These comparisons estimate the effect of intervening on selected initial signs. "
        "A separate router intervention would be needed to establish the cause of the reinforcement scores.",""]))


if __name__ == "__main__":
    main()
