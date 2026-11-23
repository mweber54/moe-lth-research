"""Verify saved outcomes and report the paired E32 sign-flip comparison."""
from collections import defaultdict
import csv
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_v2 import OUT, pilot as p


def table(name, rows):
    if rows:
        path = OUT / name
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    metrics = ("high_minus_random", "high_minus_baseline", "random_minus_baseline")
    return [dict(zip(keys, identity), pairs=len(items), **{k:mean(row[k] for row in items) for k in metrics})
            for identity, items in sorted(groups.items())]


def plots(pairs, endpoints):
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, squeeze=False)
    for i, seed in enumerate(p.SEEDS):
        for j, order in enumerate(p.ORDERS):
            ax = axes[i,j]
            subset = [r for r in pairs if r["seed"] == seed and r["order"] == order]
            for key, label in (("high_ce","High reinforcement"),("random_ce","Matched random"),("baseline_ce","Untouched IMP")):
                trajectory = [(step,mean(r[key] for r in subset if r["step"] == step))
                              for step in p.STEPS if any(r["step"] == step for r in subset)]
                if trajectory:
                    ax.plot(*zip(*trajectory), marker=".", label=label)
            ax.set(title=f"Seed {seed}, order {order}", xlabel="Recovery step", ylabel="Validation CE")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(OUT / "recovery_trajectories.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for order, offset in (("A",-.12),("B",.12)):
        for i, seed in enumerate(p.SEEDS):
            values = [r["high_minus_random"] for r in endpoints if (r["seed"],r["order"]) == (seed,order)]
            if values:
                ax.scatter([i+offset]*len(values),values,label=order if i == 0 else None)
                ax.scatter(i+offset,mean(values),marker="_",color="black",s=180)
    ax.axhline(0,color="gray",lw=.8)
    ax.set(xticks=list(range(3)),xticklabels=p.SEEDS,xlabel="Seed",ylabel="High minus random CE at step 2,500")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "endpoint_contrasts.pdf")
    plt.close(fig)


def main():
    inputs = p.read(OUT / "inputs.json")
    seal = p.digest(OUT / "matching_seal.json")
    entries = {(e["seed"],e["order"]):e for e in inputs["inputs"]}
    baseline = {}
    data = []
    paired = defaultdict(dict)
    for (seed,order), entry in entries.items():
        records = p.read(OUT / f"baseline_{seed}_{order}_imp.json")["records"]
        p.validate_losses(records,entry)
        for row in records:
            baseline[seed,order,row["step"]] = row["cross_entropy"]
            data.append(dict(seed=seed,order=order,draw="",arm="untouched",step=row["step"],
                             ce=row["cross_entropy"],delta_from_baseline=0.,attempts="",applied_updates=""))
    for job in inputs["jobs"]:
        path = OUT / "jobs" / p.job_id(job) / "result.json"
        if not path.exists():
            continue
        result = p.read(path)
        assert result["job"] == job and result["seal"] == seal
        assert result["attempts"] == 2500 and 0 < result["applied_updates"] <= 2500
        seed,order,draw,arm = (job[k] for k in ("seed","order","draw","arm"))
        p.validate_losses(result["records"],entries[seed,order])
        for row in result["records"]:
            step, ce = row["step"], row["cross_entropy"]
            paired[seed,order,draw,step][arm] = ce
            data.append(dict(seed=seed,order=order,draw=draw,arm=arm,step=step,ce=ce,
                             delta_from_baseline=ce-baseline[seed,order,step],attempts=result["attempts"],
                             applied_updates=result["applied_updates"]))
    contrasts = []
    for (seed,order,draw,step), values in sorted(paired.items()):
        if values.keys() != {"high","random"}:
            continue
        high, random = values["high"],values["random"]
        base = baseline[seed,order,step]
        contrasts.append(dict(seed=seed,order=order,draw=draw,step=step,high_ce=high,random_ce=random,
                              baseline_ce=base,high_minus_random=high-random,
                              high_minus_baseline=high-base,random_minus_baseline=random-base))
    table("per_run.csv",data)
    table("paired_results.csv",contrasts)
    endpoints = [row for row in contrasts if row["step"] == 2500]
    table("paired_endpoints.csv",endpoints)
    table("seed_order_means.csv",summarize(endpoints,("seed","order")))
    table("seed_means.csv",summarize(endpoints,("seed",)))
    if endpoints:
        plots(contrasts,endpoints)
    print(f"Verified {(len(data)-len(baseline))//len(p.STEPS)} / 36 recoveries and {len(endpoints)} / 18 complete pairs",flush=True)
    if len(endpoints) == 18:
        verdict(endpoints)


def verdict(endpoints):
    seed_order = summarize(endpoints,("seed","order"))
    by_order = {order:[r["high_minus_random"] for r in seed_order if r["order"] == order] for order in p.ORDERS}
    if all(value > 0 for values in by_order.values() for value in values):
        finding = "All six seed-by-order means were positive. These estimates support a connection-level penalty at the tested dose."
    elif min(by_order["A"]) > 0 and max(by_order["B"]) <= 0:
        finding = ("All three A means were positive and all three B means were nonpositive. "
                   "The estimates support an order-dependent effect. Recovering the same fixed flip sets under both orders "
                   "would distinguish recovery order from score-based selection.")
    else:
        finding = "The direction varied across seed-by-order groups. The table shows the size and consistency of the effect."
    lines = ["# Neuron-matched early sign reinforcement", "", finding, "",
             "| Seed | Order | Mean high − random CE | High − untouched | Random − untouched |",
             "| --- | --- | ---: | ---: | ---: |"]
    for r in seed_order:
        lines.append(f"| {r['seed']} | {r['order']} | {r['high_minus_random']:+.6f} | "
                     f"{r['high_minus_baseline']:+.6f} | {r['random_minus_baseline']:+.6f} |")
    lines += ["", "Each seed is an independent replication. The three draws in each order quantify within-seed consistency.", ""]
    (OUT / "findings.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import fcntl

    with (OUT / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
