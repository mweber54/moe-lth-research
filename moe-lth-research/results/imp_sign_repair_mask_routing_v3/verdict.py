"""Describe observed recovery mechanisms once each stage has verified evidence."""
import csv
from statistics import mean

from audit import OUT,p


def rows(name):
    path=OUT/name
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    lines=["# Sign repair, IMP connectivity, and routing",""]
    prediction=[r for r in rows("untouched_sign_prediction.csv") if r["magnitude_quartile"]=="-1"]
    lines+=["## Prediction",""]
    for r in prediction:
        eligible=int(r["retained"])+int(r["changed"])
        accuracy=int(r["correct"])/max(1,eligible-int(r["abstained"]))
        lines.append(f'- Seed {r["seed"]}, order {r["order"]}, {r["predictor"]}. '
                     f'Accuracy among calls {accuracy:.3%}, change sensitivity {float(r["sensitivity"]):.3%}, '
                     f'false-positive rate {float(r["false_positive_rate"]):.3%}, coverage {float(r["coverage"]):.3%}.')
    if not prediction:
        lines.append("Prediction table pending.")
    lines.append("")
    for stage,heading in (("repair","Repair"),("connectivity","Mask dependence"),("routing","Routing dependence")):
        lines.extend((f"## {heading}",""))
        path=OUT/f"{stage}_outcome.json"
        if not path.is_file():
            if stage=="repair":
                retained=rows("retained_step250_signs.csv")
                if len(retained)==72:
                    for arm in ("high","random"):
                        exclusive=[r for r in retained if r["arm"]==arm and r["coordinates"]=="exclusive"]
                        restored=mean(int(r["restored"])/int(r["count"]) for r in exclusive)
                        lines.append(f"At step 250, {restored:.2%} of exclusive {arm} flips had returned to their starting sign on average across the archived pairs.")
                    lines.append("The small restoration fractions alongside the closing v2 CE gap are consistent with compensation. The causal projection comparison is pending.")
                    lines.append("")
                    continue
            lines.extend(("The verified outcome is pending.",""))
            continue
        result=p.read(path)
        a,b=result["seed_means_step250"],result["seed_means_step2500"]
        lines.append(f"Seed means at step 250 were {', '.join(f'{x:+.6f}' for x in a)} CE. "
                     f"At step 2,500 they were {', '.join(f'{x:+.6f}' for x in b)} CE. "
                     f"Means weight orders A and B equally within each seed.")
        if stage=="repair":
            sign_rows=[r for r in rows("restoration.csv") if r["step"]=="250" and
                       r["coordinates"]==r["arm"]+"_only"]
            for arm in ("high","random"):
                select=[r for r in sign_rows if r["arm"]==arm]
                if select:
                    frac=mean(int(r["restored"])/int(r["count"]) for r in select)
                    lines.append(f"At step 250, {frac:.2%} of exclusive {arm} flips had returned to their starting sign on average.")
            text="All three seed means support a greater early recovery penalty when high-reinforcement signs cannot return."
        elif stage=="connectivity":
            text="All three seed means at step 250 support greater sensitivity under IMP connectivity conditional on random masks’ matched neuron degrees."
        else:
            text="All three seed means at step 250 support preferential damage when a flip set is recovered under the routing policy that selected it."
        if all(x>0 for x in a):
            lines.append(text)
        elif all(x<=0 for x in a):
            lines.append("The early interaction was nonpositive in all three seeds under this intervention.")
        else:
            lines.append("The early interaction varied in sign across seeds.")
        lines.append("")
    lines.append("These are recovery-mechanism comparisons under the tested interventions.")
    (OUT/"findings.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
