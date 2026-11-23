"""Sign retention and change detection on untouched IMP survivors."""
from collections import defaultdict
import itertools

from audit import OUT, EPSILON, p
from replay import classify
from common_v2 import load

table = load("v3_study_report",OUT/"report.py").table

np,torch=p.np,p.torch


def main():
    manifest=p.read(OUT/"audit_manifest.json")
    rows=[]
    for entry in manifest["inputs"]:
        seed,order=entry["seed"],entry["order"]
        native=next(item for item in manifest["jobs"] if
                    (item["job"]["seed"],item["job"]["order"],item["job"]["arm"])==(seed,order,"untouched"))
        start,masks=p.baseline_state(entry)
        final=p.N._read_model_payload(p.Path(native["paths"]["2500"]))["model"]
        archived=torch.load(p.OUT/f"probe_{seed}_{order}.pt",weights_only=True,mmap=True)
        endpoint=OUT/"probes/native"/f"dense_sign_{seed}_{order}.pt"
        end=torch.load(endpoint,weights_only=True,mmap=True) if endpoint.is_file() else {}
        totals=defaultdict(lambda:np.zeros(8,dtype=np.int64))
        for name,mask in masks.items():
            r,act=archived["scores"][name]
            weight=start[name][mask]
            observed=classify(final[name][mask]).numpy()
            beginning=classify(weight).numpy()
            eligible=(act[mask]>0)&act[mask].isfinite()&r[mask].isfinite()
            displacement=torch.where(eligible,(weight.sign()*r[mask]).sign(),0).numpy()
            activity=act[mask].numpy()
            estimators={"starting":beginning,"displacement":displacement}
            if name in end:
                estimators["dense_probe_end"]=end[name][mask].numpy()
            mag_bin=p.quantile_bin(weight.abs().numpy(),4) if len(weight) else np.zeros(0,dtype=int)
            act_bin=p.quantile_bin(activity,4) if len(weight) else np.zeros(0,dtype=int)
            changed=(beginning!=observed)
            valid=(beginning!=0)&(observed!=0)
            for model,predicted in estimators.items():
                valid_pred=valid&(predicted!=0)
                predicted_change=predicted!=beginning
                for m,a in [(-1,-1),*itertools.product(range(4),repeat=2)]:
                    group=np.ones(len(weight),dtype=bool) if m<0 else (mag_bin==m)&(act_bin==a)
                    usable=group&valid_pred
                    counts=(group.sum(),(group&valid&~changed).sum(),(group&valid&changed).sum(),
                            (group&~valid).sum(),(group&valid&~valid_pred).sum(),
                            (usable&(predicted==observed)).sum(),(usable&changed&predicted_change).sum(),
                            (usable&~changed&predicted_change).sum())
                    totals[model,m,a]+=counts
        for (model,m,a),counts in sorted(totals.items()):
            n,retained,changed,zero,abstain,correct,tp,fp=map(int,counts)
            rows.append(dict(seed=seed,order=order,predictor=model,magnitude_quartile=m,activity_quartile=a,
                             survivors=n,retained=retained,changed=changed,near_zero_initial_or_final=zero,
                             abstained=abstain,correct=correct,true_positive=tp,false_positive=fp,
                             coverage=(retained+changed-abstain)/max(1,retained+changed),
                             sensitivity=tp/max(1,changed),false_positive_rate=fp/max(1,retained),
                             precision=tp/(tp+fp) if tp+fp else None))
        print("Prediction",seed,order,sorted({key[0] for key in totals}),flush=True)
    table("untouched_sign_prediction.csv",rows)


if __name__=="__main__":
    main()
