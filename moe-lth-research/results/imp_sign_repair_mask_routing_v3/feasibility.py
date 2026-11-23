"""Freeze degree-control or routing-source matches using the v2 rectangle policy."""
import argparse
import gzip
import json
from pathlib import Path

from audit import OUT, SOURCE, p
from common_v2 import load

rect=load("v3_v2_rectangles",SOURCE/"rectangle_match.py")
np,torch=p.np,p.torch
TARGETS=rect.TARGETS
RULES=rect.RULES


def rank_summary(masks,first,second,dense):
    counts=np.zeros(4,dtype=np.int64)
    for name,mask in masks.items():
        w=dense[name].numpy().ravel()
        r0,a0=(t.numpy().ravel() for t in first[name])
        r1,a1=(t.numpy().ravel() for t in second[name])
        valid=(w!=0)&(a0>0)&(a1>0)&np.isfinite(r0)&np.isfinite(r1)&np.isfinite(a0)&np.isfinite(a1)
        population=np.flatnonzero(valid)
        q=len(population)//4
        if not q:
            continue
        top0=population[np.argsort(r0[population],kind="stable")[-q:]]
        top1=population[np.argsort(r1[population],kind="stable")[-q:]]
        survive=mask.numpy().ravel()
        both=np.intersect1d(top0,top1)
        counts+=(len(top0),len(both),survive[top0].sum()+survive[top1].sum(),survive[both].sum())
    return dict(dense_quartile_overlap=int(counts[1])/max(1,int(counts[0])),
                survivor_top_quartile_dice=2*int(counts[3])/max(1,int(counts[2])))


def main(stage):
    root=OUT/stage
    root.mkdir(exist_ok=True)
    archive=p.read(OUT/"audit_manifest.json")
    entries=archive["inputs"]
    degree=p.read(OUT/"degree_masks/manifest.json")["controls"] if stage=="connectivity" else []
    native_targets={(int(r["seed"]),int(r["draw"]),r["tensor"]):float(r["realized_distance"])
        for r in p.rows(p.PRIOR/"matrix_invariants.csv") if r["mask_type"]=="imp" and r["arm"]=="Bm"}
    summary={target:[] for target in TARGETS}
    sources=[]
    rankings=[]
    sampler=rect.sampler_function()
    for entry in entries:
        seed,order=entry["seed"],entry["order"]
        dense=p.N._read_model_payload(Path(entry["base"]["initial_checkpoint"]))["model"]
        if stage=="connectivity":
            parts=[("imp",entry)]+[(f'degree{ctrl["replicate"]}',dict(entry,base=dict(entry["base"],
                           mask_path=ctrl["path"],mask_sha256=ctrl["mask_hash"]))) for ctrl in degree if ctrl["seed"]==seed]
        else:
            parts=[(condition,entry) for condition in ("native","scaled")]
        fresh={}
        if stage=="routing":
            for condition in ("native","scaled"):
                probe=OUT/"probes"/condition/f"probe_{seed}_{order}.pt"
                attestation=p.read(probe.parent/f"audit_{seed}_{order}.json")
                assert attestation["probe_sha256"]==p.digest(probe)
                fresh[condition]=torch.load(probe,weights_only=True,mmap=True)["scores"]
            masks=p.N.load_masks(Path(entry["base"]["mask_path"]))
            rankings.append(dict(seed=seed,order=order,**rank_summary(masks,fresh["native"],fresh["scaled"],dense)))
        for label,variant in parts:
            path=Path(variant["base"]["mask_path"])
            masks=p.N.load_masks(path)
            assert p.N.recovery._mask_hash(masks)==variant["base"]["mask_sha256"]
            if label.startswith("degree"):
                state=p.N._masked_initial_state(Path(entry["base"]["initial_checkpoint"]),masks)
                variant["base"]["initial_model_state_hash"]=p.N._tensor_state_hash(state)
                del state
            scores=fresh[label] if stage=="routing" else torch.load(
                p.OUT/f"probe_{seed}_{order}.pt",weights_only=True,mmap=True)["scores"]
            matrices={name:(dense[name].shape,rect.matrix_inputs(dense[name],*scores[name],mask)) for name,mask in masks.items()}
            pools=list(rect.collect(variant,1,matrices,native_targets,sampler))
            del matrices
            for target in TARGETS:
                selected={arm:{} for arm in ("high","random")}
                audit=[]
                counts,level=rect.allocation(pools,target)
                for m,c in zip(pools,counts):
                    ids,row=m.audit(c)
                    audit.append(row)
                    for arm,coordinates in zip(selected,ids):
                        selected[arm][m.name]=torch.from_numpy(coordinates)
                cell=dict(label=label,**rect.pair_summary(seed,order,1,audit,target))
                summary[target].append(cell)
                if cell["passed"]:
                    directory=root/f"dose_{target}"
                    directory.mkdir(exist_ok=True)
                    stem=f"{seed}_{order}_{label}"
                    p.save_tensor(directory/f"flips_{stem}.pt",selected)
                    with gzip.open(directory/f"neurons_{stem}.json.gz","wt",compresslevel=2) as f:
                        json.dump(audit,f,separators=(",",":"))
            sources.append(dict(label=label,entry=variant,mask_sha256=p.digest(path)))
            print(stage,seed,order,label,flush=True)
            p.write(root/"diagnostics.json",summary)
            del pools,masks
        del dense
    if rankings:
        p.write(root/"ranking_audit.json",rankings)
        if any(row["survivor_top_quartile_dice"]>.90 for row in rankings):
            raise SystemExit("Ranking overlap exceeded the frozen crossover gate.")
    passing=[t for t in TARGETS if len(summary[t])==len(sources) and all(row["passed"] for row in summary[t])]
    if not passing:
        raise SystemExit("No common dose passed matching.")
    target=passing[0]
    directory=root/f"dose_{target}"
    for source in sources:
        entry=source["entry"]
        stem=f'{entry["seed"]}_{entry["order"]}_{source["label"]}'
        for key,name in (("flips",f"flips_{stem}.pt"),("diagnostic",f"neurons_{stem}.json.gz")):
            source[key]=str(directory/name)
            source[key+"_sha256"]=p.digest(directory/name)
    if stage=="routing":
        overlap=[]
        for native,scaled in zip(sources[::2],sources[1::2]):
            left,right=(torch.load(item["flips"],weights_only=True) for item in (native,scaled))
            common=sum(len(np.intersect1d(a.numpy(),right["high"][name].numpy())) for name,a in left["high"].items())
            count=sum(len(a) for a in left["high"].values())+sum(len(a) for a in right["high"].values())
            overlap.append(2*common/max(1,count))
        p.write(root/"flip_set_overlap.json",overlap)
        if any(value>.60 for value in overlap):
            raise SystemExit("Selected high flip sets overlap too much for the crossover.")
    if stage=="connectivity" and target==.34207:
        for item in (r for r in sources if r["label"]=="imp"):
            seed,order=(item["entry"][k] for k in ("seed","order"))
            archived=torch.load(SOURCE/f"flips_{seed}_{order}_imp_1.pt",weights_only=True)
            proposed=torch.load(item["flips"],weights_only=True)
            assert all(torch.equal(x,archived[arm][name]) for arm,selected in proposed.items() for name,x in selected.items())
    p.write(root/"manifest.json",dict(stage=stage,target=target,rules=RULES,dose_grid=TARGETS,
            rank_overlap_limit=.90,flip_overlap_limit=.60,sources=sources,
            diagnostics_sha256=p.digest(root/"diagnostics.json"),draw=1))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("connectivity","routing"))
    main(parser.parse_args().stage)
