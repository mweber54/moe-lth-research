"""Bounded exact local-feasibility classification for uncovered controls."""
from __future__ import annotations
import ctypes, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULTS=HERE.parent
V2=RESULTS/"imp_early_sign_reinforcement_neuron_matched_v2"
V3=RESULTS/"imp_sign_repair_mask_routing_v3"
AUDIT=RESULTS/"imp_sampler_audit_v1"
sys.path[:0]=[str(V2),str(V3)]
from common_v2 import pilot as p
np,torch=p.np,p.torch

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    return mod

rect=module("exact_followup_rectangle_match",V2/"rectangle_match.py")
local=module("exact_followup_audit",AUDIT/"audit.py")
MAX_PAIRS=100_000_000
TARGET=.075

def exact_function():
    lib=ctypes.CDLL(str(HERE/"exact_rectangle.so")); fn=lib.exact_rectangle
    ints=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags="C_CONTIGUOUS")
    doubles=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags="C_CONTIGUOUS")
    u64p=ctypes.POINTER(ctypes.c_uint64)
    fn.argtypes=[ctypes.c_int,ctypes.c_int,ints,doubles,doubles,ints,ctypes.c_int,
                 ctypes.c_uint64,ints,u64p,u64p]
    fn.restype=ctypes.c_int
    return fn

def gap(a,b): return abs(a-b)/max(abs(a),abs(b),1e-30)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    exact=exact_function()
    sampler=local.load_sampler(V2/"rectangle_sampler.so","rectangles")
    archive=json.loads((V3/"audit_manifest.json").read_text())
    controls=json.loads((V3/"degree_masks/manifest.json").read_text())["controls"]
    native={(int(r["seed"]),int(r["draw"]),r["tensor"]):float(r["realized_distance"])
            for r in p.rows(p.PRIOR/"matrix_invariants.csv") if r["mask_type"]=="imp" and r["arm"]=="Bm"}
    rows=[]
    started=time.time()
    for entry in archive["inputs"]:
        seed,order=entry["seed"],entry["order"]
        dense=p.N._read_model_payload(Path(entry["base"]["initial_checkpoint"]))["model"]
        scores=torch.load(p.OUT/f"probe_{seed}_{order}.pt",weights_only=True,mmap=True)["scores"]
        for ctrl in (x for x in controls if x["seed"]==seed):
            label=f'degree{ctrl["replicate"]}'
            variant=dict(entry,base=dict(entry["base"],mask_path=ctrl["path"],mask_sha256=ctrl["mask_hash"]))
            masks=p.N.load_masks(Path(ctrl["path"]))
            matrices={name:(dense[name].shape,rect.matrix_inputs(dense[name],*scores[name],mask))
                      for name,mask in masks.items()}
            pools=list(rect.collect(variant,1,matrices,native,sampler))
            for pool in pools:
                if len(pool.prefix)>1: continue
                shape,inputs=matrices[pool.name]
                w,score,activity,survivor,groups,high=inputs
                energy=np.ascontiguousarray(4*np.square(w.astype(np.float64)))
                activity=np.ascontiguousarray(activity,dtype=np.float64)
                groups=np.ascontiguousarray(groups,dtype=np.int32)
                high=np.ascontiguousarray(high,dtype=np.int32)
                witness=np.full(6,-1,dtype=np.int32)
                tested=ctypes.c_uint64(); eligible=ctypes.c_uint64()
                t0=time.time()
                code=exact(*shape,groups,energy,activity,high,len(high),MAX_PAIRS,witness,
                           ctypes.byref(tested),ctypes.byref(eligible))
                status={1:"witness_found",0:"provably_none",-1:"unresolved"}[code]
                row=dict(seed=seed,order=order,label=label,tensor=pool.name,shape=list(shape),
                         high_pool=len(high),eligible_survivors=int((groups>=0).sum()),
                         status=status,ordered_pairs_tested=tested.value,
                         stratum_eligible_seen=eligible.value,seconds=time.time()-t0)
                if code==1:
                    a,b,x,y,ro,co=map(int,witness)
                    row["witness"]={"a":a,"b":b,"cross_r":x,"cross_s":y,
                                    "row_orientation":bool(ro),"column_orientation":bool(co),
                                    "coordinates":{"a":[a//shape[1],a%shape[1]],"b":[b//shape[1],b%shape[1]],
                                                   "cross_r":[x//shape[1],x%shape[1]],"cross_s":[y//shape[1],y%shape[1]]},
                                    "energy":[float(energy[z]) for z in (a,b,x,y)],
                                    "activity":[float(activity[z]) for z in (a,b,x,y)],
                                    "gaps":{"energy":gap(energy[a]+energy[b],energy[x]+energy[y]),
                                            "magnitude":gap(math.sqrt(energy[a])+math.sqrt(energy[b]),math.sqrt(energy[x])+math.sqrt(energy[y])),
                                            "activity":gap(activity[a]+activity[b],activity[x]+activity[y]),
                                            "row_r":gap(energy[a],energy[x]),"row_s":gap(energy[b],energy[y]),
                                            "column_c":gap(energy[a],energy[y]),"column_d":gap(energy[b],energy[x])}}
                rows.append(row)
            (HERE/"classification_partial.json").write_text(json.dumps(rows,indent=2)+"\n")
            print(seed,order,label,"uncovered",sum(r["seed"]==seed and r["order"]==order and r["label"]==label for r in rows),flush=True)
            del pools,matrices,masks
        del dense,scores
    (HERE/"classification.json").write_text(json.dumps(rows,indent=2)+"\n")
    counts={k:sum(r["status"]==k for r in rows) for k in ("witness_found","provably_none","unresolved")}
    summary=dict(target=TARGET,scope="matrices with zero rectangles after frozen 1M-proposal sampler",
                 max_ordered_pairs_per_matrix=MAX_PAIRS,matrices=len(rows),counts=counts,
                 elapsed_seconds=time.time()-started,
                 interpretation="A witness proves local first-rectangle feasibility only; it does not satisfy model-level frozen gates.")
    (HERE/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
