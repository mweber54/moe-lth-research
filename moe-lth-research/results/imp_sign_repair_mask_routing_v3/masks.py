"""Make reproducible, well-mixed IMP degree controls."""
import ctypes
import csv

from audit import OUT, SOURCE, p

np,torch=p.np,p.torch
MASKS=OUT/"degree_masks"
TRIALS_PER_EDGE=24
MIN_MOVED=.50


def function():
    lib=ctypes.CDLL(str(OUT/"mask_swaps.so"))
    swap=lib.randomize
    array=lambda dtype:np.ctypeslib.ndpointer(dtype=dtype,ndim=1,flags="C_CONTIGUOUS")
    swap.argtypes=[ctypes.c_int,array(np.uint8),array(np.int32),ctypes.c_int,ctypes.c_uint64,ctypes.c_int]
    swap.restype=ctypes.c_int
    return swap


def main():
    MASKS.mkdir(exist_ok=True)
    swap=function()
    archive=p.read(SOURCE/"inputs.json")
    manifest=dict(trials_per_edge=TRIALS_PER_EDGE,minimum_model_moved=MIN_MOVED,controls=[])
    rows=[]
    for seed in p.SEEDS:
        entry=next(e for e in archive["inputs"] if e["seed"]==seed)
        _,imp=p.baseline_state(entry)
        controls=[]
        for replicate in (1,2):
            masks={}
            for name,mask in imp.items():
                original=mask.numpy()
                target=original.astype(np.uint8,copy=True)
                edges=np.ascontiguousarray(np.flatnonzero(target),dtype=np.int32)
                key=p.rng_seed(OUT.name,"degree-control",seed,replicate,name)
                trials=TRIALS_PER_EDGE*len(edges)
                accepted=swap(target.shape[1],target.ravel(),edges,len(edges),key,trials)
                assert all(np.array_equal(original.sum(axis),target.sum(axis)) for axis in (0,1))
                retained=int((original & target).sum())
                rows.append(dict(seed=seed,replicate=replicate,tensor=name,rng_seed=key,
                                 trials=trials,accepted=accepted,survivors=len(edges),overlap=retained,
                                 moved_fraction=1-retained/len(edges) if len(edges) else 0))
                masks[name]=torch.from_numpy(target.astype(bool))
            subset=[r for r in rows if r["seed"]==seed and r["replicate"]==replicate]
            moved=1-sum(r["overlap"] for r in subset)/sum(r["survivors"] for r in subset)
            assert moved>=MIN_MOVED,(seed,replicate,moved)
            target=MASKS/f"seed{seed}_degree{replicate}.pt"
            p.save_tensor(target,masks)
            assert p.N.recovery._mask_hash(p.N.load_masks(target))==p.N.recovery._mask_hash(masks)
            controls.append(masks)
            manifest["controls"].append(dict(seed=seed,replicate=replicate,path=str(target),sha256=p.digest(target),
                mask_hash=p.N.recovery._mask_hash(masks),fraction_imp_edges_moved=moved))
        intersection=sum(int((controls[0][name]&controls[1][name]).sum()) for name in imp)
        manifest["controls"][-1]["pairwise_control_overlap"]=intersection/sum(int(mask.sum()) for mask in imp.values())
    path=MASKS/"matrix_audit.csv"
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest["matrix_audit_sha256"]=p.digest(path)
    p.write(MASKS/"manifest.json",manifest)


if __name__=="__main__":
    main()
