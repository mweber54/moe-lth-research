"""Independently audit the saved mask payload by magnitude stratum."""
import argparse
import hashlib

from prefix_state import location,DEV
from groups import branch,torch,np,HERE,V2,json,audit_match


def audit(seed,order):
    cell=f"seed{seed}_{order}"
    folder=HERE/"groups"/cell
    if seed in DEV:
        p=branch.V1/"prefixes"/cell
        state_file=p/"native/interruption.pt"
        scores=torch.load(p/"scores.pt",mmap=True,weights_only=True)["scores"]
        old=json.loads((V2/"matching"/cell/"audit.json").read_text())
        certs={item["name"]:item["certificates"] for item in old["matrices"]}
    else:
        state_file=location(seed,order)/"state250.pt"
        scores=torch.load(location(seed,order)/"scores250.pt",mmap=True,weights_only=True)
        certs={item["name"]:item["certificates"] for item in json.loads((folder/"audit.json").read_text())["matrices"]}
    state=torch.load(state_file,map_location="cpu",mmap=True,weights_only=False)["model"]
    meta=json.loads((folder/"audit.json").read_text())
    path=folder/"sets.pt"
    assert branch.sha(path)==meta["sha256"]
    output={}
    for name,g in torch.load(path,weights_only=True).items():
        nr,nc=state[name].shape
        weights=state[name].numpy().astype(np.float64).ravel()
        f=scores[name]
        values=f["score"].numpy().ravel()
        eligible=(f["initial_sign"].numpy().ravel()!=0)&(f["absolute_update"].numpy().ravel()>0)
        ranking=np.lexsort((np.arange(len(weights)),np.abs(weights)))[:len(weights)//4]
        pool=ranking[eligible[ranking]]
        r,h=g["R"].numpy(),g["H"].numpy()
        first=np.random.default_rng(audit_match.seed64(seed,order,name)).choice(pool,13107,replace=False)
        assert np.array_equal(np.sort(first),r)
        union=np.union1d(r,h)
        assert np.array_equal(g["M"].numpy(),union)
        assert np.array_equal(g["S"].numpy(),np.unique(np.setxor1d(r,h)%nc))
        strata=[]
        for b,cert in zip(np.array_split(pool,5),certs[name],strict=True):
            rr,hh=np.intersect1d(b,r),np.intersect1d(b,h)
            counts=[]
            for a,dim in ((lambda x:x//nc,nr),(lambda x:x%nc,nc)):
                match=np.bincount(a(rr),minlength=dim)
                assert np.array_equal(match,np.bincount(a(hh),minlength=dim))
                counts.append(hashlib.sha256(match.astype("<i8").tobytes()).hexdigest())
            obj=values[hh].sum()
            assert abs(obj+cert["dual_min_objective"])<=1e-5 and abs(obj+cert["min_objective"])<=1e-5
            strata.append(dict(count=len(rr),candidates=len(b),row_counts_sha256=counts[0],column_counts_sha256=counts[1],
                               objective=obj,dual=cert["dual_min_objective"],max_degree_difference=0))
        energy=[float(np.square(weights[k]).sum()) for k in (r,h)]
        assert abs(energy[1]-energy[0])<=.05*energy[0] and len(h)==len(r)==13107
        output[name]=dict(energy=energy,strata=strata,removed=[len(r),len(h)],overlap=len(np.intersect1d(r,h)),
                          exclusive=len(h)-len(np.intersect1d(r,h)),union=len(union))
    branch.save_json(folder/"independent_audit.json",dict(sets_sha256=meta["sha256"],matrices=output))

if __name__=="__main__":
    for seed in DEV:
        for order in ("A","B"):
            audit(seed,order)
