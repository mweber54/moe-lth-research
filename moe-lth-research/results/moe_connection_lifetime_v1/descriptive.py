"""Coordinate drift and validation-source provenance, independent of outcomes."""
import json
import numpy as np
import torch

from groups import HERE,branch
from prefix_state import DEV,location,ROOT
from moe_lth.experiments.run_long_validation_extension import _sample_evenly


def coordinates(seed,order,t):
    cell=f"seed{seed}_{order}"
    path=HERE/"groups"/cell/"sets.pt"
    if t==250 and seed in DEV:
        state_file=branch.V1/"prefixes"/cell/"native/interruption.pt"
        score_file=branch.V1/"prefixes"/cell/"scores.pt"
        scores=torch.load(score_file,mmap=True,weights_only=True)["scores"]
    else:
        state_file=location(seed,order)/f"state{t}.pt"
        score_file=location(seed,order)/f"scores{t}.pt"
        scores=torch.load(score_file,mmap=True,weights_only=True)
    state=torch.load(state_file,map_location="cpu",mmap=True,weights_only=False)["model"]
    result=[]
    for n,g in torch.load(path,weights_only=True).items():
        for label,k in (("H","H"),("R","R"),("H_only","H"),("R_only","R")):
            idx=g[k]
            if label.endswith("_only"):
                idx=idx[~torch.isin(idx,g["R" if k=="H" else "H"])]
            w=state[n].flatten()[idx].double()
            score=scores[n]["score"].flatten()[idx]
            result.append(dict(name=n,group=label,count=len(idx),mean_magnitude=w.abs().mean().item(),
                               energy=w.square().sum().item(),mean_reinforcement=score.mean().item()))
    branch.save_json(path.parent/f"coordinates{t}.json",dict(sets_sha256=branch.sha(path),matrices=result))


def domains():
    meta=json.loads((ROOT/"data/multidomain_balanced/metadata.json").read_text())["validation"]
    texts=[(ROOT/meta[k]).read_text() for k in ("tiny_path","wiki_path")]
    size=meta["chunk_chars"]
    target=min(map(len,texts))
    texts=[_sample_evenly(s,target,size) for s in texts]
    chunks=[(i,s[j:j+size].encode()) for j in range(0,max(map(len,texts)),size)
            for i,s in enumerate(texts) if j<len(s)]
    data=b"\n\n".join(piece for _,piece in chunks)
    assert data==(ROOT/meta["output_path"]).read_bytes()
    labels=np.concatenate([a for i,(label,piece) in enumerate(chunks) for a in
                    ([np.full(2,2,dtype=np.uint8)] if i else [])+[np.full(len(piece),label,dtype=np.uint8)]])
    length=128
    n=(len(data)-1)//length
    seq=labels[1:1+n*length].reshape(n,length)
    rows=[np.bincount(s,minlength=3).tolist() for s in seq]
    branch.save_json(HERE/"validation_domains.json",dict(domains=["TinyStories","WikiText103","separator"],
                    target_counts_by_sequence=rows,source_sha256=branch.sha(ROOT/meta["output_path"])))

if __name__=="__main__":
    domains()
