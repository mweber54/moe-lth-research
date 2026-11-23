"""Recover raw gate telemetry for early development forecasts from their batch IDs."""
import argparse
import json
import torch
from groups import HERE,branch
from lifetime_engineering import masks,restore
from prefix_state import material


def replay(seed,order,t):
    folder=HERE/"groups"/f"seed{seed}_{order}"
    rec=json.loads((folder/f"forecast{t}.json").read_text())
    rawfile=folder/f"routes{t}.pt"
    if rawfile.exists():
        digest=rec.get("routes_sha256")
        if digest is None:
            digest=json.loads((folder/f"route_replay{t}.json").read_text())["sha256"]
        assert branch.sha(rawfile)==digest
        return
    assert seed in (7,17,29)
    saved,cfg,batches,_,model,opt,scaler,dtype=material(seed,order,t)
    groups=torch.load(folder/"sets.pt",weights_only=True)
    masks_by_arm={a:masks(model,groups,k) for a,k in (("H","R"),("R","H"),("0","M"))}
    panels={}
    with torch.no_grad():
        for row in rec["batches"]:
            u=row["update_batch"]
            for arm,mask in masks_by_arm.items():
                restore(saved,model,opt,scaler)
                branch.enforce(model,opt,mask)
                with torch.autocast("cuda",dtype=dtype):
                    out=branch.N.recovery._condition_forward(model,batches[u][0].cuda(),False,1.)
                raw=[]
                for route,expected in zip(out.routes,row["attempt"][arm]["routes"]):
                    ids=route.selected_expert_indices.flatten()
                    n=len(route.usage)
                    g=route.selected_probabilities.float().flatten()
                    assert torch.bincount(ids,minlength=n).tolist()==expected["assigned"]
                    assert torch.bincount(ids[route.accepted_mask.flatten()],minlength=n).tolist()==expected["accepted"]
                    sums=torch.zeros(n,device="cuda").scatter_add_(0,ids,g)
                    torch.testing.assert_close(sums.cpu(),torch.tensor(expected["gate_sum"]),rtol=2e-5,atol=1e-5)
                    raw.append(dict(experts=route.selected_expert_indices.to(torch.uint8).cpu(),gates=route.selected_probabilities.cpu(),
                                    accepted=route.accepted_mask.cpu()))
                if u in panels:
                    assert branch.nested_equal(raw,panels[u])
                panels[u]=raw
    branch.save_pt(rawfile,panels)
    branch.save_json(folder/f"route_replay{t}.json",dict(sha256=branch.sha(rawfile),forecast_sha256=branch.sha(folder/f"forecast{t}.json")))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("seed",type=int)
    p.add_argument("order")
    p.add_argument("t",type=int)
    a=p.parse_args()
    replay(a.seed,a.order,a.t)
