"""Ordinary-gradient lifetime forecasts and separately labelled mechanistic controls."""
import argparse
import hashlib
import copy
import json
import math
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F

from lifetime_engineering import HERE, branch, copy_rows, masks, restore, verify_rows
from panels import select
from pathway import capture, components


@torch.no_grad()
def evaluate_panel(model,batches,device,details=False):
    modes=[(m,m.training) for m in model.modules()]
    state=branch.rng_get()
    model.eval()
    per_sequence=[]
    native_sum=0.
    try:
        for x,y in batches:
            x,y=x.to(device),y.to(device)
            logits=branch.N.recovery._condition_forward(model,x,False,1.).logits.flatten(0,1)
            losses=F.cross_entropy(logits,y.flatten(),reduction="none").double().reshape_as(y)
            per_sequence.extend(losses.sum(dim=1).cpu().tolist())
            native_sum+=F.cross_entropy(logits,y.flatten(),reduction="sum").item()
        count=sum(y.numel() for _,y in batches)
        total=math.fsum(per_sequence)
        assert abs(total-native_sum)/count < 2e-6
        result=dict(token_loss_sum=total,tokens=count,native_ce=native_sum/count,ce=total/count)
        if details:
            result["sequence_loss_sums"]=per_sequence
        return result
    finally:
        for mod,mode in modes:
            mod.training=mode
        branch.rng_set(state)


def attempt(model,opt,scaler,dtype,config,batch,active,extra=None,raw_routes=None):
    log={}
    handles=[]
    def observe(module,args,output):
        route=[]
        for t in output.routes:
            if raw_routes is not None:
                raw_routes.append(dict(experts=t.selected_expert_indices.detach().to(torch.uint8).cpu(),
                                       gates=t.selected_probabilities.detach().cpu(),accepted=t.accepted_mask.detach().cpu()))
            ids=t.selected_expert_indices.flatten()
            accepted=t.accepted_mask.flatten()
            g=t.selected_probabilities.detach().float().flatten()
            n=len(t.usage)
            route.append(dict(assigned=torch.bincount(ids,minlength=n).tolist(),
                accepted=torch.bincount(ids[accepted],minlength=n).tolist(),
                gate_sum=torch.zeros(n,device=g.device).scatter_add_(0,ids,g).tolist(),
                accepted_gate_sum=torch.zeros(n,device=g.device).scatter_add_(0,ids[accepted],g[accepted]).tolist(),
                drops=int((~accepted).sum())))
        log["routes"]=route
    handles.append(model.register_forward_hook(observe))
    scale=scaler.get_scale()
    for name,(rows,values) in (extra or {}).items():
        def modify(grad,rows=rows,values=values):
            return grad.index_add(0,rows,-scale*values)
        handles.append(dict(model.named_parameters())[name].register_hook(modify))
    clip=torch.nn.utils.clip_grad_norm_
    def clipping(parameters,max_norm):
        norm=clip(parameters,max_norm)
        value=float(norm)
        log.update(preclip_norm=value,clip_factor=None if math.isnan(value) else min(1.,max_norm/(value+1e-6)))
        return norm
    try:
        with patch.object(torch.nn.utils,"clip_grad_norm_",clipping):
            log["applied"]=branch.update(model,opt,scaler,dtype,config,batch,active,torch.device("cuda"))
        log.update(scale_before=scale,scale_after=scaler.get_scale())
        return log
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def contribution(model,batch,groups):
    # Mean absolute per-edge contribution |g V_ij h_j| over all routed
    # positions. A dropped assignment contributes zero.
    values={"H":0., "R":0.}
    info={}
    handles=[]
    layer=model.blocks[-1].moe
    for i,e in enumerate(layer.experts):
        def record(mod,args,out,index=i):
            info[index]=args[0]
        handles.append(e.fc2.register_forward_hook(record))
    try:
        model.eval()
        output=branch.N.recovery._condition_forward(model,batch[0].cuda(),False,1.)
        t=output.routes[-1]
        ids=t.selected_expert_indices.flatten()
        gate=t.selected_probabilities.flatten()
        accepted=t.accepted_mask.flatten()
        for i,h in info.items():
            key=f"blocks.{len(model.blocks)-1}.moe.experts.{i}.fc2.weight"
            pos=torch.nonzero(ids==i).flatten()
            if len(pos)>len(h):
                pos=pos[gate[pos].topk(len(h)).indices]
            assert accepted[pos].all()
            W=layer.experts[i].fc2.weight
            activation=(h.double().abs()*gate[pos,None].double()).sum(0)
            for arm,other in (("H","R"),("R","H")):
                idx=groups[key][arm]
                idx=idx[~torch.isin(idx,groups[key][other])].cuda()
                values[arm]+=(W.flatten()[idx].double().abs()*activation[idx%W.shape[1]]).sum().item()
    finally:
        model.train()
        for h in handles:
            h.remove()
    return values


def score(saved,config,batches,model,opt,scaler,dtype,groups,seed,order,t,out,engineering):
    start=time.monotonic()
    device=torch.device("cuda")
    active={label:masks(model,groups,k) for label,k in (("H","R"),("R","H"),("0","M"))}
    def reset(arm):
        restore(saved,model,opt,scaler)
        branch.enforce(model,opt,active[arm])
    pairs=select(config,seed,order,t,batches)
    reset("0")
    base=copy.deepcopy(model)
    readout=copy.deepcopy(base)
    magnitudes={}
    for arm,other in (("H","R"),("R","H")):
        vals=[]
        for n,g in groups.items():
            idx=g[arm][~torch.isin(g[arm],g[other])]
            vals.append(saved["model"][n].flatten()[idx].double().abs())
        magnitudes[arm]=torch.cat(vals).mean().item()
    logs=[]
    route_panels={}
    if out.exists():
        raise FileExistsError(out)
    identity=hashlib.sha256()
    for script in ("forecast.py","pathway.py","panels.py","lifetime_engineering.py"):
        identity.update(branch.sha(HERE/script).encode())
    identity.update(branch.sha(out.parent/"sets.pt").encode())
    identity.update(branch.N.recovery._batch_sequence_hash(batches[:t]).encode())
    identity.update(str((seed,order,t,engineering)).encode())
    partial=out.parent/f"probe_work_{t}"/identity.hexdigest()
    partial.mkdir(parents=True,exist_ok=True)
    for i,(u,r) in enumerate(pairs):
        receipt=partial/f"pair{i:02d}.pt"
        if receipt.exists():
            recovered=torch.load(receipt,map_location="cpu",weights_only=False)
            row=recovered["row"]
            assert (row["update_batch"],row["readout_batch"])==(u,r)
            logs.append(row)
            route_panels[u]=recovered["routes"]
            continue
        batch,measure=batches[u], [batches[r]]
        restore(saved,model,opt,scaler)
        activation=contribution(model,measure[0],groups)
        z=dict(update_batch=u,readout_batch=r,activation=activation,pre={},partner={},whole={},attempt={},
               common=evaluate_panel(base,measure,device))
        def donor(arm,extra=None):
            reset(arm)
            raw=[]
            detail=attempt(model,opt,scaler,dtype,config,batch,active[arm],extra,raw)
            if u in route_panels:
                assert branch.nested_equal(route_panels[u],raw)
            else:
                route_panels[u]=raw
            readout.load_state_dict(base.state_dict())
            copy_rows(readout,model,groups)
            verify_rows(readout,model,base,groups)
            partner=evaluate_panel(readout,measure,device)
            branch.enforce(model,opt,active["0"])
            whole=evaluate_panel(model,measure,device)
            return detail,partner,whole
        for arm in ("H","R","0"):
            reset(arm)
            z["pre"][arm]=evaluate_panel(model,measure,device)
            z["attempt"][arm],z["partner"][arm],z["whole"][arm]=donor(arm)
        if engineering and i==0:
            _,dup,_=donor("H")
            z["duplicate_floor"]=abs(dup["ce"]-z["partner"]["H"]["ce"])
            p=z["partner"]["R"]["ce"]-z["partner"]["H"]["ce"]
            assert z["partner"]["H"]["ce"]-z["partner"]["R"]["ce"] == -p
            z["controls"]={}
            reset("0")
            off=capture(model,batch,config,scaler,dtype)
            for arm in ("H","R"):
                reset(arm)
                on=capture(model,batch,config,scaler,dtype)
                block,shuffle,audit=components(on,off,groups,scaler.get_scale())
                z["controls"][arm]=dict(audit=audit,block=donor(arm,block),shuffle=donor(arm,shuffle))
        branch.save_pt(receipt,dict(row=z,routes=route_panels[u]))
        logs.append(z)
        print(seed,order,t,"probe pair",i+1,flush=True)
    def mean(kind,arm):
        records=[row[kind][arm] for row in logs]
        return math.fsum(v["token_loss_sum"] for v in records)/sum(v["tokens"] for v in records)
    effects={}
    for kind in ("partner","whole"):
        effects[kind]=dict(I_H=mean(kind,"0")-mean(kind,"H"),I_R=mean(kind,"0")-mean(kind,"R"),
                           P=mean(kind,"R")-mean(kind,"H"))
    baseline=dict(always_H=1.,always_R=-1.,magnitude=magnitudes["H"]-magnitudes["R"],
                  activation=math.fsum(row["activation"]["H"]-row["activation"]["R"] for row in logs),
                  damage=mean("pre","R")-mean("pre","H"),whole=effects["whole"]["P"])
    route_file=out.with_name(f"routes{t}.pt")
    branch.save_pt(route_file,route_panels)
    branch.save_json(out,dict(routes_sha256=branch.sha(route_file),seed=seed,order=order,t=t,batches=logs,effects=effects,baselines=baseline,
                     magnitudes=magnitudes,elapsed_s=time.monotonic()-start))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("seed",type=int,choices=(7,17,29))
    p.add_argument("order",choices=("A","B"))
    a=p.parse_args()
    saved,cfg,train,val,model,opt,scaler,dtype=branch.load_state(a.seed,a.order,torch.device("cuda"))
    folder=HERE/"groups"/f"seed{a.seed}_{a.order}"
    assert branch.sha(folder/"sets.pt")==json.loads((folder/"audit.json").read_text())["sha256"]
    groups=torch.load(folder/"sets.pt",weights_only=True)
    score(saved,cfg,train,model,opt,scaler,dtype,groups,a.seed,a.order,250,folder/"forecast250.json",True)
