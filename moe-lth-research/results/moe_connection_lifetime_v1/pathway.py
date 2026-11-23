"""Last-layer CE feedback decomposition at the native loss scale."""
import math

import torch
import torch.nn.functional as F

from groups import branch
from lifetime_engineering import restore


def capture(model,batch,config,scaler,dtype):
    records={}
    handles=[]
    for i,expert in enumerate(model.blocks[-1].moe.experts):
        def store(mod,args,out,index=i):
            records[index]=(args[0],out)
        handles.append(expert.fc1.register_forward_hook(store))
    q={}
    for i,expert in enumerate(model.blocks[-1].moe.experts):
        def register(mod,args,out,index=i):
            out.register_hook(lambda grad: q.update({index:grad.detach()}))
        handles.append(expert.fc2.register_forward_hook(register))
    try:
        x,y=(v.cuda() for v in batch)
        with torch.autocast("cuda",dtype=dtype):
            out=branch.N.recovery._condition_forward(model,x,False,1.)
            loss=F.cross_entropy(out.logits.flatten(0,1),y.flatten())
        model.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        result={}
        for i,(x,a) in records.items():
            e=model.blocks[-1].moe.experts[i]
            assert e.dropout.p==0
            name=f"blocks.{len(model.blocks)-1}.moe.experts.{i}.fc2.weight"
            result[name]=(x.detach().to(a.dtype),a.detach(),q[i],e.fc2.weight.detach().to(a.dtype).clone(),e.fc1.weight.grad.detach().clone())
        trace=out.routes[-1]
        return result, (trace.selected_expert_indices.detach(),trace.accepted_mask.detach(), trace.selected_probabilities.detach())
    finally:
        for h in handles:
            h.remove()


def components(on,off,groups,scale):
    first, second = on[0],off[0]
    assert first.keys()==second.keys() and all(torch.equal(a,b) for a,b in zip(on[1],off[1]))
    blocked,shuffled={},{}
    residual=reference=algebra_residual=algebra_reference=0.
    generator=torch.Generator(device="cuda").manual_seed(374)
    for name,(x,a,q,v,grad) in first.items():
        x0,a0,q0,v0,grad0=second[name]
        assert torch.equal(x,x0) and torch.equal(a,a0)
        rows=groups[name]["S"].cuda()
        a=a[:,rows].double()
        jac=0.5*(1+torch.erf(a/math.sqrt(2)))+a*torch.exp(-0.5*a*a)/math.sqrt(2*math.pi)
        direct=q.double() @ (v[:,rows].double()-v0[:,rows].double())
        indirect=(q.double()-q0.double()) @ v0[:,rows].double()
        feedback_delta=q.double()@v[:,rows].double()-q0.double()@v0[:,rows].double()
        difference=direct+indirect-feedback_delta
        algebra_residual+=difference.square().sum().item()
        algebra_reference+=feedback_delta.square().sum().item()
        observed=grad[rows].double()-grad0[rows].double()
        predicted=((direct+indirect)*jac).T @ x.double()
        residual+=(predicted-observed).square().sum().item()
        reference+=observed.square().sum().item()
        perturb=(direct*jac).T @ x.double()
        perm=torch.randperm(len(x),device="cuda",generator=generator)
        scramble=(direct[perm]*jac).T @ x.double()
        key=name.replace("fc2.weight","fc1.weight")
        blocked[key]=(rows,(perturb/scale).float())
        shuffled[key]=(rows,(scramble/scale).float())
    norm=lambda entries: math.sqrt(sum(t.double().square().sum().item() for _,t in entries.values()))
    target,raw=norm(blocked),norm(shuffled)
    degenerate=not(target and raw)
    if not degenerate:
        for rows,t in shuffled.values():
            t.mul_(target/raw)
    algebra_error=math.sqrt(algebra_residual/max(algebra_reference,1e-300))
    assert algebra_error < 1e-10
    if not degenerate:
        assert math.isclose(target,norm(shuffled),rel_tol=1e-6)
    err=math.sqrt(residual/max(reference,1e-300))
    if not math.isfinite(err) or err>0.05:
        raise RuntimeError(f"feedback residual {err}")
    return blocked,shuffled,dict(relative_difference_error=err,fp64_identity_error=algebra_error,direct_norm=target,
                                  shuffled_norm=norm(shuffled),degenerate=degenerate)
