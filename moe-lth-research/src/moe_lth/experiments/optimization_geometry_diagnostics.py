"""Fixed-probe diagnostics. All gradient comparisons select surviving coordinates."""
from __future__ import annotations
import csv, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from moe_lth.models.router import RouteOverride

STEPS=(0,100,250,500,750,1000,1500,2000,2500)

def digest(obj):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str): h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x: visit(v)
        else: h.update(repr(x).encode())
    visit(obj);return h.hexdigest()

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])

def append(path,rows):
    if not rows:return
    path=Path(path);exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));
        if not exists:w.writeheader()
        w.writerows(rows)

def stats(a,b):
    a=a.detach().float().reshape(-1);b=b.detach().float().reshape(-1)
    if a.numel()!=b.numel():raise ValueError('Coordinate mismatch')
    fa=torch.isfinite(a);fb=torch.isfinite(b);ok=bool(fa.all() and fb.all());n=a.numel()
    out={'finite':ok,'elements':n,'sparse_valid_elements':int(fa.sum()),'dense_valid_elements':int(fb.sum()),'sparse_zero_elements':int((a==0).sum()),'dense_zero_elements':int((b==0).sum()),'sparse_nonfinite_fraction':float((~fa).float().mean()) if n else None,'dense_nonfinite_fraction':float((~fb).float().mean()) if n else None,'sparse_zero_fraction':float((a==0).float().mean()) if n else None,'dense_zero_fraction':float((b==0).float().mean()) if n else None,'active_surviving_parameters':int((fa & (a!=0)).sum()),'sparse_norm':None,'masked_dense_norm':None,'norm_ratio':None,'dot_product':None,'cosine':None,'normalized_l2_difference':None,'status':'nonfinite' if not ok else 'empty'}
    if not ok or not n:return out
    # Vectors are fp32; fp64 reductions avoid accumulation overflow.
    a=a.double();b=b.double();na=a.norm();nb=b.norm();dot=a.dot(b)
    out.update(sparse_norm=float(na),masked_dense_norm=float(nb),norm_ratio=float(na/nb) if nb>0 else None,dot_product=float(dot),cosine=float(dot/(na*nb)) if na>0 and nb>0 else None,normalized_l2_difference=float((a-b).norm()/(nb+1e-12)),status='ok' if na>0 and nb>0 else 'zero_norm')
    return out

def vectors(model,masks,grad=False):
    groups={};params=dict(model.named_parameters())
    for name in sorted(params):
        if '.moe.experts.' not in name:continue
        parts=name.split('.');key=(int(parts[1]),int(parts[4]));p=params[name]
        value=(p.grad if p.grad is not None else torch.zeros_like(p)) if grad else p.detach()
        mask=masks.get(name,torch.ones_like(p,dtype=torch.bool)).to(p.device).bool()
        groups.setdefault(key,[]).append(value.detach().float()[mask].cpu())
    return {k:torch.cat(v) for k,v in groups.items()}

def pass_grad(model,x,y,masks,aux,overrides=None):
    model.zero_grad(set_to_none=True)
    # Unit loss scale: diagnostic gradients need no GradScaler unscale and never touch the training scaler.
    with torch.autocast(device_type=x.device.type,enabled=False):
        out=model(x,overrides)
        loss=F.cross_entropy(out.logits.reshape(-1,out.logits.shape[-1]),y.reshape(-1))+aux*out.auxiliary_loss
    loss.backward();v=vectors(model,masks,True);model.zero_grad(set_to_none=True)
    return out,v

def measure(sparse,dense,masks,probe,aux,outdir,identity,previous=None):
    """Caller audits optimizer/scaler; this function audits both complete model states."""
    outdir=Path(outdir);device=next(sparse.parameters()).device;x,y=[v.to(device) for v in probe]
    before=[digest(m.state_dict()) for m in (sparse,dense)];modes=[[a.training for a in m.modules()] for m in (sparse,dense)];rng=rng_state()
    try:
        sparse.eval();dense.eval()
        so,sg=pass_grad(sparse,x,y,masks,aux);do,dg=pass_grad(dense,x,y,masks,aux)
        native=[{**identity,'layer':l,'expert':e,**stats(sg[l,e],dg[l,e])} for l,e in sg]
        append(outdir/'gradient_alignment_native.csv',native)
        append(outdir/'gradient_norms.csv',[{**r,'routing_variant':'native'} for r in native])
        # Shared IDs AND shared capacity priorities guarantee identical accepted occurrences.
        forced=[RouteOverride(t.selected_expert_indices.detach(),capacity_scores=t.selected_probabilities.detach()) for t in so.routes]
        sf,sfg=pass_grad(sparse,x,y,masks,aux,forced);df,dfg=pass_grad(dense,x,y,masks,aux,forced)
        for a,b,c in zip(so.routes,sf.routes,df.routes):
            assert torch.equal(a.selected_expert_indices,b.selected_expert_indices) and torch.equal(a.selected_expert_indices,c.selected_expert_indices)
            assert torch.equal(a.accepted_mask,b.accepted_mask) and torch.equal(a.accepted_mask,c.accepted_mask),'Forced capacity mismatch'
        rows=[{**identity,'layer':l,'expert':e,**stats(sfg[l,e],dfg[l,e])} for l,e in sfg]
        append(outdir/'gradient_alignment_forced_sparse_routes.csv',rows)
        append(outdir/'gradient_norms.csv',[{**r,'routing_variant':'forced_sparse'} for r in rows])
        route=[];function=[]
        with torch.no_grad():
            for l,(a,b) in enumerate(zip(so.routes,do.routes)):
                p=a.probabilities.float();q=b.probabilities.float();mid=(p+q)/2
                js=.5*((p*(p.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)+(q*(q.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1))
                for e in range(sparse.config['num_experts']):
                    aa=a.selected_experts==e;bb=b.selected_experts==e;union=(aa|bb).sum()
                    route.append({**identity,'layer':l,'expert':e,'assignment_agreement':float((a.selected_experts==b.selected_experts).float().mean()),'token_set_jaccard':float((aa&bb).sum()/union) if union else None,'utilization_difference':float(aa.float().mean()-bb.float().mean()),'probability_js':float(js.mean()),'probability_tv':float((.5*(p-q).abs().sum(-1)).mean())})
                    accepted=(a.selected_experts.reshape(-1)==e)&a.accepted_mask.reshape(-1)
                    common=so.pre_router_hidden_states[l].reshape(-1,sparse.config['d_model'])[accepted]
                    u=sparse.blocks[l].moe.experts[e](common);v=dense.blocks[l].moe.experts[e](common)
                    z=stats(u,v)
                    function.append({**identity,'layer':l,'expert':e,'input_occurrences':len(common),'common_input_source':'sparse_native_pre_router_accepted_tokens','output_cosine':z['cosine'],'normalized_output_l2':z['normalized_l2_difference'],'output_norm_ratio':z['norm_ratio'],'mean_activation_difference':float((u-v).mean()) if u.numel() else None,'finite':z['finite'],'status':z['status']})
        append(outdir/'sparse_dense_routing_overlap.csv',route);append(outdir/'functional_expert_similarity.csv',function)
        sv=vectors(sparse,masks);dv=vectors(dense,masks)
        append(outdir/'parameter_trajectory_distance.csv',[{**identity,'layer':l,'expert':e,'distance':stats(sv[l,e],dv[l,e])['normalized_l2_difference']} for l,e in sv])
        if previous is not None:
            ps,pd,step=previous
            append(outdir/'update_alignment.csv',[{**identity,'interval_start':step,'interval_end':identity['step'],'layer':l,'expert':e,**stats(sv[l,e]-ps[l,e],dv[l,e]-pd[l,e])} for l,e in sv])
        # Two fixed equal microbatches: secondary estimate, never independent replicates.
        micro=[]
        for xx,yy in zip(x.chunk(2),y.chunk(2)):
            _,g=pass_grad(sparse,xx,yy,masks,aux);micro.append(g)
        coherence=[]
        for l,e in sv:
            z=stats(micro[0][l,e],micro[1][l,e]);a=z['sparse_norm'];b=z['masked_dense_norm'];c=z['cosine']
            coherence.append({**identity,'layer':l,'expert':e,'microbatch_count':2,'mean_pairwise_cosine':c,'negative_pair_fraction':float(c<0) if c is not None else None,'gradient_norm_cv':abs(a-b)/(a+b) if a is not None and b is not None and a+b>0 else None,'status':z['status']})
        append(outdir/'gradient_coherence.csv',coherence)
        return sv,dv,identity['step']
    finally:
        for m,states in zip((sparse,dense),modes):
            m.zero_grad(set_to_none=True)
            for module,mode in zip(m.modules(),states):module.training=mode
        restore_rng(rng)
        assert before==[digest(m.state_dict()) for m in (sparse,dense)],'Diagnostic model mutation'
