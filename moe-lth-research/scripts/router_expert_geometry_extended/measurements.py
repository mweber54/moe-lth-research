"""Full-spectrum energy and the GELU analogue of paper Figure 2."""
from common import *
from scipy.stats import rankdata

def gram(w):
    x=w.detach().to(dtype=torch.float64)
    return x.T@x

def gram_compare(g,ref):
    n=float(torch.linalg.vector_norm(g));rn=float(torch.linalg.vector_norm(ref))
    trace=float(torch.trace(g));rt=float(torch.trace(ref))
    if n==0 or rn==0:
        return dict(shape_distance=None,cosine=None,trace=trace,reference_trace=rt,
                    trace_ratio=None if rt==0 else trace/rt,relative_unnormalized_distance=None if rn==0 else float(torch.linalg.vector_norm(g-ref))/rn,
                    zero_current_norm=n==0,zero_reference_norm=rn==0)
    cosine=float((g*ref).sum()/(n*rn))
    distance=float(torch.linalg.vector_norm(g/n-ref/rn))
    return dict(shape_distance=distance,cosine=cosine,trace=trace,reference_trace=rt,
                trace_ratio=trace/rt,relative_unnormalized_distance=float(torch.linalg.vector_norm(g-ref))/rn,
                zero_current_norm=False,zero_reference_norm=False)

def association(x,y,min_samples=None):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    minimum=CONFIG['minimum_samples'] if min_samples is None else min_samples
    if len(x)<minimum:return None,None,'insufficient_samples'
    if not (np.isfinite(x).all() and np.isfinite(y).all()):return None,None,'nonfinite'
    if x.std()<=CONFIG['variance_floor']:return None,None,'zero_variance_score'
    if y.std()<=CONFIG['variance_floor']:return None,None,'zero_variance_activation'
    z=(x-x.mean())/x.std();a=(y-y.mean())/y.std()
    rx=rankdata(x);ry=rankdata(y)
    spearman=float(np.corrcoef(rx,ry)[0,1]) if rx.std()>0 and ry.std()>0 else None
    return float((z*a).mean()),spearman,'valid'

def summary_distribution(v,prefix):
    v=np.asarray(v,dtype=float)
    if not len(v):return {prefix+k:None for k in ('mean','sd','min','q05','q50','q95','max')}
    vals=[v.mean(),v.std(),v.min(),*np.quantile(v,[.05,.5,.95]),v.max()]
    return {prefix+k:float(x) for k,x in zip(('mean','sd','min','q05','q50','q95','max'),vals)}

def summarize_pairs(scores,activations,selected,accepted,mode,layer):
    """Selections refer to unchanged native batch; frozen inputs are restored in token order."""
    groups=[];pooled=[];within=[]
    n,e=scores.shape
    for selection in ('proposed','accepted','all_expert'):
        masks=[]
        for expert in range(e):
            mask=np.ones(n,dtype=bool) if selection=='all_expert' else (selected==expert).any(1)
            if selection=='all_expert':mask[CONFIG['counterfactual_tokens']:]=False
            if selection=='accepted':mask=((selected==expert)&accepted).any(1)
            masks.append(mask)
        # All 31 cyclic shifts enumerate every nonmatching score row once per expert.
        correlations=np.full((e,e),np.nan);counts=np.array([m.sum() for m in masks]);zx=[];za=[]
        for expert,mask in enumerate(masks):
            all_x=scores[mask,:].astype(np.float64);x=all_x[:,expert];y=activations[mask,expert].astype(np.float64)
            pearson,spearman,status=association(x,y)
            row=dict(mode=mode,layer=layer,expert=expert,selection=selection,n_samples=int(mask.sum()),
                     pearson=pearson,spearman=spearman,status=status,negative_activation_count=int((y<0).sum()))
            row.update(summary_distribution(x,'raw_score_'));row.update(summary_distribution(y,'raw_activation_'));groups.append(row)
            if len(y)>=CONFIG['minimum_samples'] and y.std()>CONFIG['variance_floor']:
                sd=all_x.std(0);valid=sd>CONFIG['variance_floor'];cols=np.where(valid)[0]
                correlations[expert,cols]=(((all_x[:,cols]-all_x[:,cols].mean(0))/sd[cols])*((y-y.mean())/y.std())[:,None]).mean(0)
            if status=='valid':zx.append((x-x.mean())/x.std());za.append((y-y.mean())/y.std())
        actual_spearman=association(np.concatenate(zx),np.concatenate(za))[1] if zx else None
        results=[]
        for shift in range(e):
            coeffs=correlations[np.arange(e),(np.arange(e)+shift)%e];valid=np.isfinite(coeffs);total=int(counts[valid].sum())
            results.append(dict(shift=shift,pearson=float(np.average(coeffs[valid],weights=counts[valid])) if total else None,
                                spearman=actual_spearman if shift==0 else None,n_pairs=total,valid_experts=np.where(valid)[0].tolist(),
                                expert_mean_pearson=float(coeffs[valid].mean()) if total else None))
        actual=results[0];null=[r for r in results[1:] if r['pearson'] is not None]
        # Fixed valid expert intersection across identity and all shifts prevents coverage confounds.
        fixed=set(actual['valid_experts'])
        for r in results[1:]:fixed &= set(r['valid_experts'])
        ids=np.array(sorted(fixed),dtype=int);matched=[]
        for shift in range(e):
            matched.append(float(np.average(correlations[ids,(ids+shift)%e],weights=counts[ids])) if len(ids) else None)
        vals=[v for v in matched[1:] if v is not None]
        pooled.append(dict(mode=mode,layer=layer,selection=selection,pearson=actual['pearson'],spearman=actual['spearman'],
                           n_pairs=actual['n_pairs'],n_valid_experts=len(actual['valid_experts']),n_excluded_experts=e-len(actual['valid_experts']),
                           expert_mean_pearson=actual['expert_mean_pearson'],shuffled_mean_pearson=float(np.mean([r['pearson'] for r in null])) if null else None,
                           matched_shuffle_n_experts=len(fixed),matched_shuffle_n_pairs=int(counts[ids].sum()),matched_pearson=matched[0],
                           matched_shuffle_mean=float(np.mean(vals)) if vals else None,
                           pearson_above_shuffle=matched[0]-float(np.mean(vals)) if vals and matched[0] is not None else None,
                           shuffled_pearsons=json.dumps(matched[1:]),valid_experts=json.dumps(actual['valid_experts']),
                           matched_valid_experts=json.dumps(sorted(fixed))))
    for token in range(min(CONFIG['counterfactual_tokens'],n)):
        p,s,status=association(scores[token],activations[token])
        within.append(dict(mode=mode,layer=layer,probe_position_index=token,pearson=p,spearman=s,status=status,n_experts=e))
    return groups,pooled,within

def activation_probe(model,probe):
    """Bounded 1024-token inference. Native GELU including bias, before route weighting."""
    device=next(model.parameters()).device;pos=probe['positions'].to(device)
    native=model(probe['tokens'].to(device))
    result={};group_rows=[];pool_rows=[];within_rows=[];drift=[]
    for mode in ('fixed','native'):
        for layer,block in enumerate(model.blocks):
            if mode=='native':
                x=native.pre_router_hidden_states[layer].reshape(-1,model.config['d_model'])[pos]
                trace=native.routes[layer]
            else:
                x=probe['inputs'][layer].to(device).float()
                ordered=torch.empty_like(x);ordered[pos]=x
                _,_,trace=block.moe(ordered.reshape_as(native.pre_router_hidden_states[layer]))
            selected=trace.selected_expert_indices.reshape(-1,block.moe.top_k)[pos]
            accepted=trace.accepted_mask[pos]
            scores=block.moe.router.projection(x)
            # Evaluate the deployed GELU in bounded expert chunks, including native biases.
            chunks=[]
            for start in range(0,len(block.moe.experts),8):
                experts=list(block.moe.experts[start:start+8])
                weights=torch.stack([e.fc1.weight for e in experts]);biases=torch.stack([e.fc1.bias for e in experts])
                pre=torch.matmul(x.unsqueeze(0),weights.transpose(1,2))+biases[:,None,:]
                chunks.append(experts[0].activation(pre).mean(-1).T)
            acts=torch.cat(chunks,dim=-1)
            assert torch.isfinite(scores).all() and torch.isfinite(acts).all()
            xref=probe['inputs'][layer].to(device).float()
            drift.append(dict(mode=mode,layer=layer,relative_hidden_l2=float((x-xref).norm()/xref.norm()),
                              hidden_cosine=float(torch.nn.functional.cosine_similarity(x,xref).mean())))
            s,a,sel,acc=[t.detach().cpu().numpy() for t in (scores,acts,selected,accepted)]
            gs,ps,ws=summarize_pairs(s,a,sel,acc,mode,layer);group_rows+=gs;pool_rows+=ps;within_rows+=ws
            key=f'{mode}_L{layer}';result[key+'_scores']=s;result[key+'_activations']=a
            result[key+'_selected']=sel;result[key+'_accepted']=acc
    return group_rows,pool_rows,within_rows,drift,result
