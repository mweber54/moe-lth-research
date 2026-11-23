"""Isolated first-MoE activation substitution with recipient-only downstream replay."""
from __future__ import annotations
import sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_capacity_diagnostic_v1 import run as c
from scripts.imp_expert_contribution_ablation_v1 import replay
import torch
MODES=('NATIVE_RESPONSE','RECIPIENT_BASELINE_REPLAY')
ALPHAS=(1.,.9,.75,.5,0.)
TIMES=('start','end')
need=c._need

def snapshot(model):
    return dict(hash=c.state_dict_hash(model.state_dict()),grads=c._gradient_signature(model),modes=[(m,m.training) for m in model.modules()],tied=model.token_embedding.weight is model.lm_head.weight)

def verify(model,before,masks):
    need(c.state_dict_hash(model.state_dict())==before['hash'],'Model state changed')
    need(c._gradient_signature(model)==before['grads'],'Gradient state changed')
    need(before['tied'] and model.token_embedding.weight is model.lm_head.weight,'Tied identity changed')
    need(all(m.training==t for m,t in before['modes']),'Modes changed')
    for name,mask in masks.items():need(torch.count_nonzero(dict(model.named_parameters())[name][~mask.to(next(model.parameters()).device).bool()])==0,'Masked coordinates nonzero')

def capture(model,x,y,identity,chunk=1024):
    data={};handles=[]
    def residual(module,args):data['u']=args[0].detach().clone()
    def contribution(module,args,result):data['c']=result[0].detach().clone()
    def scores(module,args,result):data['scores']=result.detach().reshape(*x.shape,-1).clone()
    handles.append(model.blocks[0].moe_norm.register_forward_pre_hook(residual))
    handles.append(model.blocks[0].moe.register_forward_hook(contribution))
    handles.append(model.blocks[0].moe.router.projection.register_forward_hook(scores))
    try:out=model(x)
    finally:
        for h in handles:h.remove()
    data['z']=out.pre_router_hidden_states[0].clone()
    trace=out.routes[0];ids=trace.selected_expert_indices.reshape(-1)
    data['ids']=ids.clone();data['accepted']=trace.accepted_mask.reshape(-1).clone()
    data['gates']=trace.selected_probabilities.reshape(-1,1).clone()
    data['routing']=replay.capture(model,out,x,identity['step'],identity['batch_hash'])
    data['identity']=dict(identity,token_hash=replay.digest(x),target_hash=replay.digest(y),shape=list(x.shape))
    # Selected ungated expert output at every proposed position, including drops.
    # No all-expert activation tensor is formed. Chunking only this diagnostic
    # computation cannot change the already captured complete-batch routing.
    z=data['z'].reshape(-1,data['z'].shape[-1]);ungated=torch.empty_like(z)
    for eid,expert in enumerate(model.blocks[0].moe.experts):
        pos=torch.nonzero(ids==eid).flatten()
        for start in range(0,len(pos),chunk):ungated[pos[start:start+chunk]]=expert(z[pos[start:start+chunk]])
    data['ungated']=ungated.reshape_as(data['z'])
    for name in ('u','c','z','scores','gates','ungated'):need(torch.isfinite(data[name]).all(),f'Nonfinite capture {name}')
    return out,data

def substitute(model,x,y,recipient,donor,alpha,mode,expert_chunk_size=None):
    need(mode in MODES and alpha in ALPHAS,'Unregistered intervention')
    ri,di=recipient['identity'],donor['identity']
    for name in ('seed','model_type','batch_hash','token_hash','target_hash','shape'):
        need(ri[name]==di[name],f'Donor/recipient {name} mismatch')
    need(replay.digest(x)==ri['token_hash'] and replay.digest(y)==ri['target_hash'],'Input or target identity mismatch')
    payload=recipient['routing'];need(payload['step']==ri['step'] and payload['batch_hash']==ri['batch_hash'],'Recipient route identity mismatch')
    handles=[];observed={}
    def residual(module,args):
        need(torch.equal(args[0],recipient['u']),'Recipient residual changed before insertion')
        observed['u']=args[0]
    def replace(module,args,result):
        need(torch.equal(result[0],recipient['c']) if expert_chunk_size is None else torch.allclose(result[0],recipient['c'],atol=c.MODEL_ATOL,rtol=c.MODEL_RTOL),'Recipient first contribution changed before insertion')
        inserted=donor['c']*alpha
        need(torch.isfinite(inserted).all(),'Nonfinite insertion')
        observed['inserted']=inserted
        if alpha==0:need(torch.count_nonzero(inserted)==0,'Alpha zero nonzero')
        return inserted,result[1],result[2]
    def addition(module,args,result):
        expected=recipient['u']+observed['inserted'];need(torch.equal(result[0],expected),'First residual addition altered')
        observed['h']=result[0]
    handles.append(model.blocks[0].moe_norm.register_forward_pre_hook(residual))
    handles.append(model.blocks[0].moe.register_forward_hook(replace))
    handles.append(model.blocks[0].register_forward_hook(addition))
    try:
        if mode==MODES[1]:
            need(expert_chunk_size is None,'Replay does not support chunking')
            for i,b in enumerate(model.blocks[1:],1):
                b.moe._replay_required=True;b.moe._slot=payload['layers'][i]
        out=model(x,expert_chunk_size=expert_chunk_size)
        need(torch.equal(out.pre_router_hidden_states[0],recipient['z']),'First normalized input changed')
        if mode==MODES[1]:
            for i,trace in enumerate(out.routes[1:],1):
                slot=payload['layers'][i]
                for value,key in ((trace.selected_expert_indices,'ids'),(trace.selected_probabilities,'gates'),(trace.accepted_mask,'accepted')):
                    need(torch.equal(value.detach().cpu().reshape_as(slot[key]),slot[key]),'Downstream imposed route mismatch')
                need(model.blocks[i].moe._slot is None,'Replay slot not consumed')
    finally:
        for h in handles:h.remove()
        for b in model.blocks:b.moe._replay_required=False;b.moe._slot=None
    return out,observed

def sq(x):return float(x.double().square().sum())
def norms(rec,donor,alpha,observed):
    inserted=observed['inserted'];n=inserted.numel()
    values={'donor_contribution':donor['c'],'recipient_residual':rec['u'],'recipient_native_contribution':rec['c'],'inserted_contribution':inserted,'from_recipient_native_displacement':inserted-rec['c'],'attenuation_displacement':inserted-donor['c'],'post_addition_residual':observed['h']}
    row=dict(elements=n,positions=inserted.shape[0]*inserted.shape[1])
    for name,x in values.items():
        value=sq(x);row[name+'_sumsq']=value;row[name+'_rms']=(value/n)**.5;row[name+'_l2']=value**.5
    den=row['recipient_residual_l2'];row['inserted_to_recipient_residual_ratio']=row['inserted_contribution_l2']/den if den else None
    return row

def compare(a,b,mask=None):
    a=a.reshape(-1,a.shape[-1]).double();b=b.reshape_as(a).double()
    if mask is not None:a=a[mask.reshape(-1)];b=b[mask.reshape(-1)]
    na=a.square().sum(-1);nb=b.square().sum(-1);valid=(na>0)&(nb>0)
    ss0=float(na.sum());ss1=float(nb.sum());delta=float((b-a).square().sum());dot=float((a*b).sum())
    count=a.numel();cos=(a[valid]*b[valid]).sum(-1)/(na[valid]*nb[valid]).sqrt()
    return dict(positions=len(a),elements=count,start_sumsq=ss0,end_sumsq=ss1,displacement_sumsq=delta,dot_sum=dot,start_rms=(ss0/count)**.5 if count else None,end_rms=(ss1/count)**.5 if count else None,relative_l2=(delta/ss0)**.5 if ss0 else None,cosine_global=dot/(ss0*ss1)**.5 if ss0 and ss1 else None,cosine_sum=float(cos.sum()),cosine_valid_positions=int(valid.sum()),mean_position_cosine=float(cos.mean()) if valid.any() else None,both_zero_positions=int(((na==0)&(nb==0)).sum()),start_only_zero_positions=int(((na==0)&(nb>0)).sum()),end_only_zero_positions=int(((na>0)&(nb==0)).sum()),norm_ratio=(ss1/ss0)**.5 if ss0 else None,undefined_relative_l2=int(ss0==0),zero_cosine_exclusions=int((~valid).sum()))

def interface_rows(start,end,base):
    same=start['ids']==end['ids'];rows=[]
    for name in ('u','z','scores','gates','ungated','c'):
        for population in (('same_selected_expert','changed_selected_expert') if name=='ungated' else ('all',)):
            mask=(same if population=='same_selected_expert' else ~same) if name=='ungated' else None
            rows.append(dict(**base,kind='native_start_end_pair',quantity=name,population=population,**compare(start[name],end[name],mask)))
    rows.append(dict(**base,kind='native_start_end_pair',quantity='routing',positions=len(same),assignment_agree_count=int(same.sum()),acceptance_agree_count=int((start['accepted']==end['accepted']).sum()),start_dropped=int((~start['accepted']).sum()),end_dropped=int((~end['accepted']).sum()),start_c_to_u_ratio=(sq(start['c'])/sq(start['u']))**.5 if sq(start['u']) else None,end_c_to_u_ratio=(sq(end['c'])/sq(end['u']))**.5 if sq(end['u']) else None,gate_absolute_difference_sum=float((start['gates']-end['gates']).abs().double().sum())))
    return rows

def evaluate_pair(states,masks,config,batches,identity,device,preflight=False):
    entry=c._rng_snapshot();models={};before={};batch_rows=[];stats=[];checks={};unique=0
    try:
        for t in TIMES:
            model=c.TinyMoELanguageModel(config).to(device);model.load_state_dict(states[t]);replay.install(model)
            before[t]=snapshot(model)
            if isinstance(identity['checkpoints'][t],dict):need(before[t]['hash']==identity['checkpoints'][t]['state_sha256'],'Loaded model differs from checkpoint state')
            model.eval();models[t]=model
        c._restore_rng(entry)
        with torch.inference_mode(),c._precision_guard():
            for bi,(xx,yy) in enumerate(batches):
                tick=time.perf_counter();x,y=xx.to(device),yy.to(device);bh=c.common.batch_hash([(xx,yy)])
                base=dict(seed=identity['seed'],model_type=identity['model_type'],batch_index=bi,batch_hash=bh)
                natives={};cache={}
                for t in TIMES:
                    ci=dict(seed=identity['seed'],model_type=identity['model_type'],checkpoint=identity['checkpoints'][t],step=1000 if t=='start' else 1500,batch_hash=bh)
                    natives[t],cache[t]=capture(models[t],x,y,ci);unique+=1
                    # Verify contribution cache against an independent direct native forward.
                    if bi==0:
                        out,direct=capture(models[t],x,y,ci)
                        need(torch.equal(direct['c'],cache[t]['c']) and torch.equal(out.logits,natives[t].logits),'Donor cache direct recomputation mismatch');unique+=1
                        del out,direct
                if not preflight:stats+=interface_rows(cache['start'],cache['end'],base)
                for r in TIMES:
                    for mode in MODES:
                        zero_logits=None;zero_record=None
                        for d in TIMES:
                            for alpha in ((1.,0.) if preflight else ALPHAS):
                                if preflight and alpha==1. and r!=d:continue
                                reused=alpha==0 and d=='end'
                                if reused:
                                    record=dict(zero_record,donor=d,reused_alpha_zero=True)
                                    batch_rows.append(record)
                                    if not preflight:
                                        stats.extend(dict(row,donor=d,reused_alpha_zero=True) for row in zero_route_rows)
                                    # The linear zero insertion is independent of donor; same recipient,
                                    # mode and batch reuse the already validated alpha-zero output.
                                    if not preflight:
                                        observed=dict(inserted=cache[d]['c']*0.,h=cache[r]['u'])
                                        stats.append(dict(**base,kind='substitution',recipient=r,donor=d,alpha=alpha,mode=mode,**norms(cache[r],cache[d],alpha,observed)))
                                    continue
                                reused_baseline=bi>0 and r==d and alpha==1. and mode==MODES[0]
                                if reused_baseline:
                                    output=natives[r];observed=dict(u=cache[r]['u'],inserted=cache[r]['c'],h=cache[r]['u']+cache[r]['c'])
                                else:
                                    output,observed=substitute(models[r],x,y,cache[r],cache[d],alpha,mode);unique+=1
                                need(torch.isfinite(output.logits).all(),'Nonfinite task logits')
                                if r==d and alpha==1.:
                                    need(torch.allclose(output.logits,natives[r].logits,atol=c.MODEL_ATOL,rtol=c.MODEL_RTOL),'Same-checkpoint substitution baseline mismatch')
                                if alpha==0:
                                    zero_logits=output.logits.detach().clone()
                                    if bi==0:
                                        other,_=substitute(models[r],x,y,cache[r],cache['end'],0.,mode);unique+=1
                                        need(torch.equal(other.logits,zero_logits),'Alpha zero donor dependence');del other
                                nll=float(c._ce_vector(output.logits,y).double().sum());need(math.isfinite(nll),'Nonfinite task NLL')
                                record=dict(**base,recipient=r,donor=d,alpha=alpha,mode=mode,nll_sum=nll,target_tokens=y.numel(),reused_alpha_zero=False,reused_native_baseline=reused_baseline,nonfinite=0)
                                batch_rows.append(record)
                                if alpha==0:zero_record=record;zero_route_rows=[]
                                if not preflight:
                                    stats.append(dict(**base,kind='substitution',recipient=r,donor=d,alpha=alpha,mode=mode,**norms(cache[r],cache[d],alpha,observed)))
                                    for layer,tr in enumerate(output.routes[1:],1):
                                        ref=natives[r].routes[layer]
                                        route_row=dict(**base,kind='downstream_routing',quantity=f'layer_{layer}',recipient=r,donor=d,alpha=alpha,mode=mode,routing_kind='imposed recipient baseline' if mode==MODES[1] else 'native response',positions=tr.accepted_mask.numel(),assignment_agree_count=int((tr.selected_expert_indices==ref.selected_expert_indices).sum()),acceptance_agree_count=int((tr.accepted_mask==ref.accepted_mask).sum()),dropped=int((~tr.accepted_mask).sum()),gate_absolute_difference_sum=float((tr.selected_probabilities-ref.selected_probabilities).abs().double().sum()))
                                        stats.append(route_row)
                                        if alpha==0:zero_route_rows.append(route_row)
                                del output,observed
                        del zero_logits
                del natives,cache
                print(dict(seed=identity['seed'],model_type=identity['model_type'],batch=bi,preflight=preflight,seconds=time.perf_counter()-tick),flush=True)
    finally:
        for t,model in models.items():
            for module,training in before[t]['modes']:module.training=training
        c._restore_rng(entry)
    for t,model in models.items():verify(model,before[t],masks[t])
    need(c._rng_equal(entry,c._rng_snapshot()),'Evaluation RNG changed')
    checks=dict(status='PASS',donor_identity_and_recomputation=True,same_checkpoint_logits=True,alpha_zero_donor_independence=True,linear_insertion_and_recipient_residual_exact=True,downstream_replay_ids_gates_acceptance_exact=True,current_hidden_replay=True,model_state_gradients_rng_modes_tied_weights_masks_preserved=True,nonfinite=0)
    return dict(identity=identity,batches=batch_rows,interface=stats,checks=checks,unique_forwards=unique)
