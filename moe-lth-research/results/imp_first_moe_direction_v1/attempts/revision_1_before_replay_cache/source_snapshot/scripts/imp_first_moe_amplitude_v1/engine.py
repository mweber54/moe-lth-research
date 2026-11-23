"""Global scalar amplitude control at the previously verified first-MoE interface."""
import math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_interface_v1 import engine as old
c=old.c;torch=old.torch;need=c._need
MODES=('NATIVE_DOWNSTREAM','RECIPIENT_BASELINE_REPLAY')
VARIANTS=('START_NATIVE','START_SCALED_TO_END','END_SCALED_TO_START','END_NATIVE')
TIMES=old.TIMES

def estimate_scale(ss0,ss1,elements):
    values=dict(start_sumsq=ss0,end_sumsq=ss1,elements=elements,a0=math.sqrt(ss0/elements) if elements and ss0>=0 and math.isfinite(ss0) else None,a1=math.sqrt(ss1/elements) if elements and ss1>=0 and math.isfinite(ss1) else None)
    a0,a1=values['a0'],values['a1'];ratio=a1/a0 if a0 and a1 else None
    usable=ratio is not None and math.isfinite(ratio) and ratio>0 and math.isfinite(1/ratio) and float(torch.tensor(ratio,dtype=torch.float32))>0 and math.isfinite(float(torch.tensor(ratio,dtype=torch.float32))) and float(torch.tensor(1/ratio,dtype=torch.float32))>0 and math.isfinite(float(torch.tensor(1/ratio,dtype=torch.float32)))
    return dict(values,scale=ratio if usable else None,status='AVAILABLE' if usable else 'UNAVAILABLE',reason='' if usable else 'Zero, nonfinite or FP32-unusable calibration amplitude; no epsilon or replacement scale')

def variant_tensor(donors,name,scale):
    need(name in VARIANTS,'Unknown contribution variant')
    d='start' if name.startswith('START') else 'end';native=donors[d]['c']
    if name in (VARIANTS[0],VARIANTS[3]):value=native;factor=1.
    else:
        need(scale is not None and math.isfinite(scale) and scale>0,'Unavailable scale cannot be used')
        value=native*scale if name==VARIANTS[1] else native/scale;factor=scale if name==VARIANTS[1] else 1/scale
    need(torch.isfinite(value).all(),'Scaling produced nonfinite contribution')
    need(torch.equal(value==0,native==0),'Scaling changed exact zero support')
    actual=math.sqrt(old.sq(value));expected=abs(factor)*math.sqrt(old.sq(native))
    need(abs(actual-expected)<=1e-6*max(expected,1e-30),'Scaling RMS linearity mismatch')
    return dict(donors[d],c=value,amplitude_variant=name,global_factor=factor,native_donor_c=native)

def insert(model,x,y,recipient,variant,mode):
    need(mode in MODES,'Unknown routing mode');count=[0]
    def before(module,args):count[0]+=1
    hook=model.blocks[0].moe.register_forward_pre_hook(before)
    try:output,observed=old.substitute(model,x,y,recipient,variant,1.,old.MODES[0] if mode==MODES[0] else old.MODES[1])
    finally:hook.remove()
    need(count[0]==1,'First-MoE insertion occurred other than exactly once')
    need(torch.equal(observed['inserted'],variant['c']),'Scaled contribution changed during insertion')
    return output,observed

def stats(caches,variants):
    a,b=caches['start']['c'],caches['end']['c'];flat0=a.reshape(-1,a.shape[-1]).double();flat1=b.reshape_as(flat0).double();n0=flat0.square().sum(-1).sqrt();n1=flat1.square().sum(-1).sqrt();valid=(n0>0)&(n1>0)
    cos=(flat0[valid]*flat1[valid]).sum(-1)/(n0[valid]*n1[valid]);ratios=n1[valid]/n0[valid]
    row=dict(old.compare(a,b),kind='contribution_pair',assignment_agree_count=int((caches['start']['ids']==caches['end']['ids']).sum()),acceptance_agree_count=int((caches['start']['accepted']==caches['end']['accepted']).sum()),start_dropped=int((~caches['start']['accepted']).sum()),end_dropped=int((~caches['end']['accepted']).sum()),gate_absolute_difference_sum=float((caches['start']['gates']-caches['end']['gates']).abs().double().sum()),norm_ratio_valid_positions=int(valid.sum()),norm_ratio_zero_exclusions=int((~valid).sum()))
    variants_rows=[]
    for name,v in variants.items():
        if v is None:continue
        ss=old.sq(v['c']);variants_rows.append(dict(kind='variant_amplitude',variant=name,sumsq=ss,elements=v['c'].numel(),rms=math.sqrt(ss/v['c'].numel()),positions=v['c'].shape[0]*v['c'].shape[1]))
    return row,variants_rows,cos.cpu(),ratios.cpu()

def distribution(values):
    if not len(values):return dict(count=0,mean=None,minimum=None,p10=None,p25=None,median=None,p75=None,p90=None,maximum=None)
    values=values.double();q=torch.quantile(values,torch.tensor([0.,.1,.25,.5,.75,.9,1.],dtype=torch.float64)).tolist()
    return dict(count=len(values),mean=float(values.mean()),**dict(zip(('minimum','p10','p25','median','p75','p90','maximum'),q)))

def evaluate(states,masks,config,batches,identity,device,scale=None,prior_rows=None,calibration=False,pilot=False):
    entry=c._rng_snapshot();models={};before={};rows=[];measurements=[];batch_ss=[];cosines=[];ratios=[];forwards=0;max_prior_error=0.;targets=0
    try:
        for t in TIMES:
            model=c.TinyMoELanguageModel(config).to(device);model.load_state_dict(states[t]);old.replay.install(model);before[t]=old.snapshot(model)
            need(before[t]['hash']==identity['checkpoints'][t]['state_sha256'],'Loaded state differs from source checkpoint')
            models[t]=model;model.eval()
        c._restore_rng(entry)
        for bi,(xx,yy) in enumerate(batches):
            tick=time.perf_counter();x,y=xx.to(device),yy.to(device);bh=c.common.batch_hash([(xx,yy)]);targets+=y.numel();caches={};native={}
            with torch.inference_mode(),c._precision_guard():
                for t in TIMES:
                    tag=dict(seed=identity['seed'],model_type=identity['model_type'],checkpoint=identity['checkpoints'][t],step=1000 if t=='start' else 1500,batch_hash=bh)
                    native[t],caches[t]=old.capture(models[t],x,y,tag);forwards+=1
                if calibration:
                    # No CE or task-loss calculation is performed in calibration.
                    batch_ss.append(dict(batch_index=bi,batch_hash=bh,start_sumsq=old.sq(caches['start']['c']),end_sumsq=old.sq(caches['end']['c']),elements=caches['start']['c'].numel(),target_tokens=y.numel()))
                    vs={name:variant_tensor(caches,name,1.) for name in (VARIANTS[0],VARIANTS[3])}
                else:vs={name:variant_tensor(caches,name,scale) if scale is not None or name in (VARIANTS[0],VARIANTS[3]) else None for name in VARIANTS}
                pair,amps,cos,ratio=stats(caches,vs);cosines.append(cos);ratios.append(ratio)
                base=dict(seed=identity['seed'],model_type=identity['model_type'],population='calibration' if calibration else 'validation',batch_index=bi,batch_hash=bh)
                measurements.append(dict(**base,**pair));measurements.extend(dict(**base,**r) for r in amps)
                for r in TIMES:
                    if calibration:
                        # Save sufficient statistics for ratios/perturbations; the final
                        # population scale is applied analytically only after all batches.
                        for d in TIMES:measurements.append(dict(**base,kind='calibration_recipient_moments',recipient=r,donor=d,u_sumsq=old.sq(caches[r]['u']),native_c_sumsq=old.sq(caches[r]['c']),donor_sumsq=old.sq(caches[d]['c']),donor_native_dot=float((caches[d]['c'].double()*caches[r]['c'].double()).sum()),elements=caches[r]['c'].numel()))
                        continue
                    for p in MODES:
                        for name in VARIANTS:
                            d='start' if name.startswith('START') else 'end';v=vs[name];unscaled=name in (VARIANTS[0],VARIANTS[3]);previous_mode=old.MODES[0] if p==MODES[0] else old.MODES[1]
                            ref=prior_rows[r,d,previous_mode,bi] if prior_rows is not None else None
                            if ref is not None:need(ref['batch_hash']==bh and int(ref['target_tokens'])==y.numel(),'Prior unscaled cache batch identity mismatch')
                            if v is None:
                                rows.append(dict(**base,recipient=r,mode=p,variant=name,nll_sum=None,target_tokens=y.numel(),status='UNAVAILABLE',source='unusable_calibration_scale'));continue
                            observed=dict(u=caches[r]['u'],inserted=v['c'],h=caches[r]['u']+v['c'])
                            # Every prior unscaled cell is freshly checked on the pilot;
                            # all other pairs check one complete representative batch.
                            evaluate_new=not unscaled or pilot or bi==0 or ref is None
                            if evaluate_new:
                                output,observed=insert(models[r],x,y,caches[r],v,p);forwards+=1
                                need(torch.isfinite(output.logits).all(),'Nonfinite task logits');nll=float(c._ce_vector(output.logits,y).double().sum());need(math.isfinite(nll),'Nonfinite NLL')
                                if unscaled and ref is not None:
                                    error=abs(nll-float(ref['nll_sum']))/y.numel();need(error<=1e-6,'Prior unscaled interface CE mismatch');max_prior_error=max(max_prior_error,error)
                                if unscaled and r==d:need(torch.allclose(output.logits,native[r].logits,atol=c.MODEL_ATOL,rtol=c.MODEL_RTOL),'Own native insertion mismatch')
                                del output
                            if unscaled and ref is not None:nll=float(ref['nll_sum']);source='verified_prior_interface'
                            else:source='new_scaled_evaluation' if not unscaled else 'fresh_unscaled_no_cache'
                            rows.append(dict(**base,recipient=r,mode=p,variant=name,nll_sum=nll,target_tokens=y.numel(),status='AVAILABLE',source=source))
                            ns=old.norms(caches[r],v,1.,observed)
                            measurements.append(dict(**base,kind='recipient_variant',recipient=r,mode=p,variant=name,**ns))
                del native,caches,vs
            print(dict(seed=identity['seed'],model_type=identity['model_type'],population=base['population'],batch=bi,seconds=time.perf_counter()-tick),flush=True)
    finally:
        for t,model in models.items():
            for module,training in before[t]['modes']:module.training=training
        c._restore_rng(entry)
    for t,model in models.items():old.verify(model,before[t],masks[t])
    need(c._rng_equal(entry,c._rng_snapshot()),'RNG state changed')
    return dict(identity=identity,rows=rows,measurements=measurements,calibration_batch_sums=batch_ss,distribution=dict(cosine=distribution(torch.cat(cosines)),position_norm_ratio=distribution(torch.cat(ratios))),forwards=forwards,target_tokens=targets,checks=dict(status='PASS',state_masks_gradients_rng_modes_tied_weights_preserved=True,scale_support_and_linearity=True,recipient_residual_and_exactly_one_insertion=True,recipient_baseline_replay_exact=True,current_hybrid_expert_inputs=True,coordinate_identity_exact=True,prior_unscaled_max_batch_ce_error=max_prior_error,calibration_uses_no_losses=calibration))
