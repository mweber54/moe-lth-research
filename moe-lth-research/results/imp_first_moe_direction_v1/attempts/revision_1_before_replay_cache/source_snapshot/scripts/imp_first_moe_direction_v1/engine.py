"""FP64 common-support per-token norm/direction construction, FP32 recipient forward."""
import math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_amplitude_v1 import engine as amp
old=amp.old;c=amp.c;torch=amp.torch;need=c._need
MODES=amp.MODES;TIMES=amp.TIMES;VARIANTS=('V00','V01','V10','V11')
NORM_RTOL=2e-6;DIRECTION_L2_ATOL=2e-6;UNIT_ATOL=2e-12

def common_parts(c0,c1):
    need(c0.shape==c1.shape and c0.dtype==c1.dtype==torch.float32,'Contribution shape/dtype mismatch')
    need(torch.isfinite(c0).all() and torch.isfinite(c1).all(),'Nonfinite native contribution')
    width=c0.shape[-1];a=c0.reshape(-1,width).double();b=c1.reshape_as(a).double();n0=a.square().sum(-1).sqrt();n1=b.square().sum(-1).sqrt();nz0=n0>0;nz1=n1>0;S=nz0&nz1
    # Exact nonzero eligibility. No epsilon/floor. If the FP32 cast is unusable,
    # fail before any hybrid forward; do not choose exclusions using CE.
    d0=a[S]/n0[S,None];d1=b[S]/n1[S,None];units=[d0,d1];norms=[n0[S],n1[S]];parts={};checks=[]
    uniterr=max(float((d.square().sum(-1).sqrt()-1).abs().max()) for d in units) if S.any() else None
    need(uniterr is None or uniterr<=UNIT_ATOL,'FP64 unit-vector reconstruction failed')
    for name in VARIANTS:
        ni,di=int(name[1]),int(name[2]);value=(norms[ni][:,None]*units[di]).to(c0.dtype);actual=value.double().square().sum(-1).sqrt();target=norms[ni]
        need(torch.isfinite(value).all(),'Nonfinite norm-direction cast')
        need(not len(actual) or torch.all(actual>0),'Numerically unusable nonzero target after FP32 casting')
        relative=(actual-target).abs()/target
        direction=(value.double()/actual[:,None]-units[di]).square().sum(-1).sqrt()
        maxrel=float(relative.max()) if len(relative) else None;maxdir=float(direction.max()) if len(direction) else None
        need(maxrel is None or maxrel<=NORM_RTOL,'FP32 cast violates target-norm contract; abort without CE-based threshold changes')
        need(maxdir is None or maxdir<=DIRECTION_L2_ATOL,'FP32 cast violates direction contract')
        reconstruct=name in ('V00','V11');native=a[S] if name=='V00' else b[S]
        if reconstruct:need(torch.equal(value,native.float()),'Diagonal contribution does not reconstruct native on S')
        checks.append(dict(variant=name,norm_donor=ni,direction_donor=di,eligible_positions=int(S.sum()),target_norm_sumsq=float(target.square().sum()),target_norm_minimum=float(target.min()) if len(target) else None,maximum_absolute_norm_error=float((actual-target).abs().max()) if len(actual) else None,maximum_relative_norm_error=maxrel,mean_relative_norm_error=float(relative.mean()) if len(relative) else None,maximum_direction_l2_error=maxdir,fp64_unit_norm_maximum_error=uniterr,diagonal_bitwise_reconstruction=True if reconstruct else None,status='PASS' if len(actual) else 'NO_ELIGIBLE_POSITIONS'))
        parts[name]=value
    for pair,target in [(('V00','V01'),norms[0]),(('V10','V11'),norms[1])]:
        delta=(parts[pair[0]].double().square().sum(-1).sqrt()-parts[pair[1]].double().square().sum(-1).sqrt()).abs()/target
        error=float(delta.max()) if len(delta) else None;need(error is None or error<=NORM_RTOL,'Fixed-norm pair norms differ')
        for row in checks:
            if row['variant'] in pair:row['paired_norm_maximum_relative_error']=error
    def positive_min(n):return float(n[n>0].min()) if (n>0).any() else None
    energy0=float(n0.square().sum());energy1=float(n1.square().sum());in0=float(n0[S].square().sum());in1=float(n1[S].square().sum())
    coverage=dict(positions=len(n0),eligible_positions=int(S.sum()),both_nonzero_positions=int((nz0&nz1).sum()),start_only_nonzero_positions=int((nz0&~nz1).sum()),end_only_nonzero_positions=int((~nz0&nz1).sum()),both_zero_positions=int((~nz0&~nz1).sum()),start_nonzero_positions=int(nz0.sum()),end_nonzero_positions=int(nz1.sum()),exact_zero_exclusions=int((~S).sum()),numerical_floor_exclusions=0,numerical_unusable_positions=0,start_energy_total=energy0,end_energy_total=energy1,start_energy_within_S=in0,end_energy_within_S=in1,start_energy_outside_S=float(n0[~S].square().sum()),end_energy_outside_S=float(n1[~S].square().sum()),minimum_positive_start_norm=positive_min(n0),minimum_positive_end_norm=positive_min(n1))
    coverage=ratios(coverage)
    return dict(S=S,parts=parts,checks=checks,coverage=coverage,cosine=(d0*d1).sum(-1).cpu(),shape=c0.shape)

def ratios(row):
    row=dict(row);n=row['eligible_positions']
    for name,denom in [('eligible_fraction_all','positions'),('eligible_fraction_start_nonzero','start_nonzero_positions'),('eligible_fraction_end_nonzero','end_nonzero_positions')]:row[name]=n/row[denom] if row[denom] else None
    for t in TIMES:
        row[t+'_energy_fraction_in_S']=row[t+'_energy_within_S']/row[t+'_energy_total'] if row[t+'_energy_total'] else None
        if t+'_accepted_positions' in row:row[t+'_accepted_coverage_in_S']=row[t+'_accepted_within_S']/row[t+'_accepted_positions'] if row[t+'_accepted_positions'] else None
    return row

def hybrid(common,recipient,name):
    need(name in VARIANTS,'Unknown norm-direction variant');flat=recipient['c'].reshape(-1,recipient['c'].shape[-1]);value=flat.clone();value[common['S']]=common['parts'][name]
    need(torch.equal(value[~common['S']],flat[~common['S']]),'Unsupported positions changed')
    return dict(recipient,c=value.reshape_as(recipient['c']),norm_direction_variant=name,common_support_only=True,outside_support_background='recipient_native')

def evaluate(states,masks,config,batches,identity,device,prior_rows=None,calibration=False):
    entry=c._rng_snapshot();models={};before={};losses=[];coverage=[];norm_checks=[];interfaces=[];cosines=[];forwards=0;maxnative=0.
    try:
        for t in TIMES:
            model=c.TinyMoELanguageModel(config).to(device);model.load_state_dict(states[t]);old.replay.install(model);before[t]=old.snapshot(model);need(before[t]['hash']==identity['checkpoints'][t]['state_sha256'],'Loaded state differs from source');models[t]=model;model.eval()
        c._restore_rng(entry)
        for bi,(xx,yy) in enumerate(batches):
            tick=time.perf_counter();x,y=xx.to(device),yy.to(device);bh=c.common.batch_hash([(xx,yy)]);cache={};native={}
            with torch.inference_mode(),c._precision_guard():
                for t in TIMES:
                    tag=dict(seed=identity['seed'],model_type=identity['model_type'],checkpoint=identity['checkpoints'][t],step=1000 if t=='start' else 1500,batch_hash=bh)
                    native[t],cache[t]=old.capture(models[t],x,y,tag);forwards+=1
                common=common_parts(cache['start']['c'],cache['end']['c']);S=common['S'];cosines.append(common['cosine']);base=dict(seed=identity['seed'],model_type=identity['model_type'],population='training_probe' if calibration else 'validation',batch_index=bi,batch_hash=bh)
                cov=dict(common['coverage'])
                for t in TIMES:
                    accepted=cache[t]['accepted'];cov[t+'_accepted_positions']=int(accepted.sum());cov[t+'_accepted_within_S']=int((accepted&S).sum())
                cov.update(assignment_agree_all=int((cache['start']['ids']==cache['end']['ids']).sum()),assignment_agree_in_S=int(((cache['start']['ids']==cache['end']['ids'])&S).sum()),acceptance_agree_all=int((cache['start']['accepted']==cache['end']['accepted']).sum()),gate_absolute_difference_sum_all=float((cache['start']['gates']-cache['end']['gates']).abs().double().sum()),gate_absolute_difference_sum_in_S=float((cache['start']['gates'][S]-cache['end']['gates'][S]).abs().double().sum()))
                coverage.append(dict(**base,scope='complete_original_batch',**ratios(cov)));norm_checks.extend(dict(**base,scope='common_positions_recipient_independent',**z) for z in common['checks'])
                for recipient in TIMES:
                    for name in VARIANTS:
                        v=hybrid(common,cache[recipient],name)
                        if (recipient,name) in (('start','V00'),('end','V11')):need(torch.equal(v['c'],cache[recipient]['c']),'Own diagonal is not native contribution')
                        if calibration:
                            obs=dict(inserted=v['c'],h=cache[recipient]['u']+v['c'])
                            interfaces.append(dict(**base,recipient=recipient,mode='not_applicable_training_probe',variant=name,scope='complete_population_with_recipient_background',**old.norms(cache[recipient],v,1.,obs)))
                            continue
                        for mode in MODES:
                            output,observed=amp.insert(models[recipient],x,y,cache[recipient],v,mode);forwards+=1
                            need(torch.isfinite(output.logits).all(),'Nonfinite logits');ce=c._ce_vector(output.logits,y).reshape_as(yy).double();nll=float(ce.sum());need(math.isfinite(nll),'Nonfinite NLL')
                            sequence=ce.sum(dim=1).tolist()
                            if (recipient,name) in (('start','V00'),('end','V11')):
                                need(torch.allclose(output.logits,native[recipient].logits,atol=c.MODEL_ATOL,rtol=c.MODEL_RTOL),'Own diagonal logits differ from native')
                                if prior_rows is not None:
                                    prior_mode=old.MODES[0] if mode==MODES[0] else old.MODES[1];ref=prior_rows[recipient,recipient,prior_mode,bi]
                                    need(ref['batch_hash']==bh and int(ref['target_tokens'])==y.numel(),'Native reference coordinates differ')
                                    error=abs(nll-float(ref['nll_sum']))/y.numel();need(error<=1e-6,'Own diagonal CE differs from previous native');maxnative=max(maxnative,error)
                            losses.append(dict(**base,recipient=recipient,mode=mode,variant=name,nll_sum=nll,target_tokens=y.numel(),sequence_count=y.shape[0],sequence_length=y.shape[1],sequence_nll=sequence,scope='whole_validation_population'))
                            interfaces.append(dict(**base,recipient=recipient,mode=mode,variant=name,scope='complete_population_with_recipient_background',**old.norms(cache[recipient],v,1.,observed)))
                            del output,observed
                        del v
                del cache,native,common
            print(dict(seed=identity['seed'],model_type=identity['model_type'],population=base['population'],batch=bi,seconds=time.perf_counter()-tick),flush=True)
    finally:
        for t,model in models.items():
            for module,training in before[t]['modes']:module.training=training
        c._restore_rng(entry)
    for t,model in models.items():old.verify(model,before[t],masks[t])
    need(c._rng_equal(entry,c._rng_snapshot()),'RNG changed')
    return dict(identity=identity,losses=losses,coverage=coverage,norm_checks=norm_checks,interfaces=interfaces,cosine_distribution=amp.distribution(torch.cat(cosines)),forwards=forwards,checks=dict(status='PASS',exact_nonzero_eligibility=True,no_floor_or_epsilon=True,unusable_cast_aborts_before_hybrid_CE=True,norm_and_direction_cast_contract=True,diagonals_reconstruct_on_S=True,outside_S_recipient_native_exact=True,recipient_residual_and_once_only_insertion=True,recipient_baseline_replay_exact=True,experts_use_current_inputs=True,state_masks_gradients_rng_modes_ties_preserved=True,maximum_previous_native_ce_error=maxnative,calibration_no_losses=calibration))
