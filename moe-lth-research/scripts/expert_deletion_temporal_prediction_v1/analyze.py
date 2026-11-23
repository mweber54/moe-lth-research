"""Read-only temporal statistics. No model code is imported or executed."""
import itertools
import json
import time
import hashlib
import resource
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from prepare import ROOT, D, SOURCE, sha, save

FLOOR = 1e-12
R = 2000

def jd(x):
    def clean(v):
        if isinstance(v, dict): return {str(k):clean(w) for k,w in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)): return [clean(w) for w in v]
        if isinstance(v,(np.bool_,bool)): return bool(v)
        if isinstance(v,(np.integer,int)): return int(v)
        if isinstance(v,(np.floating,float)): return float(v) if np.isfinite(v) else None
        return v
    return json.dumps(clean(x),sort_keys=True,allow_nan=False)

def ratio(n,d):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(np.asarray(d)>0,np.asarray(n)/np.asarray(d),np.nan)

def ci(x):
    x=np.asarray(x,float); finite=x[np.isfinite(x)]
    return dict(ci95_low=float(np.quantile(finite,.025)) if len(finite) else None,
                ci95_high=float(np.quantile(finite,.975)) if len(finite) else None,
                undefined_bootstrap_fraction=float(1-len(finite)/len(x)))

def score(current, previous=None, rule='current'):
    if rule=='current': return np.asarray(current)
    if previous is None: raise ValueError('history unavailable')
    if rule=='history_mean': return (np.asarray(current)+previous)/2
    if rule=='history_max': return np.maximum(current,previous)
    raise ValueError(rule)

def select(scores,ids):
    a=np.asarray(scores,float)
    if len(a)<2 or not np.isfinite(a).all(): return None,None,True
    order=sorted(range(len(a)),key=lambda i:(a[i],ids[i]))
    lo,hi=order[0],order[-1]
    tied=bool(np.sum(np.abs(a-a[lo])<=FLOOR)>1 or np.sum(np.abs(a-a[hi])<=FLOOR)>1)
    return lo,hi,tied

def rho(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if len(a)<3 or not (np.isfinite(a).all() and np.isfinite(b).all()): return None
    if np.ptp(a)<=FLOOR or np.ptp(b)<=FLOOR: return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])

def ties(a):
    return sum(bool(np.isfinite(a[i]) and np.isfinite(a[j]) and abs(a[i]-a[j])<=FLOOR)
               for i,j in itertools.combinations(range(len(a)),2))

def bootstrap_rho(a,b):
    result=np.full(len(a),np.nan)
    if a.shape[1]<3: return result
    valid=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    valid &= (np.ptp(a,axis=1)>FLOOR)&(np.ptp(b,axis=1)>FLOOR)
    x=rankdata(a[valid],axis=1);y=rankdata(b[valid],axis=1)
    x-=x.mean(1,keepdims=True);y-=y.mean(1,keepdims=True)
    result[valid]=(x*y).sum(1)/np.sqrt((x*x).sum(1)*(y*y).sum(1))
    return result

def boot_select(a,ids):
    lo=np.full(len(a),-1,int);hi=lo.copy();tied=np.ones(len(a),bool)
    for k,row in enumerate(a):
        l,h,t=select(row,ids)
        if l is not None: lo[k]=l;hi[k]=h;tied[k]=t
    return lo,hi,tied

def picked(a,idx):
    result=np.full(len(a),np.nan);valid=idx>=0
    result[valid]=a[np.flatnonzero(valid),idx[valid]]
    return result

def policy_difference(future,current_id,rule_id):
    if current_id==rule_id: return np.zeros(np.asarray(future).shape[:-1])
    return future[...,current_id]-future[...,rule_id]

def coordinate_union(sets,selected):
    return set().union(*(sets[i] for i in selected)) if len(selected) else set()

def summarize(num,den,idx,weights):
    a=num[idx];b=den[idx]
    return ratio(a.sum(0),b.sum()),ratio(weights@a,(weights@b)[:,None])

def verify_inputs():
    started=time.perf_counter();config=json.loads((D/'analysis_config.json').read_text())
    freeze=json.loads((D/'freeze.json').read_text())
    assert sha(D/'analysis_config.json')==freeze['configuration_sha256']
    assert sha(D/'protocol.md')==freeze['protocol_sha256']
    assert sha(SOURCE/'output_seal.json')==config['original_output_seal_sha256']
    original=json.loads((SOURCE/'configuration.json').read_text())
    assert sha(SOURCE/'configuration.json')==config['original_configuration_sha256']
    verified={}
    def check(p,h):
        p=Path(p);rel=str(p.relative_to(ROOT))
        if rel not in verified:
            actual=sha(p);assert actual==h,(rel,actual,h);verified[rel]=actual
        else: assert verified[rel]==h
    for rel,h in json.loads((SOURCE/'output_seal.json').read_text())['files'].items(): check(SOURCE/rel,h)
    for rel,h in original['source_files'].items(): check(ROOT/rel,h)
    oldfreeze=json.loads((SOURCE/'freeze.json').read_text())
    for rel,h in oldfreeze['implementation'].items(): check(ROOT/rel,h)
    checkpoint_hashes=[]
    for condition,steps in original['checkpoints'].items():
        assert set(steps)=={'25','250','1000','2500'}
        for t,item in steps.items():
            check(ROOT/item['path'],item['sha256']);assert item['segment_offset']==0
            checkpoint_hashes.append(item['model_hash'])
    assert len(set(checkpoint_hashes))==8
    assert json.loads((SOURCE/'final_audit.json').read_text())['passed']
    nc=json.loads((SOURCE/'numerical_checks.json').read_text());assert nc['passed']
    # Only coordinate tensors are loaded. No checkpoint/model tensors are deserialized.
    import torch
    torch.set_num_threads(2)
    coords=torch.load(ROOT/'results/expert_local_deletion_dose_v1/intervention_coordinates.pt',map_location='cpu',weights_only=True)
    groups=torch.load(ROOT/'results/expert_role_deletion_v1/group_coordinates.pt',map_location='cpu',weights_only=True)
    manifest=pd.read_csv(SOURCE/'intervention_manifest.csv').set_index('intervention')
    panel={};identities=set()
    for item in coords:
        iid=item['intervention'];idx=item['indices'].contiguous();row=manifest.loc[iid]
        digest=hashlib.sha256((str(idx.dtype)+str(tuple(idx.shape))).encode()+idx.numpy().tobytes()).hexdigest()
        if item['kind']=='weight_union': assert digest==row.coordinate_sha256
        else: assert pd.isna(row.coordinate_sha256) and len(idx)==0
        assert item['parameter']==row.parameter and item['expert']==row.expert
        assert (len(idx)==row.coordinates if item['kind']=='weight_union' else pd.isna(row.coordinates))
        assert len(set(idx.tolist()))==len(idx)
        if item['kind']=='weight_union':
            original_union=set().union(*(set(groups[g]['indices'].tolist()) for g in item['groups']))
            assert original_union==set(idx.tolist())
        key=(int(row.expert),row.parameter,digest,item['kind'])
        assert key not in identities;identities.add(key)
        panel[iid]=dict(item,coordinate_sha256=digest,identity='|'.join(map(str,key)),coordinate_set=set(idx.tolist()))
    assert len(panel)==56 and len(manifest)==56
    probe=pd.read_csv(SOURCE/'probe_manifest.csv')
    assert len(probe)==4096 and probe.batch.nunique()==32
    assert np.all(probe.block==probe.batch//4+1)
    assert np.all(probe.partition==np.where(probe.batch<16,'discovery','confirmation'))
    assert probe.sequence_id.nunique()==4096
    checkpoint_inventory=pd.read_csv(SOURCE/'artifact_checkpoint_group_inventory.csv')
    provenance=dict(verified_files=verified,verified_file_count=len(verified),
        checkpoint_identities=original['checkpoints'],checkpoint_tensor_reads=0,
        excluded_aliases=checkpoint_inventory[checkpoint_inventory.use!='selected_once'].fillna('').to_dict('records'),
        original_numerical_checks_sha256=sha(SOURCE/'numerical_checks.json'),
        model_hashes_verified_via_exact_file_hash_and_prior_audit=True,
        coordinate_tensors_only=True,coordinate_joins=56,disjoint_probe_sequences=4096,
        original_source_case=original['source_case'],verification_seconds=time.perf_counter()-started,
        magnitude_baseline=config['magnitude_baseline'])
    save(D/'input_provenance.json',provenance)
    return config,panel,manifest,verified

def run():
    started=time.perf_counter();config,panel,manifest,verified=verify_inputs()
    print('Inputs verified; protocol already frozen',flush=True)
    df=pd.read_csv(SOURCE/'deletion_damage_by_batch.csv')
    assert len(df)==43008 and not df.duplicated(['trajectory','step','batch','intervention','subset']).any()
    assert set(df.seed)=={17} and set(df.layer)=={1}
    cancellation_error=float(np.max(np.abs(df.damage_sum-(df.deleted_loss_sum-df.baseline_loss_sum))))
    assert cancellation_error<2e-10  # CSV parse plus cancellation of large CE sums
    assert np.allclose(df.damage,ratio(df.damage_sum,df.target_tokens),equal_nan=True,atol=1e-14)
    assert (df[(df.intervention_kind=='weight_union')&(df.dose_groups==0)].damage_sum==0).all()
    # Verify labels against coordinates before any temporal grouping.
    df['coordinate_identity']=df.intervention.map(lambda k:panel[k]['identity'])
    assert df.coordinate_identity.notna().all()
    assert np.all(df.expert==df.intervention.map(lambda k:panel[k]['expert']))
    arrays={}
    for key,z in df.groupby(['trajectory','step','expert','subset']):
        ids=sorted(z.coordinate_identity.unique())
        p=lambda field:z.pivot(index='batch',columns='coordinate_identity',values=field).reindex(index=range(32),columns=ids).to_numpy(float)
        num,den=p('damage_sum'),p('target_tokens')
        assert np.isfinite(num).all() and np.isfinite(den).all() and np.all(den==den[:,[0]])
        arrays[key]=(ids,num,den[:,0])
    # Independently reproduce all original within-checkpoint phase means from sums.
    old=pd.read_csv(SOURCE/'discovery_confirmation_rankings.csv');max_error=0.;reproduced=0
    for row in old.itertuples():
        ids,num,den=arrays[(row.trajectory,row.step,row.expert,row.subset)]
        lookup={x:i for i,x in enumerate(ids)}
        ix=[lookup[panel[k]['identity']] for k in json.loads(row.intervention_ids)]
        for idx,values in [(np.arange(16),row.discovery_values),(np.arange(16,32),row.confirmation_values)]:
            got=ratio(num[idx].sum(0),den[idx].sum())[ix];want=np.array(json.loads(values),float)
            assert np.allclose(got,want,rtol=0,atol=2e-14,equal_nan=True)
            delta=np.abs(got-want);max_error=max(max_error,float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.)
            reproduced+=len(got)
    wb=np.random.default_rng(config['discovery_bootstrap_seed']).multinomial(16,np.ones(16)/16,R).astype(float)
    wf=np.random.default_rng(config['confirmation_bootstrap_seed']).multinomial(16,np.ones(16)/16,R).astype(float)
    cov=pd.read_csv(SOURCE/'coverage_by_batch_and_block.csv')
    inventory=[];scores=[];ranks=[];extremes=[];policies=[];stability=[];thresholds=[];coverage=[]
    for trajectory in config['trajectories']:
      for expert in config['experts']:
       for tr in config['transitions']:
        early,future,previous=tr['early'],tr['future'],tr['previous']
        base=dict(trajectory=trajectory,expert=expert,early_step=early,future_step=future,previous_step=previous,transition=f'{early}_to_{future}',priority=tr['priority'])
        phase_coverage={}
        for phase,t,part in [('early_discovery',early,'discovery'),('future_confirmation',future,'confirmation')]+([('previous_discovery',previous,'discovery')] if previous else []):
            b=cov[(cov.trajectory==trajectory)&(cov.step==t)&(cov.expert==expert)&(cov.partition==part)]
            blocks=b[b.level=='block'];batches=b[b.level=='batch'];counts=batches.accepted_tokens.to_numpy(float);total=counts.sum()
            good=bool((blocks.accepted_tokens>=32).all());phase_coverage[phase]=good
            coverage.append(dict(base,phase=phase,checkpoint=t,partition=part,coverage_qualified=good,
                block_accepted_counts=jd(blocks.sort_values('block').accepted_tokens.to_list()),minimum_block_accepted=int(blocks.accepted_tokens.min()),
                blocks_below_32=int((blocks.accepted_tokens<32).sum()),accepted_tokens=int(total),contributing_batches=int((counts>0).sum()),
                max_batch_accepted_fraction=float(counts.max()/total) if total else None,
                accepted_batch_HHI=float(np.square(counts/total).sum()) if total else None,
                expert_load_context_only=True))
        for subset in config['subsets']:
         # The task-CE prediction rules use WHOLE discovery damage for both future outcomes.
         allids,anum,aden=arrays[(trajectory,early,expert,'whole')]
         fids,fnum,fden=arrays[(trajectory,future,expert,subset)];assert allids==fids
         current,currentboot=summarize(anum,aden,np.arange(16),wb)
         futuremean,futureboot=summarize(fnum,fden,np.arange(16,32),wf)
         prior=priorboot=None
         if previous:
            pids,pnum,pden=arrays[(trajectory,previous,expert,'whole')];assert allids==pids
            prior,priorboot=summarize(pnum,pden,np.arange(16),wb)
         for kind,dose in [('weight_union',d) for d in config['doses']]+[('expert_output_zero',None)]:
          these=sorted(k for k,v in panel.items() if v['expert']==expert and v['kind']==kind and v['dose_groups']==dose)
          pos=[allids.index(panel[k]['identity']) for k in these];n=len(these)
          assert n==({0:1,1:3,2:4,4:4,8:1}[dose] if kind=='weight_union' else 1)
          b=dict(base,subset=subset,predictor_subset='whole',intervention_kind=kind,dose_groups=dose,distinct_unions=n)
          y=futuremean[pos];yb=futureboot[:,pos];cur=current[pos];curb=currentboot[:,pos]
          prev=prior[pos] if previous else None;prevb=priorboot[:,pos] if previous else None
          sets=[panel[k]['coordinate_set'] for k in these];allcoords=coordinate_union(sets,range(n))
          denominator=int(manifest.loc[these[0],f'{trajectory}_active_matrix_coordinates'])
          fblocks=np.array([ratio(fnum[4*j:4*j+4].sum(0),fden[4*j:4*j+4].sum())[pos] for j in range(4,8)])
          floo=np.array([ratio(np.delete(fnum[16:],np.arange(j*4,j*4+4),axis=0).sum(0),np.delete(fden[16:],np.arange(j*4,j*4+4)).sum())[pos] for j in range(4)])
          for i,iid in enumerate(these):
            if subset=='whole': inventory.append(dict(base,intervention=iid,identity=panel[iid]['identity'],parameter=panel[iid]['parameter'],
                coordinate_sha256=panel[iid]['coordinate_sha256'],intervention_kind=kind,dose_groups=dose,distinct_unions=n,
                coordinates=len(sets[i]),active_matrix_coordinates=denominator,active_matrix_fraction=len(sets[i])/denominator,
                component_groups=jd(panel[iid]['groups']),ranking_eligible=n>=3))
          rulelist=['current']+(['history_mean','history_max'] if previous else [])
          cslo,cshi,cstied=select(cur,these);cbl,cbh,cbtie=boot_select(curb,these)
          for rule in rulelist:
            s=score(cur,prev,rule);sb=score(curb,prevb,rule)
            low,high,tied=select(s,these);bl,bh,btie=boot_select(sb,these)
            qualified=phase_coverage['early_discovery'] and phase_coverage['future_confirmation'] and (rule=='current' or phase_coverage.get('previous_discovery',False))
            flags=dict(coverage_qualified=qualified,early_discovery_covered=phase_coverage['early_discovery'],future_confirmation_covered=phase_coverage['future_confirmation'],
                previous_discovery_covered=phase_coverage.get('previous_discovery'),low_early_exposure=not phase_coverage['early_discovery'])
            bb=dict(b,rule=rule,**flags)
            for i,iid in enumerate(these):
                scores.append(dict(bb,intervention=iid,coordinate_sha256=panel[iid]['coordinate_sha256'],current_discovery_damage=cur[i],
                    previous_discovery_damage=prev[i] if previous else None,prediction_score=s[i],future_confirmation_damage=y[i],
                    future_target_tokens=int(fden[16:].sum()),early_target_tokens=int(aden[:16].sum()),
                    future_block_damages=jd(fblocks[:,i]),future_leave_one_block_out=jd(floo[:,i]),**ci(yb[:,i])))
            rb=bootstrap_rho(sb,yb)
            ranks.append(dict(bb,spearman=rho(s,y),**ci(rb),score_tied_pairs=ties(s),future_tied_pairs=ties(y),
                extreme_selection_unidentified=tied,intervention_ids=jd(these),scores=jd(s),future_damages=jd(y),
                interpretation='singleton_trajectory_only' if n<2 else ('unidentified_score_tie' if tied else 'exploratory_equal_dose')))
            if low is None: continue
            fixed=yb[:,high]-yb[:,low];dynamic=picked(yb,bh)-picked(yb,bl)
            e=dict(bb,selected_high=these[high],selected_low=these[low],selected_high_hash=panel[these[high]]['coordinate_sha256'],
                selected_low_hash=panel[these[low]]['coordinate_sha256'],early_score_difference=s[high]-s[low],extreme_selection_unidentified=tied,
                future_high_minus_low=y[high]-y[low],future_block_contrasts=jd(fblocks[:,high]-fblocks[:,low]),
                future_leave_one_block_out=jd(floo[:,high]-floo[:,low]),**ci(fixed))
            e.update({f'selection_resampled_{k}':v for k,v in ci(dynamic).items()})
            extremes.append(e)
            if cslo is not None:
                difference=policy_difference(y,cslo,low);fixeddiff=policy_difference(yb,cslo,low)
                assert low!=cslo or np.all(fixeddiff==0)
                adv=yb.mean(1)-yb[:,low];dynamicadv=yb.mean(1)-picked(yb,bl)
                dynamicdiff=picked(yb,cbl)-picked(yb,bl)
                p=dict(bb,selected_low=these[low],current_selected_low=these[cslo],identical_selection=low==cslo,
                    low_selection_unidentified=bool(np.sum(np.abs(s-s[low])<=FLOOR)>1),
                    selected_future_damage=y[low],panel_uniform_future_damage=float(np.mean(y)),selection_advantage=float(np.mean(y)-y[low]),
                    current_minus_rule_future_damage=float(difference),future_regret_hindsight_optimistic=float(y[low]-np.min(y)),
                    future_advantage_by_block=jd(fblocks.mean(1)-fblocks[:,low]),future_advantage_leave_one_block_out=jd(floo.mean(1)-floo[:,low]),
                    current_minus_rule_by_block=jd(fblocks[:,cslo]-fblocks[:,low]),current_minus_rule_leave_one_block_out=jd(floo[:,cslo]-floo[:,low]))
                for prefix,values in [('advantage_',adv),('current_minus_rule_',fixeddiff),('resampled_advantage_',dynamicadv),('resampled_current_minus_rule_',dynamicdiff)]:
                    p.update({prefix+k:v for k,v in ci(values).items()})
                policies.append(p)
            for i,iid in enumerate(these):
                stability.append(dict(bb,intervention=iid,coordinate_sha256=panel[iid]['coordinate_sha256'],
                    low_selection_frequency=float(np.mean(bl==i)),high_selection_frequency=float(np.mean(bh==i)),
                    original_low=low==i,original_high=high==i,undefined_selection_fraction=float(np.mean(bl<0)),
                    unidentified_extreme_tie_fraction=float(np.mean(btie)),bootstrap_resamples=R))
          if kind=='weight_union' and dose>0:
           for rule in ['current']+(['persistent'] if previous else []):
            for tol in config['reference_tolerances']:
                ok=np.isfinite(cur)&(cur<=tol)
                if rule=='persistent': ok &= np.isfinite(prev)&(prev<=tol)
                selected=np.flatnonzero(ok);sc=coordinate_union(sets,selected)
                intervals=[ci(yb[:,i]) for i in selected]
                above=sum(v['ci95_low'] is not None and v['ci95_low']>tol for v in intervals)
                below=sum(v['ci95_high'] is not None and v['ci95_high']<tol for v in intervals)
                undef=sum(v['ci95_low'] is None for v in intervals)
                thresholds.append(dict(b,rule=rule,tolerance=tol,selected_count=len(selected),candidate_count=n,
                    selected_ids=jd([these[i] for i in selected]),selected_future_damages=jd(y[selected]),
                    selected_future_intervals=jd(intervals),selected_coordinate_union_count=len(sc),panel_coordinate_union_count=len(allcoords),
                    selected_coordinate_fraction_of_panel=len(sc)/len(allcoords) if allcoords else None,
                    selected_coordinate_fraction_of_active_matrix=len(sc)/denominator,
                    false_safe_fraction=float(np.mean(y[selected]>tol)) if len(selected) and np.isfinite(y[selected]).all() else None,
                    future_interval_above=above,future_interval_below=below,future_interval_crossing=len(selected)-above-below-undef,future_interval_undefined=undef,
                    future_damage_min=float(np.min(y[selected])) if len(selected) and np.isfinite(y[selected]).all() else None,
                    future_damage_max=float(np.max(y[selected])) if len(selected) and np.isfinite(y[selected]).all() else None,
                    status='empty_undefined' if not len(selected) else ('ceiling_no_discrimination' if len(selected)==n else 'partial_panel_qualification'),
                    coverage_qualified=phase_coverage['early_discovery'] and phase_coverage['future_confirmation'] and (rule=='current' or phase_coverage.get('previous_discovery',False))))
        print(trajectory,expert,base['transition'],'complete',flush=True)
    for filename,rows in [('temporal_intervention_inventory',inventory),('temporal_scores_and_outcomes',scores),('temporal_rank_agreement',ranks),
        ('selected_high_low_future_contrasts',extremes),('low_damage_selection_comparison',policies),('selection_stability',stability),
        ('false_safe_summary',thresholds),('temporal_coverage',coverage)]:
        pd.DataFrame(rows).to_csv(D/f'{filename}.csv',index=False)
    # All historical sealed inputs must still match after analysis.
    assert all(sha(ROOT/path)==h for path,h in verified.items())
    save(D/'numerical_checks.json',dict(passed=True,temporal_coordinate_joins_verified=True,unique_interventions=56,
        signed_sum_weighting_verified=True,maximum_loss_sum_cancellation_error=cancellation_error,zero_dose_exact=True,original_phase_means_reproduced=reproduced,
        maximum_original_mean_error=max_error,historical_sealed_files_unchanged=True,
        identical_policy_differences_exact_zero=True,model_executions=0,checkpoint_tensor_reads=0,training_updates=0,
        bootstrap_resamples=R,bootstrap_unit='paired complete batches',
        implementation_sha256=sha(Path(__file__))))
    save(D/'compute_accounting.json',dict(analysis_wall_seconds=time.perf_counter()-started,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_executions=0,training_updates=0,new_measurements=0,bootstrap_resamples=R,
        storage_bytes=sum(p.stat().st_size for p in D.rglob('*') if p.is_file())))

if __name__=='__main__': run()
