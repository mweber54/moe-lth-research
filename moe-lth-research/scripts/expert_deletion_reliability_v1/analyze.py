"""Paired whole-batch bootstrap and strictly within-expert/time/dose reliability."""
from reliability_common import *
import itertools
import pandas as pd
from scipy.stats import rankdata


def interval(x):
    x=np.asarray(x,float);good=x[np.isfinite(x)]
    return (float(np.quantile(good,.025)),float(np.quantile(good,.975)),1-len(good)/len(x)) if len(good) else (None,None,1.)


def ratio(num,den):
    a=np.asarray(num,float);b=np.asarray(den,float)
    with np.errstate(divide='ignore',invalid='ignore'):return np.where(b>0,a/b,np.nan)


def bootstrap_counts(n,seed):return np.random.default_rng(seed).multinomial(n,np.ones(n)/n,size=RESAMPLES).astype(np.float64)


def summarize_phase(num,den,indices,seed):
    a=num[indices];b=den[indices];weights=bootstrap_counts(len(indices),seed)
    means=ratio(a.sum(axis=0),b.sum())
    boot=ratio(weights@a,(weights@b)[:,None])
    return dict(mean=means,boot=boot,counts=float(b.sum()),batches=len(indices),indices=np.asarray(indices))


def spearman(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid]
    if len(a)<3 or np.ptp(a)<=FLOOR or np.ptp(b)<=FLOOR:return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


def bootstrap_rank(a,b):
    good=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    if a.shape[1]<3:return np.full(len(a),np.nan)
    good&=(np.ptp(a,axis=1)>FLOOR)&(np.ptp(b,axis=1)>FLOOR)
    out=np.full(len(a),np.nan)
    if good.any():
        x=rankdata(a[good],axis=1);y=rankdata(b[good],axis=1);x-=x.mean(1,keepdims=True);y-=y.mean(1,keepdims=True)
        out[good]=(x*y).sum(1)/np.sqrt((x*x).sum(1)*(y*y).sum(1))
    return out


def ties(x):
    x=np.asarray(x,float)
    return sum(np.isfinite(x[i]) and np.isfinite(x[j]) and abs(x[i]-x[j])<=FLOOR for i in range(len(x)) for j in range(i+1,len(x)))


def jd(value):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
        if isinstance(x,(np.integer,int)):return int(x)
        if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
        return x
    return json.dumps(clean(value),allow_nan=False,sort_keys=True)


def tolerance_state(lo,hi,tol):
    if lo is None or hi is None:return 'undefined'
    if lo>tol:return 'above_exploratory_convention'
    if hi<tol:return 'compatible_below_convention_on_this_probe'
    return 'unresolved_crossing_or_boundary'


def main():
    started=time.perf_counter();verify_frozen();assert s.read(D/'measurement_completion.json')['complete']
    torch.set_num_threads(2);df=pd.read_csv(D/'deletion_damage_by_batch.csv');cov=pd.read_csv(D/'coverage_by_batch_and_block.csv');blockcov=cov[cov.level=='block'];panel={x['intervention']:x for x in interventions()}
    rankings=[];contrasts=[];uncertainty=[];functional=[];decisions=[]
    phase_indices={'discovery':np.arange(16),'confirmation':np.arange(16,32)}
    phase_indices.update({f'block_{b+1}':np.arange(b*4,b*4+4) for b in range(8)})
    phase_seeds={name:BOOTSTRAP_SEED+i for i,name in enumerate(phase_indices)}
    for keys,z in df.groupby(['trajectory','step','expert','subset']):
        kind,t,e,subset=keys;ids=sorted(z.intervention.unique());columns={k:i for i,k in enumerate(ids)}
        pivot=lambda field:z.pivot(index='batch',columns='intervention',values=field).reindex(index=range(32),columns=ids).to_numpy(float)
        nums=pivot('damage_sum');dens=pivot('target_tokens');assert np.isfinite(nums).all() and np.all(dens==dens[:,[0]])
        den=dens[:,0];phases={name:summarize_phase(nums,den,idx,phase_seeds[name]) for name,idx in phase_indices.items()}
        c=blockcov[(blockcov.trajectory==kind)&(blockcov.step==t)&(blockcov.expert==e)].sort_values('block');assert len(c)==8
        covered=bool((c.accepted_tokens>=MIN_ACCEPTED).all());conditional_eligible=covered or subset!='baseline_accepted'
        fnumer=pivot('output_difference_energy');fdenom=pivot('baseline_output_energy');elements=pivot('output_elements');gate=pivot('gate_weighted_difference_energy');gb=pivot('baseline_gate_weighted_energy')
        fphase={}
        for phase,idx in phase_indices.items():
            num=fnumer[idx].sum(0);base=fdenom[idx].sum(0);elem=elements[idx].sum(0)
            fr=np.sqrt(ratio(num,base));fr=np.where((elem>0)&(np.sqrt(ratio(base,elem))>dc.OUTPUT_EPS),fr,np.nan);fphase[phase]=fr
            if subset=='whole':
                for i,iid in enumerate(ids):
                    vals=z[(z.intervention==iid)&(z.batch.isin(idx))].relative_output_rms_difference.dropna()
                    functional.append(dict(trajectory=kind,step=int(t),expert=int(e),intervention=iid,intervention_kind=panel[iid]['kind'],dose_groups=panel[iid]['dose_groups'],partition=phase,output_difference_energy=float(num[i]),baseline_output_energy=float(base[i]),output_elements=int(elem[i]),pooled_relative_rms=float(fr[i]) if np.isfinite(fr[i]) else None,mean_defined_batch_relative_rms=float(vals.mean()) if len(vals) else None,defined_batch_ratios=len(vals),pooled_output_rms=float(np.sqrt(num[i]/elem[i])) if elem[i] else None,pooled_baseline_rms=float(np.sqrt(base[i]/elem[i])) if elem[i] else None,gate_weighted_difference_energy=float(gate[idx,i].sum()),baseline_gate_weighted_energy=float(gb[idx,i].sum()),pooled_gate_weighted_rms=float(np.sqrt(gate[idx,i].sum()/elem[i])) if elem[i] else None))
        loo={}
        for phase in ('discovery','confirmation'):
            blocks=range(4) if phase=='discovery' else range(4,8);idx=phase_indices[phase]
            loo[phase]=np.array([ratio(nums[np.setdiff1d(idx,np.arange(b*4,b*4+4))].sum(0),den[np.setdiff1d(idx,np.arange(b*4,b*4+4))].sum()) for b in blocks])
            for i,iid in enumerate(ids):
                lo,hi,undefined=interval(phases[phase]['boot'][:,i]);mean=phases[phase]['mean'][i]
                for tol in (.0001,.001):
                    uncertainty.append(dict(trajectory=kind,step=int(t),expert=int(e),intervention=iid,intervention_kind=panel[iid]['kind'],dose_groups=panel[iid]['dose_groups'],subset=subset,partition=phase,mean_damage=float(mean) if np.isfinite(mean) else None,ci95_low=lo,ci95_high=hi,bootstrap_resamples=RESAMPLES,undefined_resample_fraction=undefined,target_tokens=int(phases[phase]['counts']),operational_coverage_eligible=conditional_eligible,leave_one_block_out=jd(loo[phase][:,i]),block_values=jd([phases[f'block_{b+1}']['mean'][i] for b in blocks]),reference_tolerance=tol,tolerance_interpretation=tolerance_state(lo,hi,tol),uncertainty_unit='paired_complete_training_batches_one_source_seed'))
        groups={}
        for iid in ids:groups.setdefault((panel[iid]['kind'],panel[iid]['dose_groups']),[]).append(iid)
        for (typ,dose),these in groups.items():
            pos=[columns[k] for k in these];discovery=phases['discovery']['mean'][pos];confirmation=phases['confirmation']['mean'][pos];n=len(these)
            rho=spearman(discovery,confirmation) if conditional_eligible else None
            br=bootstrap_rank(phases['discovery']['boot'][:,pos],phases['confirmation']['boot'][:,pos]) if conditional_eligible else np.full(RESAMPLES,np.nan)
            rlo,rhi,rundef=interval(br);valid=np.isfinite(discovery)
            hiid=loid=None;di=ci=None;selectedlo=selectedhi=None;selectedundef=None;blockvalues=[];blockintervals=[];blocksigns=[];sel_loo={}
            if n>=2 and valid.all():
                ordered=sorted(range(n),key=lambda i:(discovery[i],these[i]));li=ordered[0];hii=ordered[-1];loid=these[li];hiid=these[hii];low=pos[li];high=pos[hii]
                di=float(discovery[hii]-discovery[li]);ci=float(confirmation[hii]-confirmation[li]) if np.isfinite(confirmation).all() else None
                selectedlo,selectedhi,selectedundef=interval(phases['confirmation']['boot'][:,high]-phases['confirmation']['boot'][:,low])
                for b in range(4,8):
                    p=phases[f'block_{b+1}'];v=p['mean'][high]-p['mean'][low];blo,bhi,bundef=interval(p['boot'][:,high]-p['boot'][:,low]);blockvalues.append(float(v) if np.isfinite(v) else None);blockintervals.append([blo,bhi,bundef]);blocksigns.append(int(np.sign(v)) if np.isfinite(v) and abs(v)>FLOOR else (0 if np.isfinite(v) else None))
                sel_loo={p:loo[p][:,high]-loo[p][:,low] for p in loo}
            if not conditional_eligible:status='insufficient_conditional_block_coverage'
            elif n<2:status='undefined_singleton_no_equal_dose_contrast'
            elif di is None:status='undefined_no_observed_population'
            elif di<=FLOOR:status='unidentified_discovery_tie'
            elif selectedlo is not None and selectedlo>0:status='directional_contrast_reproduced_on_confirmation'
            elif selectedhi is not None and selectedhi<0:status='directional_contrast_reversed_candidate_context_dependence'
            else:status='unresolved_at_fixed_sampling_budget'
            dfr=spearman(fphase['discovery'][pos],fphase['confirmation'][pos]) if covered else None
            fdc=spearman(fphase['discovery'][pos],discovery) if conditional_eligible and covered else None
            fcc=spearman(fphase['confirmation'][pos],confirmation) if conditional_eligible and covered else None
            ranking=dict(trajectory=kind,step=int(t),expert=int(e),intervention_kind=typ,dose_groups=dose,subset=subset,distinct_interventions=n,operational_coverage_eligible=conditional_eligible,minimum_accepted_in_block=int(c.accepted_tokens.min()),blocks_below_minimum=int((c.accepted_tokens<MIN_ACCEPTED).sum()),spearman=rho,spearman_ci95_low=rlo,spearman_ci95_high=rhi,spearman_undefined_bootstrap_fraction=rundef,discovery_numerical_tied_pairs=ties(discovery),confirmation_numerical_tied_pairs=ties(confirmation),intervention_ids=jd(these),discovery_values=jd(discovery),confirmation_values=jd(confirmation),selected_high_discovery=hiid,selected_low_discovery=loid,discovery_selected_contrast=di,confirmation_selected_contrast=ci,confirmation_selected_ci95_low=selectedlo,confirmation_selected_ci95_high=selectedhi,confirmation_selected_undefined_resamples=selectedundef,confirmation_block_contrasts=jd(blockvalues),confirmation_block_intervals=jd(blockintervals),confirmation_block_signs=jd(blocksigns),selected_leave_one_block_out=jd(sel_loo),functional_discovery_confirmation_spearman=dfr,functional_vs_damage_discovery_spearman=fdc,functional_vs_damage_confirmation_spearman=fcc,interpretation=status)
            rankings.append(ranking)
            if subset=='whole' and typ=='weight_union' and dose>0:
                decisions.append(dict(trajectory=kind,step=int(t),expert=int(e),dose_groups=dose,A_reproducibility=status,B_confirmation_interval_distinguishes_zero=(selectedlo is not None and selectedlo>0) or (selectedhi is not None and selectedhi<0),C_functional_vs_task_confirmation_spearman=fcc,C_functional_rank_reliability=dfr,D_minimum_accepted_in_block=int(c.accepted_tokens.min()),D_blocks_below_minimum=int((c.accepted_tokens<MIN_ACCEPTED).sum()),D_coverage='adequate_all_blocks' if covered else 'insufficient_conditional_coverage',E_future_persistence_use='only_later_fixed_measurement_confirmation_no_persistence_claim' if status=='directional_contrast_reproduced_on_confirmation' and covered else 'do_not_use_for_CE_group_selection_at_this_budget',spearman=rho,selected_confirmation_contrast=ci,selected_confirmation_ci95_low=selectedlo,selected_confirmation_ci95_high=selectedhi,confirmation_block_signs=jd(blocksigns)))
            if typ!='weight_union':continue
            for ia,ib in itertools.combinations(range(n),2):
                a=pos[ia];b=pos[ib]
                for phase,p in phases.items():
                    boot=p['boot'][:,b]-p['boot'][:,a];lo,hi,undefined=interval(boot);effect=p['mean'][b]-p['mean'][a]
                    contrasts.append(dict(trajectory=kind,step=int(t),expert=int(e),dose_groups=dose,subset=subset,partition=phase,intervention_a=these[ia],intervention_b=these[ib],orientation='b_minus_a',mean_paired_contrast=float(effect) if np.isfinite(effect) else None,ci95_low=lo,ci95_high=hi,undefined_resample_fraction=undefined,complete_batches=p['batches'],target_tokens=int(p['counts']),conditional_coverage_eligible=conditional_eligible,discovery_selected_pair={these[ia],these[ib]}=={hiid,loid},interval_crosses_zero=lo is not None and hi is not None and lo<=0<=hi))
    table(D/'discovery_confirmation_rankings.csv',rankings);table(D/'paired_intervention_contrasts.csv',contrasts);table(D/'uncertainty_and_tolerance_summary.csv',uncertainty);table(D/'functional_distortion_summary.csv',functional);table(D/'condition_decisions.csv',decisions)
    a=s.read(D/'compute_accounting.json');a['analysis_wall_seconds']=time.perf_counter()-started;a['bootstrap_resamples']=RESAMPLES;a['bootstrap_seed']=BOOTSTRAP_SEED;save(D/'compute_accounting.json',a)
    save(D/'analysis_completion.json',dict(complete=True,rankings=len(rankings),paired_contrasts=len(contrasts),uncertainty_rows=len(uncertainty),condition_decisions=len(decisions),no_pooled_expert_checkpoint_dose_correlations=True,unix=time.time()))
    print('Paired-batch analysis complete',len(rankings),'fixed-stratum rankings',flush=True)

if __name__=='__main__':
    with worker():main()
