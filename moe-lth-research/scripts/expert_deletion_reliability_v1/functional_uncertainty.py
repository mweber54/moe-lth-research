"""Apply the frozen paired-batch bootstrap to pooled functional energies and C comparisons.

No new observations, selection rules, thresholds, or resampling scheme. This fills
functional uncertainty using the same frozen discovery/confirmation count vectors.
"""
from reliability_common import *
import analyze as a
import pandas as pd


def main():
    verify_frozen();started=time.perf_counter();df=pd.read_csv(D/'deletion_damage_by_batch.csv');df=df[df.subset=='whole'];summary=pd.read_csv(D/'functional_distortion_summary.csv');cov=pd.read_csv(D/'coverage_by_batch_and_block.csv');updates=[];comparisons=[]
    for (kind,t,e),z in df.groupby(['trajectory','step','expert']):
        ids=sorted(z.intervention.unique());pos={i:j for j,i in enumerate(ids)}
        pivot=lambda field:z.pivot(index='batch',columns='intervention',values=field).reindex(index=range(32),columns=ids).to_numpy(float)
        num=pivot('output_difference_energy');base=pivot('baseline_output_energy');elements=pivot('output_elements');gate=pivot('gate_weighted_difference_energy');ce=pivot('damage_sum');den=pivot('target_tokens')
        bootstrap={};points={};ceboot={};cepoints={}
        coverage=cov[(cov.level=='block')&(cov.trajectory==kind)&(cov.step==t)&(cov.expert==e)];covered=bool((coverage.accepted_tokens>=MIN_ACCEPTED).all())
        for phase,idx,seed in [('discovery',np.arange(16),BOOTSTRAP_SEED),('confirmation',np.arange(16,32),BOOTSTRAP_SEED+1)]:
            weights=a.bootstrap_counts(16,seed);n=weights@num[idx];b=weights@base[idx];el=weights@elements[idx];g=weights@gate[idx]
            relative=np.sqrt(a.ratio(n,b));relative=np.where(np.sqrt(a.ratio(b,el))>dc.OUTPUT_EPS,relative,np.nan)
            bootstrap[phase]=relative;points[phase]=np.sqrt(a.ratio(num[idx].sum(0),base[idx].sum(0)));ceboot[phase]=a.ratio(weights@ce[idx],weights@den[idx]);cepoints[phase]=a.ratio(ce[idx].sum(0),den[idx].sum(0))
            metrics={'pooled_relative_rms':relative,'pooled_output_rms':np.sqrt(a.ratio(n,el)),'pooled_baseline_rms':np.sqrt(a.ratio(b,el)),'pooled_gate_weighted_rms':np.sqrt(a.ratio(g,el))}
            for iid,j in pos.items():
                row=dict(trajectory=kind,step=int(t),expert=int(e),intervention=iid,partition=phase,functional_bootstrap_resamples=RESAMPLES)
                for metric,values in metrics.items():
                    lo,hi,undefined=a.interval(values[:,j]);row[metric+'_ci95_low']=lo;row[metric+'_ci95_high']=hi;row[metric+'_undefined_resample_fraction']=undefined
                updates.append(row)
        panel={i:next(p for p in interventions() if p['intervention']==i) for i in ids}
        groups={}
        for iid in ids:groups.setdefault((panel[iid]['kind'],panel[iid]['dose_groups']),[]).append(iid)
        for (typ,dose),these in groups.items():
            p=[pos[i] for i in these]
            for phase in ('discovery','confirmation'):
                rho=a.spearman(points[phase][p],cepoints[phase][p]) if covered else None
                reps=a.bootstrap_rank(bootstrap[phase][:,p],ceboot[phase][:,p]) if covered else np.full(RESAMPLES,np.nan)
                lo,hi,undefined=a.interval(reps)
                comparisons.append(dict(trajectory=kind,step=int(t),expert=int(e),intervention_kind=typ,dose_groups=dose,comparison='functional_vs_task_within_phase',task_population='whole_probe',partition=phase,distinct_interventions=len(these),operational_coverage_eligible=covered,spearman=rho,ci95_low=lo,ci95_high=hi,undefined_resample_fraction=undefined,bootstrap_resamples=RESAMPLES))
            rho=a.spearman(points['discovery'][p],points['confirmation'][p]) if covered else None;reps=a.bootstrap_rank(bootstrap['discovery'][:,p],bootstrap['confirmation'][:,p]) if covered else np.full(RESAMPLES,np.nan);lo,hi,undefined=a.interval(reps)
            comparisons.append(dict(trajectory=kind,step=int(t),expert=int(e),intervention_kind=typ,dose_groups=dose,comparison='functional_discovery_vs_confirmation',task_population='whole_probe',partition='independent_phase_resamples',distinct_interventions=len(these),operational_coverage_eligible=covered,spearman=rho,ci95_low=lo,ci95_high=hi,undefined_resample_fraction=undefined,bootstrap_resamples=RESAMPLES))
    before=D/'functional_distortion_summary_before_uncertainty.csv'
    if not before.exists():before.write_bytes((D/'functional_distortion_summary.csv').read_bytes())
    summary=summary.drop(columns=[k for k in summary.columns if '_ci95_' in k or '_undefined_resample_' in k or k=='functional_bootstrap_resamples'])
    merged=summary.merge(pd.DataFrame(updates),on=['trajectory','step','expert','intervention','partition'],how='left',validate='one_to_one');merged.to_csv(D/'functional_distortion_summary.csv',index=False)
    table(D/'functional_task_rank_uncertainty.csv',comparisons)
    save(D/'functional_uncertainty_provenance.json',dict(script_sha256=sha(Path(__file__)),input_batch_table_sha256=sha(D/'deletion_damage_by_batch.csv'),bootstrap_resamples=RESAMPLES,phase_seeds=[BOOTSTRAP_SEED,BOOTSTRAP_SEED+1],paired_complete_batch_resampling=True,energies_pooled_before_ratio=True,no_new_data=True,wall_seconds=time.perf_counter()-started))
    print('Functional paired-batch uncertainty complete',flush=True)
if __name__=='__main__':
    with worker():main()
