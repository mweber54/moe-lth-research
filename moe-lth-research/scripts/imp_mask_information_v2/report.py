"""Seed-level descriptive analysis; nested mask draws never count as model seeds."""
from v2_common import *
import math
import statistics
import numpy as np
from scipy.stats import t


def describe(values):
    mean=statistics.mean(values);n=len(values)
    sd=statistics.stdev(values) if n>1 else None
    half=float(t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else None
    return dict(n=n,mean=mean,sample_sd=sd,minimum=min(values),maximum=max(values),descriptive_95pct_t_lower=mean-half if half is not None else None,descriptive_95pct_t_upper=mean+half if half is not None else None)


def analysis(manifest):
    observed=[];states=[]
    for job in manifest['jobs']:
        s=job_status(job);states.append(dict(job=job['id'],phase=job['phase'],seed=job['seed'],condition=job['condition'],replicate=job['replicate'],**s))
        if s['status'] in ('COMPLETE','REUSED'):observed.append(result(job))
    # Flatten nested initial hashes for CSV, retaining full JSON in per-job records.
    flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in observed]
    table(OUT/'results.csv',flat, list(dict.fromkeys(k for r in flat for k in r)) if flat else ['job','seed','condition','final_ce'])
    table(OUT/'job_status.csv',states)
    paired=[]
    for seed in SEEDS:
        rr=[r for r in observed if r['seed']==seed]
        imp=next((r for r in rr if r['condition']=='IMP'),None)
        if imp is None:continue
        for condition in ('matrix','degree'):
            control=[r['final_ce'] for r in rr if r['condition']==condition]
            if not control:continue
            d=describe(control)
            paired.append(dict(seed=seed,condition=condition,n_masks=len(control),imp_ce=imp['final_ce'],historical_dense_ce=imp['historical_dense_ce'],historical_imp_ce=imp['historical_imp_ce'],mean_control_ce=d['mean'],sample_sd=d['sample_sd'],min_control_ce=d['minimum'],max_control_ce=d['maximum'],mean_minus_imp=d['mean']-imp['final_ce'],nested_mask_mean_descriptive_95pct_lower=d['descriptive_95pct_t_lower'],nested_mask_mean_descriptive_95pct_upper=d['descriptive_95pct_t_upper'],complete=len(control)==5,execution_rule_passed=(len(control)==5 and d['mean']-imp['final_ce']>.02) if condition=='matrix' else None))
    table(OUT/'paired_effects.csv',paired,['seed','condition','n_masks','imp_ce','historical_dense_ce','historical_imp_ce','mean_control_ce','sample_sd','min_control_ce','max_control_ce','mean_minus_imp','nested_mask_mean_descriptive_95pct_lower','nested_mask_mean_descriptive_95pct_upper','complete','execution_rule_passed'])
    aggregates={}
    for condition in ('matrix','degree'):
        pp=[r for r in paired if r['condition']==condition and r['complete']]
        if pp:
            effects=[r['mean_minus_imp'] for r in pp]
            aggregates[condition]=dict(**describe(effects),source_seeds=[r['seed'] for r in pp],complete=len(pp)==5,leave_one_seed_out_means={str(r['seed']):statistics.mean([p['mean_minus_imp'] for p in pp if p['seed']!=r['seed']]) for r in pp} if len(pp)>1 else {},unit='source training seed; random masks nested',inference='descriptive n<=5; not a proof of generality or a mechanism')
    if all(c in aggregates and aggregates[c]['complete'] for c in ('matrix','degree')):
        contrasts=[next(p['mean_control_ce'] for p in paired if p['seed']==s and p['condition']=='matrix')-next(p['mean_control_ce'] for p in paired if p['seed']==s and p['condition']=='degree') for s in SEEDS]
        aggregates['matrix_minus_degree']=dict(**describe(contrasts),source_seeds=list(SEEDS),unit='source seed',per_seed=dict(zip(map(str,SEEDS),contrasts)))
    save(OUT/'aggregate_effects.json',aggregates)
    counts={s:sum(r['status']==s for r in states) for s in sorted({r['status'] for r in states})}
    launched=sum(r['status'] in ('RUNNING','COMPLETE','FAILED') for r in states)
    startup_failures=len(list(OUT.glob('launch_failed_*.json')))
    gate=read(OUT/'phase2_gate.json') if (OUT/'phase2_gate.json').exists() else dict(status='PENDING_PHASE1')
    stop=read(OUT/'STOP.json') if (OUT/'STOP.json').exists() else None
    phase1done=all(s['status'] in ('COMPLETE','REUSED') for s in states if s['phase']==1)
    phase2done=all(s['status']=='COMPLETE' for s in states if s['phase']==2)
    scientifically_complete=phase1done and (phase2done or gate['status'] in ('NOT_ADMITTED_EFFECT_RULE','NOT_ADMITTED_MIXING')) and not stop
    overall='COMPLETE' if scientifically_complete else 'STOPPED_FAILURE' if stop else 'RUNNING_OR_PREPARED'
    save(OUT/'completion.json',dict(status=overall,counts=counts,new_jobs_launched=launched,reused_jobs=counts.get('REUSED',0),recovery_failures=counts.get('FAILED',0),recorded_startup_failures=startup_failures,phase1_verified=sum(s['status'] in ('COMPLETE','REUSED') for s in states if s['phase']==1),phase1_total=30,phase2_complete=sum(s['status']=='COMPLETE' for s in states if s['phase']==2),phase2_total_if_admitted=25,phase2_gate=gate,scientifically_complete=bool(scientifically_complete),updated=time.time()))
    lines=['# IMP mask information v2 findings','',f'Status: {overall}. Phase 1 verified endpoints: {sum(s["status"] in ("COMPLETE","REUSED") for s in states if s["phase"]==1)}/30. Phase 2 completed: {sum(s["status"]=="COMPLETE" for s in states if s["phase"]==2)}/25 conditional endpoints.',f'Job counts: {counts}. Phase 2 gate: {gate["status"]}.',f'New jobs actually claimed: {launched}; reused: {counts.get("REUSED",0)}; recovery failures: {counts.get("FAILED",0)}; recorded pre-worker startup failures: {startup_failures} (see launch_failed_*.json and launch_correction.json).','', '## Direct observations','', 'E16, nominal 80% sparsity, seeds 7/17/29/43/61 were fixed before new outcomes, irrespective of historical dense gaps. Seed17 IMP and five matrix controls are compatible pilot reuse. Global/expert controls were not rerun. All new recoveries use 2500 attempted updates; applied updates and AMP skips appear in results.csv.']
    for p in paired:
        sd='unavailable' if p['sample_sd'] is None else f"{p['sample_sd']:.6f}"
        lines.append(f"- Seed {p['seed']} {p['condition']}: {p['n_masks']}/5 draws; IMP CE {p['imp_ce']:.6f}; mean control CE {p['mean_control_ce']:.6f}; SD {sd}; mean minus IMP {p['mean_minus_imp']:+.6f}; historical matched dense CE {p['historical_dense_ce']:.6f}.")
    lines+=['','## Descriptive uncertainty','', 'Random draws are nested within source seeds. Within-seed intervals describe variation over masks under fixed training seeds and do not increase independent model replication. Across-seed intervals use equal-weight paired effects with a t reference and at most five source seeds; they are descriptive and limited to this architecture/sparsity/protocol.']
    for condition,d in aggregates.items():lines.append(f"- {condition}: {d.get('n')} seed effects, mean {d['mean']:+.6f}; descriptive 95% t interval [{d['descriptive_95pct_t_lower']}, {d['descriptive_95pct_t_upper']}].")
    lines+=['','## Execution decision','', 'The degree phase is admitted only if all five seed-level matrix-minus-IMP means exceed 0.02. This is an execution rule, not a significance test. Finite-chain diagnostics must also pass for every diagnostic and production matrix before any degree recovery. No uniform-sampling or independent-stationary-draw claim is made.']
    if gate.get('limitation'):lines.append(gate['limitation'])
    if stop:lines.append('Failure stop: '+str(stop))
    lines+=['','## Interpretation and unresolved alternatives','', 'Any replicated matrix-control deficit is evidence about mask placement beyond matrix survivor counts for these E16/80% seed runs, not a general result or discovered mechanism. If degree-preserving masks recover better, connectivity counts may explain part of the matrix-random deficit; a remaining deficit means these counts are insufficient under this protocol. Neither outcome identifies indispensable edges, proves routed-token information, or explains sparse outperformance by itself. Finite swap-chain mixing, initialization, optimization and the narrow architecture/sparsity scope remain limitations. Do not declare equivalence from nonsignificant differences.','', 'Exact provenance: protocol.md, registration.json, manifest.json, seal.json, reuse_checks.json, host_checks/, per-attempt stage metadata, mask_validation.csv and degree_diagnostics.json when admitted. Every observed endpoint and original dense CE is in results.csv; each canonical curve is referenced by path. No further expansion is scheduled.']
    (OUT/'findings.md').write_text('\n'.join(lines)+'\n')
    if observed:figures(observed,paired)
    return paired


def figures(observed,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    colors={'IMP':'black','matrix':'tab:orange','degree':'tab:blue'}
    fig,axes=plt.subplots(1,5,figsize=(16,4),sharey=True)
    for ax,seed in zip(axes,SEEDS):
        for i,c in enumerate(('IMP','matrix','degree')):
            rr=[r for r in observed if r['seed']==seed and r['condition']==c]
            ax.scatter([i]*len(rr),[r['final_ce'] for r in rr],c=colors[c])
        ax.axhline(float(read(OUT/'manifest.json')['cohorts'][str(seed)]['dense_comparison_ce']),ls='--',c='gray',label='historical matched dense')
        ax.set_xticks(range(3),['IMP','matrix','degree']);ax.set_title(f'Seed {seed}')
    axes[0].set_ylabel('Final validation CE (nats/token)');fig.tight_layout();fig.savefig(dest/'endpoint_comparisons.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for c in ('matrix','degree'):
        pp=[p for p in paired if p['condition']==c and p['complete']]
        ax.scatter([p['seed'] for p in pp],[p['mean_minus_imp'] for p in pp],label=c)
    ax.axhline(.02,c='gray',ls='--',label='execution threshold');ax.axhline(0,c='black');ax.set_xlabel('Source seed');ax.set_ylabel('Mean control CE minus IMP CE');ax.legend();fig.tight_layout();fig.savefig(dest/'paired_seed_effects.png',dpi=160);plt.close(fig)
    curve_rows=[];fig,axes=plt.subplots(1,5,figsize=(18,4),sharey=True)
    for ax,seed in zip(axes,SEEDS):
        for r in [r for r in observed if r['seed']==seed]:
            curve=[json.loads(x) for x in Path(r['curves_path']).read_text().splitlines()]
            for x in curve:curve_rows.append(dict(job=r['job'],seed=seed,condition=r['condition'],replicate=r['replicate'],step=x['step'],validation_ce=x['loss']))
            ax.plot([x['step'] for x in curve],[x['loss'] for x in curve],color=colors[r['condition']],alpha=1 if r['condition']=='IMP' else .4)
        ax.set_title(f'Seed {seed}');ax.set_xlabel('Attempted updates');ax.set_yscale('log')
    axes[0].set_ylabel('Validation CE (log scale)');fig.tight_layout();fig.savefig(dest/'recovery_curves.png',dpi=160);plt.close(fig)
    table(OUT/'recovery_curves.csv',curve_rows,['job','seed','condition','replicate','step','validation_ce'])

if __name__=='__main__':
    with lock(OUT/'report.lock'):analysis(read(OUT/'manifest.json'))
