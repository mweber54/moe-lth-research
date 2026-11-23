"""Seed-level analysis; incomplete draws are never averaged as a full cohort."""
from run import *
import statistics
import math


def analyze():
    manifest=read(DEST/'registration.json')
    results=[];curves=[];missing=[]
    for job in manifest['jobs']:
        directory=DEST/'attempts'/job['id'];path=directory/'result.json'
        if not path.exists():
            missing.append(dict(job=job['id'],status=read(directory/'status.json') if (directory/'status.json').exists() else 'MISSING'))
            continue
        result=read(path);results.append(result)
        for row in read(directory/'curves.json'):curves.append(dict(job=job['id'],seed=result['seed'],sparsity=result['sparsity'],kind=result['kind'],draw=result['draw'],**row))
    inventory=[]
    for variant in manifest.get('variants',[]):
        donor=DEST/'attempts'/f"{variant['id']}_donor"
        inventory.append(dict(variant=variant['id'],status='VALIDATED_COMPLETE' if (donor/'result.json').exists() else 'MISSING_COMPLETE_TRAINING_TRACE',trace_inventory=str(donor/'trace_inventory.json'),provenance=str(donor/'provenance.json')))
    save(DEST/'donor_inventory.json',inventory)
    table(DEST/'outcomes.csv',results,fields=['variant','seed','sparsity','kind','draw','final_ce','applied_updates','amp_skips','wall_seconds','checkpoint'])
    table(DEST/'learning_curves.csv',curves,fields=['job','seed','sparsity','kind','draw','step','ce'])
    save(DEST/'missing_failed.json',missing)
    effects=[]
    def find(seed,sp,kind,draw=None):
        return next((x for x in results if x['seed']==seed and x['sparsity']==sp and x['kind']==kind and x['draw']==draw),None)
    for seed in SEEDS:
        for sparsity in (.8,.85):
            for step in (2500,5000):
                row=dict(seed=seed,sparsity=sparsity,step=step,status='MISSING',sparse_penalty=None,dense_penalty=None,differential_penalty=None)
                def penalty(sp):
                    identity=find(seed,sp,'identity');draws=[find(seed,sp,'shuffle',d) for d in PERMUTATIONS]
                    if identity is None or any(x is None for x in draws):return None
                    def value(result):
                        job=result['variant']+'_'+result['kind']+(f"_{result['draw']}" if result['draw'] is not None else '')
                        return next(x['ce'] for x in read(DEST/'attempts'/job/'curves.json') if x['step']==step)
                    return statistics.mean(value(x) for x in draws)-value(identity)
                sparse=penalty(sparsity);dense=penalty(0.)
                if sparse is not None and dense is not None:row.update(status='COMPLETE',sparse_penalty=sparse,dense_penalty=dense,differential_penalty=sparse-dense)
                effects.append(row)
    table(DEST/'per_seed_differential_penalties.csv',effects)
    summaries=[]
    for sp in (.8,.85):
        for step in (2500,5000):
            values=[x['differential_penalty'] for x in effects if x['sparsity']==sp and x['step']==step and x['status']=='COMPLETE']
            if len(values)==5:
                mean=statistics.mean(values);sd=statistics.stdev(values);half=2.776445105*sd/math.sqrt(5)
                summaries.append(dict(sparsity=sp,step=step,n_source_seeds=5,mean=mean,sd=sd,minimum=min(values),maximum=max(values),descriptive_t95_low=mean-half,descriptive_t95_high=mean+half,positive_seeds=sum(x>0 for x in values)))
    save(DEST/'descriptive_summaries.json',summaries)
    lines=['# Findings','',f'Completed runs: {len(results)}/{len(manifest["jobs"])}. Missing or failed: {len(missing)}.','', 'The primary cohort is incomplete; no cohort effect is inferred.' if missing else 'All registered runs have outcomes. See per-seed contrasts before descriptive aggregates.','', 'The intervention exchanges complete routing slots, including coefficient and drop identities. The causal comparison is shuffled versus identity replay; native recovery is context. No percentage of an earlier freezing penalty is identified.']
    if (DEST/'pilot_feasibility.json').exists():lines.extend(['', 'Pilot feasibility: '+json.dumps(read(DEST/'pilot_feasibility.json'))])
    (DEST/'findings.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,sp in zip(axes,(0.,.8),strict=True):
            for job in manifest['jobs']:
                if job['seed']!=17 or job['sparsity']!=sp:continue
                values=[x for x in curves if x['job']==job['id']]
                if values:ax.plot([x['step'] for x in values],[x['ce'] for x in values],label=job['kind']+str(job['draw'] or ''))
            ax.set(title=f'Seed17 sparsity {sp}',xlabel='Attempted updates',ylabel='Full validation CE',yscale='log')
            if ax.lines:ax.legend()
        (DEST/'figures').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(DEST/'figures/pilot_curves.png');plt.close(fig)
    except ImportError:save(DEST/'figure_status.json',dict(status='MISSING',reason='matplotlib unavailable'))
    save(DEST/'completion.json',dict(scientifically_complete=not missing,complete_runs=len(results),expected_runs=len(manifest['jobs']),missing_failed=len(missing)))


def pilot_decision():
    manifest=read(DEST/'registration.json');rows=[]
    for sp in (0.,.8):
        variant=next(v for v in manifest['variants'] if v['seed']==17 and v['sparsity']==sp)
        base=DEST/'attempts'
        paths=[base/f'{variant["id"]}_{kind}'/'result.json' for kind in ('donor','identity','shuffle_101')]
        if not all(p.exists() for p in paths):return None
        native,identity,shuffle=[read(p) for p in paths]
        # Criteria were registered before pilot outcomes and never inspect penalty sign.
        initial=read(base/f'{variant["id"]}_identity'/'curves.json')[0]['ce']
        learning=(initial-identity['final_ce'])/(initial-native['final_ce'])
        passed=(all(math.isfinite(x['final_ce']) for x in (native,identity,shuffle)) and learning>=.9 and identity['final_ce']<=1.5*native['final_ce'] and all(x['amp_skips']<=500 for x in (identity,shuffle)))
        for suffix in ('identity','shuffle_101'):
            trajectory=read(base/f"{variant['id']}_{suffix}"/'curves.json')
            passed=passed and all(math.isfinite(x['ce']) for x in trajectory)
        rows.append(dict(sparsity=sp,native_ce=native['final_ce'],identity_ce=identity['final_ce'],shuffle_ce=shuffle['final_ce'],native_improvement_retained=learning,passed=passed))
    result=dict(expand=all(x['passed'] for x in rows),rows=rows,criteria=manifest['pilot_criteria'],reason='Identity learning-regime and numerical feasibility only; shuffled penalty sign is not a criterion')
    save(DEST/'pilot_feasibility.json',result)
    (DEST/'pilot_feasibility.md').write_text('# Pilot feasibility\n\n'+json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    pilot_decision();analyze()
