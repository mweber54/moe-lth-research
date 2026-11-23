"""Reporting-only monitor. Waits for all workers to finish; never launches training."""
from run import *
import statistics


def finalize():
    results=[read(path) for path in (DEST/'attempts').glob('*/result.json')]
    complete=read(DEST/'completion.json')['scientifically_complete']
    effects=[]
    with (DEST/'per_seed_differential_penalties.csv').open() as stream:effects=list(csv.DictReader(stream))
    lines=['# Controlled token-to-routing-slot matching findings','',f'Status: {"COMPLETE" if complete else "STOPPED / INCOMPLETE"}. Completed {len(results)} of75 registered runs.','']
    if (DEST/'STOP.json').exists():lines.extend(['Stop record: '+json.dumps(read(DEST/'STOP.json')),''])
    if (DEST/'pilot_feasibility.json').exists():lines.extend(['Pilot feasibility: '+json.dumps(read(DEST/'pilot_feasibility.json')),''])
    lines.extend(['## Every source seed','', '| Seed | Sparsity | Step | Sparse penalty | Dense penalty | Differential | Status |','|---|---|---|---|---|---|---|'])
    for row in effects:lines.append('| '+' | '.join(str(row[x]) for x in ('seed','sparsity','step','sparse_penalty','dense_penalty','differential_penalty','status'))+' |')
    summaries=read(DEST/'descriptive_summaries.json')
    lines.extend(['','## Descriptive seed-level summaries','',json.dumps(summaries,indent=2),''])
    if complete:
        for summary in summaries:
            if summary['step']!=5000:continue
            mean=summary['mean'];positive=summary['positive_seeds']
            lines.append(f"IMP{int(summary['sparsity']*100)}: mean differential penalty {mean:+.6f} nats; {positive}/5 source seeds positive. Descriptive95% t interval [{summary['descriptive_t95_low']:+.6f}, {summary['descriptive_t95_high']:+.6f}].")
        consistent=all(x['positive_seeds']==5 for x in summaries if x['step']==5000) and len(summaries)==4
        if consistent:lines.extend(['','The consistent positive differential supports greater sparse sensitivity to token-to-routing-slot matching under this replay protocol. Recommend one subsequent constrained intervention that holds token-specific coefficients and drop identities fixed while reassigning accepted expert identities under preserved exposure counts. That follow-up was not launched.'])
        else:lines.extend(['','Mixed, zero or negative seed effects must remain visible. These results do not establish uniform greater sparse sensitivity across this cohort.'])
    else:lines.extend(['No full-cohort conclusion is supported. Missing/failed draws are not averaged into complete-seed contrasts.'])
    lines.extend(['','Ordinary native recovery and replay are separate learning regimes. The causal comparison is shuffled replay versus identity replay. Complete slots move expert identity, token-specific coefficient and drop identity together; assignment identity alone is not identified. This experiment does not establish IMP-encoded token specialization, the cause of historical sparse wins, novelty, generality, or a percentage of an earlier freezing penalty.','', 'Raw per-condition endpoints, complete validation/training curves, named task gradients, component displacements, trace inventories, checksums, exposure counts and gate distributions remain in the isolated attempt directories. Missing/failed conditions are explicit in missing_failed.json.'])
    (DEST/'findings.md').write_text('\n'.join(lines)+'\n')
    native_comparison=[]
    for seed in SEEDS:
        for sp in (.8,.85):
            for kind in ('donor','identity'):
                dense=next((x for x in results if x['seed']==seed and x['sparsity']==0 and x['kind']==kind),None)
                sparse=next((x for x in results if x['seed']==seed and x['sparsity']==sp and x['kind']==kind),None)
                native_comparison.append(dict(seed=seed,sparsity=sp,regime='native_recovery' if kind=='donor' else 'identity_replay',dense_ce=dense['final_ce'] if dense else None,sparse_ce=sparse['final_ce'] if sparse else None,sparse_minus_dense=sparse['final_ce']-dense['final_ce'] if dense and sparse else None,strict_match=sparse['final_ce']<=dense['final_ce'] if dense and sparse else None))
    table(DEST/'native_vs_replay_matching.csv',native_comparison)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,3,figsize=(14,18),sharex=True)
    for i,seed in enumerate(SEEDS):
        for j,sp in enumerate((0.,.8,.85)):
            ax=axes[i,j]
            for result in results:
                if result['seed']!=seed or result['sparsity']!=sp:continue
                job=result['variant']+'_'+result['kind']+(f"_{result['draw']}" if result['draw'] is not None else '')
                curve=read(DEST/'attempts'/job/'curves.json')
                ax.plot([x['step'] for x in curve],[x['ce'] for x in curve],label=result['kind']+str(result['draw'] or ''),alpha=.8)
            ax.set(title=f'Seed{seed}, sparsity{sp}',yscale='log',xlabel='Attempted updates',ylabel='Full validation CE')
            if ax.lines:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(DEST/'figures/all_learning_curves.png');fig.savefig(DEST/'figures/all_learning_curves.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for sp,color in ((.8,'steelblue'),(.85,'darkorange')):
        values=[x for x in effects if float(x['sparsity'])==sp and int(x['step'])==5000 and x['status']=='COMPLETE']
        ax.scatter([SEEDS.index(int(x['seed']))+(-.1 if sp==.8 else .1) for x in values],[float(x['differential_penalty']) for x in values],label=f'IMP{int(sp*100)}',color=color)
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(5),SEEDS);ax.set(xlabel='Source seed',ylabel='Differential penalty (nats)');ax.legend();fig.tight_layout();fig.savefig(DEST/'figures/seed_differential_penalties.png');fig.savefig(DEST/'figures/seed_differential_penalties.pdf');plt.close(fig)
    save(DEST/'reporting_complete.json',dict(scientifically_complete=complete,unix=time.time()))


if __name__=='__main__':
    with lock(DEST/'reporting_monitor.lock',False):
        while not all((DEST/'workers'/f'{host}.finished.json').exists() for host in HOSTS):time.sleep(30)
        finalize()
