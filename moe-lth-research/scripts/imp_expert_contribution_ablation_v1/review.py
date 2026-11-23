"""Validate the prespecified complete grid and render its descriptive interpretation."""
import csv,hashlib,json,statistics,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/imp_expert_contribution_ablation_v1'
SEEDS=[7,17,29,43,61]
MODES=['NATIVE_RESPONSE','BASELINE_REPLAY']
QS=['BASELINE','ZERO_L0','ZERO_L1','ZERO_L2','ZERO_L3','ZERO_ALL']
def rows(name):
    with (OUT/(name+'.csv')).open() as f:return list(csv.DictReader(f))
def summary(values):
    return f'mean {statistics.mean(values):.6f}, median {statistics.median(values):.6f}, min {min(values):.6f}, max {max(values):.6f}, sample SD {statistics.stdev(values):.6f}'
def main():
    assert json.loads((OUT/'completion.json').read_text())['scientifically_complete']
    losses=rows('absolute_losses');effects=rows('seed_level_ablation_effects');temporal=rows('seed_level_temporal_contrasts');gaps=rows('sparse_unpruned_contrasts');contrasts=rows('routing_mode_contrasts')
    idx={(int(r['seed']),r['model_type'],int(r['offset']),r['mode'],r['intervention']):float(r['task_ce']) for r in losses}
    assert len(losses)==len(idx)==240
    expected={(s,m,t,p,q) for s in SEEDS for m in ['sparse','unpruned'] for t in [0,500] for p in MODES for q in QS}
    assert set(idx)==expected
    for s,m,t,p,q in expected:
        if q in ['BASELINE','ZERO_ALL']:assert abs(idx[s,m,t,MODES[0],q]-idx[s,m,t,MODES[1],q])<=1e-6
    def T(s,m,p,q):return next(r for r in temporal if (int(r['seed']),r['model_type'],r['mode'],r['intervention'])==(s,m,p,q))
    def A(s,m,t,p,q):return idx[s,m,t,p,q]-idx[s,m,t,p,'BASELINE']
    for s,m,t,p,q in expected:
        d=float(T(s,m,p,q)['delta_A']);rr=idx[s,m,0,p,q]-idx[s,m,500,p,q]
        rb=idx[s,m,0,p,'BASELINE']-idx[s,m,500,p,'BASELINE']
        assert abs(d-(rb-rr))<1e-12
    for r in gaps:
        s=int(r['seed']);p=r['mode'];q=r['intervention']
        assert abs(float(r['X'])-(float(T(s,'sparse',p,q)['delta_A'])-float(T(s,'unpruned',p,q)['delta_A'])))<1e-12
    batches=rows('per_batch_losses')
    assert len(batches)==20*31*12
    for s,m,t,p,q in expected:
        rs=[r for r in batches if (int(r['seed']),r['model_type'],int(r['offset']),r['mode'],r['intervention'])==(s,m,t,p,q)]
        assert len(rs)==31 and sum(int(r['target_tokens']) for r in rs)==492416
        assert abs(sum(float(r['nll_sum']) for r in rs)/492416-idx[s,m,t,p,q])<1e-12
    checks=json.loads((OUT/'numerical_checks.json').read_text());assert checks['status']=='PASS'
    checks['independent_reporting_validation']=dict(status='PASS',logical_cells=240,per_batch_records=len(batches),target_tokens_per_cell=492416,temporal_and_sparse_contrast_identities=True,baseline_and_zero_all_mode_ce_equivalence=True)
    (OUT/'numerical_checks.json').write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n')
    initial=(OUT/'findings.md').read_text()
    answer=['\n## Answers to the research questions\n']
    for p in MODES:
        answer.append('\n### '+p+'\n')
        for m in ['sparse','unpruned']:
            ds=[float(T(s,m,p,'ZERO_ALL')['delta_A']) for s in SEEDS]
            answer.append(f"{m}: ZERO_ALL removal becomes more harmful in {sum(x>0 for x in ds)}/5 seeds and less harmful in {sum(x<0 for x in ds)}/5. Individual delta_A in seed order 7,17,29,43,61: {', '.join(f'{x:.6f}' for x in ds)}. {summary(ds)}.\n")
            rs=[float(T(s,m,p,'ZERO_ALL')['R_intervention']) for s in SEEDS]
            answer.append(f"Absolute CE improvement visible under ZERO_ALL for {m}: {', '.join(f'{x:.6f}' for x in rs)}; {summary(rs)}. Experts were present during historical training, so this is not a training-without-experts counterfactual.\n")
            for q in QS[1:5]:
                ds=[float(T(s,m,p,q)['delta_A']) for s in SEEDS]
                answer.append(f"{m} {q}: sensitivity increases in {sum(x>0 for x in ds)}/5 and decreases in {sum(x<0 for x in ds)}/5; seed values {', '.join(f'{x:.6f}' for x in ds)}; {summary(ds)}.\n")
        xs=[float(next(r['X'] for r in gaps if (int(r['seed']),r['mode'],r['intervention'])==(s,p,'ZERO_ALL'))) for s in SEEDS]
        answer.append(f"Sparse-specific change X: {', '.join(f'{x:.6f}' for x in xs)}; {summary(xs)}.\n")
    answer.append('\n### Routing response and harmful contributions\n')
    for m in ['sparse','unpruned']:
        for q in QS[1:]:
            ds=[float(next(r['delta_A_native_minus_replay'] for r in contrasts if (int(r['seed']),r['model_type'],r['intervention'])==(s,m,q))) for s in SEEDS]
            persistent=sum((float(T(s,m,MODES[0],q)['delta_A'])>0)==(float(T(s,m,MODES[1],q)['delta_A'])>0) for s in SEEDS)
            answer.append(f"{m} {q}: native and replay delta_A signs agree in {persistent}/5 seeds; native-minus-replay seed values {', '.join(f'{x:.6f}' for x in ds)}. This contrast includes changes in assignments, gates and capacity.\n")
    harmful=[r for r in effects if float(r['A'])<0 and r['intervention']!='BASELINE']
    answer.append(f'Negative A (removal improves CE) occurs in {len(harmful)} intervention cells. Exact seed, model, offset, mode, intervention and signed damage follow.\n')
    for r in harmful:answer.append(f"- Seed {r['seed']}, {r['model_type']}, offset {r['offset']}, {r['mode']}, {r['intervention']}: A={float(r['A']):.6f}.\n")
    mismatch=[]
    for s in SEEDS:
        for m in ['sparse','unpruned']:
            for p in MODES:
                for t in [0,500]:
                    joint=A(s,m,t,p,'ZERO_ALL');singles=[A(s,m,t,p,q) for q in QS[1:5]]
                    mismatch.append(joint-sum(singles))
    answer.append(f'Joint minus summed single-layer damage spans {min(mismatch):.6f} to {max(mismatch):.6f} CE across checkpoints and modes. Joint and single-layer interventions are distinct and must not be interpreted as independent component contributions.\n')
    if '\n## Answers to the research questions\n' in initial:initial=initial.split('\n## Answers to the research questions\n')[0]
    (OUT/'findings.md').write_text(initial+'\n'.join(answer))
    figures(idx,T,gaps)
    accounting=json.loads((OUT/'compute_accounting.json').read_text());accounting.update(cohort_model_forwards=20*31*12,full_native_precohort_forwards=31,profile_model_forwards=12,training_updates=0,new_masks=0,permanent_parameter_changes=0)
    (OUT/'compute_accounting.json').write_text(json.dumps(accounting,indent=2,sort_keys=True)+'\n')
    with tarfile.open(OUT/'summary.tar.gz','w:gz') as archive:
        for name in ['findings.md','completion.json','numerical_checks.json','absolute_losses.csv','seed_level_ablation_effects.csv','seed_level_temporal_contrasts.csv','sparse_unpruned_contrasts.csv','routing_mode_contrasts.csv']:archive.add(OUT/name,arcname=name)
    print('Validated 240 cells and 7440 per-batch records; summary archive refreshed.')

def figures(L,T,gaps):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=OUT/'figures';d.mkdir(exist_ok=True)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for mi,m in enumerate(['sparse','unpruned']):
        for pi,p in enumerate(MODES):
            ax=axes[mi,pi]
            for s,color in zip(SEEDS,colors):
                for q,style in [('BASELINE','-'),('ZERO_ALL','--')]:ax.plot([0,500],[L[s,m,t,p,q] for t in [0,500]],style,marker='o',color=color,label=f'{s} {q}')
            ax.set_title(m+' '+p);ax.set_xlabel('Recovery offset');ax.set_ylabel('Task CE');ax.legend(fontsize=6,ncol=2)
    fig.tight_layout();fig.savefig(d/'1.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for mi,m in enumerate(['sparse','unpruned']):
        for pi,p in enumerate(MODES):
            ax=axes[mi,pi]
            for s,color in zip(SEEDS,colors):ax.plot([0,500],[L[s,m,t,p,'ZERO_ALL']-L[s,m,t,p,'BASELINE'] for t in [0,500]],marker='o',color=color,label=str(s))
            ax.set_title(m+' '+p);ax.set_xlabel('Recovery offset');ax.set_ylabel('ZERO_ALL damage A');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(d/'2.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for mi,m in enumerate(['sparse','unpruned']):
        for pi,p in enumerate(MODES):
            ax=axes[mi,pi]
            for s,color in zip(SEEDS,colors):ax.plot(range(4),[float(T(s,m,p,q)['delta_A']) for q in QS[1:5]],marker='o',color=color,label=str(s))
            ax.set_xticks(range(4),QS[1:5]);ax.set_title(m+' '+p);ax.set_ylabel('delta_A');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(d/'3.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,p in zip(axes,MODES):
        for s,color in zip(SEEDS,colors):ax.plot(range(5),[float(next(r['X'] for r in gaps if (int(r['seed']),r['mode'],r['intervention'])==(s,p,q))) for q in QS[1:]],marker='o',color=color,label=str(s))
        ax.set_xticks(range(5),QS[1:]);ax.set_title(p);ax.set_ylabel('X: sparse-minus-unpruned delta_A');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(d/'4.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,m in zip(axes,['sparse','unpruned']):
        for s,color in zip(SEEDS,colors):
            for q,marker in zip(QS[1:],['o','s','^','v','*']):ax.scatter(float(T(s,m,MODES[0],q)['delta_A']),float(T(s,m,MODES[1],q)['delta_A']),color=color,marker=marker,label=f'{s} {q}')
        lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],color='gray',lw=.7);ax.set_title(m);ax.set_xlabel('Native-response delta_A');ax.set_ylabel('Baseline-replay delta_A');ax.legend(fontsize=5,ncol=3)
    fig.tight_layout();fig.savefig(d/'5.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
