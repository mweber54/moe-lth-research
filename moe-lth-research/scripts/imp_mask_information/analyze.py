"""Descriptive pilot analysis; a single checkpoint is not a seed-level sample."""
import csv
import json
import math
import statistics
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from audit import OUT, rows, write_csv


def analyze():
    observations=rows(OUT/'pilot_results.csv')
    imp=next((float(r['final_ce']) for r in observations if r['condition']=='IMP'),None)
    summaries=[]
    for condition in ['global','expert','matrix']:
        values=[float(r['final_ce']) for r in observations if r['condition']==condition]
        if not values or imp is None: continue
        summaries.append(dict(condition=condition,n_masks=len(values),mean_ce=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None,min_ce=min(values),max_ce=max(values),imp_ce=imp,delta_random_minus_imp=statistics.mean(values)-imp,imp_percentile=100*sum(x<=imp for x in values)/len(values),independent_training_seeds=1,seed_level_ci='not_estimable_from_one_checkpoint'))
    if summaries: write_csv(OUT/'pilot_analysis.csv',summaries)
    status=json.loads((OUT/'pilot_status.json').read_text()) if (OUT/'pilot_status.json').exists() else dict(status='prepared' if not observations else 'running')
    if len(observations)<16 and status.get('status')=='complete': raise ValueError('Incomplete pilot marked complete')
    plan=json.loads((OUT/'pilot_plan.json').read_text()) if (OUT/'pilot_plan.json').is_file() else None
    text=['# IMP mask information pilot', '',f"Status: {status['status']}. Completed recoveries: {len(observations)}/16.",'', '## Direct observations','']
    if plan:
        c=plan['configuration']
        text += [f"Pilot checkpoint: {c['configuration_id']}. Estimated total GPU occupancy: {plan['estimate']['estimated_gpu_hours']:.2f} hours; contention can increase runtime.", '']
    if status.get('active_condition'):
        text += [f"Active condition: {status['active_condition']}, replicate {status.get('active_replicate',0)}.", '']
    if imp is None: text+=['No completed baseline reproduction or randomized-mask endpoint is available. No mask advantage can be estimated.']
    else:
        if (OUT/'reproduction_check.json').is_file():
            check=json.loads((OUT/'reproduction_check.json').read_text())
            text += [f"Baseline reproduction gate passed: {check['passed']}; absolute CE discrepancy {check['absolute_ce_discrepancy']:.6f}."]
        text += [f'IMP rerun final validation CE: {imp:.6f} nats/token.']
        for s in summaries:
            text += [f"{s['condition']}: {s['n_masks']} masks, mean CE {s['mean_ce']:.6f}, sample SD {s['sample_sd']}, random-minus-IMP {s['delta_random_minus_imp']:+.6f}."]
    text += ['', '## Statistical findings','', 'The pilot has one independently trained source checkpoint. Mask realizations are nested within it. No seed-level confidence interval, statistical significance, or practical equivalence is established. The prespecified equivalence margin is +/-0.02 nats/token. Partial results are provisional.', '', '## Mechanistic hypotheses and unresolved alternatives','', 'Any observed contrast concerns connection placement beyond the specified sparsity allocation. It cannot establish indispensable individual connections, routed-token information, or disrupted early learning. Architecture, training seed, sparsity, router protocol and recovery duration remain possible moderators. Selection of a strong IMP endpoint limits generalization. Expand to independent seeds and nonwinning controls only after review; no automatic expansion.', '', 'See protocol.md, checkpoint_inventory.csv, audit_details.json and pilot_plan.json for provenance. Source IMP and dense endpoints are prior observations, not new pilot controls. The 5,000-step fresh-optimizer extensions remain separate.']
    if status.get('error'): text+=['',f"Stop reason: {status['error']}"]
    (OUT/'summary.md').write_text('\n'.join(text)+'\n')
    if not summaries: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    conditions=[s['condition'] for s in summaries]
    def save(name):
        plt.tight_layout();plt.savefig(dest/name,dpi=160);plt.close()
    plt.figure();plt.bar(conditions,[s['mean_ce'] for s in summaries],yerr=[s['sample_sd'] or 0 for s in summaries]);plt.axhline(imp,color='black',label='reproduced IMP');plt.ylabel('Final validation CE');plt.legend();save('final_ce.png')
    plt.figure()
    for i,c in enumerate(conditions):
        values=[float(r['final_ce']) for r in observations if r['condition']==c]
        plt.scatter([i]*len(values),values,label=c)
    plt.xticks(range(len(conditions)),conditions);plt.axhline(imp,color='black',label='IMP');plt.ylabel('Final validation CE (each mask)');plt.legend();save('random_distributions.png')
    plt.figure()
    for c in ['IMP']+conditions:
        r=next(r for r in observations if r['condition']==c)
        curve=[json.loads(line) for line in Path(r['curves_path']).read_text().splitlines()]
        # Canonical curve fields are step and loss (see source trainer).
        x=[p['step'] for p in curve]
        y=[p.get('loss',p.get('validation_loss')) for p in curve]
        if any(v is None for v in y): raise ValueError('Unrecognized canonical validation curve')
        plt.plot(x,y,label=f'{c} mask 0')
    plt.yscale('log');plt.xlabel('Recovery attempted steps');plt.ylabel('Validation CE (log scale)');plt.legend();save('recovery_curves.png')
    for field,xlabel,name in [('sparsity','Realized expert sparsity','advantage_vs_sparsity.png'),('num_experts','Expert count','advantage_across_architectures.png')]:
        plt.figure()
        for s in summaries:
            row=next(r for r in observations if r['condition']==s['condition'])
            plt.scatter(float(row[field]),s['delta_random_minus_imp'],label=s['condition'])
        plt.axhline(0,color='black');plt.ylabel('Mean random CE minus IMP CE');plt.xlabel(xlabel);plt.title('Single-checkpoint pilot; no cross-configuration inference');plt.legend();save(name)

if __name__=='__main__': analyze()
