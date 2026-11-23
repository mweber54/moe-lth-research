"""Readable CE zoom plots and concise interpretation of the frozen measurements.

Presentation supplement after visual review. No intervention or metric definition
changes, new measurements, predictors, thresholds or extra populations.
"""
from dose_common import *
import pandas as pd


def main():
    started=time.perf_counter();verify_registration();sealed=s.read(D/'output_seal.json')
    for filename in ('measurements.csv','ab_reliability.csv','compute_accounting.json'):
        assert sha(D/filename)==sealed['files'][filename]
    d=pd.read_csv(D/'measurements.csv');r=pd.read_csv(D/'ab_reliability.csv')
    weights=d[(d.intervention_kind=='weight_union')&(d.subset=='whole')]
    ranks=r[(r.scope=='within_expert_equal_dose')&(r.coverage=='adequate_in_both_probes')&(r.subset=='whole')&(r.dose_groups.isin([1,2,4]))]
    summary={}
    for kind in ('sparse','unpruned'):
        ce=ranks[(ranks.trajectory==kind)&(ranks.metric=='damage')].spearman.dropna()
        functional=ranks[(ranks.trajectory==kind)&(ranks.metric=='relative_output_rms_difference')].spearman.dropna()
        summary[kind]=dict(defined_panels=len(ce),median_CE_spearman=float(ce.median()),median_relative_RMS_spearman=float(functional.median()),min_relative_RMS_spearman=float(functional.min()),max_relative_RMS_spearman=float(functional.max()))
    # Plot y ranges follow measured effects, so reference lines cannot flatten small effects.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(s.EXPERTS,('tab:blue','tab:orange','tab:green','tab:red')))
    fig,axes=plt.subplots(4,2,figsize=(12,13))
    for row,t in enumerate(s.STEPS):
      for col,kind in enumerate(('sparse','unpruned')):
        ax=axes[row,col];z=weights[(weights.trajectory==kind)&(weights.step==t)]
        for e in s.EXPERTS:
          for p,style in [('A','-'),('B','--')]:
            q=z[(z.expert==e)&(z.probe==p)];g=q.groupby('dose_groups').damage.mean()
            ax.plot(g.index,g,style+'o',color=colors[e],label=f'e{e} {p}',ms=3)
            ax.scatter(q.dose_groups,q.damage,color=colors[e],s=9,alpha=.25)
        ax.axhline(0,color='black',lw=.4);low,high=ax.get_ylim()
        for threshold in (.0001,.001):
            if low<=threshold<=high:ax.axhline(threshold,color='grey',ls=':',lw=.7)
        ax.set_ylim(low,high);ax.set_xticks(DOSES);ax.set_title(f'{kind}, step {t}');ax.set_xlabel('Groups deleted from one expert fc1');ax.set_ylabel('Signed whole-probe CE damage');ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0));ax.grid(alpha=.12)
        if row==0:ax.legend(fontsize=6,ncol=2)
    fig.suptitle('Whole-probe CE zoom · native fixed A/B, unique unions\nData-scaled axes; 0.0001/0.001 reference lines shown only if in range; output controls separate')
    fig.tight_layout();fig.savefig(D/'figures/whole_probe_ce_zoom.png',dpi=160);plt.close(fig)
    table=[]
    for kind in ('sparse','unpruned'):
        for t,z in weights[(weights.trajectory==kind)&(weights.dose_groups==8)].groupby('step'):
            table.append(f'| {kind} | {t} | {z.damage.mean():.6g} | {z.damage.min():.6g} to {z.damage.max():.6g} |')
    accepted=d[(d.intervention_kind=='weight_union')&(d.subset=='baseline_accepted')].damage
    whole=weights.damage
    eight=weights[weights.dose_groups==8]
    control=d[(d.intervention_kind=='expert_output_zero')&(d.subset=='whole')].damage
    checks=s.read(D/'numerical_checks.json');repeats=sum(c['repeat_tested'] for c in checks['checks'])
    accounting=s.read(D/'compute_accounting.json')
    verdict=f'''# Concise verdict: expert-local deletion-dose check

The numerical pipeline works, and expert-function dependence is distinguishable.
Task-CE rankings across A/B are not yet consistently reliable at equal dose.
**Improve CE measurement precision and low-traffic coverage before expanding a
CE-based dispensability study.** Cross-seed replication of the functional dose
response is reasonable as a measurement audit; it would not resolve the current
within-seed A/B task-loss ranking problem by itself. No expansion was launched.

- **Measurable and reproducible?** All {repeats} representative repeat checks were
  bitwise identical. At equal doses of 1/2/4 groups, adequate-coverage within-expert
  A/B CE rank correlations have median {summary['sparse']['median_CE_spearman']:.3g}
  for sparse and {summary['unpruned']['median_CE_spearman']:.3g} for unpruned, with
  observed reversals (range −1 to 1). Functional relative-output-RMS ranks have
  median 1 in both trajectories (range 0.8–1). Eight-group interventions change
  expert-output RMS by {eight.relative_output_rms_difference.min():.2%}–{eight.relative_output_rms_difference.max():.2%}
  of baseline where coverage exists. This establishes reproducible function changes,
  not equally reliable CE-based importance ordering.
- **Does whole-probe averaging conceal conditional effects?** Yes. Weight-union
  whole-probe damage ranges from {whole.min():.6g} to {whole.max():.6g} nats/token;
  baseline-accepted damage ranges from {accepted.min():.6g} to {accepted.max():.6g}.
  The separate output-zero control reaches {control.max():.6g} whole-probe nats/token.
  Sparse step 25 has zero accepted positions for experts 8/16/24 and only 17 on A,
  16 on B for expert 0. These cases remain explicit and do not demonstrate low
  expert dependence. Conditional subsets are fixed from baseline; they are not
  isolated expert losses and may have very small denominators.
- **Does dependence change over recovery?** Task sensitivity changes, without a
  universal monotone trend. Eight-group unpruned whole-probe damage is much larger
  at step 25 than later; sparse effects are small, signed, and more variable at
  step 2500. Fixed-coordinate temporal contrasts and traffic are stored per expert.
  Changes in task loss, context mix and accepted traffic prevent a specialization
  interpretation. Equal-dose function-output changes remain observable throughout
  covered cases.
- **What next?** Improve the task-CE probe precision/coverage first, using a separately
  declared measurement budget, without choosing a dose or threshold to force a trend.
  The method is ready for engineering replication, but these data do not justify
  stable CE-based group ranking, a specialization predictor or permanent pruning.

Eight-group whole-probe damage, means/ranges across the four experts and A/B:

| Trajectory | Step | Mean damage | Range |
|---|---:|---:|---:|
{chr(10).join(table)}

Zero dose, native baseline/cache equivalence, exact restoration, model hashes,
RNG, gradients, masks and native selection/gates passed. Four fixed orderings reduce
to 52 unique weight unions plus four separate output controls, with no global cap.
The 0.001 and 0.0001 reference lines were not used for selection. Both A/B have been
examined previously; this is one seed with nested groups, experts, orderings and
checkpoints, not independent replications or a generalization test.

Measurement/table serialization took {accounting['measure_wall_seconds']:.2f} seconds
on {accounting['gpu']}, with {accounting['cached_single_group_measurements_reused']}
single-group CE measurements reused. New cache storage is
{accounting['new_cache_bytes']/1e6:.2f} MB; all outputs are approximately
{accounting['storage_bytes_before_final_seal']/1e6:.2f} MB. See final_audit.json for
final measured storage and compute_accounting.json for forward counts and timing.
'''
    (D/'verdict.md').write_text(verdict)
    save(D/'presentation_supplement.json',dict(purpose='Concise interpretation and data-scaled CE zoom following visual review; no new effects/thresholds/interventions',script=str(Path(__file__).relative_to(ROOT)),script_sha256=sha(Path(__file__)),measurements_sha256=sha(D/'measurements.csv'),reliability_sha256=sha(D/'ab_reliability.csv'),rank_summary=summary,wall_seconds=time.perf_counter()-started))
    for p in [D/'verdict.md',D/'presentation_supplement.json',D/'figures/whole_probe_ce_zoom.png']:
        sealed['files'][str(p.relative_to(D))]=sha(p)
    save(D/'output_seal.json',sealed)
    print('Concise verdict and data-scaled CE zoom saved',flush=True)
if __name__=='__main__':
    with worker():main()
