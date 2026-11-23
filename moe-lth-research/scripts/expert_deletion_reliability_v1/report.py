"""Report fixed-stratum results and figures without pooled ranking claims."""
from reliability_common import *
import pandas as pd


def fmt(x):return 'undefined' if pd.isna(x) else f'{x:.5g}'


def main():
    started=time.perf_counter();verify_frozen();assert s.read(D/'analysis_completion.json')['complete']
    ranks=pd.read_csv(D/'discovery_confirmation_rankings.csv');decisions=pd.read_csv(D/'condition_decisions.csv');cov=pd.read_csv(D/'coverage_by_batch_and_block.csv');unc=pd.read_csv(D/'uncertainty_and_tolerance_summary.csv');functional=pd.read_csv(D/'functional_distortion_summary.csv');contrasts=pd.read_csv(D/'paired_intervention_contrasts.csv')
    plots(ranks,decisions,cov,unc,functional)
    covered=decisions[(decisions.D_blocks_below_minimum==0)&(decisions.dose_groups.isin([1,2,4]))]
    reproduced=covered[covered.A_reproducibility=='directional_contrast_reproduced_on_confirmation'];reversed_=covered[covered.A_reproducibility=='directional_contrast_reversed_candidate_context_dependence']
    if len(covered)==0:conclusion='insufficient coverage'
    elif len(reproduced)==0 and len(reversed_)==0:conclusion='adequate coverage but unresolved task-damage discrimination at this budget'
    elif len(reproduced)==len(covered):conclusion='sufficient measurement reliability in explicitly named conditions to justify a later fixed cross-seed study'
    else:conclusion='mixed results requiring a narrower claim'
    blockcov=cov[cov.level=='block'];checks=s.read(D/'numerical_checks.json');tested=[c for c in checks['checks'] if c['repeat_tested']];repeat=max((c['repeat_loss_max_abs'] for c in tested),default=0.)
    lines=['# Fixed-budget expert deletion reliability','',f'**Conclusion: {conclusion}.**','',
      'This is seed 17, the same nonmatching 32E IMP80 engineering case, with its matched unpruned recovery. No training, optimizer update, backward pass, recovery, mask search, permanent pruning, new source seed, predictor or follow-up launch occurred. Historical reports and checkpoints remain unchanged.','',
      '## Fixed design and probe provenance','',
      'Exactly 32 additional complete 128×128 training batches were frozen before new deletion measurements. The user authorized re-batching intact canonical 128-token sequences; historical shuffled minibatch membership is not claimed. The fixed availability-only sampler uses seed 171032 and excludes previous A/B windows, overlapping input-plus-target source spans and duplicate full-window content. All 4,096 new windows are source-disjoint, including shifted targets. The same batches are used for all eight trajectory/checkpoint combinations. Blocks 1–4 are discovery; 5–8 confirmation. Both are training-distribution contexts, possibly seen during training, not independent generalization or source-seed samples. See probe_manifest.csv and probe_overlap_audit.json.','',
      'The original layer, experts, coordinate groups, orderings and 0/1/2/4/8-group unions were retained: 52 unique weight unions plus four separate output-zero controls. Deletions are coordinate-matched across models, not density-matched. Step 2500 is verified pre-reset. Historical/rerun endpoint aliases count once. Source code, coordinates, numerical definitions and metadata were reused; prior A/B losses were not copied onto these new contexts.','',
      '## A–E: per-condition decision','',
      f'Among {len(covered)} operationally covered whole-probe conditions with at least three distinct same-dose unions, {len(reproduced)} discovery-selected contrasts have a confirmation paired-batch 95% interval entirely above zero, and {len(reversed_)} entirely below zero. These are descriptive sampled-context intervals in one source seed, with multiple dependent panels and no multiplicity-adjusted or seed-level success claim. The full per-expert table below and condition_decisions.csv are primary; there is no global PASS.','',
      'A: discovery/confirmation reproducibility is judged within the same trajectory, checkpoint, expert and dose. B: a contrast interval crossing zero is unresolved at this budget, not evidence of harmlessness. C: functional-distortion/CE associations and cross-phase functional rankings are recorded in discovery_confirmation_rankings.csv for that same stratum; a functional output change need not order task damage. D: all eight blocks need at least 32 accepted positions for conditional-ranking eligibility; low-coverage whole-probe measurements remain visible. E: any reproduced contrast supports only a later explicitly fixed measurement confirmation in that named condition, not future persistence or permanent-pruning safety.','',
      '| Trajectory | Step | Expert | Groups | Accepted minimum/block | Discovery–confirmation rho | Selected confirmation contrast [95% interval] | A/B/E interpretation |','|---|---:|---:|---:|---:|---:|---|---|']
    for r in decisions.itertuples():lines.append(f'| {r.trajectory} | {r.step} | {r.expert} | {int(r.dose_groups)} | {r.D_minimum_accepted_in_block} | {fmt(r.spearman)} | {fmt(r.selected_confirmation_contrast)} [{fmt(r.selected_confirmation_ci95_low)}, {fmt(r.selected_confirmation_ci95_high)}] | {r.A_reproducibility} |')
    lines+=['','Each row retains expert identity. The eight-group dose contains one unique union per expert, so it cannot identify an equal-dose ranking or high-minus-low comparison. The 1-group panel has three unique unions; 2/4-group panels have four. Small panels and overlapping unions require caution even when Spearman is ±1. Zero dose and output-zero controls are not mixed into dose rankings.','',
      '## Sampling precision, near ties and context variation','',
      'All fixed pairwise contrasts—not only the discovery-selected extremes—are in paired_intervention_contrasts.csv. It contains discovery, confirmation and each of the eight blocks, with contrast sizes, paired complete-batch bootstrap intervals, target counts and zero-denominator frequencies. Selection used discovery only, with frozen ID tie-breaking; confirmation never reselected either extreme. discovery_confirmation_rankings.csv retains underlying intervention damages, numerical ties, bootstrap rank intervals/undefined frequencies, confirmation block signs/intervals and phase-specific leave-one-block-out sensitivities.','',
      'Every bootstrap uses 2,000 fixed-seed resamples of complete batches, paired across all interventions. Conditional ratios resample loss sums and accepted counts, then divide; conditional means with different denominators are not averaged. Undefined replicates remain undefined. These intervals measure variation over sampled training contexts in one seed, not model/seed generalization. Negative rank correlations alone do not establish context-dependent importance: compare the paired contrasts, their uncertainty, coverage and leave-one-block-out estimates before treating context dependence as a candidate explanation. No semantic or specialization labels were fitted.','',
      'uncertainty_and_tolerance_summary.csv reports continuous effects and intervals first, followed by interpretations relative to 0.0001 and 0.001 nats/token. An interval below a convention is compatible with below-convention damage on this probe only. Crossing is unresolved; exceeding is evidence relative to that convention. None defines harmlessness, permanent dispensability or pruning safety.','',
      '## Coverage and functional distortion','',
      f'{int((blockcov.accepted_tokens<MIN_ACCEPTED).sum())} of {len(blockcov)} expert/trajectory/checkpoint/block cells fall below the operational minimum of 32 accepted positions. This threshold is a coverage flag, not proof of precision. Contributing complete-batch counts, maximum batch fractions, accepted-position HHI and effective batch counts are stored for all cells. No expert, checkpoint or batch was replaced after seeing coverage.','',
      'Whole-model CE changes can propagate beyond baseline-accepted positions through attention and later layers; accepted-position CE is not an isolated expert loss. Empty conditional populations are undefined, not zero. Negative damage is retained.','',
      'functional_distortion_summary.csv stores the squared output-difference numerator, baseline-output energy denominator, element counts and gate-weighted energies separately. Pooled RMS ratios form sqrt(sum numerator / sum denominator), with the original near-zero baseline rule, and are explicitly distinguished from the average of defined batch ratios. Distortion-versus-task-damage correlations stay within expert/checkpoint/dose and are not manufactured by pooling doses.','',
      '## Numerical checks and execution','',
      f'{len(tested)} representative repeats had maximum token-loss disagreement {repeat:.6g}. Zero-dose equivalence, unmodified native baseline equivalence, exact intervention restoration, full model hashes, original IMP masks, RNG, module modes, no gradients and native targeted-layer routing/gates were checked. Deterministic repeats are not evidence of agreement across contexts.','']
    accounting=s.read(D/'compute_accounting.json')
    if (D/'engineering_pause.json').exists():
        accounting['adder_original_worker_elapsed_seconds']=s.read(D/'engineering_pause.json')['elapsed_seconds']
        accounting['combined_main_worker_elapsed_seconds']=accounting.get('viper_interrupted_worker_elapsed_seconds',0)+accounting['adder_original_worker_elapsed_seconds']+accounting['measure_wall_seconds']
        if (D/'prefix_engineering_pause.json').exists():
            accounting['adder_integer_engine_worker_elapsed_seconds']=s.read(D/'prefix_engineering_pause.json')['elapsed_seconds']
            accounting['combined_main_worker_elapsed_seconds']+=accounting['adder_integer_engine_worker_elapsed_seconds']
            accounting['prefix_engine_validation']=s.read(D/'prefix_validation.json')
            if (D/'return_to_viper_transition.json').exists():
                accounting['adder_prefix_engine_worker_elapsed_seconds']=s.read(D/'return_to_viper_transition.json')['elapsed_seconds']
                accounting['combined_main_worker_elapsed_seconds']+=accounting['adder_prefix_engine_worker_elapsed_seconds']
                accounting['final_worker_host']='viper.cs.kent.edu'
                accounting['final_worker_wall_seconds']=accounting['measure_wall_seconds']
                accounting.pop('adder_resumed_measurement_wall_seconds',None)
        accounting['integer_dispatch_acceleration']=s.read(D/'acceleration_validation.json')
    accounting['report_wall_seconds_before_seal']=time.perf_counter()-started;accounting['storage_bytes_before_seal']=sum(p.stat().st_size for p in D.rglob('*') if p.is_file());save(D/'compute_accounting.json',accounting)
    if (D/'prefix_validation.json').exists():
        lines += ['A second exact-compatible optimization reused only the unmodified complete-batch prefix strictly upstream of layer1. Native layer1 and every downstream layer were still evaluated for each deletion, with complete-batch routing/capacity. All original profile effects, sequence statistics and functional energies matched bitwise. Prefix tensors were ephemeral GPU working memory, never saved as hidden-data caches. New cache identities also bind prefix_engine_sha256.', '']
    if (D/'acceleration_validation.json').exists():
        lines += ['An engineering acceleration replaced dynamic integer assignment-index synchronization with stable integer grouping; native router probabilities, top-k capacity selection, gates, expert order, expert computation and index_add remained unchanged. All 56 registered profile interventions, token-loss baselines, paired sequence sums, CE statistics and output energies matched bitwise against the original implementation. Original frozen sources and statistical definitions were preserved. New caches additionally bind routing_engine_sha256; earlier exact-compatible caches retain their original implementation identities. No new probe contexts were introduced.', '']
    lines += [f'The sole worker moved from contended Viper to the established Adder H100 host after an exact all-intervention check on registered batch0, then returned when the original GPU became nearly idle. Baseline routing/gates, all CE statistics and paired sequence sums matched bitwise; output energies also matched exactly. The original Viper profile is retained as a contention-sensitive estimate. Combined main-worker elapsed time is {accounting.get("combined_main_worker_elapsed_seconds",accounting["measure_wall_seconds"]):.1f}s, plus the original profile and host compatibility check. The transfer changed no contexts, groups, numerical protocol or statistical definitions.', '', f'One-batch profile estimated {accounting["profile"]["estimated_measurement_seconds"]:.1f}s for measurement and {accounting["profile"]["estimated_compact_cache_bytes"]/1e6:.1f} MB of compact caches. Actual main pass including table serialization: {accounting["measure_wall_seconds"]:.1f}s on {accounting["gpu"]}; analysis {accounting["analysis_wall_seconds"]:.1f}s; report/figures before sealing {accounting["report_wall_seconds_before_seal"]:.1f}s. Cache storage {accounting["new_cache_bytes"]/1e6:.1f} MB; all outputs before seal {accounting["storage_bytes_before_seal"]/1e6:.1f} MB. Profiling used one of the registered 32 batches and that cache was reused. No full hidden tensors or logits were saved.','',
      f'Final category: **{conclusion}**. No cross-seed expansion or later experiment follows automatically. This assay provides no specialization, lottery-ticket-mechanism, permanent-dispensability or pruning-algorithm claim.']
    (D/'findings.md').write_text('\n'.join(lines)+'\n')
    save(D/'completion.json',dict(measurement_complete=True,analysis_complete=True,report_complete=True,conclusion=conclusion,additional_seeds_launched=False,training_updates=0,unix=time.time()))
    save(D/'report_implementation.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    save(D/'output_seal.json',dict(files={str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.md','.png','.pt') and p.name not in ('output_seal.json','final_audit.json','progress.json','heartbeat.json','finisher_status.json') and not p.name.startswith('failure_')}))
    print('Fixed-budget report complete:',conclusion,flush=True)


def plots(ranks,decisions,cov,unc,functional):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=D/'figures';out.mkdir(exist_ok=True)
    weights=ranks[ranks.intervention_kind=='weight_union'];primary=weights[weights.subset=='whole'];u=unc[(unc.reference_tolerance==.0001)&(unc.subset=='whole')];cb=cov[cov.level=='block']
    fig,axes=plt.subplots(2,4,figsize=(16,6))
    for row,kind in enumerate(('sparse','unpruned')):
      for col,t in enumerate(s.STEPS):
        z=cb[(cb.trajectory==kind)&(cb.step==t)].pivot(index='expert',columns='block',values='accepted_tokens').reindex(index=s.EXPERTS,columns=range(1,9));ax=axes[row,col]
        ax.imshow(np.log1p(z.to_numpy()),aspect='auto',cmap='viridis',vmin=0,vmax=np.log1p(cb.accepted_tokens.max()))
        for i in range(4):
          for j in range(8):ax.text(j,i,str(int(z.iloc[i,j]))+('*' if z.iloc[i,j]<32 else ''),ha='center',va='center',fontsize=6,color='white')
        ax.set_title(f'{kind} step {t}');ax.set_yticks(range(4),s.EXPERTS);ax.set_xticks(range(8),range(1,9));ax.axvline(3.5,color='red');ax.set_xlabel('Block (* below 32)');ax.set_ylabel('Expert')
    fig.suptitle('Accepted-position coverage; counts shown, log color; red splits discovery/confirmation');fig.tight_layout();fig.savefig(out/'coverage.png',dpi=160);plt.close(fig)
    for kind in ('sparse','unpruned'):
      for t in s.STEPS:
        fig,axes=plt.subplots(4,4,figsize=(13,11))
        for row,e in enumerate(s.EXPERTS):
          for col,dose in enumerate((1,2,4,8)):
            z=primary[(primary.trajectory==kind)&(primary.step==t)&(primary.expert==e)&(primary.dose_groups==dose)].iloc[0];ids=json.loads(z.intervention_ids);a=json.loads(z.discovery_values);b=json.loads(z.confirmation_values);ax=axes[row,col]
            for iid,x,y in zip(ids,a,b):
                if x is None or y is None:continue
                ci=u[(u.trajectory==kind)&(u.step==t)&(u.expert==e)&(u.intervention==iid)].set_index('partition')
                lo,hi=ci.loc['discovery',['ci95_low','ci95_high']];bl,bh=ci.loc['confirmation',['ci95_low','ci95_high']]
                if pd.notna(lo) and pd.notna(hi):ax.hlines(y,lo,hi,color='tab:blue',alpha=.35)
                if pd.notna(bl) and pd.notna(bh):ax.vlines(x,bl,bh,color='tab:blue',alpha=.35)
                ax.scatter(x,y,s=17)
            low=min(ax.get_xlim()[0],ax.get_ylim()[0]);high=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([low,high],[low,high],':',color='grey');ax.set_title(f'e{e}, {dose} groups; rho={fmt(z.spearman)}',fontsize=8);ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0));ax.set_xlabel('Discovery',fontsize=8);ax.set_ylabel('Confirmation',fontsize=8)
        fig.suptitle(f'{kind}, step {t}: whole-probe damage at each fixed expert/dose\nPaired-batch 95% intervals; no pooling; singleton rank undefined');fig.tight_layout();fig.savefig(out/f'{kind}_{t}_discovery_confirmation.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,7))
    for row,kind in enumerate(('sparse','unpruned')):
      for col,t in enumerate(s.STEPS):
        z=decisions[(decisions.trajectory==kind)&(decisions.step==t)&(decisions.dose_groups.isin([1,2,4]))].sort_values(['expert','dose_groups']);ax=axes[row,col]
        for i,r in enumerate(z.itertuples()):
            color='grey' if r.D_blocks_below_minimum else {1:'tab:blue',2:'tab:orange',4:'tab:green'}[r.dose_groups]
            if pd.notna(r.selected_confirmation_ci95_low):ax.vlines(i,r.selected_confirmation_ci95_low,r.selected_confirmation_ci95_high,color=color)
            ax.scatter(i,r.selected_confirmation_contrast,color=color,s=15)
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(len(z)),[f'e{int(r.expert)}/{int(r.dose_groups)}' for r in z.itertuples()],rotation=65,fontsize=6);ax.set_title(f'{kind}, step {t}');ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    fig.suptitle('Discovery-selected high minus low on confirmation, paired-batch 95% intervals\nGray = conditional coverage deficient; selections never use confirmation');fig.tight_layout();fig.savefig(out/'selected_confirmation_contrasts.png',dpi=160);plt.close(fig)
    uall=unc[(unc.reference_tolerance==.0001)&(unc.partition=='confirmation')&(unc.intervention_kind=='weight_union')]
    fig,axes=plt.subplots(2,4,figsize=(16,7))
    for row,kind in enumerate(('sparse','unpruned')):
      for col,t in enumerate(s.STEPS):
        ax=axes[row,col]
        for dose in (1,2,4,8):
            z=uall[(uall.trajectory==kind)&(uall.step==t)&(uall.dose_groups==dose)].pivot(index=['expert','intervention'],columns='subset',values='mean_damage');ax.scatter(z.whole,z.baseline_accepted,label=f'{dose} groups',s=15)
        ax.set_title(f'{kind}, step {t}');ax.set_xlabel('Whole CE');ax.set_ylabel('Baseline-accepted CE');ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0));ax.legend(fontsize=6)
    fig.suptitle('Confirmation whole versus baseline-accepted damage; doses explicit, no pooled correlation');fig.tight_layout();fig.savefig(out/'whole_vs_conditional.png',dpi=150);plt.close(fig)
    f=functional[(functional.partition=='confirmation')&(functional.intervention_kind=='weight_union')]
    for dose in (1,2,4,8):
        fig,axes=plt.subplots(2,4,figsize=(16,7))
        for row,kind in enumerate(('sparse','unpruned')):
          for col,t in enumerate(s.STEPS):
            ax=axes[row,col]
            for e in s.EXPERTS:
                z=f[(f.trajectory==kind)&(f.step==t)&(f.expert==e)&(f.dose_groups==dose)].set_index('intervention');d=uall[(uall.trajectory==kind)&(uall.step==t)&(uall.expert==e)&(uall.subset=='whole')].set_index('intervention');z=z.join(d[['mean_damage']]);ax.scatter(z.pooled_relative_rms,z.mean_damage,label=f'e{e}',s=18)
            ax.set_title(f'{kind}, step {t}, {dose} groups');ax.set_xlabel('Pooled relative output RMS');ax.set_ylabel('Whole CE damage');ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0));ax.legend(fontsize=6)
        fig.suptitle('Confirmation functional distortion versus task damage; expert colors, no pooled fit');fig.tight_layout();fig.savefig(out/f'functional_vs_damage_dose{dose}.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    with worker():main()
