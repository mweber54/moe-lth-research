"""Artifact-backed figures and restrained interpretation; no training entry points."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[2]


def rho(x,y):
    x,y=np.asarray(x),np.asarray(y)
    return float(stats.spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else np.nan


def tensor_hash(state):
    h=hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode());h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def counts(mask):
    output=[]
    for up in sorted(k for k in mask if k.endswith('.fc1.weight')):
        down=up.replace('.fc1.weight','.fc2.weight')
        if mask[up].dtype!=torch.bool or mask[down].dtype!=torch.bool:
            raise ValueError('Nonboolean mechanistic control')
        l=int(up.split('.')[1]);e=int(up.split('.experts.')[1].split('.')[0])
        u=mask[up].double().mean(1).numpy();d=mask[down].double().mean(0).numpy()
        output.append(pd.DataFrame(dict(layer=l,expert=e,neuron=np.arange(len(u)),up=u,down=d,retention=(u+d)/2)))
    return pd.concat(output,ignore_index=True).sort_values(['layer','expert','neuron']).reset_index(drop=True)


def mechanistic_controls(out,runs):
    """No new interventions: original dense scores on historical altered supports."""
    source=ROOT/'results/imp_mask_information_v2/results.csv'
    table=pd.read_csv(source)
    result=[]
    for seed,g in table.groupby('seed'):
        native=g[g.condition=='IMP'].iloc[0]
        base_mask=torch.load(native.mask_path,map_location='cpu',weights_only=True)
        base=counts(base_mask)
        if tensor_hash(base_mask)!=native.mask_hash:
            raise RuntimeError('Historical IMP control mask identity mismatch')
        for name in ['reference_step_0','reference_step_2500','discovery_dense_final']:
            profiles=pd.read_parquet(out/f'e16_seed{seed}'/name/'profiles.parquet')
            pdf=profiles[profiles.domain=='pooled'].sort_values(['layer','expert','neuron']).reset_index(drop=True)
            for row in g.itertuples():
                mask=torch.load(row.mask_path,map_location='cpu',weights_only=True)
                if tensor_hash(mask)!=row.mask_hash:
                    raise RuntimeError('Historical randomized control mask identity mismatch')
                c=counts(mask)
                if not c[['layer','expert','neuron']].equals(base[['layer','expert','neuron']]):
                    raise RuntimeError('Control coordinate mismatch')
                p=pdf.ramp.to_numpy();ret=c.retention.to_numpy()
                weighted=float(p@ret/p.sum()) if p.sum() else np.nan
                baseweighted=float(p@base.retention.to_numpy()/p.sum()) if p.sum() else np.nan
                exact_degrees=bool(np.array_equal(c[['up','down']].to_numpy(),base[['up','down']].to_numpy()))
                if row.condition=='degree' and not exact_degrees:
                    raise RuntimeError('Saved degree control fails exact neuron retention preservation')
                result.append(dict(seed=int(seed),job=row.job,condition=row.condition,replicate=int(row.replicate),profile_name=name,
                                   mask_path=row.mask_path,mask_hash=row.mask_hash,source_table=str(source),curves_path=row.curves_path,
                                   final_ce=row.final_ce,historical_dense_ce=row.historical_dense_ce,
                                   recovery_gap=(row.final_ce-row.historical_dense_ce)/row.historical_dense_ce,
                                   control_minus_imp_ce=row.final_ce-native.final_ce,exact_neuron_retention_preserved=exact_degrees,
                                   maximum_neuron_retention_change=float(np.abs(ret-base.retention.to_numpy()).max()),
                                   ramp_weighted_retention=weighted,ramp_weighted_retention_minus_imp=weighted-baseweighted,
                                   interpretation='same original dense RAMP profile applied to control support; altered network function not profiled'))
    frame=pd.DataFrame(result)
    frame.to_csv(out/'historical_mask_controls.csv',index=False)
    summary=frame.groupby(['seed','condition','profile_name'],as_index=False).agg(control_minus_imp_ce=('control_minus_imp_ce','mean'),
          ramp_score_difference=('ramp_weighted_retention_minus_imp','mean'),nested_masks=('job','count'),
          all_neuron_retentions_preserved=('exact_neuron_retention_preserved','all'))
    summary.to_csv(out/'historical_mask_controls_seed_summary.csv',index=False)
    # Contextual joins use their own protocol outcomes; never pool with 2500-step H2.
    context=[]
    adaptation_hashes=json.loads((ROOT/'results/imp_adaptation_factorial_v1/input_hashes.json').read_text())
    source_pairs=pd.read_csv(out/'verified_recovery_pairs.csv')
    adaptation=pd.read_csv(ROOT/'results/imp_adaptation_factorial_v1/per_condition_endpoints.csv')
    for seed,g in adaptation.groupby('seed'):
        for condition,h in g.groupby('condition'):
            dense=h[h.sparsity==0]
            if len(dense)!=1:
                raise RuntimeError('Adaptation cohort lacks unique condition-matched dense reference')
            dense=dense.iloc[0]
            for row in h[h.sparsity>0].itertuples():
                base=runs[(runs.architecture=='E32')&(runs.seed==seed)&(runs.sparsity==row.sparsity)&(runs.profile_name=='reference_step_0')&(runs.domain=='pooled')]
                if len(base)!=1:
                    raise RuntimeError('Ambiguous contextual RAMP identity')
                pair=source_pairs[(source_pairs.architecture=='E32')&(source_pairs.seed==seed)&(source_pairs.sparsity==row.sparsity)].iloc[0]
                if adaptation_hashes.get(pair.mask_path)!=pair.mask_file_sha256 or adaptation_hashes.get(pair.initial_checkpoint)!=pair.initialization_id:
                    raise RuntimeError('Adaptation original mask/initialization provenance differs')
                context.append(dict(seed=int(seed),condition=condition,sparsity=row.sparsity,budget='5000; separate adaptation protocol',
                      source_table='results/imp_adaptation_factorial_v1/per_condition_endpoints.csv',
                      sparse_checkpoint=row.checkpoint,dense_checkpoint=dense.checkpoint,
                      sparse_ce=row.final_ce,dense_ce=dense.final_ce,recovery_gap=(row.final_ce-dense.final_ce)/dense.final_ce,
                      fixed_initial_ramp_weighted_retention=float(base.iloc[0].ramp_weighted_retention),
                      interpretation='fixed initial summary cannot encode changed trainable parameter groups; no new profiling of adaptation endpoints'))
    pd.DataFrame(context).to_csv(out/'adaptation_context_separate_protocol.csv',index=False)
    sign_source=ROOT/'results/imp_sign_recovery_stage1_v1/per_seed_gate.csv'
    sign=pd.read_csv(sign_source)
    sign['source_table']=str(sign_source)
    sign['protocol']='historical sign-only Stage1; 3 source seeds; fixed order/draw nested controls; separate from primary H2'
    sign['ramp_conditioned_sign_effect']='not identified: transformed states not profiled or matched by RAMP'
    sign.to_csv(out/'sign_only_context_separate_protocol.csv',index=False)



def make_figures(out,runs,summary,dec,stability):
    endpoint=runs[runs.domain=='pooled'].copy()
    endpoint['endpoint_group']=np.where(endpoint.recovery_gap<=0,'at_or_better_paired_dense','above_paired_dense')
    endpoint.groupby(['architecture','sparsity','profile_name','endpoint_group'],as_index=False).agg(
        independent_seeds=('seed','nunique'),mean_ramp_alignment=('ramp_within_expert_rho','mean'),
        mean_initial_magnitude_alignment=('initial_magnitude_within_expert_rho','mean'),
        mean_recovery_gap=('recovery_gap','mean')).to_csv(out/'endpoint_alignment_groups.csv',index=False)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    manifest=[]
    def save(fig,name,inputs,selection):
        fig.tight_layout()
        for ext in ('pdf','png'):
            fig.savefig(figures/f'{name}.{ext}',dpi=240,bbox_inches='tight')
        plt.close(fig)
        manifest.append(dict(figure=name,underlying_artifacts=[str(p) for p in inputs],selection=selection,
                             source_run_manifest=str(out/'verified_recovery_pairs.csv'),calibration_manifest=str(out/'calibration_manifest.json')))
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharey=True)
    inputs=[]
    for col,e in enumerate((8,16,32)):
        for row,s in enumerate((.8,.85)):
            ax=axes[row,col]
            for seed in (7,17,29,43,61):
                p=out/f'e{e}_seed{seed}/discovery_dense_final/neurons_s{s:.2f}.parquet';inputs.append(p)
                df=pd.read_parquet(p)
                sample=df.iloc[np.linspace(0,len(df)-1,min(1200,len(df)),dtype=int)]
                ax.scatter(np.log10(sample.ramp+1e-12),sample.retention,s=2,alpha=.13,rasterized=True)
            ax.set_title(f'E{e}, {s:.0%} sparse');ax.set_xlabel('log10 RAMP')
            if col==0:ax.set_ylabel('Combined neuron retention')
    save(fig,'figure1a_ramp_retention',inputs,'All 5 seeds; unpruned discovery ancestor; systematic neuron sample for display only')
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),sharey=True)
    d=dec[(dec.profile_name=='discovery_dense_final')&(dec.domain=='pooled')&(dec.feature=='ramp')]
    for ax,e in zip(axes,(8,16,32)):
        for s in (.8,.85):
            sub=d[(d.architecture==f'E{e}')&(d.sparsity==s)].groupby(['seed','decile']).retention.mean().reset_index()
            agg=sub.groupby('decile').retention.agg(['mean','std','count'])
            half=stats.t.ppf(.975,agg['count']-1)*agg['std']/np.sqrt(agg['count'])
            ax.errorbar(agg.index,agg['mean'],yerr=half,label=f'{s:.0%}',marker='o',markersize=3,capsize=2)
        ax.set_title(f'E{e}');ax.set_xlabel('Within-layer RAMP decile')
    axes[0].set_ylabel('Retention (seed mean, descriptive 95% CI)');axes[-1].legend()
    save(fig,'figure1b_retention_deciles',[out/'retention_deciles.parquet'],'Pooled; unpruned discovery ancestor; layer mean per seed, then equal seed average')
    fig,axes=plt.subplots(3,3,figsize=(11,7))
    inputs=[]
    for col,e in enumerate((8,16,32)):
        p=out/f'e{e}_seed7/discovery_dense_final/neurons_s0.80.parquet';inputs.append(p)
        df=pd.read_parquet(p).query('layer == 0')
        for row,feature in enumerate(('ramp','up_retention','down_retention')):
            z=[]
            for expert,g in df.groupby('expert'):
                g=g.sort_values(['ramp','neuron'])
                values=g[feature].to_numpy()
                if feature=='ramp':values=stats.rankdata(values)/len(values)
                z.append(values)
            image=axes[row,col].imshow(z,aspect='auto',origin='lower',vmin=0,vmax=1,interpolation='nearest',cmap='viridis')
            axes[row,col].set_title(f"E{e} {dict(ramp='RAMP rank',up_retention='Up retention',down_retention='Down retention')[feature]}");axes[row,col].set_xlabel('Neuron rank within expert')
            axes[row,col].set_ylabel('Expert')
    fig.subplots_adjust(left=.065,right=.89,bottom=.08,top=.95,hspace=.75,wspace=.42)
    color_axis=fig.add_axes([.925,.18,.015,.64])
    fig.colorbar(image,cax=color_axis,label='Rank / retention fraction')
    for ext in ('pdf','png'):fig.savefig(figures/f'figure2_neuron_structure.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    manifest.append(dict(figure='figure2_neuron_structure',underlying_artifacts=[str(p) for p in inputs],selection='Prespecified seed7, layer0, 80%; sorted within expert; ramp rank and raw retention',source_run_manifest=str(out/'verified_recovery_pairs.csv')))
    seed_results=pd.read_csv(out/'h1_seed_results.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),sharey=True)
    features=['ramp','activation_only','down_norm','routing_activation','initial_magnitude','checkpoint_magnitude']
    markers=['o','s','^']
    for ax,scope in zip(axes,('within_expert','within_layer')):
        for j,e in enumerate((8,16,32)):
            sub=seed_results[(seed_results.architecture==f'E{e}')&(seed_results.sparsity==.8)&(seed_results.profile_name=='discovery_dense_final')&(seed_results.domain=='pooled')&(seed_results.scope==scope)]
            for i,f in enumerate(features):
                values=sub[sub.feature==f].retention_rho.to_numpy();x=i+(j-1)*.22
                ax.scatter(np.full(len(values),x),values,s=16,alpha=.7,marker=markers[j],label=f'E{e}' if i==0 else None)
        ax.axhline(0,color='black',lw=.7);ax.set_title(scope.replace('_',' '));ax.set_xticks(range(len(features)),features,rotation=45,ha='right')
    axes[0].set_ylabel('Mean group Spearman rho per independent seed');axes[-1].legend()
    save(fig,'figure3_alignment_seeds',[out/'h1_seed_results.csv'],'All 5 seed points per architecture; 80%; pooled; discovery ancestor; no neuron-based inferential intervals')
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharey='row')
    for col,e in enumerate((8,16,32)):
        for row,s in enumerate((.8,.85)):
            ax=axes[row,col]
            sub=runs[(runs.architecture==f'E{e}')&(runs.sparsity==s)&(runs.domain=='pooled')&(runs.profile_name=='reference_step_0')]
            ax.scatter(sub.ramp_weighted_retention,sub.recovery_gap,s=25)
            for r in sub.itertuples():ax.annotate(str(r.seed),(r.ramp_weighted_retention,r.recovery_gap),xytext=(3,2),textcoords='offset points',fontsize=7)
            ax.axhline(0,color='black',lw=.7);ax.set_title(f'E{e}, {s:.0%}');ax.set_xlabel('Initial RAMP-weighted retention');ax.xaxis.set_major_locator(MaxNLocator(4))
            if col==0:ax.set_ylabel('Relative recovery gap')
    save(fig,'figure4_sparse_recovery',[out/'mask_run_results.csv'],'Initialization only; all independent seeds; separate architecture and sparsity; no fitted trend')
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    seeds=pd.read_csv(out/'h1_seed_results.csv')
    for ax,e in zip(axes,(8,16,32)):
        sub=seeds[(seeds.architecture==f'E{e}')&(seeds.sparsity==.8)&(seeds.domain=='pooled')&(seeds.feature=='ramp')&(seeds.scope=='within_expert')&(seeds.profile_name.str.startswith('reference_step_'))]
        for seed,g in sub.groupby('seed'):
            g=g.sort_values('profile_step');ax.plot(g.profile_step,g.retention_rho,marker='o',ms=3,label=str(seed),alpha=.75)
        ax.set_xlabel('Available dense reference checkpoint step');ax.set_title(f'E{e}');ax.axhline(0,c='black',lw=.7)
    axes[0].set_ylabel('RAMP vs eventual IMP80 retention (seed means)');axes[-1].legend(title='Seed',fontsize=7)
    save(fig,'figure5_importance_evolution',[out/'h1_seed_results.csv',out/'checkpoint_rank_stability.csv'],'Actual steps 0,250,1250,2500; no interpolation of missing states; eventual masks used only as target')
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,e in zip(axes,(8,16,32)):
        sub=stability[stability.group.str.startswith(f'e{e}_') & stability.profile_name.str.startswith('reference_step_')].copy()
        sub['step']=sub.profile_name.str.rsplit('_',n=1).str[-1].astype(int)
        for group,g in sub.groupby('group'):
            q=g.groupby('step').ramp_rank_stability.mean().sort_index()
            ax.plot(q.index,q.values,marker='o',ms=3,alpha=.75,label=group.split('seed')[-1])
        ax.set_xlabel('Available dense reference checkpoint step');ax.set_title(f'E{e}')
    axes[0].set_ylabel('Within-layer RAMP rank correlation to final');axes[-1].legend(title='Seed',fontsize=7)
    save(fig,'figure5b_rank_stability',[out/'checkpoint_rank_stability.csv'],'Within-layer rank stability averaged within each independent seed; final reference step2500')
    controls=pd.read_csv(out/'historical_mask_controls_seed_summary.csv')
    fig,ax=plt.subplots(figsize=(5,3.5))
    for i,condition in enumerate(('matrix','degree')):
        df=controls[(controls.condition==condition)&(controls.profile_name=='reference_step_0')]
        ax.scatter(np.full(len(df),i),df.control_minus_imp_ce,s=25)
        for r in df.itertuples():ax.annotate(str(r.seed),(i,r.control_minus_imp_ce),xytext=(5,0),textcoords='offset points')
    ax.axhline(0,c='black',lw=.7);ax.set_xticks([0,1],['Matrix-random','Degree-preserving']);ax.set_ylabel('Control CE − IMP CE (nested-mask mean per seed)')
    save(fig,'supplement_historical_mask_controls',[out/'historical_mask_controls.csv',out/'historical_mask_controls_seed_summary.csv'],'Historical E16/80 controls, 5 draws per seed; not new H3 interventions')
    (figures/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))


def format_table(frame,cols):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.4f}' if np.isfinite(x) else 'undefined'
        return str(x)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(r[c]) for c in cols)+' |' for r in frame.to_dict('records'))


def make_report(out,runs,summary,associations):
    relative=out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    h1=summary[(summary.profile_name=='discovery_dense_final')&(summary.domain=='pooled')&(summary.feature=='ramp')&(summary.metric=='retention_rho')].sort_values(['architecture','sparsity','scope'])
    base=summary[(summary.profile_name=='discovery_dense_final')&(summary.domain=='pooled')&(summary.scope=='within_expert')&(summary.sparsity==.8)&(summary.metric=='retention_rho')].sort_values(['architecture','feature'])
    h2=associations[(associations.profile_name=='reference_step_0')&(associations.domain=='pooled')&(associations.feature=='ramp_weighted_retention')]
    gaps=runs[(runs.profile_name=='reference_step_0')&(runs.domain=='pooled')].groupby(['architecture','sparsity']).agg(independent_seeds=('seed','nunique'),mean_relative_gap=('recovery_gap','mean'),min_gap=('recovery_gap','min'),max_gap=('recovery_gap','max')).reset_index()
    controls=pd.read_csv(out/'historical_mask_controls_seed_summary.csv')
    control=controls[(controls.condition=='degree')&(controls.profile_name=='reference_step_0')]
    cal=json.loads((out/'calibration_manifest.json').read_text())
    final_ramp=h1[h1.scope=='within_expert']['mean']
    initial_ramp=summary[(summary.profile_name=='reference_step_0')&(summary.domain=='pooled')&(summary.feature=='ramp')&(summary.metric=='retention_rho')&(summary.scope=='within_expert')]
    initial_magnitude=base[base.feature=='initial_magnitude']['mean']
    timing=summary[(summary.profile_name.str.startswith('reference_step_'))&(summary.domain=='pooled')&(summary.feature=='ramp')&(summary.metric=='retention_rho')&(summary.scope=='within_expert')&(summary.sparsity==.8)].sort_values(['architecture','profile_step'])
    text=f'''# RAMP–IMP correspondence and sparse recoverability

Research date: 2026-11-13. Experiment `ramp_imp_correspondence_v1`. Status: retrospective CPU analysis complete; no recovery training launched. Pre-implementation audit: [ramp_imp_audit.md](ramp_imp_audit.md).

## Measured findings

Across architectures/sparsities, discovery-final within-expert RAMP–retention Spearman means are {final_ramp.min():.3f}–{final_ramp.max():.3f}: weak alignment. Initialization RAMP means are {initial_ramp['mean'].min():.3f}–{initial_ramp['mean'].max():.3f}. Initial magnitude alone reaches {initial_magnitude.min():.3f}–{initial_magnitude.max():.3f} at IMP80, exceeding the discovery-final RAMP alignment. Recovery prediction is unestablished with five independent seeds per stratum; no superiority over simple baselines is demonstrated. Degree-preserving controls show a concrete limitation of neuron-density summaries despite matched original-state RAMP scores.

The final-state pattern is closer to Outcome C than a strong-support Outcome A: weak RAMP alignment coexists with some IMP recoveries reaching paired dense validation CE. This is not proof that RAMP has zero recovery information, nor confirmation of Outcome B; its early scores and the limited calibration/sample size need separate interpretation.

## Scope and interpretation

Analyzed 30 historical IMP recoveries: E8/E16/E32, seeds 7/17/29/43/61, 80%/85% sparsity. The two masks within a seed are dependent. Five seeds per architecture/sparsity are the replication units; architectures reuse seed labels and are not pooled. All results, including failures to reach dense loss, remain included. Loss references are exact protocol-matched dense recoveries, not the dense checkpoint chosen for profiling. Recovery gap is (sparse CE − paired dense CE) / paired dense CE.

RAMP–retention correlations are descriptive associations. The H2 sample is too small for a defensible multivariable predictor; no fitted predictive model, feature selection, or neuron-level recovery regression is used. Leave-one-seed-out correlation ranges measure sensitivity, not generalization accuracy. The report does not establish that RAMP predicts recovery beyond simpler baselines.

A sharper limitation is directly testable: historical degree-preserving randomized supports have exactly the same up/down neuron retention as IMP, hence identical RAMP-weighted retention for **any fixed original-state RAMP profile**. Yet their final recovery losses differ. This means these neuron-retention summaries cannot explain all connection-level recoverability. It does not imply that profiling the altered networks would yield identical RAMP.

## Architecture and faithful score

The existing ExpertFFN is conventional: `fc2(dropout(GELU(fc1(x))))`, with bias terms. No neuron gate projection exists; the MoE router is separate. Adapt [Zhou et al., Eq. (3)](https://arxiv.org/html/2611.13975v1#S3.SS2) by taking actual GELU(fc1(x)) as intermediate activation in eval mode. RAMP = sqrt(sum (accepted routing coefficient × activation)^2) × norm(fc2 column). Biases are unpruned and output bias is outside the individual neuron score. No shared experts exist. Native top-k routing uses selected raw softmax coefficients without renormalization, including after capacity truncation. Rejected edges contribute zero. Secondary routes are included. Actual native traces and layer inputs drive a bounded eval replay, validated against actual fc2-input hooks and a direct calculation. Primary RAMP is unnormalized; per-dispatch and per-routing-energy scores are separately named baselines.

FP32 CPU native activations and float64 multiplication/reductions are used. Historical recovery used CUDA FP16; fresh CPU RAMP is a measurement on the saved state, not reproduction of historical FP16 dispatch bit for bit. Every profile records its checkpoint hash, config, calibration hash, source hashes, and software version. No historical mask or result is changed. Unsupported expert architectures fail explicitly; the generic accumulator accepts gated activations but only this repository's conventional FFN adapter is validated for full-model profiling.

## Calibration

Fixed selection seed {cal['selection_seed']}: {cal['sequences']} sequences × {cal['sequence_length']} tokens = {cal['tokens']} input tokens, equal exact-source TinyStories/WikiText quotas, batch size {cal['batch_size']}. Sampling is from canonical training-split blocks using `load_token_splits` and `TokenBlockDataset`; it is not an independent held-out calibration corpus. Every chosen 128-token sequence is rejected if it appears anywhere in the entire canonical validation byte stream. This verifies exact sequence disjointness, not absence of shorter phrases or document-level source overlap. Unique source-substring membership supplies domain labels; ambiguous/cross-boundary blocks are excluded. Exact token IDs, source paths/byte offsets, block indices, tokenizer implementation hash, file hashes, and token hash are in `{relative}/calibration_manifest.json`.

All checkpoints use identical token sequences and batch partitions. Source subsets accumulate contributions under the same pooled native capacity competition. Subset results are not equivalent to standalone source-only dispatch. This small {cal['tokens']}-token calibration can leave rarely routed experts unexposed; undefined within-expert correlations are omitted with valid-group and unexposed-expert counts retained. There is no domain-specialization claim or outcome-based calibration tuning.

## H1: support correspondence

Unpruned discovery round_00 is the directly ancestral dense state, separately named `discovery_dense_final`. It is compared with the eventual nested IMP masks without claiming that its dense magnitude ranking alone generated those later-round masks. Each group uses exact fc1 rows and fc2 columns. Pearson correlation of average ranks gives Spearman correlation with ties. Constant score/retention groups return undefined, not zero. Group means are formed per seed, then descriptive 95% Student-t intervals across five seeds. These intervals are not multiplicity-adjusted confirmatory tests.

{format_table(h1,['architecture','sparsity','scope','independent_seeds','mean','ci95_low','ci95_high'])}

Within-expert baseline comparison at IMP80 on the same discovery dense states:

{format_table(base,['architecture','feature','independent_seeds','mean','ci95_low','ci95_high'])}

Full up/down correlations, within-layer comparisons, paired baseline differences, retention deciles, top-decile enrichment, AUC and average precision are in `h1_seed_summary.csv`, `h1_baseline_differences.csv`, and `h1_groups.parquet`. High-retention labels use retention >= within-layer 90th percentile, retaining tied labels; prevalence is saved. Average-rank deciles leave tied sets together and may leave empty bins. Random rankings use a fixed independent generator; utilization is constant within expert and cannot produce a within-expert rank correlation. Exposure-adjusted baselines distinguish frequently selected experts from per-dispatch contribution; adjustment is not substituted for Eq. (3).

## H2: sparse recovery

{format_table(gaps,['architecture','sparsity','independent_seeds','mean_relative_gap','min_gap','max_gap'])}

Prespecified initial-state RAMP-weighted retention versus relative recovery gap, separately by architecture and sparsity:

{format_table(h2,['architecture','sparsity','independent_seeds','spearman_rho','loo_rho_min','loo_rho_max'])}

`h2_descriptive_associations.csv` contains the same information-budget univariate summaries for random, utilization, activation, down-norm, routing-weighted activation, and initial/current magnitude baselines. Positive correlation means larger score associates with worse recovery (positive gap). There are only five independent seeds per stratum and many exploratory summaries; a high sample correlation is not evidence of a reliable predictor or incremental value. No across-neuron train/test split is used.

Mask summaries are descriptive: importance-weighted coordinate density, retained-weight-weighted mean importance, importance share in neurons with >=50% retention, and mean within-expert/layer alignment. Individual contribution magnitudes can add in squared norm, while neuron output vectors can cancel or compensate; these summaries are not fractions of preserved function. Strong-retention shares can be zero or nearly constant at these sparsities. All H2 scores are conditioned on historical masks, so even initialization-state summaries do not constitute a deployable early mask-selection predictor.

## Timing and mechanistic integration

Actual reference trajectory steps 0,250,1250,2500 and unpruned discovery round_00 final are profiled. Rewind is step 0 in the eligible recovery cohort. The reference trajectory and IMP ancestor are distinct trajectories even when both are at step 2500. No unavailable state is fabricated. Measured early-to-final support alignment at IMP80:

{format_table(timing,['architecture','profile_step','independent_seeds','mean','ci95_low','ci95_high'])}

Reference early RAMP uses only weights and activations at that state; current-magnitude features never use later weights. Initial magnitude is available at all timings. Later checkpoints are retrospective explanation, not evidence for pruning during training. `checkpoint_rank_stability.csv` reports rank correlation to the actual final reference state; Figure 5 compares early RAMP with eventual IMP support. Earliest sampled step is not the true onset of predictability.

Historical E16 IMP80 degree controls, averaged across five nested randomized masks per seed:

{format_table(control,['seed','nested_masks','all_neuron_retentions_preserved','ramp_score_difference','control_minus_imp_ce'])}

All saved control masks are reloaded, tensor hashes checked, and exact up/down neuron retentions verified. `historical_mask_controls.csv` preserves every mask and curve path; the separate seed summary prevents treating random draws as independent training seeds. The original-profile neuron-retention score cannot distinguish these degree controls. Matrix-random controls are also retained. Their recovery protocol is kept separate from primary 2500-step endpoints; a historical dense value is labeled as such. No causal preservation of function is inferred from matched scores.

The existing E32 adaptation factorial is joined separately in `adaptation_context_separate_protocol.csv`, using condition-matched dense endpoints at its own 5000-step budget. Fixed initial RAMP summaries are identical across changes to trainable parameter groups, while later recovery can differ. Thus an initial functional contribution metric alone does not encode the allowed optimization/adaptation process. Frozen router parameters do not freeze token exposure as upstream states change. New router/exposure perturbations were not introduced.

The completed initial-value-placement study (`results/imp_initial_value_placement_v1/findings.md`, `source_level_summary.csv`, `endpoint_losses.csv`) reports that shuffling surviving values harms original IMP more than sampled random supports across five sources. That intervention combines sign, magnitude, and placement. A distinct completed sign-only Stage1 cohort (`results/imp_sign_recovery_stage1_v1/completion.json`, 54 runs, 3 source seeds) reports IMP-minus-random sign-shuffling interactions at step2000 of 0.1744, 0.1785 and 0.1947 CE for seeds7/17/29. Its fixed-order/draw uncertainty is conditional on sources; `sign_only_context_separate_protocol.csv` preserves those historical summaries without pooling protocols. This experiment has not separately profiled those transformed states or matched neurons by RAMP to identify sign effects; no claim that RAMP explains or removes the historical sign-specific effect is justified. `endpoint_alignment_groups.csv` additionally separates at-or-better-dense endpoints (gap<=0) from positive-gap endpoints without calling either group a certified winning ticket; E8/85 and other strata can lack one group, and the small groups preclude a reliable group-effect estimate. Primary comparisons use continuous relative gaps, with no outcome-selected seeds or threshold tuning. No individual final-mask sign statistic is treated as evidence of causal sign importance.

## Artifacts, figures and reproducibility

Primary artifact paths and hashes are in `verified_recovery_pairs.csv`; per-checkpoint manifests identify exact states. `e*_seed*/*/neurons_s*.parquet` is the consolidated partitioned pooled neuron/checkpoint/mask dataset (one row per pairing); profile files contain source-subset RAMP/exposure and pairing files contain initial/current surviving magnitude/sign statistics. Stable seed, initialization, mask, discovery, rewind, recovery, and checkpoint identifiers are included. `mask_run_results.csv` has one row per mask/checkpoint/domain. `h1_groups.parquet` and `retention_deciles.parquet` preserve group and decile calculations. The calibration sample is registered before any outcomes enter analysis. Partial reruns require matching registration, input hashes, and completed artifact hashes. `engineering_amendment.json` preserves the original profiling source/registration and records a correction confined to the unexecuted H3 apportionment helper; the profiling implementation, calibration and H1/H2 choices are unchanged.

Figures 1a/1b: scatter and retention-decile curves; Figure 2: seed7/layer0 RAMP-rank and up/down structure; Figure 3: independent-seed alignment points against baselines; Figure 4: mask-level initial score versus relative recovery gap; Figures 5/5b: actual dense-checkpoint timing and rank stability. PDF vector exports and 240-dpi PNGs are under `{relative}/figures`. Exact source runs and underlying data paths are listed per figure in `figures/figure_manifest.json`. Supplement: existing mask controls, not H3. Figure 6 is deliberately absent because controlled new recoveries were not executed.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/run.py --output {relative}
.venv/bin/python -m pytest tests/test_ramp_imp_correspondence.py tests/test_ramp_imp_pipeline.py -q
# Final sparse-weight and canonical-stream validation supplement:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/supplement.py --output {relative}
# Validate final bundle hashes/counts:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/validate_results.py --output {relative}
# Rebuild summaries and figures from validated saved profiles:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/run.py --output {relative} --summarize-only
```

`recovered_neuron_statistics/*.parquet` supplies separately timed final sparse surviving-weight statistics; join by architecture/seed/sparsity/mask/layer/expert/neuron. These measurements are excluded from initial/early features. `reconstruction_validation.json` verifies canonical train/validation stream reconstruction and profile payload steps/configs; `supplement_manifest.json` records all supplemental artifact hashes.

The analysis uses pandas, installed in the repository environment through existing dataset dependencies; no packages or datasets were downloaded. Inference uses current source including pre-existing local architecture edits; source hashes preserve that version. Unit/integration test results are recorded separately in `test_validation.txt`. Calibration, masks, checkpoint payloads, historical sparse endpoints, curve values and data identities have explicit validation checks. A missing/mismatched artifact halts rather than substituting a new model. `completion.json` records completion only after all summary/control/plot stages succeed.

## Validation and remaining scope

The completed run passes 33 unit/integration/regression tests. `validation_summary.json` verifies all75 profile bundles, 11,468,800 pooled neuron/checkpoint/mask rows, 30 final sparse supports, all15 reconstructed canonical train/validation streams, checkpoint payload/config identities, and original dense reference hashes. All25 historical degree-control masks preserve neuron retention exactly. No unresolved numerical or provenance check failed.

Intermediate discovery masks01–07 are inventoried and supported by the generic mapping utility but are outside this report’s primary 80%/85% final-recovery analysis. Perturbed sign/router endpoint profiles are not claimed; their historical findings are contextual only.

## Decision and limitations

Interpret H1 using the displayed effect sizes and baseline comparisons, without treating correlation as causal importance. H2 remains descriptive; five seeds cannot demonstrate robust predictive value or superiority over magnitude/utilization baselines. The degree-control evidence directly rules out a complete account of recoverability based solely on original-profile importance and neuron connection density. The requested A/B/C outcomes therefore need not be exhaustive: support alignment and failure to capture connection topology can coexist. A strong positive H1 with inconclusive H2 is not yet a confirmed Outcome B (absence of predictive value).

Limitations include one conventional FFN family varied by expert count, one mixed dataset and protocol, small in-training calibration, capacity/batch dependence, CPU/FP16 numerical differences, limited seeds, dependence of nested masks, source seeds varying both initialization and training order, no independently repeated discoveries per seed, no untouched test or equivalence criterion, and no intervention-specific RAMP measurement. No new recovery experiment, causal claim, domain-specialization claim, or winning-ticket certification follows automatically.

A narrowly controlled H3 is scientifically reasonable to distinguish ranking value from connection topology, but the current associations do not establish expected benefit. The prepared design in `scripts/ramp_imp_correspondence_v1/H3_design.md` preserves exact expert-local unstructured budgets and all recovery controls. It is not launched. No conclusion should force the observations to support routing-conditioned optimization; the measured limitations indicate which connection-level and adaptation information RAMP alone lacks.
'''
    (ROOT/'reports/ramp_imp_correspondence.md').write_text(text)
