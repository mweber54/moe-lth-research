"""Fixed amplitude/content contrasts; no fitted scales or outcome-selected conditions."""
import math,statistics,tarfile
from pathlib import Path
from .engine import c,ROOT,MODES,VARIANTS,TIMES
OUT=ROOT/'results/imp_first_moe_amplitude_v1'
METRICS=('B_original','B_scale_start','B_content_end','B_content_start','B_scale_end','I')
LIMITS='''The first MoE layer and this follow-up question were selected after previous diagnostics. The new amplitude intervention is sealed before its outcomes; this is not project-level preregistration. Calibration uses an already sealed four-complete-batch training probe previously used in training and diagnostics, not held-out generalization data. Validation was repeatedly examined and is not an independent test set. Source seed (n=5) is the independent replication unit; tokens, batches, recipient contexts, modes and variants are nested measurements. One training-only scalar per seed/model type changes only global contribution amplitude. Residual differences can include directions, token-specific amplitude allocation, zero support, assignments and gates; they are not an isolated direction effect. Global norms/cosines describe interventions, not why they help. Recipient context includes its own residual and subsequent computation, not downstream weights alone. Native/replay differences include downstream routing responses and do not isolate a routing mechanism. Sparse-minus-unpruned effects compare each trajectory's own intervention and are not equal absolute-amplitude perturbations. No equivalence claim, percentage of recovery explained, expert weight learning, training counterfactual, standalone sparse recoverability, online pruning criterion or generalization beyond this architecture/mask family/window is established. No follow-up is launched.'''
PROTOCOL='''# First-MoE global amplitude control\n\nAll five seeds (7,17,29,43,61), abrupt IMP80 and matched unpruned shared-only/router_experts_frozen checkpoints at steps 1000 and 1500, are exactly the states verified in imp_first_moe_interface_v1. Seed 17 is the predetermined engineering pilot and is always retained.\n\nAt block 0, u=input+attention(attention_norm(input)); z=moe_norm(u); c=MoE(z); h=u+c. Replace only complete gated c, including activation, both expert biases, top-1 full-softmax confidence and whole-batch capacity acceptance. Donor c0/c1 comes from each donor's own unmodified native forward at identical complete-batch token/target coordinates. Keep recipient u. No donor residual, second gate, alignment, rotation, per-token/expert normalization or parameter change is used.\n\nCalibration reuses the sealed training probe (four complete native training batches, selected by lowest SHA256 index ranks from first 2500 verified training batches). Batch indices, actual bytes/hashes, probe identity and previous usage are audited before scale computation. Sum c squared in FP64 over all contextual elements including rejected-position zeros. a0=sqrt(SS0/n), a1=sqrt(SS1/n), r=a1/a0. Exactly one r per seed/model type, independent of any validation loss. Zero/nonfinite/FP32-unusable amplitudes produce explicit unavailable scaled comparisons, not epsilons or zero effects. Seal all ten scales and source/config/probe hashes before new validation interventions.\n\nFour variants are A=START_NATIVE=c0; B=START_SCALED_TO_END=r*c0; C=END_SCALED_TO_START=c1/r; D=END_NATIVE=c1. Both recipient contexts and modes are retained (160 cells). NATIVE_DOWNSTREAM is primary. RECIPIENT_BASELINE_REPLAY fixes only downstream layers 1–3 to the exact recipient baseline IDs, selected gates and accepted/drop decisions, evaluating experts on current hybrid inputs. The donor's layer-0 routing is contained in c.\n\nThe canonical full validation sequence, boundaries and target weighting are reverified against source artifacts. FP32, TF32 disabled, eval/inference. Task CE=sum NLL/actual targets; auxiliary loss excluded. A/D reuse prior per-batch interface NLL only after exact checkpoint, code, config, mask, batch, target and mode checks. Seed17 pilot freshly checks all A/D cells over the full population; every other pair checks representative complete-batch A/D cells and freshly computes native donor baselines. Reuse is provenance, not new replication. B/C are new evaluations. No validation amplitude correction or scale refit is allowed.\n\nContrasts per seed/model/recipient/mode: B_original=A-D; B_scale_start=A-B; B_content_end=B-D; B_content_start=A-C; B_scale_end=C-D; I=B_scale_end-B_scale_start. Verify B_original=B_scale_start+B_content_end=B_content_start+B_scale_end independently from exported absolute losses and paired batch NLL. These are arithmetic activation-intervention decompositions, not causal percentages. Report same-seed sparse-minus-unpruned and native-minus-replay contrasts. Individual seeds precede descriptive mean, median, min, max and sample SD. Missing values stay unavailable/null, never zero.\n\nCalibration scaled RMS and recipient perturbation norms can be derived from FP64 native moments (SS and cross-dot) without storing duplicated activations; these are labeled analytic moments. Actual FP32 scaling linearity/support are checked in synthetic tests and every validation batch. Validation RMS/mismatch comes from actually scaled tensors and is never used to revise r. Exact per-position cosine and norm-ratio distributions store only small one-value-per-position temporary vectors, excluding zero vectors explicitly; no full activation/logit arrays are persisted.\n\nUse existing GPU admission, exclusive shared-home lock and unique attempt directories. Atomically cache complete seed/model calibration and validation results with source/spec/calibration/checkpoint/mask/data/variant/mode identities. Dispatch depends only on numerical/provenance/resource feasibility. Preserve all failed attempts and original studies.\n\n'''+LIMITS

def difference(x,y):return x-y if x is not None and y is not None else None
def contrasts(losses):
    A,B,C,D=[losses.get(name) for name in VARIANTS]
    original=difference(A,D);start=difference(A,B);ce=difference(B,D);cs=difference(A,C);end=difference(C,D)
    return dict(B_original=original,B_scale_start=start,B_content_end=ce,B_content_start=cs,B_scale_end=end,I=difference(end,start))

def summarize(rows,groups,fields,kind):
    out=[]
    for key in sorted(set(tuple(str(r.get(g,'')) for g in groups) for r in rows)):
        selected=[r for r in rows if tuple(str(r.get(g,'')) for g in groups)==key]
        c._need(len(selected)==5 and len({r['seed'] for r in selected})==5,'Nested measurements incorrectly treated as replications')
        for metric in fields:
            vals=[r[metric] for r in selected if r.get(metric) is not None]
            out.append(dict(kind=kind,**dict(zip(groups,key)),metric=metric,n_available=len(vals),n_source_seeds=5,mean=statistics.mean(vals) if vals else None,median=statistics.median(vals) if vals else None,minimum=min(vals) if vals else None,maximum=max(vals) if vals else None,sample_sd=statistics.stdev(vals) if len(vals)>1 else None))
    return out

def pair_summary(result):
    rows=[r for r in result['measurements'] if r['kind']=='contribution_pair'];keys=('positions','elements','start_sumsq','end_sumsq','displacement_sumsq','dot_sum','cosine_sum','cosine_valid_positions','both_zero_positions','start_only_zero_positions','end_only_zero_positions','zero_cosine_exclusions','assignment_agree_count','acceptance_agree_count','start_dropped','end_dropped','gate_absolute_difference_sum','norm_ratio_valid_positions','norm_ratio_zero_exclusions')
    totals={k:math.fsum(r[k] for r in rows) if k.endswith('sumsq') or k in ('dot_sum','cosine_sum','gate_absolute_difference_sum') else sum(r[k] for r in rows) for k in keys};a,b,n=totals['start_sumsq'],totals['end_sumsq'],totals['elements']
    totals.update(global_cosine=totals['dot_sum']/math.sqrt(a*b) if a and b else None,a0=math.sqrt(a/n) if n else None,a1=math.sqrt(b/n) if n else None,relative_l2=math.sqrt(totals['displacement_sumsq']/a) if a else None,cosine_distribution=result['distribution']['cosine'],norm_ratio_distribution=result['distribution']['position_norm_ratio'],norm_ratio_population='both contributions nonzero; all zero-vector exclusions counted')
    return totals

def pooled_measurements(result,scale):
    ident=result['identity'];population='calibration' if result['calibration_batch_sums'] else 'validation';pair=pair_summary(result)
    base=dict(seed=ident['seed'],model_type=ident['model_type'],population=population)
    nested={k:v for k,v in pair.items() if not isinstance(v,dict)}
    for prefix,key in [('cosine','cosine_distribution'),('position_norm_ratio','norm_ratio_distribution')]:
        nested.update({prefix+'_'+k:v for k,v in pair[key].items()})
    output=[dict(**base,kind='contribution_pair',**nested)]
    if population=='calibration':
        for name in VARIANTS:
            donor='start' if name.startswith('START') else 'end';ss=pair['start_sumsq'] if donor=='start' else pair['end_sumsq']
            factor=1. if name in (VARIANTS[0],VARIANTS[3]) else scale if name==VARIANTS[1] else 1/scale if scale else None
            output.append(dict(**base,kind='variant_amplitude',variant=name,rms=math.sqrt(ss/pair['elements'])*factor if factor is not None else None,method='analytic FP64 native sufficient moments',status='AVAILABLE' if factor is not None else 'UNAVAILABLE'))
        moments=[r for r in result['measurements'] if r['kind']=='calibration_recipient_moments']
        for recipient in TIMES:
            for name in VARIANTS:
                donor='start' if name.startswith('START') else 'end';rs=[r for r in moments if r['recipient']==recipient and r['donor']==donor]
                factor=1. if name in (VARIANTS[0],VARIANTS[3]) else scale if name==VARIANTS[1] else 1/scale if scale else None
                n=sum(r['elements'] for r in rs);u=math.fsum(r['u_sumsq'] for r in rs);ds=math.fsum(r['donor_sumsq'] for r in rs);ns=math.fsum(r['native_c_sumsq'] for r in rs);dot=math.fsum(r['donor_native_dot'] for r in rs)
                perturb=0. if donor==recipient and factor==1. else math.fsum([factor*factor*ds,ns,-2*factor*dot]) if factor is not None else None
                c._need(perturb is None or perturb>=0,'Invalid analytic squared displacement')
                output.append(dict(**base,kind='recipient_variant',recipient=recipient,mode='not_applicable_calibration',variant=name,contribution_rms=math.sqrt(ds/n)*factor if factor is not None else None,recipient_residual_rms=math.sqrt(u/n),contribution_to_residual_rms_ratio=math.sqrt(ds/u)*factor if factor is not None and u else None,perturbation_rms=math.sqrt(perturb/n) if perturb is not None else None,method='analytic FP64 native moments, not additional interventions',status='AVAILABLE' if factor is not None else 'UNAVAILABLE'))
    else:
        for name in VARIANTS:
            rs=[r for r in result['measurements'] if r['kind']=='variant_amplitude' and r['variant']==name]
            output.append(dict(**base,kind='variant_amplitude',variant=name,rms=math.sqrt(math.fsum(r['sumsq'] for r in rs)/sum(r['elements'] for r in rs)) if rs else None,method='actually scaled FP32 validation tensors',status='AVAILABLE' if rs else 'UNAVAILABLE'))
        for recipient in TIMES:
            for mode in MODES:
                for name in VARIANTS:
                    rs=[r for r in result['measurements'] if (r['kind'],r.get('recipient'),r.get('mode'),r.get('variant'))==('recipient_variant',recipient,mode,name)]
                    n=sum(r['elements'] for r in rs);ss=lambda k:math.fsum(r[k] for r in rs)
                    output.append(dict(**base,kind='recipient_variant',recipient=recipient,mode=mode,variant=name,contribution_rms=math.sqrt(ss('inserted_contribution_sumsq')/n) if n else None,recipient_residual_rms=math.sqrt(ss('recipient_residual_sumsq')/n) if n else None,contribution_to_residual_rms_ratio=math.sqrt(ss('inserted_contribution_sumsq')/ss('recipient_residual_sumsq')) if n and ss('recipient_residual_sumsq') else None,perturbation_rms=math.sqrt(ss('from_recipient_native_displacement_sumsq')/n) if n else None,status='AVAILABLE' if n else 'UNAVAILABLE',method='actually inserted validation tensors'))
    return output

def write_outputs(calibrations,readouts,scales):
    absolute=[];batches=[];seed_rows=[];paired=[];stats=[];amplitude=[]
    sm={(r['seed'],r['model_type']):r for r in scales}
    for result in calibrations+readouts:
        scale=sm[result['identity']['seed'],result['identity']['model_type']]['scale'];stats+=pooled_measurements(result,scale)
    for result in readouts:
        ident=result['identity'];seed,m=ident['seed'],ident['model_type'];rows=result['rows'];batches+=rows
        for r in TIMES:
            for p in MODES:
                losses={}
                for name in VARIANTS:
                    rs=[z for z in rows if (z['recipient'],z['mode'],z['variant'])==(r,p,name)];count=sum(z['target_tokens'] for z in rs);nll=math.fsum(z['nll_sum'] for z in rs) if all(z['nll_sum'] is not None for z in rs) else None;loss=nll/count if nll is not None else None;losses[name]=loss
                    absolute.append(dict(seed=seed,model_type=m,recipient=r,mode=p,variant=name,task_ce=loss,nll_sum=nll,target_tokens=count,batches=len(rs),status='AVAILABLE' if loss is not None else 'UNAVAILABLE',source='verified_prior_interface' if name in (VARIANTS[0],VARIANTS[3]) else 'new_scaled_evaluation',calibration_scale=sm[seed,m]['scale']))
                seed_rows.append(dict(seed=seed,model_type=m,recipient=r,mode=p,**contrasts(losses)))
                for bi in range(len(ident['validation_batch_hashes'])):
                    rs={name:next(z for z in rows if (z['recipient'],z['mode'],z['variant'],z['batch_index'])==(r,p,name,bi)) for name in VARIANTS}
                    paired.append(dict(seed=seed,model_type=m,recipient=r,mode=p,batch_index=bi,batch_hash=rs[VARIANTS[0]]['batch_hash'],target_tokens=rs[VARIANTS[0]]['target_tokens'],**contrasts({name:row['nll_sum'] for name,row in rs.items()})))
        pop=[z for z in stats if z['population']=='validation' and z['kind']=='variant_amplitude' and (z['seed'],z['model_type'])==(seed,m)];rms={z['variant']:z['rms'] for z in pop}
        for target,left,right in [('end',VARIANTS[1],VARIANTS[3]),('start',VARIANTS[2],VARIANTS[0])]:
            ratio=rms[left]/rms[right] if rms[left] is not None and rms[right] else None
            amplitude.append(dict(seed=seed,model_type=m,nominal_amplitude_target=target,calibration_scale=sm[seed,m]['scale'],scaled_variant=left,reference_variant=right,scaled_validation_rms=rms[left],reference_validation_rms=rms[right],rms_ratio=ratio,relative_rms_mismatch=ratio-1 if ratio is not None else None,absolute_rms_discrepancy=difference(rms[left],rms[right]),adjusted_after_validation=False))
    sparse=[];routing=[]
    for s in c.SEEDS:
        for r in TIMES:
            for p in MODES:
                rs={m:next(z for z in seed_rows if (z['seed'],z['model_type'],z['recipient'],z['mode'])==(s,m,r,p)) for m in c.MODEL_TYPES}
                sparse.append(dict(seed=s,recipient=r,mode=p,**{k:difference(rs['sparse'][k],rs['unpruned'][k]) for k in METRICS}))
            for m in c.MODEL_TYPES:
                rs={p:next(z for z in seed_rows if (z['seed'],z['model_type'],z['recipient'],z['mode'])==(s,m,r,p)) for p in MODES}
                routing.append(dict(seed=s,model_type=m,recipient=r,**{k:difference(rs[MODES[0]][k],rs[MODES[1]][k]) for k in METRICS}))
    summaries=summarize(seed_rows,['model_type','recipient','mode'],METRICS,'within_model')+summarize(sparse,['recipient','mode'],METRICS,'sparse_minus_unpruned')+summarize(routing,['model_type','recipient'],METRICS,'native_minus_replay')
    for name,rs in [('absolute_losses',absolute),('seed_level_contrasts',seed_rows),('descriptive_seed_summary',summaries),('validation_amplitude_checks',amplitude),('interface_statistics',stats),('per_batch_losses',batches),('per_batch_nll_contrasts',paired),('sparse_unpruned_contrasts',sparse),('routing_mode_contrasts',routing)]:c._csv_write(OUT/(name+'.csv'),rs)
    verify_exports()
    findings(seed_rows,sparse,amplitude,scales,summaries)
    figures(absolute,seed_rows,amplitude,scales)

def verify_exports():
    absolute=c._read_csv(OUT/'absolute_losses.csv');idx={(int(r['seed']),r['model_type'],r['recipient'],r['mode'],r['variant']):float(r['task_ce']) if r['task_ce'] else None for r in absolute}
    expected={(s,m,r,p,v) for s in c.SEEDS for m in c.MODEL_TYPES for r in TIMES for p in MODES for v in VARIANTS};c._need(len(absolute)==160 and set(idx)==expected,'Incomplete or duplicate 160-cell grid')
    def near(a,b):
        if a is None or b is None:c._need(a is None and b is None,'Missing value serialized as an effect')
        else:c._need(abs(a-b)<1e-12,'Exported contrast identity failed')
    seed=c._read_csv(OUT/'seed_level_contrasts.csv');lookup={}
    for row in seed:
        key=(int(row['seed']),row['model_type'],row['recipient'],row['mode']);A,B,C,D=[idx[(*key,name)] for name in VARIANTS]
        direct={'B_original':difference(A,D),'B_scale_start':difference(A,B),'B_content_end':difference(B,D),'B_content_start':difference(A,C),'B_scale_end':difference(C,D),'I':difference(difference(C,D),difference(A,B))}
        parsed={k:float(row[k]) if row[k] else None for k in METRICS};lookup[key]=parsed
        for k in METRICS:near(parsed[k],direct[k])
        if all(parsed[k] is not None for k in METRICS):near(parsed['B_original'],parsed['B_scale_start']+parsed['B_content_end']);near(parsed['B_original'],parsed['B_content_start']+parsed['B_scale_end'])
    for row in c._read_csv(OUT/'sparse_unpruned_contrasts.csv'):
        s,r,p=int(row['seed']),row['recipient'],row['mode']
        for k in METRICS:near(float(row[k]) if row[k] else None,difference(lookup[s,'sparse',r,p][k],lookup[s,'unpruned',r,p][k]))
    for row in c._read_csv(OUT/'routing_mode_contrasts.csv'):
        s,m,r=int(row['seed']),row['model_type'],row['recipient']
        for k in METRICS:near(float(row[k]) if row[k] else None,difference(lookup[s,m,r,MODES[0]][k],lookup[s,m,r,MODES[1]][k]))
    batches=c._read_csv(OUT/'per_batch_losses.csv');paired=c._read_csv(OUT/'per_batch_nll_contrasts.csv')
    for key in expected:
        s,m,r,p,v=key;rs=[z for z in batches if (int(z['seed']),z['model_type'],z['recipient'],z['mode'],z['variant'])==key];tokens=sum(int(z['target_tokens']) for z in rs);nll=math.fsum(float(z['nll_sum']) for z in rs) if all(z['nll_sum'] for z in rs) else None
        source=next(z for z in absolute if (int(z['seed']),z['model_type'],z['recipient'],z['mode'],z['variant'])==key);c._need(len(rs)==int(source['batches']) and tokens==int(source['target_tokens']),'Token/batch coverage mismatch');near(idx[key],nll/tokens if nll is not None else None)
    for key,values in lookup.items():
        s,m,r,p=key;rs=[z for z in paired if (int(z['seed']),z['model_type'],z['recipient'],z['mode'])==key];n=sum(int(z['target_tokens']) for z in rs)
        for k in METRICS:near(values[k],math.fsum(float(z[k]) for z in rs)/n if all(z[k] for z in rs) else None)
    return dict(status='PASS',logical_cells=160,paired_batch_nll_identities=True,all_six_contrasts_and_two_decompositions=True,missing_values_preserved=True)

def fmt(x):return 'unavailable' if x is None else f'{x:.6f}'
def findings(seed,sparse,amplitude,scales,summaries):
    text=['# First-MoE amplitude control\n',LIMITS,'\nPositive B_content means a recovered-contribution advantage at the nominally matched amplitude. Negative effects are retained. Individual seed values appear first.\n']
    for p in MODES:
        text.append('\n## '+p+'\n')
        for m in c.MODEL_TYPES:
            for r in TIMES:
                text.append('\n### '+m+' / '+r+' recipient\n')
                rows=[z for z in seed if (z['model_type'],z['recipient'],z['mode'])==(m,r,p)]
                text.append('| Seed | Original benefit | Scale start | Content end-target | Content start-target | Scale end | Interaction |\n|---:|---:|---:|---:|---:|---:|---:|')
                for row in rows:text.append('| '+str(row['seed'])+' | '+' | '.join(fmt(row[k]) for k in METRICS)+' |')
                for k in METRICS:
                    z=next(z for z in summaries if z['kind']=='within_model' and (z['model_type'],z['recipient'],z['mode'],z['metric'])==(m,r,p,k));text.append(f"{k}: n={z['n_available']}/5, mean {fmt(z['mean'])}, median {fmt(z['median'])}, range [{fmt(z['minimum'])}, {fmt(z['maximum'])}], sample SD {fmt(z['sample_sd'])}.\n")
                for k in ('B_content_end','B_content_start'):text.append(f"{k} positive in {sum(z[k] is not None and z[k]>0 for z in rows)}/5 seeds; scaling alone leaves the signed residual differences above.\n")
                both=sum(z['B_content_end'] is not None and z['B_content_start'] is not None and z['B_content_end']>0 and z['B_content_start']>0 for z in rows)
                text.append(f"Recovered contribution retains an advantage at both nominal amplitude targets in {both}/5 seeds. The two target-specific contrasts and I report context/target interaction; no equivalence or causal percentage is inferred.\n")
        text.append('\n### Paired sparse-minus-unpruned residual advantages\n')
        for r in TIMES:
            for k in ('B_content_end','B_content_start'):
                rs=[z for z in sparse if (z['recipient'],z['mode'])==(r,p)];text.append(f"{r} recipient {k}, seed order 7,17,29,43,61: "+', '.join(fmt(z[k]) for z in rs)+f"; positive in {sum(z[k] is not None and z[k]>0 for z in rs)}/5 seeds. These are trajectory-specific, not equal absolute-amplitude perturbations.\n")
    text.append('\n## Calibration and validation mismatch\n')
    for row in scales:text.append(f"Seed {row['seed']} {row['model_type']}: training-only r={fmt(row['scale'])}; a0={fmt(row['a0'])}, a1={fmt(row['a1'])}; {row['status']}. Four complete, previously examined training batches; no validation CE in this estimate.\n")
    for row in amplitude:text.append(f"Seed {row['seed']} {row['model_type']} target {row['nominal_amplitude_target']}: actual validation scaled/reference RMS ratio {fmt(row['rms_ratio'])}; relative mismatch {fmt(row['relative_rms_mismatch'])}; no scale adjustment.\n")
    vals=[abs(z['relative_rms_mismatch']) for z in amplitude if z['relative_rms_mismatch'] is not None]
    if vals:text.append(f"Largest absolute relative validation RMS mismatch across the nominal pairs is {max(vals):.6f}. The exact mismatch is shown for every seed/target. No prespecified equivalence margin was introduced: interpret the contrasts as nominal training-calibrated matching, with this remaining global-amplitude limitation, not exact validation matching.\n")
    text.append('\n## Verdict and interpretation\n')
    for p in MODES:
        for m in c.MODEL_TYPES:
            rows=[z for z in seed if z['mode']==p and z['model_type']==m];both=sum(z['B_content_end'] is not None and z['B_content_start'] is not None and z['B_content_end']>0 and z['B_content_start']>0 for z in rows)
            text.append(f"{m}, {p}: positive recovered-contribution residual advantage at both targets in {both}/10 seed-by-recipient combinations. Where both remain positive, global amplitude alone is insufficient under these training-calibrated interventions; residual differences include more than direction. Otherwise the per-seed target/context interaction remains the conclusion.\n")
    text.append('Native versus replay contrast differences are fully reported in routing_mode_contrasts.csv and Figure 5. They assess sensitivity to downstream routing response, not an isolated routing mechanism. Calibration and validation direction/support/gate diagnostics are descriptive. No subsequent experiment or training is launched.\n')
    (OUT/'findings.md').write_text('\n'.join(text))

def figures(absolute,seed,amplitude,scales):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    out=OUT/'figures';out.mkdir(exist_ok=True)
    def save(fig,n):fig.tight_layout();fig.savefig(out/f'{n}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for i,m in enumerate(c.MODEL_TYPES):
        for j,(r,p) in enumerate((r,p) for r in TIMES for p in MODES):
            ax=axes[i,j]
            for s,color in zip(c.SEEDS,colors):
                values=[next(z['task_ce'] for z in absolute if (z['seed'],z['model_type'],z['recipient'],z['mode'],z['variant'])==(s,m,r,p,v)) for v in VARIANTS];ax.plot(range(4),[x if x is not None else float('nan') for x in values],marker='o',color=color,label=str(s))
            ax.set_xticks(range(4),['A: start','B: start ↑','C: end ↓','D: end'],rotation=30);ax.set_title(f'{m} {r}\n{p}',fontsize=8);ax.set_ylabel('Absolute task CE');ax.legend(fontsize=6)
    save(fig,1)
    for n,fields,title in [(2,('B_content_end','B_content_start'),'Residual recovered-contribution benefit'),(3,('B_scale_start','B_scale_end'),'Global amplitude effects')]:
        fig,axes=plt.subplots(2,4,figsize=(17,8))
        for i,m in enumerate(c.MODEL_TYPES):
            for j,(r,p) in enumerate((r,p) for r in TIMES for p in MODES):
                ax=axes[i,j]
                for s,color in zip(c.SEEDS,colors):
                    row=next(z for z in seed if (z['seed'],z['model_type'],z['recipient'],z['mode'])==(s,m,r,p));ax.plot([0,1],[row[k] if row[k] is not None else float('nan') for k in fields],marker='o',color=color,label=str(s))
                ax.set_xticks([0,1],fields,rotation=15,fontsize=7);ax.set_title(f'{m} {r}\n{p}',fontsize=8);ax.set_ylabel(title+' (CE)');ax.axhline(0,color='gray',lw=.5);ax.legend(fontsize=6)
        save(fig,n)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,m in zip(axes,c.MODEL_TYPES):
        for s,color in zip(c.SEEDS,colors):
            scale=next(z['scale'] for z in scales if (z['seed'],z['model_type'])==(s,m))
            for target,marker in [('start','o'),('end','s')]:
                row=next(z for z in amplitude if (z['seed'],z['model_type'],z['nominal_amplitude_target'])==(s,m,target))
                if scale is not None and row['relative_rms_mismatch'] is not None:ax.scatter(scale,row['relative_rms_mismatch'],marker=marker,color=color,label=f'{s} {target}')
        ax.set_title(m);ax.set_xlabel('Training-calibrated r');ax.set_ylabel('Validation relative RMS mismatch');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=6,ncol=2)
    save(fig,4)
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for i,m in enumerate(c.MODEL_TYPES):
        for j,(r,k) in enumerate((r,k) for r in TIMES for k in ('B_content_end','B_content_start')):
            ax=axes[i,j]
            for s,color in zip(c.SEEDS,colors):
                vals=[next(z[k] for z in seed if (z['seed'],z['model_type'],z['recipient'],z['mode'])==(s,m,r,p)) for p in MODES]
                if all(v is not None for v in vals):ax.scatter(*vals,color=color,label=str(s))
            lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],color='gray',lw=.6);ax.set_title(f'{m} {r}\n{k}',fontsize=8);ax.set_xlabel('Native downstream');ax.set_ylabel('Recipient baseline replay');ax.legend(fontsize=6)
    save(fig,5)

def bundle():
    with tarfile.open(OUT/'summary.tar.gz','w:gz') as tar:
        for name in ('protocol.md','calibration_scales.csv','findings.md','completion.json','numerical_checks.json','absolute_losses.csv','seed_level_contrasts.csv','descriptive_seed_summary.csv','validation_amplitude_checks.csv','interface_statistics.csv','sparse_unpruned_contrasts.csv','routing_mode_contrasts.csv'):tar.add(OUT/name,arcname=name)
