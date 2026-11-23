"""Fixed common-support norm/direction estimands and descriptive source-seed reporting."""
import csv,math,statistics,tarfile
from pathlib import Path
from .engine import c,ROOT,MODES,TIMES,VARIANTS,ratios
OUT=ROOT/'results/imp_first_moe_direction_v1'
METRICS=('D_start_norm','D_end_norm','N_start_dir','N_end_dir','B_common','I')
LIMITS='''This follow-up and layer 0 were chosen after prior interface/amplitude results; the new configuration and numerical eligibility rule are sealed before intervention losses. Validation has been repeatedly examined and is not an independent test set. Source seeds (n=5) are the replication unit; tokens, sequences, batches, contexts and cells are nested observations. Directions are those of the complete native gated contribution in native residual coordinates, with no learned alignment or transformation. On S only, norm and direction donors are crossed; outside S every variant equals the recipient's own native contribution. Unsupported positions are not tested and are not shown harmless. Other-diagonal hybrids are common-support substitutions, not necessarily full-donor transplants. Cross-recipient contrasts include residual context, downstream parameters and unsupported-position backgrounds; they are not downstream-weight attribution. Each sparse/unpruned trajectory uses its own norms/directions and is not an equal-sized cross-model perturbation. Downstream replay fixes each recipient's own decisions and computes experts on current inputs; mode differences do not isolate one routing mechanism. Algebraic decompositions are not percentages of training recovery. Frozen experts did not learn new weights. Direction effects may reflect changed assignments or different inputs to the same frozen experts; they do not isolate upstream cause, semantic content, all recovery, winning tickets, normal-training attainability, or an online pruning criterion. No subsequent experiment is launched.'''
PROTOCOL='''# First-MoE per-token norm–direction intervention\n\nUse all five source seeds, abrupt IMP80 sparse and matched unpruned shared-only/router_experts_frozen trajectories at global steps 1000 and 1500, exactly matching the completed interface/amplitude studies. Seed17 is the predetermined engineering pilot and remains regardless of signs.\n\nThe native first block computes u=input+attention(attention_norm(input)); z=moe_norm(u); c=MoE(z); h=u+c. c includes expert biases, GELU, selected full-softmax confidence, top-1 assignment and complete-batch capacity acceptance. Capture both donors from unmodified native forwards before any hybrids. Preserve recipient u and all attention/normalization/embedding parameters. Insert exactly once at the verified MoE-output boundary. Recipient-baseline replay controls only downstream layers 1–3 and evaluates experts on current hybrid inputs.\n\nEligibility is retrospective but outcome independent: compute n0,n1 in FP64; candidate S=(n0>0)&(n1>0). Prefer exact nonzero, with no epsilon or numerical floor. Sealed training-only probes and dtype-aware tests must establish FP64 unit reconstruction and FP32 hybrid target-norm/direction preservation. If a cast is numerically unusable, abort before the hybrid CE; do not invent directions, silently exclude positions, adjust a threshold or weaken tolerances. Synthetic unrepresentable subnormal casts must be refused; stable near-zero examples must pass. The evidence and final rule are written to direction_eligibility.json and hash-sealed before any new validation loss.\n\nOn S, V00=n0*d0, V01=n0*d1, V10=n1*d0, V11=n1*d1, constructed in FP64 then cast to FP32. Norm relative errors and paired norm matching must be <=2e-6, direction L2 error <=2e-6, FP64 unit norm error <=2e-12. Native diagonals must reconstruct bitwise after casting on S. Outside S all four are exact copies of recipient-native c, holding unsupported backgrounds identical within each recipient. No first-layer routing, support, donor input or norm is fitted. The starting recipient V00 and recovered recipient V11 reproduce native logits/CE. The other diagonal is labeled common-support substitution.\n\nThe complete original validation population, boundaries, order, targets and partial batch are reverified from source artifacts. FP32 forwards, TF32 disabled, eval/inference; token NLL is accumulated in FP64. Full-population CE is primary. Retain per-batch losses plus paired batch and sequence NLL contrasts with actual token counts; sequences are nested, not new seed replication. Do not infer position-local expert loss: attention/later layers can transmit effects.\n\nD_start_norm=L00-L01; D_end_norm=L10-L11; N_start_dir=L00-L10; N_end_dir=L01-L11; B_common=L00-L11; I=D_end_norm-D_start_norm=N_end_dir-N_start_dir. Verify B_common=D_start_norm+N_end_dir=N_start_dir+D_end_norm independently from exported absolute CE and paired NLL. Report each seed/model/recipient/mode, paired sparse-minus-unpruned, recipient differences in D and native-minus-replay. No clamping, significance fishing, fitted classifier or causal percentage.\n\nReport raw batch coverage/exclusions and full-population pooled coverage: S/all positions, S/start nonzero, S/end nonzero, exact zeros, separate zero categories, numerical-floor exclusions (zero if none), accepted-position coverage, and contribution energy within/outside S. Fractions use explicit denominators; validation summaries pool counts/energy, not unweighted batch fractions. Norm diagnostics are shared across recipients on S, not independent cells. Interface norms include the complete recipient-specific outside-S background; per-position cosine distribution is conditional on S. Full-donor CE is reused as context, never subtracted to call a remainder a causal support contribution.\n\nUse source/probe/rule/code/checkpoint/mask/batch/context/mode/variant identities in atomic complete pair caches. Profile the fixed seed17 pilot before remaining dispatch. Admission and locks follow existing shared-home conventions, with unique invocation attempts and no killed jobs. Scientific scope never depends on favorable losses. Preserve any failure and source revision explicitly. No training, backward/optimizer operation, mask discovery, learned transformation or later experiment.\n\n'''+LIMITS

def contrasts(v):
    a,b,cc,d=[v[k] for k in VARIANTS]
    ds=a-b;de=cc-d;ns=a-cc;ne=b-d
    return dict(D_start_norm=ds,D_end_norm=de,N_start_dir=ns,N_end_dir=ne,B_common=a-d,I=de-ds)

def seed_summaries(rows,groups,fields,kind):
    out=[]
    for key in sorted(set(tuple(str(r.get(g,'')) for g in groups) for r in rows)):
        rs=[r for r in rows if tuple(str(r.get(g,'')) for g in groups)==key];c._need(len(rs)==5 and len({r['seed'] for r in rs})==5,'Incorrect independent replication count')
        for field in fields:
            vals=[r[field] for r in rs if r.get(field) is not None]
            out.append(dict(kind=kind,**dict(zip(groups,key)),metric=field,n_available=len(vals),n_source_seeds=5,mean=statistics.mean(vals) if vals else None,median=statistics.median(vals) if vals else None,minimum=min(vals) if vals else None,maximum=max(vals) if vals else None,sample_sd=statistics.stdev(vals) if len(vals)>1 else None))
    return out

def pooled_coverage(result):
    rs=result['coverage'];first=rs[0];skip={'seed','model_type','population','batch_index','batch_hash','scope'}
    minimum={'minimum_positive_start_norm','minimum_positive_end_norm'}
    counts={k:sum(r[k] for r in rs) for k in first if k not in skip and k not in minimum and not k.startswith('eligible_fraction') and 'energy_fraction' not in k and 'accepted_coverage' not in k}
    for k in minimum:
        valid=[r[k] for r in rs if r[k] is not None];counts[k]=min(valid) if valid else None
    return dict(seed=first['seed'],model_type=first['model_type'],population=first['population'],scope='pooled_complete_population',batch_index='ALL',batches=len(rs),**ratios(counts))

def pooled_interfaces(result):
    output=[];rs=result['interfaces']
    for key in sorted(set((r['recipient'],r['mode'],r['variant']) for r in rs)):
        selected=[r for r in rs if (r['recipient'],r['mode'],r['variant'])==key];n=sum(r['elements'] for r in selected)
        row=dict(seed=result['identity']['seed'],model_type=result['identity']['model_type'],population=selected[0]['population'],recipient=key[0],mode=key[1],variant=key[2],scope='pooled_complete_population_with_recipient_background',batch_index='ALL',elements=n,positions=sum(r['positions'] for r in selected),batches=len(selected))
        for field in [k for k in selected[0] if k.endswith('_sumsq')]:
            ss=math.fsum(r[field] for r in selected);name=field.removesuffix('_sumsq');row.update({field:ss,name+'_rms':math.sqrt(ss/n),name+'_l2':math.sqrt(ss)})
        row['inserted_to_recipient_residual_ratio']=row['inserted_contribution_l2']/row['recipient_residual_l2'] if row['recipient_residual_l2'] else None;output.append(row)
    output.append(dict(seed=result['identity']['seed'],model_type=result['identity']['model_type'],population=rs[0]['population'],scope='cosine_distribution_conditional_on_S',**{f'cosine_{k}':v for k,v in result['cosine_distribution'].items()}))
    return output

def write_outputs(calibrations,readouts,full_context):
    absolute=[];batch=[];paired=[];sequence=[];seed=[];coverage=[];checks=[];interfaces=[];pooled=[]
    for result in calibrations+readouts:
        coverage+=result['coverage'];pooled.append(pooled_coverage(result));checks+=result['norm_checks'];interfaces+=result['interfaces']+pooled_interfaces(result)
    coverage+=pooled
    for result in readouts:
        ident=result['identity'];s,m=ident['seed'],ident['model_type'];rs=result['losses']
        batch.extend({k:v for k,v in row.items() if k!='sequence_nll'} for row in rs)
        for r in TIMES:
            for p in MODES:
                vals={}
                for name in VARIANTS:
                    selected=[z for z in rs if (z['recipient'],z['mode'],z['variant'])==(r,p,name)];n=sum(z['target_tokens'] for z in selected);ss=math.fsum(z['nll_sum'] for z in selected);vals[name]=ss/n
                    absolute.append(dict(seed=s,model_type=m,recipient=r,mode=p,variant=name,norm_donor='start' if name[1]=='0' else 'end',direction_donor='start' if name[2]=='0' else 'end',task_ce=ss/n,nll_sum=ss,target_tokens=n,batches=len(selected),intervention_scope='native_reconstruction' if (r,name) in (('start','V00'),('end','V11')) else 'common_support_hybrid_recipient_native_outside_S',eligibility_sha256=ident['eligibility_sha256']))
                seed.append(dict(seed=s,model_type=m,recipient=r,mode=p,**contrasts(vals)))
                for bi in range(len(ident['validation_batch_hashes'])):
                    selected={name:next(z for z in rs if (z['recipient'],z['mode'],z['variant'],z['batch_index'])==(r,p,name,bi)) for name in VARIANTS};example=selected['V00']
                    base=dict(seed=s,model_type=m,recipient=r,mode=p,batch_index=bi,batch_hash=example['batch_hash'])
                    paired.append(dict(**base,target_tokens=example['target_tokens'],scope='batch_nested_in_source_seed',**contrasts({name:z['nll_sum'] for name,z in selected.items()})))
                    for qi in range(example['sequence_count']):sequence.append(dict(**base,sequence_index=qi,target_tokens=example['sequence_length'],scope='sequence_nested_in_source_seed',**contrasts({name:z['sequence_nll'][qi] for name,z in selected.items()})))
    sparse=[];routing=[];recipient=[]
    lookup={(z['seed'],z['model_type'],z['recipient'],z['mode']):z for z in seed}
    for s in c.SEEDS:
        for r in TIMES:
            for p in MODES:sparse.append(dict(seed=s,recipient=r,mode=p,**{k:lookup[s,'sparse',r,p][k]-lookup[s,'unpruned',r,p][k] for k in METRICS}))
            for m in c.MODEL_TYPES:routing.append(dict(seed=s,model_type=m,recipient=r,**{k:lookup[s,m,r,MODES[0]][k]-lookup[s,m,r,MODES[1]][k] for k in METRICS}))
        for m in c.MODEL_TYPES:
            for p in MODES:recipient.append(dict(seed=s,model_type=m,mode=p,contrast='recovered_minus_starting_recipient_context',D_start_norm=lookup[s,m,'end',p]['D_start_norm']-lookup[s,m,'start',p]['D_start_norm'],D_end_norm=lookup[s,m,'end',p]['D_end_norm']-lookup[s,m,'start',p]['D_end_norm']))
    summary=seed_summaries(seed,['model_type','recipient','mode'],METRICS,'within_model')+seed_summaries(sparse,['recipient','mode'],METRICS,'sparse_minus_unpruned')+seed_summaries(routing,['model_type','recipient'],METRICS,'native_minus_replay')+seed_summaries(recipient,['model_type','mode'],('D_start_norm','D_end_norm'),'recipient_context_difference')+seed_summaries(pooled,['model_type','population'],('eligible_fraction_all','eligible_fraction_start_nonzero','eligible_fraction_end_nonzero','start_energy_fraction_in_S','end_energy_fraction_in_S'),'coverage_denominators_are_per_seed_populations')
    for name,rows in [('absolute_losses',absolute),('seed_level_contrasts',seed),('descriptive_seed_summary',summary),('coverage_and_exclusions',coverage),('norm_preservation_checks',checks),('interface_statistics',interfaces),('per_batch_losses',batch),('per_batch_nll_contrasts',paired),('per_sequence_nll_contrasts',sequence),('sparse_unpruned_contrasts',sparse),('routing_mode_contrasts',routing),('recipient_direction_contrasts',recipient),('full_donor_context',full_context)]:c._csv_write(OUT/(name+'.csv'),rows)
    verified=verify_exports();findings(seed,sparse,routing,recipient,pooled,summary);figures(absolute,seed,pooled)
    return verified

def verify_exports():
    absolute=c._read_csv(OUT/'absolute_losses.csv');L={(int(z['seed']),z['model_type'],z['recipient'],z['mode'],z['variant']):float(z['task_ce']) for z in absolute};expected={(s,m,r,p,v) for s in c.SEEDS for m in c.MODEL_TYPES for r in TIMES for p in MODES for v in VARIANTS};c._need(len(absolute)==160 and set(L)==expected,'Incomplete/duplicate logical grid')
    def near(x,y):c._need(abs(float(x)-y)<=1e-12,'Exported CE/contrast identity mismatch')
    look={}
    for z in c._read_csv(OUT/'seed_level_contrasts.csv'):
        key=int(z['seed']),z['model_type'],z['recipient'],z['mode'];a,b,cc,d=[L[(*key,v)] for v in VARIANTS];look[key]={k:float(z[k]) for k in METRICS}
        direct=dict(D_start_norm=a-b,D_end_norm=cc-d,N_start_dir=a-cc,N_end_dir=b-d,B_common=a-d,I=(cc+b)-(d+a))
        for k in METRICS:near(z[k],direct[k])
        near(z['I'],float(z['N_end_dir'])-float(z['N_start_dir']));near(z['B_common'],float(z['D_start_norm'])+float(z['N_end_dir']));near(z['B_common'],float(z['N_start_dir'])+float(z['D_end_norm']))
    for z in c._read_csv(OUT/'sparse_unpruned_contrasts.csv'):
        s,r,p=int(z['seed']),z['recipient'],z['mode']
        for k in METRICS:near(z[k],look[s,'sparse',r,p][k]-look[s,'unpruned',r,p][k])
    for z in c._read_csv(OUT/'routing_mode_contrasts.csv'):
        s,m,r=int(z['seed']),z['model_type'],z['recipient']
        for k in METRICS:near(z[k],look[s,m,r,MODES[0]][k]-look[s,m,r,MODES[1]][k])
    for z in c._read_csv(OUT/'recipient_direction_contrasts.csv'):
        s,m,p=int(z['seed']),z['model_type'],z['mode']
        for k in ('D_start_norm','D_end_norm'):near(z[k],look[s,m,'end',p][k]-look[s,m,'start',p][k])
    batches=c._read_csv(OUT/'per_batch_losses.csv')
    for key in expected:
        selected=[z for z in batches if (int(z['seed']),z['model_type'],z['recipient'],z['mode'],z['variant'])==key];a=next(z for z in absolute if (int(z['seed']),z['model_type'],z['recipient'],z['mode'],z['variant'])==key)
        c._need(len(selected)==int(a['batches']) and len(set(int(z['batch_index']) for z in selected))==len(selected),'Batch grid duplicated');n=sum(int(z['target_tokens']) for z in selected);c._need(n==int(a['target_tokens']),'Incorrect target coverage');near(L[key],math.fsum(float(z['nll_sum']) for z in selected)/n)
    totals={}
    # Stream sequence-level contrasts; no pooled token-level statistical tests.
    for filename in ('per_batch_nll_contrasts','per_sequence_nll_contrasts'):
        grouped={}
        with (OUT/(filename+'.csv')).open() as stream:
            for z in csv.DictReader(stream):
                key=int(z['seed']),z['model_type'],z['recipient'],z['mode'];entry=grouped.setdefault(key,dict(tokens=0,values={k:[] for k in METRICS}));entry['tokens']+=int(z['target_tokens'])
                for k in METRICS:entry['values'][k].append(float(z[k]))
        for key,z in grouped.items():
            for k in METRICS:near(look[key][k],math.fsum(z['values'][k])/z['tokens'])
        totals[filename]=sum(len(z['values']['I']) for z in grouped.values())
    return dict(status='PASS',logical_cells=160,all_six_estimands=True,two_common_benefit_identities=True,two_interaction_identities=True,paired_batch_and_sequence_nll_identities=True,record_counts=totals)

def fmt(x):return 'undefined' if x is None else f'{x:.6f}'
def findings(seed,sparse,routing,recipient,coverage,summaries):
    text=['# First-MoE per-token direction study\n',LIMITS,'\nPositive direction benefits mean recovered directions lower whole-population CE at fixed per-token norm targets on S, with recipient-native backgrounds elsewhere. Per-seed values precede aggregates.\n']
    for p in MODES:
        text.append('\n## '+p+'\n')
        for m in c.MODEL_TYPES:
            for r in TIMES:
                rows=[z for z in seed if (z['model_type'],z['recipient'],z['mode'])==(m,r,p)];text.append(f'\n### {m}, {r} recipient\n');text.append('| Seed | D start norm | D end norm | N start dir | N end dir | B common | I |\n|---:|---:|---:|---:|---:|---:|---:|')
                for z in rows:text.append('| '+str(z['seed'])+' | '+' | '.join(fmt(z[k]) for k in METRICS)+' |')
                for k in METRICS:
                    z=next(z for z in summaries if z['kind']=='within_model' and (z['model_type'],z['recipient'],z['mode'],z['metric'])==(m,r,p,k));text.append(f"{k}: mean {fmt(z['mean'])}, median {fmt(z['median'])}, range [{fmt(z['minimum'])}, {fmt(z['maximum'])}], sample SD {fmt(z['sample_sd'])}; n=5 source seeds.\n")
                for k in ('D_start_norm','D_end_norm','N_start_dir','N_end_dir'):text.append(f"{k} positive in {sum(z[k]>0 for z in rows)}/5 seeds, negative in {sum(z[k]<0 for z in rows)}/5.\n")
                text.append(f"Norm–direction interaction I positive in {sum(z['I']>0 for z in rows)}/5 and negative in {sum(z['I']<0 for z in rows)}/5; norm allocation is not assumed independent of chosen direction.\n")
        text.append('\n### Paired sparse-minus-unpruned direction benefits\n')
        for r in TIMES:
            rows=[z for z in sparse if (z['recipient'],z['mode'])==(r,p)]
            for k in ('D_start_norm','D_end_norm'):text.append(f"{r} recipient {k}: seed order 7,17,29,43,61 = "+', '.join(fmt(z[k]) for z in rows)+f"; sparse larger in {sum(z[k]>0 for z in rows)}/5 seeds. Each model has its own norms, directions and S.\n")
    text.append('\n## Common-position and energy coverage\n')
    for z in coverage:
        if z['population']!='validation':continue
        text.append(f"Seed {z['seed']} {z['model_type']}: S {z['eligible_positions']}/{z['positions']} positions ({z['eligible_fraction_all']:.6f}); S/start nonzero {fmt(z['eligible_fraction_start_nonzero'])}, S/end nonzero {fmt(z['eligible_fraction_end_nonzero'])}; start/end contribution energy fractions {fmt(z['start_energy_fraction_in_S'])}/{fmt(z['end_energy_fraction_in_S'])}; exact-zero exclusions {z['exact_zero_exclusions']}, numerical-floor exclusions {z['numerical_floor_exclusions']}; start-only/end-only/both-zero counts {z['start_only_nonzero_positions']}/{z['end_only_nonzero_positions']}/{z['both_zero_positions']}. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.\n")
    text.append('\n## Routing and recipient dependence\n')
    for m in c.MODEL_TYPES:
        for r in TIMES:
            for k in ('D_start_norm','D_end_norm'):
                a=[next(z[k] for z in seed if (z['seed'],z['model_type'],z['recipient'],z['mode'])==(s,m,r,MODES[0])) for s in c.SEEDS];b=[next(z[k] for z in seed if (z['seed'],z['model_type'],z['recipient'],z['mode'])==(s,m,r,MODES[1])) for s in c.SEEDS]
                text.append(f"{m}, {r}, {k}: native/replay positive-sign agreement in {sum((x>0)==(y>0) for x,y in zip(a,b))}/5 seeds. Native-minus-replay seed differences: "+', '.join(fmt(x-y) for x,y in zip(a,b))+'.\n')
    for m in c.MODEL_TYPES:
        for p in MODES:
            rows=[z for z in recipient if z['model_type']==m and z['mode']==p]
            for k in ('D_start_norm','D_end_norm'):text.append(f"{m} {p}, recovered-minus-starting recipient {k}: "+', '.join(fmt(z[k]) for z in rows)+'. These include residual, downstream-parameter and unsupported-background differences, not downstream weights alone.\n')
    text.append('\n## Supported interpretation\n')
    for p in MODES:
        for m in c.MODEL_TYPES:
            rows=[z for z in seed if z['model_type']==m and z['mode']==p]
            for k in ('D_start_norm','D_end_norm'):text.append(f"{m}, {p}, {k}: positive in {sum(z[k]>0 for z in rows)}/10 seed-by-recipient combinations. Where positive: on common positions with defined starting and recovered directions, replacing starting contribution direction with recovered direction improves CE under the tested fixed per-token norm and recipient conditions. Mixed signs remain context/norm-target dependence.\n")
    text.append('Full-donor CE context is preserved in full_donor_context.csv. No full-minus-common remainder is called a causal support effect. Whole-population loss includes effects transmitted by attention and later layers, including to unsupported positions; it is not an isolated expert-position loss. No subsequent experiment is launched.\n')
    (OUT/'findings.md').write_text('\n'.join(text))

def figures(absolute,seed,coverage):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True);colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    def save(fig,n):fig.tight_layout();fig.savefig(folder/f'{n}.png',dpi=170);plt.close(fig)
    for number,fields,ylabel in [(1,VARIANTS,'Absolute whole-population task CE'),(2,('D_start_norm','D_end_norm'),'Recovered direction benefit (CE)'),(3,('N_start_dir','N_end_dir'),'Recovered norm benefit (CE)')]:
        fig,axes=plt.subplots(2,4,figsize=(17,8))
        for i,m in enumerate(c.MODEL_TYPES):
            for j,(r,p) in enumerate((r,p) for r in TIMES for p in MODES):
                ax=axes[i,j]
                for s,color in zip(c.SEEDS,colors):
                    if number==1:vals=[next(z['task_ce'] for z in absolute if (z['seed'],z['model_type'],z['recipient'],z['mode'],z['variant'])==(s,m,r,p,v)) for v in fields]
                    else:
                        z=next(z for z in seed if (z['seed'],z['model_type'],z['recipient'],z['mode'])==(s,m,r,p));vals=[z[k] for k in fields]
                    ax.plot(range(len(fields)),vals,marker='o',color=color,label=str(s))
                ax.set_xticks(range(len(fields)),fields,rotation=20,fontsize=7);ax.set_title(f'{m} {r}\n{p}',fontsize=8);ax.set_ylabel(ylabel)
                if number!=1:ax.axhline(0,color='gray',lw=.6)
                ax.legend(fontsize=6)
        save(fig,number)
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for i,m in enumerate(c.MODEL_TYPES):
        for j,(r,k) in enumerate((r,k) for r in TIMES for k in ('D_start_norm','D_end_norm')):
            ax=axes[i,j]
            for s,color in zip(c.SEEDS,colors):
                vals=[next(z[k] for z in seed if (z['seed'],z['model_type'],z['recipient'],z['mode'])==(s,m,r,p)) for p in MODES];ax.scatter(*vals,color=color,label=str(s))
            lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],color='gray',lw=.6);ax.set_title(f'{m} {r}\n{k}',fontsize=8);ax.set_xlabel('Native downstream');ax.set_ylabel('Recipient baseline replay');ax.legend(fontsize=6)
    save(fig,4)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    fields=('eligible_fraction_all','start_energy_fraction_in_S','end_energy_fraction_in_S')
    for ax,m in zip(axes,c.MODEL_TYPES):
        for s,color in zip(c.SEEDS,colors):
            z=next(z for z in coverage if (z['seed'],z['model_type'],z['population'])==(s,m,'validation'));ax.plot(range(3),[z[k] for k in fields],marker='o',color=color,label=str(s))
        ax.set_xticks(range(3),['Positions','Start energy','End energy']);ax.set_ylim(0,1.02);ax.set_ylabel('Fraction represented in S');ax.set_title(m);ax.legend(fontsize=7)
    save(fig,5)
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,(m,p) in zip(axes,((m,p) for m in c.MODEL_TYPES for p in MODES)):
        for s,color in zip(c.SEEDS,colors):ax.plot([0,1],[next(z['I'] for z in seed if (z['seed'],z['model_type'],z['recipient'],z['mode'])==(s,m,r,p)) for r in TIMES],marker='o',color=color,label=str(s))
        ax.set_xticks([0,1],TIMES);ax.set_title(m+'\n'+p,fontsize=8);ax.set_ylabel('Norm–direction interaction I');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=6)
    save(fig,6)

def bundle():
    with tarfile.open(OUT/'summary.tar.gz','w:gz') as archive:
        for name in ('protocol.md','direction_eligibility.json','findings.md','completion.json','numerical_checks.json','absolute_losses.csv','seed_level_contrasts.csv','descriptive_seed_summary.csv','coverage_and_exclusions.csv','norm_preservation_checks.csv','sparse_unpruned_contrasts.csv','routing_mode_contrasts.csv','recipient_direction_contrasts.csv','full_donor_context.csv'):archive.add(OUT/name,arcname=name)
