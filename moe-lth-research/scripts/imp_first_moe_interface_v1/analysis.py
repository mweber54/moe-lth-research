"""Prespecified estimands, independent exported-CE checks and descriptive reporting."""
import json,statistics,tarfile
from pathlib import Path
from .engine import c,MODES,ALPHAS,TIMES,ROOT
OUT=ROOT/'results/imp_first_moe_interface_v1'
LIMITS='''Layer 0 was chosen post hoc after examining the prior contribution-ablation study. This new design and implementation were sealed before new outcomes; this is not project-level preregistration. Validation was repeatedly examined and is not an independent test. Source seeds (n=5) are replication units; batches, tokens, alphas, layers and cells are nested measurements. The recipient context includes its own pre-addition residual representation and subsequent computation, not downstream weights alone. Donor and recipient share seed/model type; no alignment, norm matching, weight transplantation or training is performed. Usefulness is confined to tested recipient contexts, not universally better semantic information. Q is an evaluation compatibility interaction that can reflect coordinate mismatch, scaling, routing and subsequent use. Reciprocal harm does not prove semantic co-specialization. Frozen expert/router weights and biases do not learn. Fractional attenuation has different absolute sizes across donors, but fixed donor/alpha perturbations are identical across recipients. Norms and cosines do not establish usefulness or explain CE causally. Sparse-versus-unpruned donor perturbations can also differ in absolute magnitude. No percentage of recovery, training counterfactual, ticket mechanism, safe pruning rule or generalization beyond this architecture/mask family/window is identified. Nonmonotone curves and negative effects are retained.'''
DEFINITION='''# First-MoE interface\n\nThe model is pre-normalized. Block 0 computes a = attention(attention_norm(input)); u = input + a; z = moe_norm(u); c = MoE(z); h = u + c. c includes fc1/fc2 biases, GELU, selected full-softmax confidence, top-1 assignment and whole-batch capacity acceptance. A forward hook replaces only the first MoE module's returned contribution tensor with alpha*c_d. The block's native residual addition remains active. Donor c_d is captured from its own unmodified forward on identical token/target positions and the complete original batch. F_r includes blocks 1–3, final_norm and the tied embedding/output head. No projection, alignment, gating again or rescaling is applied.\n\nNATIVE_RESPONSE lets downstream routes respond. RECIPIENT_BASELINE_REPLAY fixes only layers 1–3 to the exact recipient's own baseline IDs, selected coefficients and acceptance/order, while experts consume current inputs. Layer 0 is never overwritten by recipient replay. Internal native router probabilities used by replay are not labeled imposed selections. Routing is not held identical across the two recipients.\n\nNative interface statistics compare u, z, raw pre-temperature projection scores, selected confidence, ungated selected expert output including dropped positions, and complete c. Ungated differences are split by unchanged versus changed selected expert. Diagnostic selected-expert computation can be chunked after complete-batch routing has been captured; actual model dispatch is unchunked. Relative L2 uses the start tensor as denominator; global cosine and mean per-position cosine exclude zero denominators and record exclusions. Pair sufficient statistics support pooled validation metrics, rather than averages of batch RMS/relative changes. Substitution statistics retain sums of squares and elements, actual inserted/native/attenuation displacements, and post-addition residuals. Norm ratios use L2 norms on identical positions. Alpha-zero donor aliases have explicit logical rows but no independent status.\n\nTask CE = sum actual token NLL / sum actual target counts (492416 over the original 31 batches including partial final batch); auxiliary loss is excluded. FP32, TF32 off, eval/inference.\n\nV(r)=L(r,start,1)-L(r,end,1). Q=V(end)-V(start). S_start=L(start,end,1)-L(start,start,1); S_end=L(end,start,1)-L(end,end,1). A(r,d,a)=L(r,d,a)-L(r,d,1). T(d,a)=A(end,d,a)-A(start,d,a). Positive V means recovered donor helps; positive S means reciprocal substitution hurts; positive A means attenuation hurts; positive T means fixed-donor attenuation hurts more in the recovered recipient. Alpha .9 is the modest-attenuation primary contrast; .75,.5,0 are the fixed response curve. Sparse contrasts are same-seed sparse minus unpruned; routing contrasts are native minus replay. Diagonal A(end,end,0)-A(start,start,0) must equal prior ZERO_L0 damage change and is distinct from fixed-donor T. All identities are independently rechecked from exported absolute CE.\n\n'''+LIMITS

def estimands(absolute):
    L={(int(r['seed']),r['model_type'],r['mode'],r['recipient'],r['donor'],float(r['alpha'])):float(r['task_ce']) for r in absolute}
    sub=[];atten=[];sensitivity=[];sparse=[];routing=[]
    for seed in c.SEEDS:
        for m in c.MODEL_TYPES:
            for p in MODES:
                def loss(r,d,a=1.):return L[seed,m,p,r,d,a]
                vs=loss('start','start')-loss('start','end');ve=loss('end','start')-loss('end','end')
                diag=(loss('end','end',0.)-loss('end','end'))-(loss('start','start',0.)-loss('start','start'))
                sub.append(dict(seed=seed,model_type=m,mode=p,V_start=vs,V_end=ve,Q=ve-vs,S_start=-vs,S_end=ve,diagonal_zero_damage_change=diag,**{f'L_{r}_{d}':loss(r,d) for r in TIMES for d in TIMES}))
                for d in TIMES:
                    for alpha in ALPHAS:
                        damage={r:loss(r,d,alpha)-loss(r,d) for r in TIMES}
                        for r in TIMES:atten.append(dict(seed=seed,model_type=m,mode=p,recipient=r,donor=d,alpha=alpha,A=damage[r],task_ce=loss(r,d,alpha),unattenuated_ce=loss(r,d)))
                        sensitivity.append(dict(seed=seed,model_type=m,mode=p,donor=d,alpha=alpha,T=damage['end']-damage['start'],A_start=damage['start'],A_end=damage['end'],modest_primary=alpha==.9))
    for seed in c.SEEDS:
        for p in MODES:
            rs={m:next(r for r in sub if (r['seed'],r['model_type'],r['mode'])==(seed,m,p)) for m in c.MODEL_TYPES}
            for name in ('V_start','V_end','Q'):
                s,u=rs['sparse'][name],rs['unpruned'][name];sparse.append(dict(seed=seed,mode=p,metric=name,sparse_value=s,unpruned_value=u,sparse_minus_unpruned=s-u))
            for d in TIMES:
                for alpha in ALPHAS:
                    rs={m:next(r for r in sensitivity if (r['seed'],r['model_type'],r['mode'],r['donor'],r['alpha'])==(seed,m,p,d,alpha)) for m in c.MODEL_TYPES}
                    s,u=rs['sparse']['T'],rs['unpruned']['T'];sparse.append(dict(seed=seed,mode=p,metric='T',donor=d,alpha=alpha,sparse_value=s,unpruned_value=u,sparse_minus_unpruned=s-u))
        for m in c.MODEL_TYPES:
            for table,metrics,extras in ((sub,('V_start','V_end','Q','S_start','S_end','diagonal_zero_damage_change'),()),(atten,('A',),('recipient','donor','alpha')),(sensitivity,('T',),('donor','alpha'))):
                keys=sorted(set(tuple(r[k] for k in extras) for r in table if r['seed']==seed and r['model_type']==m))
                for key in keys:
                    rs={p:next(r for r in table if (r['seed'],r['model_type'],r['mode'])==(seed,m,p) and tuple(r[k] for k in extras)==key) for p in MODES}
                    for metric in metrics:routing.append(dict(seed=seed,model_type=m,metric=metric,**dict(zip(extras,key)),native_value=rs[MODES[0]][metric],replay_value=rs[MODES[1]][metric],native_minus_replay=rs[MODES[0]][metric]-rs[MODES[1]][metric]))
    return sub,atten,sensitivity,sparse,routing

def pooled_interface(stats):
    out=[]
    pairs=[r for r in stats if r['kind']=='native_start_end_pair']
    keys=set((r['seed'],r['model_type'],r['quantity'],r.get('population','')) for r in pairs)
    for seed,m,q,pop in sorted(keys):
        rs=[r for r in pairs if (r['seed'],r['model_type'],r['quantity'],r.get('population',''))==(seed,m,q,pop)]
        base=dict(seed=seed,model_type=m,kind='validation_native_pair',quantity=q,population=pop,batches=len(rs),positions=sum(r['positions'] for r in rs))
        if q=='routing':
            for name in ('assignment_agree_count','acceptance_agree_count','start_dropped','end_dropped','gate_absolute_difference_sum'):base[name]=sum(r[name] for r in rs)
            n=base['positions'];base.update(assignment_agreement=base['assignment_agree_count']/n,acceptance_agreement=base['acceptance_agree_count']/n,start_drop_rate=base['start_dropped']/n,end_drop_rate=base['end_dropped']/n)
        else:
            for name in ('elements','start_sumsq','end_sumsq','displacement_sumsq','dot_sum','cosine_sum','cosine_valid_positions','both_zero_positions','start_only_zero_positions','end_only_zero_positions','zero_cosine_exclusions'):base[name]=sum(r[name] for r in rs)
            a,b,n=base['start_sumsq'],base['end_sumsq'],base['elements'];v=base['cosine_valid_positions']
            base.update(start_rms=(a/n)**.5 if n else None,end_rms=(b/n)**.5 if n else None,relative_l2=(base['displacement_sumsq']/a)**.5 if a else None,norm_ratio=(b/a)**.5 if a else None,cosine_global=base['dot_sum']/(a*b)**.5 if a and b else None,mean_position_cosine=base['cosine_sum']/v if v else None)
        out.append(base)
    for row in out:
        if row['quantity']=='c':
            u=next(z for z in out if (z['seed'],z['model_type'],z['quantity'])==(row['seed'],row['model_type'],'u'))
            row['start_c_to_u_ratio']=(row['start_sumsq']/u['start_sumsq'])**.5 if u['start_sumsq'] else None
            row['end_c_to_u_ratio']=(row['end_sumsq']/u['end_sumsq'])**.5 if u['end_sumsq'] else None
    # Pooled substitution norms preserve actual perturbation size for each cell.
    ns=[r for r in stats if r['kind']=='substitution']
    for key in sorted(set((r['seed'],r['model_type'],r['recipient'],r['donor'],r['alpha'],r['mode']) for r in ns)):
        rs=[r for r in ns if (r['seed'],r['model_type'],r['recipient'],r['donor'],r['alpha'],r['mode'])==key]
        base=dict(zip(('seed','model_type','recipient','donor','alpha','mode'),key));base.update(kind='validation_substitution',positions=sum(r['positions'] for r in rs),elements=sum(r['elements'] for r in rs),batches=len(rs))
        for name in [k for k in rs[0] if k.endswith('_sumsq')]:
            value=sum(r[name] for r in rs);prefix=name.removesuffix('_sumsq');base.update({name:value,prefix+'_l2':value**.5,prefix+'_rms':(value/base['elements'])**.5})
        base['inserted_to_recipient_residual_ratio']=base['inserted_contribution_l2']/base['recipient_residual_l2'] if base['recipient_residual_l2'] else None
        out.append(base)
    return out

def summaries(tables):
    out=[]
    for table,groups,metrics in tables:
        keys=sorted(set(tuple(str(r.get(g,'')) for g in groups) for r in table))
        for key in keys:
            rs=[r for r in table if tuple(str(r.get(g,'')) for g in groups)==key]
            c._need(len(rs)==5 and len(set(r['seed'] for r in rs))==5,'Summary replication unit is not five distinct seeds')
            for metric in metrics:
                vals=[r[metric] for r in rs if r.get(metric) is not None]
                if not vals:continue
                out.append(dict(zip(groups,key),metric=metric,n=len(vals),mean=statistics.mean(vals),median=statistics.median(vals),minimum=min(vals),maximum=max(vals),sample_sd=statistics.stdev(vals) if len(vals)>1 else None))
    return out

def verify_exported():
    """Re-read exported absolute CE; independently verify every requested identity."""
    absolute=c._read_csv(OUT/'absolute_losses.csv')
    L={(int(r['seed']),r['model_type'],r['mode'],r['recipient'],r['donor'],float(r['alpha'])):float(r['task_ce']) for r in absolute}
    expected={(s,m,p,r,d,a) for s in c.SEEDS for m in c.MODEL_TYPES for p in MODES for r in TIMES for d in TIMES for a in ALPHAS}
    c._need(len(absolute)==400 and set(L)==expected,'Logical grid incomplete or duplicated')
    for s,m,p,r,d,a in expected:
        if a==0:c._need(L[s,m,p,r,'start',a]==L[s,m,p,r,'end',a],'Alpha zero donor CE differs')
    def near(actual,expected):c._need(abs(float(actual)-expected)<1e-12,'Exported estimand identity mismatch')
    for row in c._read_csv(OUT/'contribution_substitution_effects.csv'):
        s,m,p=int(row['seed']),row['model_type'],row['mode'];ss=L[s,m,p,'start','start',1.];se=L[s,m,p,'start','end',1.];es=L[s,m,p,'end','start',1.];ee=L[s,m,p,'end','end',1.]
        for field,value in dict(V_start=ss-se,V_end=es-ee,Q=(es+se)-(ee+ss),S_start=se-ss,S_end=es-ee,diagonal_zero_damage_change=L[s,m,p,'end','end',0.]-ee-L[s,m,p,'start','start',0.]+ss).items():near(row[field],value)
    for row in c._read_csv(OUT/'attenuation_effects.csv'):
        s,m,p,r,d,a=int(row['seed']),row['model_type'],row['mode'],row['recipient'],row['donor'],float(row['alpha']);near(row['A'],L[s,m,p,r,d,a]-L[s,m,p,r,d,1.])
    for row in c._read_csv(OUT/'recipient_context_sensitivity.csv'):
        s,m,p,d,a=int(row['seed']),row['model_type'],row['mode'],row['donor'],float(row['alpha']);near(row['T'],(L[s,m,p,'end',d,a]+L[s,m,p,'start',d,1.])-(L[s,m,p,'end',d,1.]+L[s,m,p,'start',d,a]))
    def independent(seed,m,p,metric,r='',d='',alpha=1.):
        f=lambda rr,dd,aa=1.:L[seed,m,p,rr,dd,aa]
        v0=f('start','start')-f('start','end');v1=f('end','start')-f('end','end')
        if metric=='A':return f(r,d,alpha)-f(r,d)
        if metric=='T':return f('end',d,alpha)-f('end',d)-f('start',d,alpha)+f('start',d)
        return dict(V_start=v0,V_end=v1,Q=v1-v0,S_start=-v0,S_end=v1,diagonal_zero_damage_change=f('end','end',0.)-f('end','end')-f('start','start',0.)+f('start','start'))[metric]
    for row in c._read_csv(OUT/'sparse_unpruned_contrasts.csv'):
        s,p,metric=int(row['seed']),row['mode'],row['metric'];d=row.get('donor','');alpha=float(row.get('alpha') or 1.)
        sv=independent(s,'sparse',p,metric,d=d,alpha=alpha);uv=independent(s,'unpruned',p,metric,d=d,alpha=alpha)
        near(row['sparse_value'],sv);near(row['unpruned_value'],uv);near(row['sparse_minus_unpruned'],sv-uv)
    for row in c._read_csv(OUT/'routing_mode_contrasts.csv'):
        s,m,metric=int(row['seed']),row['model_type'],row['metric'];r=row.get('recipient','');d=row.get('donor','');alpha=float(row.get('alpha') or 1.)
        nv=independent(s,m,MODES[0],metric,r,d,alpha);rv=independent(s,m,MODES[1],metric,r,d,alpha)
        near(row['native_value'],nv);near(row['replay_value'],rv);near(row['native_minus_replay'],nv-rv)
    batches=c._read_csv(OUT/'per_batch_losses.csv');c._need(len(batches)==12400,'Per-batch grid incomplete')
    for key in expected:
        s,m,p,r,d,a=key;rs=[b for b in batches if (int(b['seed']),b['model_type'],b['mode'],b['recipient'],b['donor'],float(b['alpha']))==key]
        c._need(len(rs)==31 and len(set(int(b['batch_index']) for b in rs))==31 and sum(int(b['target_tokens']) for b in rs)==492416,'Batch coverage invalid');near(L[key],sum(float(b['nll_sum']) for b in rs)/492416)
    # Prior diagonal result is an independent registered contrast, in both modes.
    prior=c._read_csv(ROOT/'results/imp_expert_contribution_ablation_v1/seed_level_temporal_contrasts.csv');maxerr=0.
    for row in c._read_csv(OUT/'contribution_substitution_effects.csv'):
        p='BASELINE_REPLAY' if row['mode']==MODES[1] else 'NATIVE_RESPONSE'
        old=next(r for r in prior if (r['seed'],r['model_type'],r['mode'],r['intervention'])==(row['seed'],row['model_type'],p,'ZERO_L0'))
        error=abs(float(row['diagonal_zero_damage_change'])-float(old['delta_A']));c._need(error<=1e-6,'Prior diagonal ZERO_L0 change mismatch');maxerr=max(maxerr,error)
    return dict(status='PASS',logical_cells=400,per_batch_records=12400,all_estimand_identities=True,alpha_zero_donor_equivalence=True,previous_diagonal_max_ce_error=maxerr)

def write_outputs(readouts):
    absolute=[];batches=[];stats=[]
    for result in readouts:
        ident=result['identity'];rs=result['batches'];c._need(result['checks']['status']=='PASS','Unvalidated pair')
        c._need(len(rs)==1240 and len(set((r['recipient'],r['donor'],r['alpha'],r['mode'],r['batch_index']) for r in rs))==1240,'Pair incomplete')
        batches+=rs;stats+=result['interface']
        for r in TIMES:
            for d in TIMES:
                for alpha in ALPHAS:
                    for p in MODES:
                        selected=[b for b in rs if (b['recipient'],b['donor'],b['alpha'],b['mode'])==(r,d,alpha,p)];n=sum(b['target_tokens'] for b in selected);c._need(n==492416 and len(selected)==31,'Invalid validation counts')
                        nll=sum(b['nll_sum'] for b in selected)
                        absolute.append(dict(seed=ident['seed'],model_type=ident['model_type'],recipient=r,donor=d,recipient_checkpoint=json.dumps(ident['checkpoints'][r],sort_keys=True),donor_checkpoint=json.dumps(ident['checkpoints'][d],sort_keys=True),alpha=alpha,mode=p,task_ce=nll/n,nll_sum=nll,target_tokens=n,batches=len(selected),reused_alpha_zero=alpha==0 and d=='end',precision='FP32_TF32off',seal_sha256=ident['seal_sha256']))
    sub,atten,sensitivity,sparse,routing=estimands(absolute)
    pooled=pooled_interface(stats);stats+=pooled
    aggregate=summaries([(sub,['model_type','mode'],['V_start','V_end','Q','S_start','S_end','diagonal_zero_damage_change']),(atten,['model_type','mode','recipient','donor','alpha'],['A']),(sensitivity,['model_type','mode','donor','alpha'],['T']),(sparse,['mode','metric','donor','alpha'],['sparse_minus_unpruned']),(routing,['model_type','metric','recipient','donor','alpha'],['native_minus_replay']),([r for r in pooled if r['kind']=='validation_native_pair'],['model_type','kind','quantity','population'],['start_rms','end_rms','relative_l2','norm_ratio','cosine_global'])])
    for name,rs in [('absolute_losses',absolute),('contribution_substitution_effects',sub),('attenuation_effects',atten),('recipient_context_sensitivity',sensitivity),('sparse_unpruned_contrasts',sparse),('routing_mode_contrasts',routing),('interface_statistics',stats),('per_batch_losses',batches),('aggregate_summaries',aggregate)]:c._csv_write(OUT/(name+'.csv'),rs)
    verified=verify_exported()
    findings(absolute,sub,atten,sensitivity,sparse,pooled,aggregate)
    figures(absolute,sub,atten,sensitivity,sparse)
    return verified

def describe(vals):return f"mean {statistics.mean(vals):.6f}, median {statistics.median(vals):.6f}, range [{min(vals):.6f}, {max(vals):.6f}], sample SD {statistics.stdev(vals):.6f}"
def findings(absolute,sub,atten,sensitivity,sparse,pooled,aggregate):
    lines=['# First-MoE interface experiment\n',LIMITS,'\nAll effects below use source seed as the replication unit. Individual seeds appear before descriptive summaries. Positive V means the recovered contribution helps; positive T means increased recipient-context sensitivity to the same donor perturbation.\n']
    for p in MODES:
        lines.append('\n## '+p+'\n')
        for m in c.MODEL_TYPES:
            lines.append('\n### '+m+'\n')
            for seed in c.SEEDS:
                row=next(r for r in sub if (r['seed'],r['model_type'],r['mode'])==(seed,m,p))
                lines.append(f"Seed {seed}: native start CE={row['L_start_start']:.6f}, native end CE={row['L_end_end']:.6f}; start<-end CE={row['L_start_end']:.6f}, end<-start CE={row['L_end_start']:.6f}; V(start)={row['V_start']:.6f}, V(end)={row['V_end']:.6f}, Q={row['Q']:.6f}; S_start={row['S_start']:.6f}, S_end={row['S_end']:.6f}; prior diagonal damage change={row['diagonal_zero_damage_change']:.6f}.\n")
            selected=[r for r in sub if r['model_type']==m and r['mode']==p]
            for metric in ('V_start','V_end','Q'):lines.append(metric+': '+describe([r[metric] for r in selected])+'.\n')
            patterns={key:0 for key in ('both recipients','end recipient only','start recipient only','neither recipient')}
            for row in selected:
                key='both recipients' if row['V_start']>0 and row['V_end']>0 else 'end recipient only' if row['V_end']>0 else 'start recipient only' if row['V_start']>0 else 'neither recipient';patterns[key]+=1
            lines.append('Recovered-contribution benefit by seed: '+', '.join(f'{k}: {v}/5' for k,v in patterns.items())+'. Zero values, if present, mean no measured CE benefit, not evidence of equivalence.\n')
            lines.append(f"Reciprocal substitutions hurt in both directions for {sum(r['S_start']>0 and r['S_end']>0 for r in selected)}/5 seeds. Such compatibility can involve residual/output coordinates, scaling and subsequent use.\n")
            for d in TIMES:
                for alpha in (.9,.75,.5,0.):
                    rs=[r for r in sensitivity if (r['model_type'],r['mode'],r['donor'],r['alpha'])==(m,p,d,alpha)]
                    vals=[r['T'] for r in rs]
                    lines.append(f"Fixed {d} donor, alpha={alpha}: T in seed order 7,17,29,43,61 = "+', '.join(f'{v:.6f}' for v in vals)+f"; {describe(vals)}; increased in {sum(v>0 for v in vals)}/5 seeds. {'This is the prespecified modest-attenuation contrast.' if alpha==.9 else 'Response-curve point, not an independent replication.'}\n")
            for d in TIMES:
                modest=[r['T'] for r in sensitivity if (r['model_type'],r['mode'],r['donor'],r['alpha'])==(m,p,d,.9)]
                removal=[r['T'] for r in sensitivity if (r['model_type'],r['mode'],r['donor'],r['alpha'])==(m,p,d,0.)]
                lines.append(f"Modest attenuation versus removal for fixed {d} donor: alpha=.9 T is positive in {sum(v>0 for v in modest)}/5 seeds (mean {statistics.mean(modest):.6f}); alpha=0 T is positive in {sum(v>0 for v in removal)}/5 (mean {statistics.mean(removal):.6f}). These are different perturbation sizes, not independent replications or a locally linear extrapolation.\n")
            # No monotonicity or optimum assumption; all signs are retained.
            negative=[r for r in atten if r['model_type']==m and r['mode']==p and r['alpha']!=1 and r['A']<0]
            lines.append(f"Attenuation improves CE (negative A) in {len(negative)} of 80 nontrivial attenuation cells; full signed curves are in attenuation_effects.csv. Native alpha=1 is not assumed optimal.\n")
        lines.append('\n### Same-seed sparse-minus-unpruned contrasts\n')
        for metric in ('V_start','V_end','Q','T'):
            for d in (TIMES if metric=='T' else ('',)):
                rs=[r for r in sparse if r['mode']==p and r['metric']==metric and (metric!='T' or r['donor']==d and r['alpha']==.9)]
                vals=[r['sparse_minus_unpruned'] for r in rs];lines.append(f"{metric} {d} {'alpha=.9' if metric=='T' else ''}: seed values "+', '.join(f'{v:.6f}' for v in vals)+f'; {describe(vals)}; sparse is larger in {sum(v>0 for v in vals)}/5 seeds.\n')
    lines.append('\n## Interface magnitude and routing interpretation\n')
    for m in c.MODEL_TYPES:
        for seed in c.SEEDS:
            pair=next(r for r in pooled if r['kind']=='validation_native_pair' and r['quantity']=='c' and (r['seed'],r['model_type'])==(seed,m))
            sizes=[next(r for r in pooled if r['kind']=='validation_substitution' and (r['seed'],r['model_type'],r['recipient'],r['donor'],r['alpha'],r['mode'])==(seed,m,'start',d,.9,MODES[0])) for d in TIMES]
            lines.append(f"Seed {seed} {m}: contribution start/end RMS {pair['start_rms']:.6f}/{pair['end_rms']:.6f}, end/start norm ratio {pair['norm_ratio']:.6f}, relative L2 change {pair['relative_l2']:.6f}, global cosine {pair['cosine_global']:.6f}; actual alpha=.9 attenuation displacement RMS start/end donor {sizes[0]['attenuation_displacement_rms']:.6f}/{sizes[1]['attenuation_displacement_rms']:.6f}. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.\n")
    for m in c.MODEL_TYPES:
        for d in TIMES:
            a=[next(r['T'] for r in sensitivity if (r['seed'],r['model_type'],r['mode'],r['donor'],r['alpha'])==(s,m,MODES[0],d,.9)) for s in c.SEEDS]
            b=[next(r['T'] for r in sensitivity if (r['seed'],r['model_type'],r['mode'],r['donor'],r['alpha'])==(s,m,MODES[1],d,.9)) for s in c.SEEDS]
            lines.append(f"{m}, fixed {d} donor alpha=.9: native/replay T signs agree in {sum((x>0)==(y>0) for x,y in zip(a,b))}/5 seeds. Native-minus-replay differences: "+', '.join(f'{x-y:.6f}' for x,y in zip(a,b))+'. Replay fixes each recipient’s own downstream decisions, not a common routing policy across recipients.\n')
    lines.append('\nAll five seeds are retained. The per-seed ranges above show concentration and heterogeneity without successful-seed filtering, thresholds, significance tests, or fitted classifiers. The previous diagonal ZERO_L0 result is reproduced independently and must not be substituted for fixed-donor T. Absolute CE matrices and attenuation curves are shown alongside changes in six figures.\n')
    (OUT/'findings.md').write_text('\n'.join(lines))

def figures(absolute,sub,atten,sensitivity,sparse):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    def save(fig,n):fig.tight_layout();fig.savefig(folder/f'{n}.png',dpi=170);plt.close(fig)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    fig,axes=plt.subplots(4,5,figsize=(16,11))
    for row,(p,m) in enumerate((p,m) for p in MODES for m in c.MODEL_TYPES):
        for col,s in enumerate(c.SEEDS):
            ax=axes[row,col];a=np.array([[next(z['task_ce'] for z in absolute if (z['seed'],z['model_type'],z['mode'],z['recipient'],z['donor'],z['alpha'])==(s,m,p,r,d,1.)) for d in TIMES] for r in TIMES]);ax.imshow(a,cmap='viridis');ax.set_xticks([0,1],TIMES);ax.set_yticks([0,1],TIMES)
            for i in range(2):
                for j in range(2):ax.text(j,i,f'{a[i,j]:.4f}',ha='center',va='center',color='white',fontsize=9)
            ax.set_title(f'{m} seed {s}\n{p}',fontsize=7);ax.set_xlabel('Donor');ax.set_ylabel('Recipient')
    save(fig,1)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,m in enumerate(c.MODEL_TYPES):
        for j,p in enumerate(MODES):
            ax=axes[i,j]
            for s,color in zip(c.SEEDS,colors):
                row=next(r for r in sub if (r['seed'],r['model_type'],r['mode'])==(s,m,p));ax.plot([0,1],[row['V_start'],row['V_end']],marker='o',color=color,label=str(s))
            ax.set_xticks([0,1],TIMES);ax.set_title(m+' '+p,fontsize=9);ax.set_ylabel('V: recovered contribution benefit');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=7)
    save(fig,2)
    fig,axes=plt.subplots(4,4,figsize=(17,13))
    for row,(m,p) in enumerate((m,p) for m in c.MODEL_TYPES for p in MODES):
        for col,(d,metric) in enumerate((d,metric) for d in TIMES for metric in ('task_ce','A')):
            ax=axes[row,col]
            for s,color in zip(c.SEEDS,colors):
                for r,style in [('start','--'),('end','-')]:
                    vals=[next(z[metric] for z in atten if (z['seed'],z['model_type'],z['mode'],z['recipient'],z['donor'],z['alpha'])==(s,m,p,r,d,a)) for a in ALPHAS];ax.plot(ALPHAS,vals,style,marker='.',color=color,label=f'{s} {r}')
            ax.set_title(f'{m} {p}\n{d} donor',fontsize=7);ax.set_xlabel('Alpha');ax.set_ylabel(metric);ax.axhline(0,color='gray',lw=.5)
            if row==0:ax.legend(fontsize=5,ncol=2)
    save(fig,3)
    fig,axes=plt.subplots(2,4,figsize=(17,7))
    for i,m in enumerate(c.MODEL_TYPES):
        for j,(p,d) in enumerate((p,d) for p in MODES for d in TIMES):
            ax=axes[i,j]
            for s,color in zip(c.SEEDS,colors):ax.plot(ALPHAS,[next(z['T'] for z in sensitivity if (z['seed'],z['model_type'],z['mode'],z['donor'],z['alpha'])==(s,m,p,d,a)) for a in ALPHAS],marker='.',color=color,label=str(s))
            ax.set_title(f'{m} {p}\n{d} donor',fontsize=7);ax.set_xlabel('Alpha');ax.set_ylabel('Fixed-donor T');ax.axhline(0,color='gray',lw=.6);ax.axvline(.9,color='gray',lw=.5);ax.legend(fontsize=6)
    save(fig,4)
    fig,axes=plt.subplots(1,2,figsize=(12,5));labels=['V_start','V_end','Q','T start .9','T end .9']
    for ax,p in zip(axes,MODES):
        for s,color in zip(c.SEEDS,colors):
            vals=[next(z['sparse_minus_unpruned'] for z in sparse if (z['seed'],z['mode'],z['metric'])==(s,p,metric)) for metric in labels[:3]]+[next(z['sparse_minus_unpruned'] for z in sparse if (z['seed'],z['mode'],z['metric'],z.get('donor'),z.get('alpha'))==(s,p,'T',d,.9)) for d in TIMES];ax.plot(range(5),vals,marker='o',color=color,label=str(s))
        ax.set_xticks(range(5),labels,rotation=20);ax.set_title(p,fontsize=9);ax.set_ylabel('Sparse minus unpruned CE contrast');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=7)
    save(fig,5)
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for i,m in enumerate(c.MODEL_TYPES):
        for j,metric in enumerate(('V_start','V_end','Q')):
            ax=axes[i,j]
            for s,color in zip(c.SEEDS,colors):
                vals=[next(r[metric] for r in sub if (r['seed'],r['model_type'],r['mode'])==(s,m,p)) for p in MODES];ax.scatter(*vals,color=color,label=str(s))
            lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],color='gray',lw=.7);ax.set_title(m+' '+metric);ax.set_xlabel('Native response');ax.set_ylabel('Recipient baseline replay');ax.legend(fontsize=7)
    save(fig,6)

def bundle():
    with tarfile.open(OUT/'summary.tar.gz','w:gz') as archive:
        for name in ('findings.md','completion.json','numerical_checks.json','absolute_losses.csv','contribution_substitution_effects.csv','attenuation_effects.csv','recipient_context_sensitivity.csv','sparse_unpruned_contrasts.csv','routing_mode_contrasts.csv','interface_statistics.csv'):archive.add(OUT/name,arcname=name)
