"""Bounded, checkpoint-only contribution ablation. No training APIs are called."""
from __future__ import annotations
import argparse, hashlib, json, statistics, sys, time, tarfile
from contextlib import contextmanager
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.imp_capacity_diagnostic_v1 import run as c
from scripts.imp_expert_contribution_ablation_v1 import replay
import torch
OUT = ROOT/'results/imp_expert_contribution_ablation_v1'
MODES = ('NATIVE_RESPONSE','BASELINE_REPLAY')
QS = ('BASELINE','ZERO_L0','ZERO_L1','ZERO_L2','ZERO_L3','ZERO_ALL')
write = c._json_write
need = c._need

@contextmanager
def intervention(model, q):
    """Replace only the complete MoE module output, before block residual addition."""
    handles=[]; norms={}; residuals={}
    selected = set(range(len(model.blocks))) if q=='ZERO_ALL' else ({int(q[-1])} if q.startswith('ZERO_L') else set())
    def hook(layer):
        def apply(module, args, result):
            contribution, aux, trace = result
            value = torch.zeros_like(contribution) if layer in selected else contribution
            norms[layer]={'pre_sumsq':float(contribution.double().square().sum()), 'post_sumsq':float(value.double().square().sum()),'elements':value.numel()}
            if layer in selected: need(torch.count_nonzero(value)==0,'Nonzero ablated contribution')
            return value, aux, trace
        return apply
    for i,b in enumerate(model.blocks):
        def norm_input(module,args,i=i):
            x=args[0]; residuals[i]={'residual_sumsq':float(x.double().square().sum()),'residual_elements':x.numel()}
        handles.append(b.moe_norm.register_forward_pre_hook(norm_input))
        handles.append(b.moe.register_forward_hook(hook(i)))
    try: yield norms,residuals
    finally:
        for h in handles: h.remove()

def run_forward(model,x,q,mode,payload=None,key=None):
    with intervention(model,q) as (norms,residuals):
        output = model(x) if mode==MODES[0] else replay.forward(model,x,payload,key['step'],key['batch_hash'])
    return output,[dict(layer=i,**norms[i],**residuals[i]) for i in range(len(model.blocks))]

def evaluate(state,masks,config,batches,identity,device,profile=False):
    entry=c._rng_snapshot(); model=c.TinyMoELanguageModel(config).to(device); model.load_state_dict(state)
    replay.install(model); c._restore_rng(entry)
    before=c.state_dict_hash(model.state_dict()); grads=c._gradient_signature(model)
    modes=[(m,m.training) for m in model.modules()]
    tied=model.lm_head.weight is model.token_embedding.weight
    records=[]; routing=[]; predictions=[]; norms=[]; checks={}; seconds=[]
    try:
        model.eval()
        with torch.inference_mode(),c._precision_guard():
            for bi,(xx,yy) in enumerate(batches):
                x,y=xx.to(device),yy.to(device); batch_hash=c.common.batch_hash([(xx,yy)])
                key={'step':identity['offset']+1000,'batch_hash':batch_hash}
                start=time.perf_counter()
                baseline,bnorm=run_forward(model,x,'BASELINE',MODES[0])
                payload=replay.capture(model,baseline,x,key['step'],batch_hash)
                base_logits=baseline.logits; native_routes=baseline.routes
                all_logits=None
                for mode in MODES:
                    for q in QS:
                        tick=time.perf_counter()
                        if mode==MODES[0] and q=='BASELINE': output,ns=baseline,bnorm
                        else: output,ns=run_forward(model,x,q,mode,payload,key)
                        logits=output.logits
                        need(torch.isfinite(logits).all(),'Nonfinite logits')
                        if q=='BASELINE':
                            need(torch.allclose(logits,base_logits,atol=c.MODEL_ATOL,rtol=c.MODEL_RTOL),'Replay/inactive baseline prediction mismatch')
                        if q=='ZERO_ALL':
                            if mode==MODES[0]: all_logits=logits.detach().clone()
                            else: need(torch.allclose(logits,all_logits,atol=c.MODEL_ATOL,rtol=c.MODEL_RTOL),'ZERO_ALL routing modes disagree')
                        if q.startswith('ZERO_L'):
                            for layer in range(int(q[-1])+1):
                                need(torch.equal(output.pre_router_hidden_states[layer],baseline.pre_router_hidden_states[layer]),'Single-layer upstream hidden state changed')
                        for layer,(route,ref) in enumerate(zip(output.routes,native_routes)):
                            ids=route.selected_expert_indices; gates=route.selected_probabilities; accepted=route.accepted_mask
                            if mode==MODES[1]:
                                for a,b in ((ids,ref.selected_expert_indices),(gates,ref.selected_probabilities),(accepted,ref.accepted_mask)):
                                    need(torch.equal(a,b),'Imposed route mismatch')
                            proposed=torch.bincount(ids.flatten(),minlength=model.blocks[layer].moe.num_experts).tolist()
                            accepted_load=torch.bincount(ids.reshape_as(accepted)[accepted],minlength=len(proposed)).tolist()
                            routing.append(dict(**identity,batch_index=bi,mode=mode,intervention=q,layer=layer,routing_kind='imposed routing' if mode==MODES[1] else 'native response',proposed_loads=json.dumps(proposed),accepted_loads=json.dumps(accepted_load),drop_rate=float((~accepted).float().mean()),assignment_agreement=float((ids==ref.selected_expert_indices).float().mean()),gate_mean_absolute_difference=float((gates-ref.selected_probabilities).abs().mean()),accepted_agreement=float((accepted==ref.accepted_mask).float().mean()),tokens=ids.numel()))
                        nll=float(c._ce_vector(logits,y).double().sum()); count=y.numel()
                        js,agree,n=c._js_sum_and_agreement(base_logits,logits)
                        records.append(dict(**identity,batch_index=bi,batch_sha256=batch_hash,mode=mode,intervention=q,nll_sum=nll,target_tokens=count,nonfinite=0))
                        predictions.append(dict(**identity,batch_index=bi,mode=mode,intervention=q,js_sum=js,agreement_count=agree,target_tokens=n,undefined_statistics=0))
                        norms.extend(dict(**identity,batch_index=bi,mode=mode,intervention=q,pre_contribution_rms=(row['pre_sumsq']/row['elements'])**.5,post_contribution_rms=(row['post_sumsq']/row['elements'])**.5,residual_rms=(row['residual_sumsq']/row['residual_elements'])**.5,**row) for row in ns)
                        if device.type=='cuda': torch.cuda.synchronize()
                        seconds.append(time.perf_counter()-tick)
                        if output is not baseline: del output
                del baseline,base_logits,all_logits
                print(json.dumps({'checkpoint':identity['checkpoint_id'],'batch':bi,'seconds':time.perf_counter()-start}),flush=True)
    finally:
        for m,training in modes: m.training=training
        c._restore_rng(entry)
    need(before==c.state_dict_hash(model.state_dict()),'Model mutated')
    need(grads==c._gradient_signature(model),'Gradients mutated')
    need(c._rng_equal(entry,c._rng_snapshot()),'RNG mutated')
    need(all(m.training==t for m,t in modes),'Modes mutated')
    need(sum(b['target_tokens'] for b in records)==len(MODES)*len(QS)*sum(y.numel() for _,y in batches),'Token coverage mismatch')
    need(tied and model.lm_head.weight is model.token_embedding.weight,'Tied identity changed')
    for name,mask in masks.items(): need(torch.count_nonzero(dict(model.named_parameters())[name][~mask.to(device).bool()])==0,'Masked coordinates nonzero')
    checks=dict(status='PASS',baseline_replay_predictions=True,zero_all_modes=True,upstream_unchanged=True,imposed_routes_exact=True,states_gradients_rng_modes_ties_masks_preserved=True,finite=True)
    return dict(identity=identity,batches=records,routing=routing,predictions=predictions,norms=norms,checks=checks,forward_seconds=seconds)

LIMITS='''This prospective diagnostic is motivated by previously examined results; it is not project-level preregistration. The validation set has been repeatedly examined and is not an independent test. Five source seeds are replication units. Complete contribution removal creates potentially out-of-distribution internal states. Lower damage does not prove dispensability; higher damage does not prove semantic specialization. Improvement under ZERO_ALL does not establish that training without experts would achieve it: experts were present throughout historical training. Shared-only recovery does not allocate causal percentages to embeddings, attention, normalization, routing or output head. The tied embedding/output matrix has both input and prediction roles. Contribution RMS alone is not functional importance. No safety of further weight pruning, mechanism of ticket formation, or explanation of four final sparse/dense matches is established. Native-minus-replay includes gates and capacity response, not isolated assignment effects. Single-layer effects must not be summed as independent contributions.'''
METRICS='''A=L_intervention-L_baseline; delta_A=A(500)-A(0); X=delta_A_sparse-delta_A_unpruned. Positive A means removal hurts; positive delta_A means harm increases. R=L(0)-L(500); delta_A=R_baseline-R_intervention. G=L_sparse-L_unpruned; closure=G(0)-G(500). Native-minus-replay contrasts are reported with signs. CE=sum token NLL / actual target count. JSD uses natural logarithms and all token distributions; prediction agreement uses argmax. RMS=sqrt(sum squares/elements); residual means the post-attention, pre-MoE-addition residual (input to moe_norm). Imposed route statistics describe cached IDs, gates and acceptance; native probabilities computed internally in replay are not summarized as imposed routing. Batch diagnostics are not independent replications. Undefined statistics are counted explicitly; no clamping or outcome-dependent selection.\n\n'''+LIMITS

def seal(audit=True):
    OUT.mkdir(parents=True,exist_ok=True)
    # Audit implementation writes exclusively into this study's output directory.
    c.OUT=OUT
    provenance,inventory,variants=c.audit_sources()
    config=dict(analysis_id=OUT.name,seeds=list(c.SEEDS),model_types=list(c.MODEL_TYPES),offsets=[0,500],modes=list(MODES),interventions=list(QS),condition='router_experts_frozen',expected_logical_cells=240,validation=dict(target_tokens=492416,batches=31,rebatching=False,precision='FP32',tf32=False),tolerances=dict(ce_absolute=1e-6,logit_atol=1e-6,logit_rtol=1e-5),primary='ZERO_ALL delta_A and X',baseline_reference_sha256=c._sha256(ROOT/'results/imp_capacity_diagnostic_v1/absolute_losses.csv'),replay_reference='each exact checkpoint and complete original validation batch; never cross time or model',limitations=LIMITS,cache_unit='complete checkpoint; identity includes every batch, mode and intervention')
    files=[Path(__file__),Path(replay.__file__),Path(c.__file__),ROOT/'src/moe_lth/models/moe_layer.py',ROOT/'src/moe_lth/models/transformer.py',ROOT/'src/moe_lth/models/router.py',ROOT/'src/moe_lth/models/experts.py',ROOT/'tests/test_imp_expert_contribution_ablation_v1.py']
    sealed=dict(config=config,source_hashes={str(p.relative_to(ROOT)):c._sha256(p) for p in files},provenance_sha256=c._sha256(OUT/'checkpoint_and_validation_provenance.json'))
    path=OUT/'source_configuration_seal.json'
    if path.exists(): need(c._json_read(path)==sealed,'Source/configuration seal changed; preserve old attempt before correction')
    else: write(OUT/'analysis_config.json',config); (OUT/'metric_definitions.md').write_text(METRICS);write(path,sealed)
    return provenance,variants,c._sha256(path)

def outputs(readouts):
    absolute=[]; batches=[]; routing=[]; pred=[]; norms=[]
    for r in readouts:
        need(r['checks']['status']=='PASS','Unvalidated readout')
        need(len(r['batches'])==372,'Incomplete logical batch grid')
        need({(b['mode'],b['intervention'],b['batch_index']) for b in r['batches']}=={(p,q,i) for p in MODES for q in QS for i in range(31)},'Duplicate or absent logical batch cells')
        batches+=r['batches'];routing+=r['routing'];norms+=r['norms']
        for mode in MODES:
            for q in QS:
                rows=[b for b in r['batches'] if b['mode']==mode and b['intervention']==q]
                n=sum(b['target_tokens'] for b in rows); need(n==492416 and len(rows)==31,'Incomplete cache coverage')
                loss=sum(b['nll_sum'] for b in rows)
                absolute.append(dict(**r['identity'],mode=mode,intervention=q,task_ce=loss/n,total_nll_sum=loss,target_tokens=n,batches=len(rows)))
                ps=[p for p in r['predictions'] if p['mode']==mode and p['intervention']==q]
                pred.append(dict(**r['identity'],mode=mode,intervention=q,prediction_agreement=sum(p['agreement_count'] for p in ps)/n,mean_js_divergence=sum(p['js_sum'] for p in ps)/n,target_tokens=n,nonfinite=0,undefined_statistics=0))
    idx={(r['seed'],r['model_type'],r['offset'],r['mode'],r['intervention']):r['task_ce'] for r in absolute}
    def L(s,m,t,p,q): return idx[s,m,t,p,q]
    effects=[]; temporal=[]; gaps=[]; contrasts=[]; aggregate=[]
    for s in c.SEEDS:
        for m in c.MODEL_TYPES:
            for p in MODES:
                for q in QS:
                    for t in (0,500): effects.append(dict(seed=s,model_type=m,mode=p,intervention=q,offset=t,A=L(s,m,t,p,q)-L(s,m,t,p,'BASELINE')))
                    a0=L(s,m,0,p,q)-L(s,m,0,p,'BASELINE'); a1=L(s,m,500,p,q)-L(s,m,500,p,'BASELINE')
                    rb=L(s,m,0,p,'BASELINE')-L(s,m,500,p,'BASELINE'); rq=L(s,m,0,p,q)-L(s,m,500,p,q)
                    need(abs((a1-a0)-(rb-rq))<1e-12,'Temporal identity failed')
                    temporal.append(dict(seed=s,model_type=m,mode=p,intervention=q,delta_A=a1-a0,R_baseline=rb,R_intervention=rq))
        for p in MODES:
            for q in QS:
                ds=[r['delta_A'] for r in temporal if r['seed']==s and r['mode']==p and r['intervention']==q]
                g0=L(s,'sparse',0,p,q)-L(s,'unpruned',0,p,q);g1=L(s,'sparse',500,p,q)-L(s,'unpruned',500,p,q)
                bg0=L(s,'sparse',0,p,'BASELINE')-L(s,'unpruned',0,p,'BASELINE');bg1=L(s,'sparse',500,p,'BASELINE')-L(s,'unpruned',500,p,'BASELINE')
                gaps.append(dict(seed=s,mode=p,intervention=q,X=ds[0]-ds[1],G_start=g0,G_end=g1,gap_closure=g0-g1,baseline_gap_closure=bg0-bg1))
        for m in c.MODEL_TYPES:
            for q in QS:
                es=[next(r['A'] for r in effects if (r['seed'],r['model_type'],r['mode'],r['intervention'],r['offset'])==(s,m,p,q,t)) for t in (0,500) for p in MODES]
                contrasts.append(dict(seed=s,model_type=m,intervention=q,A_native_minus_replay_start=es[0]-es[1],A_native_minus_replay_end=es[2]-es[3],delta_A_native_minus_replay=(es[2]-es[3])-(es[0]-es[1])))
    for table,groups,metrics in [(temporal,['model_type','mode','intervention'],['delta_A','R_baseline','R_intervention']),(gaps,['mode','intervention'],['X','gap_closure']),(effects,['model_type','mode','intervention','offset'],['A'])]:
        for key in sorted(set(tuple(r[g] for g in groups) for r in table)):
            for metric in metrics:
                vs=[r[metric] for r in table if tuple(r[g] for g in groups)==key]
                aggregate.append(dict(zip(groups,key),metric=metric,n=len(vs),mean=statistics.mean(vs),median=statistics.median(vs),minimum=min(vs),maximum=max(vs),sample_sd=statistics.stdev(vs)))
    for name,rows in [('absolute_losses',absolute),('seed_level_ablation_effects',effects),('seed_level_temporal_contrasts',temporal),('sparse_unpruned_contrasts',gaps),('routing_mode_contrasts',contrasts),('per_batch_losses',batches),('routing_diagnostics',routing),('prediction_diagnostics',pred),('contribution_norms',norms),('aggregate_summaries',aggregate)]: c._csv_write(OUT/(name+'.csv'),rows)
    findings=['# Expert-contribution ablation\n',LIMITS,'\nIndividual seeds precede cohort summaries below. Positive delta_A means increased harm.\n']
    for p in MODES:
        findings.append('\n## '+p+'\n')
        for s in c.SEEDS:
            rows=[r for r in temporal if r['seed']==s and r['mode']==p and r['intervention']=='ZERO_ALL']
            x=next(r['X'] for r in gaps if r['seed']==s and r['mode']==p and r['intervention']=='ZERO_ALL')
            findings.append(f"Seed {s}: "+'; '.join(f"{r['model_type']} delta_A={r['delta_A']:.6f}, R_baseline={r['R_baseline']:.6f}, R_ZERO_ALL={r['R_intervention']:.6f}" for r in rows)+f'; X={x:.6f}.\n')
        for m in c.MODEL_TYPES:
            for q in QS[1:]:
                a=next(r for r in aggregate if r.get('model_type')==m and r.get('mode')==p and r.get('intervention')==q and r['metric']=='delta_A')
                findings.append(f"{m} {q}: delta_A mean {a['mean']:.6f}, median {a['median']:.6f}, min {a['minimum']:.6f}, max {a['maximum']:.6f}, sample SD {a['sample_sd']:.6f}.\n")
    negative=[r for r in effects if r['A']<0]
    findings.append(f'\nNet harmful expert contributions (negative A) occur in {len(negative)} logical cells; inspect signed seed-level table.\n')
    for s in c.SEEDS:
        for m in c.MODEL_TYPES:
            for p in MODES:
                for t in (0,500):
                    a={r['intervention']:r['A'] for r in effects if (r['seed'],r['model_type'],r['mode'],r['offset'])==(s,m,p,t)}
                    findings.append(f"Nonadditivity seed {s} {m} {p} offset {t}: joint minus sum single-layer damage={a['ZERO_ALL']-sum(a[q] for q in QS[1:5]):.6f}.\n")
    findings.append('\nRouting response contrasts for A and delta_A are in routing_mode_contrasts.csv. Absolute CE scales are in absolute_losses.csv and Figure 1; no percentages of causal recovery are assigned.\n')
    (OUT/'findings.md').write_text('\n'.join(findings))
    plots(absolute,effects,temporal,gaps,contrasts)
    write(OUT/'completion.json',dict(status='COMPLETE',scientifically_complete=True,validated_readouts=20,logical_cells=len(absolute),seeds=list(c.SEEDS)))

def plots(absolute,effects,temporal,gaps,contrasts):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for number,(rows,metric,title) in enumerate([(absolute,'task_ce','Absolute CE: baseline versus ZERO_ALL'),(effects,'A','ZERO_ALL ablation damage'),(temporal,'delta_A','Layer-specific temporal sensitivity'),(gaps,'X','Sparse-specific temporal sensitivity'),(contrasts,'delta_A_native_minus_replay','Native minus replay temporal sensitivity')],1):
        fig,ax=plt.subplots(figsize=(14,6))
        selected=[r for r in rows if (r['intervention'] in ('BASELINE','ZERO_ALL') if number==1 else r['intervention']=='ZERO_ALL' if number==2 else r['intervention']!='BASELINE')]
        labels=[]
        for i,r in enumerate(selected):
            label=' '.join(str(r.get(k,'')) for k in ('seed','model_type','mode','intervention','offset'))
            ax.scatter(i,r[metric],s=12); labels.append(label)
        ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=5);ax.set_title(title);ax.set_ylabel(metric);ax.axhline(0,color='gray',linewidth=.5);fig.tight_layout();fig.savefig(folder/f'{number}.png',dpi=160);plt.close(fig)

def bundle():
    names=['findings.md','completion.json','numerical_checks.json','absolute_losses.csv','seed_level_ablation_effects.csv','seed_level_temporal_contrasts.csv','sparse_unpruned_contrasts.csv','routing_mode_contrasts.csv']
    with tarfile.open(OUT/'summary.tar.gz','w:gz') as archive:
        for n in names:
            if (OUT/n).exists(): archive.add(OUT/n,arcname=n)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    provenance,variants,seal_hash=seal()
    if not args.evaluate:
        write(OUT/'completion.json',dict(status='SOURCE_AUDIT_COMPLETE',scientifically_complete=False));return
    import subprocess
    test=subprocess.run([sys.executable,'-m','pytest','tests/test_imp_expert_contribution_ablation_v1.py','tests/test_model.py','-q'],cwd=ROOT,capture_output=True,text=True)
    write(OUT/'numerical_checks.json',dict(status='PRECOHORT_PASS' if test.returncode==0 else 'FAIL',synthetic_output=test.stdout+test.stderr,return_code=test.returncode,cohort_evaluated=False))
    need(test.returncode==0,'Synthetic numerical tests failed')
    try: admission=c.gpu_admission_check()
    except RuntimeError as e:
        write(OUT/'completion.json',dict(status='RESOURCE_BLOCKED',scientifically_complete=False,reason=str(e),validated_readouts=0,logical_cells=0))
        write(OUT/'compute_accounting.json',dict(cohort_started=False,profile_available=False,reason=str(e)))
        (OUT/'findings.md').write_text('Evaluation has not run: '+str(e)+'\n\nNo scientific conclusions are available.\n\n'+LIMITS)
        bundle();print(str(e),flush=True);return
    write(OUT/'gpu_admission.json',admission)
    readouts=[]; started=time.perf_counter()
    with c.exclusive_claim(OUT/'evaluation.lock'):
        c.gpu_admission_check();device=torch.device('cuda')
        for seed in c.SEEDS:
            v=variants[seed];batches,vhash=c._validation_batches(seed,v)
            for m in c.MODEL_TYPES:
                for offset in (0,500):
                    state,masks,cid,chash=c._state_for(seed,m,offset,v)
                    identity=dict(seed=seed,model_type=m,offset=offset,checkpoint_id=cid,checkpoint_sha256=chash,state_sha256=c.state_dict_hash(state),mask_sha256=c._mask_hash(masks),validation_sha256=vhash,seal_sha256=seal_hash,precision='FP32_TF32off')
                    key=hashlib.sha256(json.dumps(dict(identity=identity,modes=MODES,interventions=QS,batches=[c.common.batch_hash([b]) for b in batches]),sort_keys=True).encode()).hexdigest()
                    path=OUT/'cache'/f'{key}.json'
                    if path.exists():
                        r=c._json_read(path);need(r['identity']==identity and r['checks']['status']=='PASS','Invalid cache')
                        need('registered_ce_absolute_error' in r['checks'] and r['checks']['registered_ce_absolute_error']<=1e-6,'Cache baseline not verified')
                    else:
                        if not (OUT/'precohort_native_baseline.json').exists():
                            entry=c._rng_snapshot(); pilot_model=c.TinyMoELanguageModel(v['config']['model']).to(device);pilot_model.load_state_dict(state);pilot_model.eval();total=0.;count=0
                            with torch.inference_mode(),c._precision_guard():
                                for x,y in batches:
                                    ce=c._ce_vector(pilot_model(x.to(device)).logits,y.to(device));total+=float(ce.double().sum());count+=y.numel()
                            c._restore_rng(entry);del pilot_model
                            registered=next(row for row in c._read_csv(c.ROOT/'results/imp_capacity_diagnostic_v1/absolute_losses.csv') if int(row['seed'])==seed and row['model_type']==m and int(row['offset'])==offset and row['policy']=='NATIVE')
                            error=abs(total/count-float(registered['task_ce']));need(error<=1e-6,'Precohort registered baseline CE mismatch')
                            write(OUT/'precohort_native_baseline.json',dict(identity=identity,task_ce=total/count,registered_ce=float(registered['task_ce']),absolute_error=error,target_tokens=count,batches=len(batches),status='PASS'))
                        if not (OUT/'evaluation_profile.json').exists():
                            tick=time.perf_counter();pilot=evaluate(state,masks,v['config']['model'],batches[:1],identity,device,True);elapsed=time.perf_counter()-tick
                            write(OUT/'evaluation_profile.json',dict(one_batch_seconds=elapsed,projected_cohort_seconds=elapsed*31*20,logical_cells=240,batches=31,peak_memory_bytes=torch.cuda.max_memory_allocated(),status='PASS'))
                        r=evaluate(state,masks,v['config']['model'],batches,identity,device)
                        baseline=next(row for row in c._read_csv(c.ROOT/'results/imp_capacity_diagnostic_v1/absolute_losses.csv') if int(row['seed'])==seed and row['model_type']==m and int(row['offset'])==offset and row['policy']=='NATIVE')
                        bs=[b for b in r['batches'] if b['mode']==MODES[0] and b['intervention']=='BASELINE']; ce=sum(b['nll_sum'] for b in bs)/sum(b['target_tokens'] for b in bs)
                        need(abs(ce-float(baseline['task_ce']))<=1e-6,'Registered baseline CE mismatch');r['checks']['registered_ce_absolute_error']=abs(ce-float(baseline['task_ce']))
                        write(path,r)
                    readouts.append(r)
                    write(OUT/'progress.json',dict(validated_readouts=len(readouts),expected=20))
    write(OUT/'numerical_checks.json',dict(status='PASS',synthetic_output=test.stdout,cohort_checks=[dict(identity=r['identity'],checks=r['checks']) for r in readouts]))
    write(OUT/'compute_accounting.json',dict(elapsed_seconds=time.perf_counter()-started,unique_checkpoints=20,logical_cells=240,baseline_cache_reused=False,routing_caches='within-checkpoint within-batch only',profile=c._json_read(OUT/'evaluation_profile.json')))
    outputs(readouts);bundle()
if __name__=='__main__':
    try: main()
    except Exception as e:
        write(OUT/'attempts'/f'failure_{time.time_ns()}.json',dict(error=repr(e),scientifically_complete=False))
        write(OUT/'completion.json',dict(status='FAILED_INVARIANT_OR_ENGINEERING',scientifically_complete=False,error=repr(e)))
        raise
