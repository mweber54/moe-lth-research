"""One-worker fixed32-batch inference panel with compact resumable batch caches."""
from reliability_common import *
import argparse,traceback


def native(m,batch,experts,stats,zero_expert=None):
    start=time.perf_counter();out=dc.evaluate(m,[batch],experts,zero_expert);torch.cuda.synchronize();stats['native_complete_batch_forwards']+=1;stats['native_evaluation_wall_seconds']+=time.perf_counter()-start;return out


def measure_batch(m,kind,t,b,batch,reg,stats):
    path=cache_path(kind,t,b);identity=cache_identity(kind,t,b)
    if path.exists():
        cached=load(path);assert cached['identity']==identity and cached['verified'];stats['cached_batches_reused']+=1;return cached
    j=s.jobs()[kind];masks=s.load_masks(j['mask']) if j['mask'] else {};expected=reg['checkpoints'][j['id']][str(t)]['model_hash'];assert s.state_dict_hash(m.state_dict())==expected
    baseline=native(m,batch,s.EXPERTS,stats);zero=native(m,batch,s.EXPERTS,stats);dc.check_baseline(baseline,zero)
    if b in (0,16):
        plain=s.evaluate(m,[batch],details=True);dc.check_baseline(baseline,plain);stats['native_complete_batch_forwards']+=1
    coverage={e:coverage_stats(baseline,e) for e in s.EXPERTS};total=sum(p.numel() for p in m.parameters());active=total-sum(x.numel()-int(x.sum()) for x in masks.values());eligible=sum(int(x.sum()) for x in masks.values()) if masks else sum(p.numel() for n,p in m.named_parameters() if '.experts.' in n and n.endswith(('.fc1.weight','.fc2.weight')))
    rows=[];seqstats={};checks=[];functionrows=[];panel=interventions()
    first_fours={e:next(x['intervention'] for x in panel if x['expert']==e and x['dose_groups']==4) for e in s.EXPERTS}
    for item in panel:
        e=item['expert'];n=item['parameter'];iid=item['intervention'];idx=item['indices'];control=item['kind']=='expert_output_zero';dose=item['dose_groups'];p=dict(m.named_parameters())[n];before=p.detach().clone();local=int(masks[n].sum()) if masks else p.numel()
        repeat=False;repeat_error=None
        if not control and dose==0:
            out=zero
        else:
            with s.deletion(m,{} if control else {n:idx}):
                out=native(m,batch,[e],stats,zero_expert=e if control else None)
                for key in ('ids','accepted','gates'):assert torch.equal(baseline[key],out[key]),key
                first4=first_fours[e]
                if b in (0,16) and (control or iid==first4):
                    again=native(m,batch,[e],stats,zero_expert=e if control else None);repeat=True;repeat_error=float((out['loss'].double()-again['loss'].double()).abs().max());assert repeat_error==0
                    assert all(torch.equal(a['output'],q['output']) for a,q in zip(out['outputs'][e],again['outputs'][e]))
        assert torch.equal(before,p) and p.grad is None and not m.blocks[1].moe.experts[e]._forward_hooks
        accepted=(baseline['ids']==e)&baseline['accepted'];delta=out['loss'].double()-baseline['loss'].double()
        if not control and dose==0:assert torch.count_nonzero(delta)==0
        outblocks=out['outputs'][e];baseblocks=baseline['outputs'][e];assert all(torch.equal(a['positions'],q['positions']) and torch.equal(a['gates'],q['gates']) for a,q in zip(baseblocks,outblocks))
        energy=energies(baseblocks,[x['output'] for x in outblocks]);subsets,sequence=subset_statistics(delta,baseline['loss'],out['loss'],accepted)
        meta=dict(seed=17,trajectory=kind,step=t,expert=e,layer=1,batch=b,block=b//4+1,partition='discovery' if b<16 else 'confirmation',intervention=iid,intervention_kind=item['kind'],dose_groups=dose,coordinates=None if control else len(idx),active_matrix_coordinates=local,local_active_weight_fraction=None if control else len(idx)/local,model_parameters=total,model_active_parameters=active,model_wide_total_fraction=None if control else len(idx)/total,model_wide_active_fraction=None if control else len(idx)/active,all_active_expert_weights=eligible,model_wide_active_expert_fraction=None if control else len(idx)/eligible,baseline_auxiliary=float(baseline['auxiliary_batch_means'][0]),deleted_auxiliary=float(out['auxiliary_batch_means'][0]))
        rows.extend(dict(**meta,**sub,**coverage[e],**energy) for sub in subsets)
        functionrows.append(dict(**meta,**energy));seqstats[iid]=sequence
        checks.append(dict(intervention=iid,restored_exactly=True,repeat_tested=repeat,repeat_loss_max_abs=repeat_error,target_layer_native=True,no_gradient=True,hooks_removed=True))
    mask_check(m,masks);assert s.state_dict_hash(m.state_dict())==expected
    compactbase=dict(loss=baseline['loss'],ids=baseline['ids'].to(torch.int16),accepted=baseline['accepted'],gates=baseline['gates'],auxiliary_batch_means=baseline['auxiliary_batch_means'])
    cached=dict(identity=identity,verified=True,rows=rows,baseline=compactbase,paired_sequence_statistics=seqstats,checks=checks,coverage=coverage,functional=functionrows,full_model_hash_preserved=True,mask_integrity=True,zero_dose_equal=True,RNG_modes_gradients_preserved=True)
    s.tensor_save(path,cached);save(D/'progress.json',dict(trajectory=kind,step=t,batch=b,block=b//4+1,unix=time.time()));return cached


def emit_tables():
    rows=[];coverage=[];checks=[]
    for path in sorted((D/'cache').glob('*.pt')):
        cache=load(path);assert cache['verified'];ident=cache['identity'];checkpoint=ident['checkpoint'];meta=dict(trajectory='sparse' if 'sparse_' in path.name else 'unpruned',step=int(path.name.split('_')[1]),batch=ident['batch']['batch'],block=ident['batch']['block'],partition=ident['batch']['partition'])
        rows.extend(cache['rows']);coverage.extend(dict(**meta,expert=e,**v) for e,v in cache['coverage'].items())
        checks.extend(dict(**meta,**v) for v in cache['checks'])
    table(D/'deletion_damage_by_batch.csv',rows)
    # Blocks aggregate conditional sums/counts, never means of differently weighted batches.
    import pandas as pd
    frame=pd.DataFrame(rows);blockrows=[];coverageframe=pd.DataFrame(coverage);coverageout=[]
    for keys,z in coverageframe.groupby(['trajectory','step','expert','block']):
        kind,t,e,block=keys;counts=z.accepted_tokens.to_numpy();n=int(counts.sum());props=int(z.proposed_tokens.sum());hhi=float(np.sum((counts/n)**2)) if n else None
        coverageout.append(dict(level='block',trajectory=kind,step=int(t),expert=int(e),block=int(block),partition='discovery' if block<=4 else 'confirmation',batch=None,accepted_tokens=n,proposed_tokens=props,capacity_drops=props-n,acceptance_rate=n/props if props else None,contributing_batches=int((counts>0).sum()),complete_batches=len(z),max_batch_accepted_fraction=float(counts.max()/n) if n else None,accepted_batch_HHI=hhi,effective_contributing_batches=1/hhi if hhi else None,coverage='adequate' if n>=MIN_ACCEPTED else 'insufficient_accepted_positions'))
    coverageout.extend(dict(level='batch',**r,contributing_batches=int(r['accepted_tokens']>0),complete_batches=1,coverage='nonempty' if r['accepted_tokens'] else 'empty') for r in coverage)
    table(D/'coverage_by_batch_and_block.csv',coverageout)
    for keys,z in frame.groupby(['trajectory','step','expert','intervention','block','subset']):
        kind,t,e,iid,block,subset=keys;n=int(z.target_tokens.sum());first=z.iloc[0]
        frows=[{k:(None if pd.isna(r[k]) else r[k]) for k in ('output_elements','output_difference_energy','baseline_output_energy','gate_weighted_difference_energy','baseline_gate_weighted_energy','relative_output_rms_difference')} for r in z.to_dict('records')]
        blockrows.append(dict(trajectory=kind,step=int(t),expert=int(e),intervention=iid,intervention_kind=first.intervention_kind,dose_groups=int(first.dose_groups) if pd.notna(first.dose_groups) else None,block=int(block),partition='discovery' if block<=4 else 'confirmation',subset=subset,target_tokens=n,baseline_loss_sum=float(z.baseline_loss_sum.sum()),deleted_loss_sum=float(z.deleted_loss_sum.sum()),damage_sum=float(z.damage_sum.sum()),damage=float(z.damage_sum.sum())/n if n else None,baseline_ce=float(z.baseline_loss_sum.sum())/n if n else None,deleted_ce=float(z.deleted_loss_sum.sum())/n if n else None,accepted_tokens=int(z.accepted_tokens.sum()),contributing_batches=int((z.accepted_tokens>0).sum()),**aggregate_energies(frows)))
    table(D/'deletion_damage_by_block.csv',blockrows)
    save(D/'numerical_checks.json',dict(passed=True,cache_batches=len(list((D/'cache').glob('*.pt'))),zero_dose_equivalence=True,native_baseline_equivalence_on_batches_0_16=True,paired_FP64_sums=True,model_hashes_restored=True,IMP_mask_integrity=True,unchanged_RNG_modes_gradients=True,training_updates=0,checks=checks))


def run(stage):
    reg=verify_frozen();assert reg['batches']==32;probes=load(D/'probes.pt');started=time.perf_counter();torch.set_num_threads(2);assert torch.cuda.is_available()
    stats=dict(native_complete_batch_forwards=0,native_evaluation_wall_seconds=0.,cached_batches_reused=0)
    if stage=='profile':
        assert not (D/'profile.json').exists(),'Already profiled'
        kind='sparse';t=1000;m=s.model_for(s.jobs()[kind],t);before=time.perf_counter();measure_batch(m,kind,t,0,probes[0],reg,stats);elapsed=time.perf_counter()-before
        cachebytes=cache_path(kind,t,0).stat().st_size
        profile=dict(profile_context='registered_batch0_only',trajectory=kind,step=t,full_panel_batch_seconds=elapsed,estimated_measurement_seconds=elapsed*256,estimated_compact_cache_bytes=cachebytes*256,gpu=torch.cuda.get_device_name(),one_GPU_worker=True,stats=stats,profile_wall_seconds=time.perf_counter()-started,panel_or_probes_changed=False)
        save(D/'profile.json',profile);print('Profile full-panel batch',round(elapsed,3),'seconds; estimate',round(elapsed*256,1),'seconds',flush=True);return
    assert (D/'profile.json').exists(),'Profile must precede full pass'
    for kind,j in s.jobs().items():
      for t in s.STEPS:
        m=s.model_for(j,t)
        for b,batch in enumerate(probes):
            measure_batch(m,kind,t,b,batch,reg,stats)
            print(kind,t,'batch',b+1,'/32 verified',flush=True)
        del m;torch.cuda.empty_cache()
    emit_tables();accounting=s.read(D/'compute_accounting.json');accounting.update(profile=s.read(D/'profile.json'),measure_wall_seconds=time.perf_counter()-started,measurement_stats=stats,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),new_cache_bytes=sum(p.stat().st_size for p in (D/'cache').glob('*.pt')),storage_bytes_at_measure_end=sum(p.stat().st_size for p in D.rglob('*') if p.is_file()));save(D/'compute_accounting.json',accounting)
    save(D/'measurement_completion.json',dict(complete=True,unique_context_batches=32,checkpoint_batch_caches=256,training_updates=0,unix=time.time()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['profile','measure']);a=ap.parse_args()
    with worker(),s.protected():
        try:run(a.stage)
        except BaseException as e:
            save(D/f'failure_{a.stage}_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()));raise
