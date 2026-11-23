"""Prepare and execute the fixed, checkpoint-only expert-local dose panel."""
from dose_common import *
import argparse
import csv
import traceback

PROTOCOL='''# Expert-local deletion dose and reliability check v1

This is a measurement study using the completed Stage 1 engineering pilot, not a
specialization study. Seed 17 remains a NONMATCHING 32E IMP80 engineering case.
Use only its verified sparse and matched unpruned recovery steps 25,250,1000,2500;
2500 is the pre-reset endpoint. Historical/rerun/reset endpoint aliases count once.
Layer index 1, experts 0,8,16,24, fc1.weight and the original eight disjoint groups
are fixed. No additional seeds, training, optimizer, permanent pruning, recovery,
cluster refit, specialization predictor, or Stage 2 is launched.

Four group permutations use fixed numpy seeds 170901,170902,170903,170904, identically
across experts, time and trajectories. Prefix unions have 0,1,2,4,8 groups. Deduplicate
identical (expert,coordinate-set) unions before evaluation and preserve every
ordering/dose alias separately. The empty union and all-eight union are each unique
per expert. No previous global union cap applies: each intervention affects exactly
ONE expert fc1 matrix. Coordinates match between sparse and unpruned models, not
relative density. Report both sparse mask-active matrix denominator and unpruned
full-matrix denominator; actual nonzero counts are separately recorded per snapshot.

Reuse the complete, original 2 x 128 x 128 A and B training-only batches. Both have
already been examined and are diagnostic populations, not independent generalization
data. Preserve native sequence structure, gates, top1 routing and capacity. Eval,
FP32, TF32 disabled, inference mode, signed actual-token-weighted task CE; auxiliary
loss stays separate. Use original baseline acceptance for conditional and complement
subsets; attention/later layers can spread effects beyond accepted positions.

Capture baseline expert inputs and outputs in the actual native dispatch order in
each complete batch. For cached single-group CE interventions replay only the expert
function on these *same native-dispatch rows*, not routing on isolated tokens. These
rows and gates cannot change under a layer-1 fc1 intervention. Verify against
representative full-model interventions at every expert/checkpoint/probe. Raw output
RMS difference = sqrt(mean((deleted_output-baseline_output)^2)) over accepted positions
and output channels. Relative difference divides by baseline RMS only if >1e-12;
otherwise undefined_near_zero_baseline. No accepted positions -> undefined, never zero.
Gate-weighted RMS = sqrt(mean((baseline_gate * output_difference)^2)), same denominator.
No change to capacity population or gate normalization. Coverage <32 accepted positions
is low_traffic, retained rather than removed. This minimum is descriptive, not a test.

A separate positive control returns zeros from the selected expert module using a
scoped forward hook, preserving native router/gate/capacity calculations at that
layer. It zeros the expert output contribution, including fc2 bias; it is NOT a
weight-deletion dose and is excluded from dose curves and weight-dose correlations.

For each union compute continuous whole/accepted/complement CE damage and per-sequence
paired damage/counts. Nonadditivity = union damage - sum of cached Stage 1 component
single-group damages, for each subset separately. No component subtraction is defined
for the output-zero positive control. Exact token-paired deltas and expert metrics are
cached with source, checkpoint, probe, implementation and intervention identities.

Reliability is A/B paired agreement and ranking. Main strata: each trajectory/time/
equal group dose, pooling unique expert/unions; also report within-expert equal-dose
rankings. No duplicate ordering aliases count as separate observations. A single union
per expert at eight groups has no within-expert rank reliability estimate. Report MAE,
RMSE, bias, ranges, sign agreement, Spearman and pairwise order concordance with ties
explicit; undefined constant/small panels remain undefined. Use a representative
repeatability floor (max measured repeat difference, at least 1e-12) only to flag ties
and numerically zero outcomes, not as a damage-selection threshold. Separately summarize
adequate-coverage paired panels; retain all low-traffic panels in primary outputs.
Do not pool doses to manufacture ranking reliability. No p-values or seed-level
confidence claims: one seed, nested experts/groups/orderings/checkpoints/sequences.
0.001 and 0.0001 nats/token are descriptive plot reference lines only. Do not choose
one to obtain a trend. Continuous outcomes and conditional/functional effects lead.

Validate zero dose, cached-baseline equivalence, representative single and 4-group
repeatability, positive-control repeatability, exact selected-coordinate restoration
after EVERY intervention, full model hash after each expert panel, unchanged RNG,
module modes/gradients and TF32 flags, sparse mask integrity, unchanged targeted-layer
routing/gates, and hook removal. Scoped finally blocks restore even on failures.
Workers use study/host advisory locks and 15-second heartbeats. Write only new results
under expert_local_deletion_dose_v1, preserve historical artifacts, and record actual
stage wall-time, CUDA-event forward time, forward counts, peak memory and storage.
'''


def prepare():
    started=time.perf_counter();assert not (D/'registration.json').exists(),'Already prepared'
    checked=source_audit();groups=load(s.D/'group_coordinates.pt');panel,aliases,orders=build_panel(groups)
    masks=s.load_masks(s.jobs()['sparse']['mask']);rows=[]
    for item in panel:
        e=item['expert'];active=int(masks[s.name(e)].sum());total=masks[s.name(e)].numel()
        aliases_here=[a for a in aliases if a['intervention']==item['intervention']]
        rows.append(dict(intervention=item['intervention'],intervention_kind=item['kind'],layer=1,expert=e,parameter=item['parameter'],dose_groups=item['dose_groups'],component_groups=';'.join(item['groups']),coordinates=len(item['indices']) if item['kind']=='weight_union' else None,coordinate_sha256=digest(item['indices']) if item['kind']=='weight_union' else None,sparse_active_matrix_coordinates=active,unpruned_active_matrix_coordinates=total,total_matrix_coordinates=total,sparse_active_matrix_fraction=len(item['indices'])/active if item['kind']=='weight_union' else None,unpruned_active_matrix_fraction=len(item['indices'])/total if item['kind']=='weight_union' else None,ordering_seeds=';'.join(str(a['ordering_seed']) for a in aliases_here),logical_ordering_occurrences=len(aliases_here),density_comparison='coordinate_matched_not_density_matched',global_cap_applied=False))
    table(D/'intervention_manifest.csv',rows);table(D/'ordering_aliases.csv',aliases)
    s.tensor_save(D/'intervention_coordinates.pt',panel)
    hashes=s.read(s.D/'checkpoint_and_probe_hashes.json')
    stage1_files={r['path']:r['sha256'] for r in checked}
    for p in [ROOT/'scripts/expert_role_deletion_v1/study.py',s.D/'output_seal.json',s.D/'final_audit.json']:
        stage1_files[str(p.relative_to(ROOT))]=sha(p)
    registration=dict(seed=17,source_case='NONMATCHING IMP80 engineering case',steps=list(s.STEPS),layer=1,experts=list(s.EXPERTS),order_seeds=list(ORDER_SEEDS),orders=orders,group_doses=list(DOSES),unique_interventions=len(panel),logical_weight_interventions=len(aliases),output_rms_floor=OUTPUT_EPS,minimum_accepted_coverage=MIN_ACCEPTED,thresholds_are_reference_lines_only=[.001,.0001],stage1_files=stage1_files,checkpoint_provenance=hashes['checkpoints'],duplicate_endpoint_aliases=hashes['duplicates'],training_updates=0,global_cap_applied=False)
    save(D/'registration.json',registration);(D/'protocol.md').write_text(PROTOCOL)
    save(D/'input_seal.json',dict(registration_sha256=sha(D/'registration.json'),coordinates_sha256=sha(D/'intervention_coordinates.pt'),protocol_sha256=sha(D/'protocol.md'),implementation=implementation_files()))
    save(D/'source_audit.json',dict(passed=True,source_files_verified=checked,stage1_modified=False,checkpoints='verified again when each model is loaded',step2500_pre_reset_verified=True,groups_and_A_B_preserved=True,aliases_deduplicated=True))
    save(D/'compute_accounting.json',dict(prepare_wall_seconds=time.perf_counter()-started,host=socket.gethostname(),additional_seeds_launched=False,training_updates=0))
    print('Prepared',len(panel),'unique expert interventions from',len(aliases),'ordering/dose entries plus four output controls',flush=True)


def timed_evaluate(m,batches,experts,stats,zero_expert=None):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();out=evaluate(m,batches,experts,zero_expert);end.record();torch.cuda.synchronize()
    stats['native_evaluations']+=1;stats['complete_batch_forwards']+=len(batches);stats['cuda_event_evaluation_seconds']+=start.elapsed_time(end)/1000
    return out


def assert_routes(baseline,out):
    for key in ('ids','accepted','gates'):assert torch.equal(baseline[key],out[key]),'Affected-layer routing changed: '+key


def mask_check(model,masks):
    state=model.state_dict()
    for n,mask in masks.items():assert torch.count_nonzero(state[n][~mask.to(state[n].device).bool()])==0


def measure():
    started=time.perf_counter();reg=verify_registration();assert torch.cuda.is_available()
    groups=load(s.D/'group_coordinates.pt');panel=load(D/'intervention_coordinates.pt');batches=s.probes();checks=[];reuse=[]
    stats=dict(native_evaluations=0,complete_batch_forwards=0,cuda_event_evaluation_seconds=0.,cached_panels_reused=0,cached_single_group_measurements_reused=0)
    torch.cuda.reset_peak_memory_stats()
    for kind,j in s.jobs().items():
      masks=s.load_masks(j['mask']) if j['mask'] else {}
      for t in s.STEPS:
        paths={p:D/'cache'/f'{kind}_{t}_{p}.pt' for p in batches}
        m=None
        for probe,bs in batches.items():
            identity=dict(stage1_cache_sha256=reg['stage1_files'][str(s.cache_path(kind,t,probe).relative_to(ROOT))],checkpoint=reg['checkpoint_provenance'][j['id']][str(t)],input_seal_sha256=sha(D/'input_seal.json'),probe=probe,kind=kind,step=t)
            path=paths[probe]
            if path.exists():
                cached=load(path);assert cached['identity']==identity and cached['verified'];stats['cached_panels_reused']+=1;checks.extend(cached['checks']);reuse.extend(cached['reuse']);continue
            if m is None:m=s.model_for(j,t)
            expected=reg['checkpoint_provenance'][j['id']][str(t)]['model_hash']
            assert s.state_dict_hash(m.state_dict())==expected
            old=load(s.cache_path(kind,t,probe));assert old['verified'] and old['identity']==s.cache_identity(kind,t,probe)
            baseline=timed_evaluate(m,bs,s.EXPERTS,stats);check_baseline(old['baseline'],baseline)
            # One real zero-dose pass verifies the empty union for all four expert panels.
            with s.deletion(m,{s.name(s.EXPERTS[0]):torch.empty(0,dtype=torch.long)}):zero=timed_evaluate(m,bs,s.EXPERTS,stats)
            check_baseline(baseline,zero)
            for e in s.EXPERTS:
                assert all(torch.equal(a['output'],b['output']) for a,b in zip(baseline['outputs'][e],zero['outputs'][e]))
                replay=replay_outputs(m,e,baseline)
                assert all(torch.equal(a['output'],b) for a,b in zip(baseline['outputs'][e],replay))
            interventions={};panel_checks=[];panel_reuse=[]
            for e in s.EXPERTS:
                n=s.name(e);parameter=dict(m.named_parameters())[n];active=parameter.numel() if kind=='unpruned' else int(masks[n].sum());nonzero=int(torch.count_nonzero(parameter))
                first_single=True;first_four=True
                for item in [x for x in panel if x['expert']==e]:
                    idx=item['indices'];iid=item['intervention'];dose=item['dose_groups'];control=item['kind']=='expert_output_zero'
                    before=parameter.detach().clone();cache_source='new_native_evaluation';repeat_error=0.;native_replay_checked=False;repeat_tested=False
                    component=sum((old['paired_token_damage'][k].double() for k in item['groups']),torch.zeros_like(baseline['loss'],dtype=torch.float64)) if not control else None
                    if not control and dose==0:
                        delta=torch.zeros_like(baseline['loss'],dtype=torch.float64)
                        outs=[b['output'] for b in baseline['outputs'][e]];cache_source='verified_zero_dose_and_baseline'
                    elif not control and dose==1:
                        key=item['groups'][0];delta=old['paired_token_damage'][key].double();cache_source='stage1_single_group_cache';stats['cached_single_group_measurements_reused']+=1
                        with s.deletion(m,{n:idx}):
                            outs=replay_outputs(m,e,baseline)
                            if first_single:
                                out=timed_evaluate(m,bs,[e],stats);assert_routes(baseline,out)
                                assert torch.equal(delta,(out['loss']-baseline['loss']).double())
                                assert all(torch.equal(b['output'],z) for b,z in zip(out['outputs'][e],outs))
                                again=timed_evaluate(m,bs,[e],stats);assert_routes(baseline,again)
                                repeat_error=float((again['loss'].double()-out['loss'].double()).abs().max())
                                assert torch.equal(again['loss'],out['loss'])
                                assert all(torch.equal(a['output'],b['output']) for a,b in zip(out['outputs'][e],again['outputs'][e]))
                                first_single=False;native_replay_checked=True;repeat_tested=True
                    else:
                        with s.deletion(m,{} if control else {n:idx}):
                            out=timed_evaluate(m,bs,[e],stats,zero_expert=e if control else None);assert_routes(baseline,out)
                            delta=(out['loss'].double()-baseline['loss'].double());outs=[b['output'] for b in out['outputs'][e]]
                            if control or (dose==4 and first_four):
                                again=timed_evaluate(m,bs,[e],stats,zero_expert=e if control else None);assert_routes(baseline,again)
                                repeat_error=float((again['loss'].double()-out['loss'].double()).abs().max())
                                assert torch.equal(again['loss'],out['loss'])
                                assert all(torch.equal(a['output'],b['output']) for a,b in zip(out['outputs'][e],again['outputs'][e]))
                                repeat_tested=True
                                if not control:first_four=False
                    assert torch.equal(parameter,before),'Intervention restoration failed'
                    assert all(p.grad is None for p in m.parameters())
                    # All native hooks have been removed, including output-zero controls.
                    assert not m.blocks[1].moe.experts[e]._forward_hooks
                    metrics=output_metrics(baseline['outputs'][e],outs)
                    meta=dict(seed=17,trajectory=kind,step=t,probe=probe,layer=1,expert=e,intervention=iid,intervention_kind=item['kind'],dose_groups=dose,component_groups=';'.join(item['groups']),coordinates=len(idx) if not control else None,active_matrix_coordinates=active,actual_nonzero_matrix_coordinates=nonzero,total_matrix_coordinates=parameter.numel(),active_matrix_fraction=len(idx)/active if not control else None,coordinate_matched=True,ce_source=cache_source)
                    rows=measurement_rows(meta,delta,baseline,metrics,component)
                    interventions[iid]=dict(delta=delta,rows=rows,output_metrics=metrics,metadata=meta)
                    panel_reuse.append(dict(trajectory=kind,step=t,probe=probe,intervention=iid,ce_source=cache_source))
                    panel_checks.append(dict(trajectory=kind,step=t,probe=probe,expert=e,intervention=iid,restored_exactly=True,gradient_none=True,hooks_removed=True,targeted_layer_routing_preserved=True,repeat_tested=repeat_tested,repeat_loss_max_abs=repeat_error,native_replay_checked=native_replay_checked))
                assert s.state_dict_hash(m.state_dict())==expected
                mask_check(m,masks)
            restored=timed_evaluate(m,bs,s.EXPERTS,stats);check_baseline(baseline,restored)
            for e in s.EXPERTS:assert all(torch.equal(a['output'],b['output']) for a,b in zip(baseline['outputs'][e],restored['outputs'][e]))
            assert s.state_dict_hash(m.state_dict())==expected
            cache=dict(identity=identity,verified=True,baseline=baseline,interventions=interventions,checks=panel_checks,reuse=panel_reuse)
            s.tensor_save(path,cache);checks.extend(panel_checks);reuse.extend(panel_reuse)
            save(D/'progress.json',dict(trajectory=kind,step=t,probe=probe,unix=time.time()))
            print(kind,t,probe,len(interventions),'interventions verified',flush=True)
        del m;torch.cuda.empty_cache()
    save(D/'numerical_checks.json',dict(passed=True,zero_dose_equivalent=True,source_baseline_bitwise_equal=True,model_hashes_restored=True,mask_integrity=True,RNG_checked_every_evaluation=True,no_gradients=True,training_updates=0,checks=checks))
    table(D/'cache_reuse.csv',reuse)
    emit_tables()
    accounting=s.read(D/'compute_accounting.json');accounting.update(measure_wall_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),**stats);save(D/'compute_accounting.json',accounting)


def emit_tables():
    rows=[];seqrows=[];candidates=load(s.D/'probe_candidates.pt');n=s.read(s.D/'probe_config.json')['batches_per_probe']
    for path in sorted((D/'cache').glob('*.pt')):
        cache=load(path);assert cache['verified'];probe=cache['identity']['probe'];batches=candidates[:n] if probe=='A' else candidates[2:2+n];sequence_ids=[i for b in batches for i in b['sequence_ids']]
        for item in cache['interventions'].values():
            rows.extend(item['rows']);meta=item['metadata'];base=cache['baseline'];accepted=(base['ids']==meta['expert'])&base['accepted']
            seqrows.extend(dict(**{k:meta[k] for k in ['seed','trajectory','step','probe','layer','expert','intervention','intervention_kind','dose_groups']},**r) for r in sequence_rows(item['delta'],accepted,sequence_ids))
    table(D/'measurements.csv',rows);table(D/'paired_sequence_damage.csv',seqrows)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','measure']);args=ap.parse_args();torch.set_num_threads(2)
    with worker(),s.protected():
        try:globals()[args.stage]()
        except BaseException as e:
            save(D/f'failure_{args.stage}_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()));raise
if __name__=='__main__':main()
