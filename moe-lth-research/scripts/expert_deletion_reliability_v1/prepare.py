"""Freeze 32 source-disjoint native-shape training batches and all analyses."""
from reliability_common import *
from torch.utils.data import DataLoader

PROTOCOL='''# Fixed-budget deletion-damage reliability v1

Question: can deletions at a fixed checkpoint/expert/dose be ordered reproducibly
across additional training contexts? Measurement only, no specialization or pruning
algorithm. Seed17 remains a NONMATCHING 32E IMP80 engineering case. Retain verified
sparse/matched-unpruned steps25/250/1000/2500, pre-reset2500, layer1 experts0/8/16/24,
original fc1 groups, four orderings, 52 distinct weight unions and four separate
expert-output-zero controls. Coordinate-matched, not density-matched. No global cap.
Aliases with identical endpoint model hashes count once. No model/seed expansion.

The user explicitly authorized deterministic re-batching of INTACT canonical
128-token sequences. This preserves original source-window sequence boundaries,
complete128x128 native-shape batches, native collation/routing/capacity, but does NOT
claim to preserve historical shuffled minibatch membership. Use the canonical byte
TokenBlockDataset and data configuration. Seed171032 uniformly permutes source
sequence IDs, then greedily accepts the first4096 whose129-token input+target span
is disjoint from prior A/B spans and all previously accepted spans, also excluding
identical full-window contents at different source positions. No routing/loss is
consulted. Strict overlaps include the one-byte shifted-target boundary. If4096
unique valid windows are unavailable, stop and report the limitation; do not relax.
Pack in accepted order into exactly32 complete batches. Hash all sequences/batches.
Identical batches at all checkpoints/trajectories. Blocks1..8 contain four batches
each, in frozen order. Discovery blocks1..4; confirmation blocks5..8. These training
contexts may have been seen during training; confirmation is not independent test
or source-seed replication. No further contexts are evaluated. Profiling reuses
one of these32 batches and its measurements are cached for the main pass.

FP32, TF32-off, eval/inference, native top1 gates and capacity, task-only CE;
auxiliary loss separate. Compute paired token deleted-minus-baseline in FP64 before
FP64 aggregation; never clip negative damage. Conditional positions are baseline
accepted positions, fixed for all interventions. Effects may propagate outside
through attention/later layers. Store compact sequence/subset paired sums/counts,
batch sums and baseline routes/losses, not hidden tensors/logits. No old CE is reused
on these new contexts. Reuse only verified code, masks, coordinates and definitions.
Output RMS is on native-dispatch baseline accepted positions/channels. Save squared
difference/base energies and gate-weighted energies separately. Pool energies before
sqrt(numerator/denominator); do not replace by average batch ratios. Baseline RMS
<=1e-12 gives undefined relative distortion. No accepted output -> undefined.

Coverage is reported for EVERY expert/checkpoint/trajectory/batch/block. Threshold:
>=32 accepted positions PER FOUR-BATCH BLOCK for conditional ranking eligibility;
all eight blocks must qualify for the discovery/confirmation conditional ranking.
Still save raw low-coverage loss estimates and intervals. Report contributing batch
count, maximum batch fraction, HHI and effective contributing-batch count. Coverage
is an operational flag, not precision. Whole-probe results remain valid when traffic
is low, but zeros do not prove dispensability. Never replace experts or extend budget.

Ranks NEVER pool checkpoints, experts or doses. Each trajectory/time/expert/dose has
3 unique single-group unions,4 two-group,4 four-group,1 eight-group; zero and output
control are separate singleton cases. Spearman undefined if <3 unions, constant
range <=1e-12, or ineligible conditional coverage. Preserve all underlying values,
tie counts and distinct unions. Overlapping unions remain dependent observations.
Discovery selects maximal/minimal mean damage only. Tie break by lexicographic ID:
lowest first, highest last, so constant panels expose arbitrary tied choices. With
<2 unions selection is undefined. Evaluate the SAME pair on confirmation; report
confirmation/discovery contrast, each block contrast/sign/CI, and all fixed lexicographic
pairwise contrasts in both partitions and all eight blocks. Do not reselect with
confirmation. Report LOO block estimates for each phase and selected contrast.

Bootstrap complete batches ONLY, retaining paired all-intervention sums. Fixed
seed1732000, 2000 multinomial-resampled-batch count vectors per partition/block;
same count vectors for all interventions/subsets within that set. Discovery and
confirmation use independently seeded vectors. Conditional estimates resample sums
AND counts and divide AFTER summing; zero-denominator replicates undefined, report
frequency. Percentile2.5/97.5 intervals are descriptive sampled-context intervals in
one seed, not source-seed confidence or generalization. Bootstrap rank intervals
pair independent discovery/confirmation draws, preserving within-phase intervention
pairing; undefined replicates reported. Block CIs use four complete batches, not
individual tokens. LOO removes one whole four-batch block within each phase.

Continuous effects/intervals lead. Reference tolerances0.0001 and0.001: interval
entirely above -> exceeds exploratory convention; entirely below -> compatible
with below-convention damage on these contexts; crossing -> unresolved. No claim
of harmlessness, permanent-pruning safety or dispensability from a small effect.
Numerical repeatability is separate from context agreement. Negative ranks alone
are not context dependence. Inspect paired contrasts, intervals and coverage; avoid
semantic/domain labels or fitted specialization predictors. Functional distortion
rank agreement across phases and distortion-vs-CE correlations are also restricted
to the SAME expert/time/dose; singleton/constant panels undefined.

Condition verdicts retain expert identity: selected confirmation interval >0 and
positive discovery contrast -> reproduced directional contrast in these contexts;
interval <0 -> reversed directional contrast, candidate context dependence only;
interval crosses0 -> unresolved at this budget. Low coverage and <2/<3 union cases
explicit. This is descriptive, not a global PASS, and selected effects can be noisy.
Report A-E per fixed condition; final conclusion must preserve heterogeneous cases.

Profile sparse step1000 batch0 full panel before the full pass, after this freeze.
One GPU worker, locks,15-second heartbeat, atomic per-batch caches and completion.
Cache identity binds checkpoint, exact batch, coordinate/intervention seal, numerical
mode and implementation. Real empty deletion, repeat checks on batch0 and16 of every
checkpoint, native baseline check, scoped finally restoration and exact weight checks
per intervention; full hash/mask check per batch. All evaluation checks RNG/modes/
gradients. No backward, optimizer, training, recovery, mask search, permanent pruning,
new seeds, extra contexts or automatic follow-up. Engineering fixes must preserve
scientific definitions and retain old code/provenance.
'''


def main():
    start=time.perf_counter();assert not (D/'freeze.json').exists(),'Already frozen'
    dc.verify_registration();checked=dc.source_audit()
    source_files={r['path']:r['sha256'] for r in checked}
    for p,h in s.read(dc.D/'output_seal.json')['files'].items():
        path=dc.D/p;assert sha(path)==h,p;source_files[str(path.relative_to(ROOT))]=h
    for p in [dc.D/'output_seal.json',dc.D/'input_seal.json',dc.D/'final_audit.json',dc.D/'intervention_coordinates.pt',ROOT/'scripts/expert_role_deletion_v1/study.py',ROOT/'scripts/expert_local_deletion_dose_v1/dose_common.py']:
        source_files[str(p.relative_to(ROOT))]=sha(p)
    cfg=s.jobs()['sparse']['config']
    with s.protected():tr,_=s.build_dataloaders(cfg['data'],128,17,reshuffle_each_epoch=False)
    dataset=tr.dataset;old=load(s.D/'probe_candidates.pt');old_ids={i for b in old for i in b['sequence_ids']};old_hashes={content_hash(dataset,i) for i in old_ids}
    source_hash=sha(ROOT/cfg['data']['train_path']);assert source_hash==s.read(s.D/'checkpoint_and_probe_hashes.json')['train_corpus_sha256']
    chosen,rejections=choose_sequences(dataset,old_ids,old_hashes)
    audit=dict(dataset_windows=len(dataset),prior_AB_windows=len(old_ids),required_new_windows=4096,selected_windows=len(chosen),source_corpus_sha256=source_hash,selection_seed=PROBE_SEED,rebatching_authorized=True,historical_minibatch_membership_preserved=False,rejections=rejections,overlap_includes_shifted_target=True,duplicate_content_excluded=True,available=len(chosen)==4096)
    save(D/'probe_overlap_audit.json',audit)
    if len(chosen)!=4096:
        (D/'findings.md').write_text('# Insufficient probe availability\n\nOnly '+str(len(chosen))+' source-disjoint, content-unique intact windows were available under the fixed sampler;4096 required. No model diagnostics were run, no design relaxation occurred.\n')
        save(D/'completion.json',dict(measurement_complete=False,reason='insufficient_probe_availability'));return
    ids=[i for i,h in chosen];hs=[h for i,h in chosen];assert len(set(ids))==4096 and len(set(hs))==4096
    assert not any(i-1 in set(ids) or i+1 in set(ids) or i in old_ids or i-1 in old_ids or i+1 in old_ids for i in ids)
    sampler=[ids[i:i+128] for i in range(0,4096,128)]
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=0,generator=torch.Generator().manual_seed(PROBE_SEED))
    batches=[];probes=[];rows=[]
    for b,(x,y) in enumerate(loader):
        assert tuple(x.shape)==(128,128) and tuple(y.shape)==(128,128)
        block=b//4+1;phase='discovery' if block<=4 else 'confirmation';seq=sampler[b]
        assert torch.equal(x,torch.stack([dataset[i][0] for i in seq]))
        ctx=hs[b*128:(b+1)*128];context_hash=hashlib.sha256(('\n'.join(str(i)+':'+h for i,h in zip(seq,ctx))).encode()).hexdigest()
        probes.append(dict(batch=b,block=block,partition=phase,sequence_ids=seq,inputs_sha256=digest(x),targets_sha256=digest(y),sequence_contexts_sha256=context_hash))
        batches.append((x,y))
        rows.extend(dict(batch=b,block=block,partition=phase,row=j,sequence_id=i,source_start=i*128,source_end_inclusive=i*128+128,context_content_sha256=h,source_corpus_sha256=source_hash) for j,(i,h) in enumerate(zip(seq,ctx)))
    s.tensor_save(D/'probes.pt',batches);table(D/'probe_manifest.csv',rows)
    inventory=[];cpmeta=s.read(s.D/'checkpoint_and_probe_hashes.json')
    for kind,j in s.jobs().items():
        for t in s.STEPS:
            p=s.cp_path(j,t);expected=cpmeta['checkpoints'][j['id']][str(t)];assert sha(p)==expected['sha256'];cp=load(p);assert cp['step']==t and s.state_dict_hash(cp['model'])==expected['model_hash']
            side=s.read(p.with_suffix('.json'));assert side['segment_offset']==0
            inventory.append(dict(artifact_type='checkpoint',trajectory=kind,step=t,path=str(p.relative_to(ROOT)),sha256=sha(p),model_hash=expected['model_hash'],use='selected_once',bytes=p.stat().st_size))
        before=s.read(s.cp_path(j,2500).with_suffix('.json'));after=s.read(s.cp_path(j,2500).with_name('step_2500_after_reset.json'));assert before['optimizer_entries']>0 and after['optimizer_entries']==0 and before['model_hash']==after['model_hash']
    for alias in cpmeta['duplicates']:inventory.append(dict(artifact_type='endpoint_alias',trajectory=alias['kind'],step=2500,path=alias['path'],sha256=alias['sha256'],model_hash=alias['model_hash'],use='excluded_same_model_hash'))
    for p in [s.D/'group_manifest.csv',dc.D/'intervention_manifest.csv',dc.D/'ordering_aliases.csv',dc.D/'numerical_checks.json',dc.D/'final_audit.json',dc.D/'compute_accounting.json']:
        inventory.append(dict(artifact_type='source_manifest_or_audit',path=str(p.relative_to(ROOT)),sha256=sha(p),use='verified_reuse_code_or_coordinates_not_new_probe_losses'))
    table(D/'artifact_checkpoint_group_inventory.csv',inventory)
    panel=interventions();rebuilt,aliases,orders=dc.build_panel(load(s.D/'group_coordinates.pt'))
    assert len(panel)==len(rebuilt)==56
    for a,b in zip(panel,rebuilt):assert a['intervention']==b['intervention'] and torch.equal(a['indices'],b['indices'])
    table(D/'intervention_manifest.csv',list(__import__('csv').DictReader((dc.D/'intervention_manifest.csv').open())))
    reg=dict(seed=17,source_case='NONMATCHING IMP80 engineering case',probe_seed=PROBE_SEED,bootstrap_seed=BOOTSTRAP_SEED,bootstrap_resamples=RESAMPLES,batches=32,batch_shape=[128,128],blocks=8,batches_per_block=4,rebatching_authorization='user explicitly allowed deterministic re-batching of intact canonical sequences',probes=probes,checkpoints=cpmeta['checkpoints'],source_files=source_files,interventions=56,minimum_block_accepted=32,rank_minimum_distinct_unions=3,tie_floor=FLOOR,reference_tolerances=[.0001,.001],numerics='FP32_eval_inference_TF32off_native',training_updates=0)
    save(D/'configuration.json',reg);(D/'protocol.md').write_text(PROTOCOL)
    files={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')};ih=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    save(D/'freeze.json',dict(configuration_sha256=sha(D/'configuration.json'),protocol_sha256=sha(D/'protocol.md'),probes_sha256=sha(D/'probes.pt'),implementation=files,implementation_sha256=ih,unix=time.time()))
    save(D/'compute_accounting.json',dict(prepare_wall_seconds=time.perf_counter()-start,unique_probe_batches=32,profile='pending',training_updates=0,host=socket.gethostname()))
    print('Frozen32 complete source-disjoint batches, eight blocks,56 interventions',flush=True)
if __name__=='__main__':
    with worker():main()
