"""Stage 1 only: fixed checkpoint deletion interventions, never optimizer updates.

Run prepare, profile, measure, report in that order. Historical artifacts are read-only.
"""
from __future__ import annotations
import argparse, contextlib, csv, fcntl, hashlib, json, os, random, socket, sys, threading, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.router_age import state_dict_hash
from moe_lth.pruning.masks import load_masks
from moe_lth.data import build_dataloaders
D = ROOT/'results/expert_role_deletion_v1'
H = ROOT/'results/high_sparsity_coupling_v1'
STEPS = (25, 250, 1000, 2500)
EXPERTS = (0, 8, 16, 24)
TOLERANCES = (.0001, .001, .005)
SEED = 170625


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def digest(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256((str(t.dtype)+str(tuple(t.shape))).encode()+t.numpy().tobytes()).hexdigest()
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n'); q.replace(p)
def tensor_save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True); q=p.with_suffix('.tmp')
    torch.save(x,q); q.replace(p)
def table(p, rows, fields=None):
    rows=list(rows); fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix('.tmp')
    with q.open('w') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    q.replace(p)
def load(p): return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def name(e): return f'blocks.1.moe.experts.{e}.fc1.weight'
def jobs():
    cohort=read(H/'cohort.json')
    return {kind:next(c for c in cohort['conditions'] if c['id']==f'e32_seed17_{suffix}')
            for kind,suffix in [('sparse','final_s0.80'),('unpruned','dense')]}
def attempt(j): return Path(read(H/'reruns'/j['id']/'completed.json')['attempt'])
def cp_path(j,t): return attempt(j)/'snapshots'/f'step_{t:04d}.pt'
def model_for(j,t):
    p=cp_path(j,t); expected=read(D/'checkpoint_and_probe_hashes.json')['checkpoints'][j['id']][str(t)]
    assert sha(p)==expected['sha256']
    cp=load(p); assert cp['step']==t
    assert state_dict_hash(cp['model'])==expected['model_hash']
    with protected(): m=TinyMoELanguageModel(j['config']['model']).cuda().float()
    m.load_state_dict(cp['model'],strict=True); m.eval()
    assert all(p.grad is None and p.dtype==torch.float32 for p in m.parameters())
    if j['mask']:
        masks=load_masks(j['mask'])
        for n,mask in masks.items(): assert torch.count_nonzero(cp['model'][n][~mask.bool()])==0
    return m


@contextlib.contextmanager
def protected(model=None):
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    modes=[(m,m.training) for m in model.modules()] if model is not None else []
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    grads={n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()} if model is not None else {}
    try:
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        if model is not None: model.eval()
        yield
    finally:
        random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
        if states[3]: torch.cuda.set_rng_state_all(states[3])
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        for m,flag in modes:m.training=flag
        if model is not None:
            for n,p in model.named_parameters():
                g=grads[n];assert (p.grad is None)==(g is None)
                if g is not None:assert torch.equal(g,p.grad)


@contextlib.contextmanager
def deletion(model, groups):
    ps=dict(model.named_parameters()); old={}
    try:
        with torch.no_grad():
            for n,idx in groups.items():
                flat=ps[n].view(-1);idx=idx.to(flat.device)
                old[n]=(idx,flat[idx].clone());flat[idx]=0
        yield
    finally:
        with torch.no_grad():
            for n,(idx,values) in old.items():ps[n].view(-1)[idx]=values


def evaluate(model,batches,details=False):
    losses=[]; ids=[]; accepted=[]; gates=[]; features=[]; aux=[]
    with protected(model),torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for x,y in batches:
            out=model(x.cuda());z=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.cuda().flatten(),reduction='none')
            assert torch.isfinite(z).all();losses.append(z.cpu());aux.append(float(out.auxiliary_loss))
            if details:
                tr=out.routes[1]
                ids.append(tr.selected_expert_indices.cpu().reshape(-1));accepted.append(tr.accepted_mask.cpu().reshape(-1))
                gates.append(tr.selected_probabilities.cpu().reshape(-1));features.append(out.pre_router_hidden_states[1].cpu().reshape(-1,out.pre_router_hidden_states[1].shape[-1]))
    result={'loss':torch.cat(losses),'auxiliary_batch_means':aux}
    if details: result.update(ids=torch.cat(ids),accepted=torch.cat(accepted),gates=torch.cat(gates),features=torch.cat(features))
    return result


def inventory():
    # Metadata-only inventory precedes checkpoint loads and all model diagnostics.
    sources=[
      ('high_sparsity_coupling_v1','cohort.json','verified IMP80 recovery snapshots and matched dense; selected; historical/rerun aliases deduplicated by model hash'),
      ('router_expert_geometry_v1','registration.json','R0/R50/R100 are conditions, not time; fixed-source geometry not historical routing; no raw metric reuse'),
      ('imp_mask_information_v1','protocol.md','mask structure only; no probe-matched deletion measurements'),
      ('imp_mask_information_v2','registration.json','IMP masks distinct from one-shot masks; no native deletion reuse'),
      ('token_expert_matching_v1/evaluation_repair_v1','corrected_interpretation.md','native frozen-router and matched-policy readouts are not interchangeable; no reuse'),
      ('high_sparsity_coupling_v1/functional_geometry_extension_v1','completion.json','fixed-source functional geometry extension; different populations; no raw reuse'),
      ('high_sparsity_coupling_v1/functional_geometry_extension_v1/audit_v1','audit_summary.md','reviewed metric definitions and audit; provenance only'),
      ('imp_adaptation_factorial_v1','manifest.json','verified recovery paths/configuration; adaptation conditions not deletion results'),
      ('expert_exposure_weight_development_v1','findings.md','original unpruned source trajectories; exposure logs are not fixed contextual assignment logs'),
      ('magnitude_growth_pruning_v1','findings.md','different cohort, dose and recovery intervention; outcomes not reused'),
      ('gradient_sensitivity_pruning_v1','findings.md','scoring/matching stopped; no pruning outcomes; gradient scores lack matching study provenance'),
      ('gradient_sensitivity_matching_audit_v1','findings.md','matching feasibility audit, not pruning outcomes'),
      ('exposure_stratified_pruning_v1','findings.md','frozen feasibility gate failed; no pruning outcomes'),
      ('imp_extended_recovery','README.md','fresh AdamW/scaler boundary at 2500; second segment excluded')]
    rows=[]
    for directory,entry,reason in sources:
        p=ROOT/'results'/directory/entry
        rows.append(dict(artifact=directory,path=str(p.relative_to(ROOT)),exists=p.exists(),sha256=sha(p) if p.exists() else '',use=reason))
    table(D/'artifact_inventory.csv',rows)
    return rows


def prepare():
    assert not (D/'checkpoint_and_probe_hashes.json').exists(),'Preparation already frozen'
    D.mkdir(parents=True,exist_ok=True);inventory()
    from moe_lth.pruning.router_age import state_dict_hash
    js=jobs(); checkpoints={}; duplicates=[]
    for kind,j in js.items():
        checkpoints[j['id']]={}
        for t in STEPS:
            p=cp_path(j,t);meta=read(p.with_suffix('.json'));h=sha(p)
            assert h==meta['snapshot_sha256'] and meta['segment_offset']==0
            cp=load(p);mh=state_dict_hash(cp['model']);assert cp['step']==t and mh==meta['model_hash']
            checkpoints[j['id']][str(t)]={'path':str(p.relative_to(ROOT)),'sha256':h,'model_hash':mh,'segment_offset':0}
        before=read(cp_path(j,2500).with_suffix('.json'));after=read(cp_path(j,2500).with_name('step_2500_after_reset.json'))
        assert before['optimizer_entries']>0 and after['optimizer_entries']==0 and before['model_hash']==after['model_hash']
        # Independently compare historical endpoint weights, never count aliases as replication.
        historical=Path(j['boundary']);hist=load(historical);mh=state_dict_hash(hist['model'])
        assert mh==before['model_hash']
        duplicates.append(dict(kind=kind,path=str(historical),sha256=sha(historical),model_hash=mh,duplicate_of=str(cp_path(j,2500)),after_reset_same_weights=True))
    mask_path=Path(js['sparse']['mask']);masks=load_masks(mask_path)
    groups={};manifest=[];rng=np.random.default_rng(SEED)
    for e in EXPERTS:
        n=name(e);active=torch.where(masks[n].view(-1).bool())[0].numpy();size=len(active)//100
        assert size>0 and 8*size<=len(active)
        chosen=rng.permutation(active)[:8*size].reshape(8,size)
        for g,indices in enumerate(chosen):
            key=f'e{e}_g{g}';idx=torch.tensor(np.sort(indices));groups[key]={'expert':e,'parameter':n,'indices':idx}
            manifest.append(dict(group=key,expert=e,layer=1,parameter=n,coordinates=size,active_coordinates=len(active),total_coordinates=masks[n].numel(),fraction_active=size/len(active),fraction_total=size/masks[n].numel(),coordinate_sha256=digest(idx)))
    tensor_save(D/'group_coordinates.pt',groups);table(D/'group_manifest.csv',manifest)
    # Reproduce native first complete training batches, then recover exact sequence IDs.
    cfg=js['sparse']['config']
    with protected():
        tr,_=build_dataloaders(cfg['data'],128,17,reshuffle_each_epoch=False)
        # DataLoader consumes a base-seed draw before RandomSampler's permutation.
        gen=torch.Generator().manual_seed(17);torch.empty((),dtype=torch.int64).random_(generator=gen)
        order=torch.randperm(len(tr.dataset),generator=gen).tolist()
        iterator=iter(tr);candidates=[];used=set()
        for bi in range(min(512,len(tr))):
            x,y=next(iterator);seq=order[bi*128:(bi+1)*128]
            assert torch.equal(x,torch.stack([tr.dataset[i][0] for i in seq]))
            # Require disjoint raw windows including the one-token target overlap.
            if any(i in used or i-1 in used or i+1 in used for i in seq):continue
            candidates.append({'native_batch_index':bi,'sequence_ids':seq,'batch':(x,y)})
            used.update(seq)
            if len(candidates)==4:break
        assert len(candidates)==4,'Need four disjoint complete native batches'
    tensor_save(D/'probe_candidates.pt',candidates)
    probe_rows=[dict(native_batch_index=b['native_batch_index'],sequence_ids=b['sequence_ids'],input_sha256=digest(b['batch'][0]),target_sha256=digest(b['batch'][1])) for b in candidates]
    hashes={'checkpoints':checkpoints,'duplicates':duplicates,'mask':{'path':str(mask_path.relative_to(ROOT)),'sha256':sha(mask_path)},'group_coordinates_sha256':sha(D/'group_coordinates.pt'),'probe_candidates_sha256':sha(D/'probe_candidates.pt'),'train_corpus_sha256':sha(ROOT/cfg['data']['train_path']),'probe_candidates':probe_rows}
    save(D/'checkpoint_and_probe_hashes.json',hashes)
    save(D/'probe_config.json',dict(status='awaiting_cost_profile',seed=SEED,training_only=True,training_seen_possible=True,B='held out from selection, not independent generalization',sequence_length=128,batch_size=128,candidate_native_batch_indices=[b['native_batch_index'] for b in candidates],window_identity='train corpus SHA256 + sequence_id*128 byte offset; contextual identity adds token position',disjoint_including_target_overlap=True,maximum_batches_per_probe=2,numerical_mode='eval FP32, TF32 off, native gates/top1/capacity',clustering=dict(K=16,seed=SEED,preprocessing='per-feature standardize using A reference only; scale floor 1e-6',minimum_cluster_count=32,small_clusters='retain and flag; never merge/refit',minimum_expert_tokens=32,reference='earliest matched unpruned layer1 router inputs',shuffles=64)))
    (D/'protocol.md').write_text(PROTOCOL)
    save(D/'implementation_seal.json',{'study_sha256':sha(Path(__file__)),'protocol_sha256':sha(D/'protocol.md'),'model_files':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/moe_lth/models').glob('*.py')}})
    print('Inventory and pilot scope frozen',flush=True)


PROTOCOL='''# Expert role and persistent deletion sensitivity: Stage 1 engineering pilot

Seed17, 32E IMP80 is retained as a NONMATCHING recovery under the previous
5000-step criterion; this is an engineering case, not a winner. Four preferred
existing checkpoints 25,250,1000,2500 are available with no substitutions.
2500 is the pre-reset endpoint; the fresh optimizer/scaler segment is excluded.
Layer index1, experts0/8/16/24, fc1.weight only, eight disjoint uniformly sampled
groups per expert, floor(1% surviving mask coordinates). Fixed membership in both
trajectories: equal-coordinate deletion, not equal-relative-sparsity comparison.
Primary tolerances 0.001 nats/token, sensitivity 0.0001 and0.005, all exploratory.
Continuous signed task CE damage is primary; auxiliary loss is recorded separately.

H1 (more persistent low-damage groups) and H2 (roles add predictive information)
are separate and neither assumed true. Single A selection uses current damage;
persistent A selection uses previous/current only. Future outcomes always next B.
All groups remain in the prospective panel, including retrospectively rejected
ones. With four times only two two-history forecasts exist. No predictive model,
high-dimensional fit, or significance claim is supported by this one-seed pilot.
Prespecified descriptive comparisons: step; load/gates; magnitude/movement;
past damage; past damage plus separate persistence/differentiation. Eight groups
share an expert role measurement and are not independent role replications.

A/B contain complete deterministic native training batches with original sequence
order/capacity, disjoint raw byte windows including targets. Training may have seen
both. B is locked against candidate/dose/threshold/cluster decisions. Profile one
baseline and one first uniform-group deletion on A before fixing budget: if their
combined cost <=2 seconds choose two batches per probe, otherwise one. Upper bound
two batches each, fixed independently of effect direction. K16 clustering on the
earliest unpruned reference A only, feature standardization floor1e-6, seed170625,
n_init1, Lloyd max_iter100. Freeze centers and reference A/B labels; small clusters
(<32) retained and flagged. Expert populations <32 flagged unavailable for divergence.
Differentiation of routing over fixed context groups is Jensen-Shannon divergence
in nats against overall probe distribution; these are not semantic domains.

Proposed and accepted populations remain separate; capacity drops and gates are
baseline measurements. Adjacent contextual-set Jaccard empty/empty is undefined;
retention intersection/previous, inflow new/current. Shuffle references permute
whole per-token expert/acceptance/gate tuples within each complete batch, preserving
loads/capacity totals,64 deterministic replicates, independently by time.

Ablations start from baseline each time and restore in finally; baseline accepted
conditional positions are fixed before deletion. Attention/later layers spread
effects; conditional damage is not isolated expert loss. Token and sequence-paired
losses are stored but tokens are not independent source replications. Zero dose,
repeat and restoration controls required. Primary group dose never retuned.
Union diagnostics use persistent past-only A selection, canonical expert/group
order, at most floor(1% ALL eligible expert-weight coordinates across all MoE layers,
fc1+fc2); never split a group to fill the cap. Evaluate fixed union at decision and
next checkpoints on A/B; compare actual union damage with sum of singles. No
selection -> explicit undefined, no fake zero. No permanent deletion or recovery.

Checkpoint file and model hashes deduplicate historical/rerun/reset aliases. Raw
old metrics reused only if all provenance matches; none currently qualify. Cache
identity binds checkpoint/probe/group/mode/implementation. New results only.
Workers hold per-host and per-study advisory locks and emit15-second heartbeat.
No training, optimizer, mask search, five-seed expansion, or Stage2 is authorized.
Associations are observational. Low damage at this dose/probe does not prove
future-training dispensability or permanent-pruning recoverability. Final IMP mask
is a diagnostic cohort definition, unavailable to an eventual dense-training method.
'''


def verify_inputs():
    seal=read(D/'implementation_seal.json');assert sha(Path(__file__))==seal['study_sha256']
    assert sha(D/'protocol.md')==seal['protocol_sha256']
    for p,h in seal['model_files'].items():assert sha(ROOT/p)==h
    hashes=read(D/'checkpoint_and_probe_hashes.json')
    for field,path in [('mask',ROOT/hashes['mask']['path'])]:assert sha(path)==hashes[field]['sha256']
    for filename,key in [('group_coordinates.pt','group_coordinates_sha256'),('probe_candidates.pt','probe_candidates_sha256')]:assert sha(D/filename)==hashes[key]


def probes():
    candidates=load(D/'probe_candidates.pt');n=read(D/'probe_config.json')['batches_per_probe']
    return {'A':[c['batch'] for c in candidates[:n]],'B':[c['batch'] for c in candidates[2:2+n]]}


def profile():
    verify_inputs();assert torch.cuda.is_available()
    assert read(D/'probe_config.json')['status']=='awaiting_cost_profile'
    j=jobs()['sparse'];m=model_for(j,25);b=[load(D/'probe_candidates.pt')[0]['batch']]
    groups=load(D/'group_coordinates.pt');g=groups['e0_g0'];before=state_dict_hash(m.state_dict())
    torch.cuda.synchronize();start=time.perf_counter();baseline=evaluate(m,b);torch.cuda.synchronize();base_s=time.perf_counter()-start
    start=time.perf_counter()
    with deletion(m,{g['parameter']:g['indices']}):evaluate(m,b)
    torch.cuda.synchronize();ablation_s=time.perf_counter()-start
    assert state_dict_hash(m.state_dict())==before
    repeated=evaluate(m,b);assert torch.equal(baseline['loss'],repeated['loss'])
    with deletion(m,{}):zero=evaluate(m,b)
    assert torch.equal(zero['loss'],baseline['loss'])
    n=2 if base_s+ablation_s<=2 else 1
    config=read(D/'probe_config.json');config.update(status='frozen',batches_per_probe=n,profile_rule='two if baseline+ablation <=2 sec, else one; upper bound two; no effect inspection')
    save(D/'probe_config.json',config)
    save(D/'compute_accounting.json',dict(profile_baseline_seconds=base_s,profile_ablation_seconds=ablation_s,batches_per_probe=n,gpu=torch.cuda.get_device_name(),host=socket.gethostname(),estimated_primary_forward_seconds=(base_s+ablation_s)/2*2*4*2*n*33,five_seed_expansion_launched=False))
    save(D/'numerical_checks.json',dict(profile_state_restored=True,baseline_repeat_bitwise_equal=True,zero_dose_bitwise_equal=True,fp32=True,tf32=False,production_checks='pending'))
    print(f'Cost profile {base_s:.3f}s baseline, {ablation_s:.3f}s deletion; frozen {n} batches/probe',flush=True)


def jsd(counts,total):
    if counts.sum()==0:return None
    p=counts/counts.sum();q=total/total.sum();m=(p+q)/2
    return float(.5*np.sum(p[p>0]*np.log(p[p>0]/m[p>0]))+.5*np.sum(q[q>0]*np.log(q[q>0]/m[q>0])))
def overlap(a,b):
    inter=int(np.logical_and(a,b).sum());union=int(np.logical_or(a,b).sum())
    return dict(jaccard=inter/union if union else None,retention=inter/int(a.sum()) if a.sum() else None,inflow=int(np.logical_and(b,~a).sum())/int(b.sum()) if b.sum() else None)
def shuffle_route(r,rng,batch_tokens=16384):
    idx=np.concatenate([rng.permutation(np.arange(i,min(i+batch_tokens,len(r['ids'])))) for i in range(0,len(r['ids']),batch_tokens)])
    return {k:r[k].numpy()[idx] for k in ('ids','accepted','gates')}
def population(r,e,accepted):
    ids=np.asarray(r['ids']);mask=ids==e
    return mask & np.asarray(r['accepted']) if accepted else mask


def role_rows(baselines,labels):
    rows=[];rng=np.random.default_rng(SEED)
    for kind in ('sparse','unpruned'):
      for probe in ('A','B'):
        labs=labels[probe];total=np.bincount(labs,minlength=16);prev=None
        for t in STEPS:
          r=baselines[(kind,t,probe)]
          shuffles=[shuffle_route(r,rng) for _ in range(64)]
          prior_shuffles=[shuffle_route(prev,rng) for _ in range(64)] if prev is not None else None
          for e in EXPERTS:
            for accepted in (False,True):
              mask=population(r,e,accepted);count=int(mask.sum());proposed=int(population(r,e,False).sum());gates=r['gates'].numpy()[mask]
              counts=np.bincount(labs[mask],minlength=16)
              reference=[jsd(np.bincount(labs[population(s,e,accepted)],minlength=16),total) for s in shuffles]
              valid=[v for v in reference if v is not None]
              row=dict(seed=17,source='matched_recovery',kind=kind,layer=1,expert=e,step=t,probe=probe,population='accepted' if accepted else 'proposed',load=count,proposed_load=proposed,accepted_load=int(population(r,e,True).sum()),capacity_drops=proposed-int(population(r,e,True).sum()),acceptance_rate=int(population(r,e,True).sum())/proposed if proposed else None,gate_mean=float(gates.mean()) if count else None,gate_min=float(gates.min()) if count else None,gate_max=float(gates.max()) if count else None,gate_sum=float(gates.sum()),context_group_counts=json.dumps(counts.tolist()),differentiation_js_nats=jsd(counts,total) if count>=32 else None,coverage='adequate' if count>=32 else 'unavailable_low_traffic',shuffled_js_mean=float(np.mean(valid)) if valid and count>=32 else None,jaccard=None,retention=None,inflow=None,shuffled_jaccard_mean=None)
              if prev is not None:
                row.update(overlap(population(prev,e,accepted),mask))
                refs=[overlap(population(p,e,accepted),population(s,e,accepted))['jaccard'] for p,s in zip(prior_shuffles,shuffles)]
                refs=[v for v in refs if v is not None];row['shuffled_jaccard_mean']=float(np.mean(refs)) if refs else None
              rows.append(row)
          prev=r
    return rows


def damage_rows(kind,t,probe,key,g,baseline,deleted,weights,previous):
    diff=(deleted['loss'].double()-baseline['loss'].double()).numpy();e=g['expert']
    accepted=population(baseline,e,True);n=g['parameter'];idx=g['indices'];w=weights[n].flatten();v=w[idx].double()
    active=w[w!=0].double();den=float(active.abs().mean()) if len(active) else 0
    attrs=dict(seed=17,source='matched_recovery',kind=kind,step=t,probe=probe,layer=1,expert=e,group=key,coordinates=len(idx),absolute_magnitude=float(v.abs().mean()),normalized_magnitude=float(v.abs().mean())/den if den else None,squared_weight_energy=float((v*v).sum()),recent_movement=float((v-previous[n].flatten()[idx].double()).abs().mean()) if previous else None,row_occupancy=len(torch.unique(idx//weights[n].shape[1])),column_occupancy=len(torch.unique(idx%weights[n].shape[1])),sensitivity_score=None,sensitivity_provenance='unavailable_no_matching_provenance')
    return [dict(**attrs,subset=subset,target_tokens=int(mask.sum()),damage=float(diff[mask].mean()) if mask.any() else None,baseline_ce=float(baseline['loss'].numpy()[mask].astype(np.float64).mean()) if mask.any() else None) for subset,mask in [('whole',np.ones(len(diff),bool)),('baseline_accepted',accepted),('complement',~accepted)]]


def cache_identity(kind,t,probe):
    j=jobs()[kind];h=read(D/'checkpoint_and_probe_hashes.json')
    return dict(kind=kind,step=t,probe=probe,checkpoint=h['checkpoints'][j['id']][str(t)],groups=h['group_coordinates_sha256'],probes=h['probe_candidates_sha256'],config_sha256=sha(D/'probe_config.json'),implementation_sha256=sha(Path(__file__)),mode='FP32_eval_native_TF32off')
def cache_path(kind,t,probe):return D/'cache'/f'{kind}_{t}_{probe}.pt'


def measure():
    verify_inputs();assert read(D/'probe_config.json')['status']=='frozen'
    batches=probes();groups=load(D/'group_coordinates.pt');js=jobs();started=time.monotonic();baselines={};rows=[];checks=[]
    for kind,j in js.items():
      previous=None
      for t in STEPS:
        missing=[p for p in batches if not cache_path(kind,t,p).exists()]
        m=model_for(j,t) if missing else None
        weights=load(cp_path(j,t))['model']
        for probe,bs in batches.items():
          path=cache_path(kind,t,probe);identity=cache_identity(kind,t,probe)
          if path.exists():
            cached=load(path);assert cached['identity']==identity and cached['verified'];base=cached['baseline'];rows.extend(cached['rows'])
          else:
            before=state_dict_hash(m.state_dict());base=evaluate(m,bs,True);repeated=evaluate(m,bs)
            assert torch.equal(base['loss'],repeated['loss'])
            with deletion(m,{}):zero=evaluate(m,bs)
            assert torch.equal(base['loss'],zero['loss']);gr=[];paired={}
            for key,g in groups.items():
                with deletion(m,{g['parameter']:g['indices']}):deleted=evaluate(m,bs)
                gr.extend(damage_rows(kind,t,probe,key,g,base,deleted,weights,previous))
                paired[key]=deleted['loss']-base['loss']
            restored=evaluate(m,bs);assert torch.equal(base['loss'],restored['loss'])
            assert state_dict_hash(m.state_dict())==before
            cached=dict(identity=identity,verified=True,baseline=base,rows=gr,paired_token_damage=paired,paired_sequence_damage={k:v.reshape(-1,128).double().mean(1) for k,v in paired.items()})
            tensor_save(path,cached);rows.extend(gr)
          baselines[(kind,t,probe)]=base;checks.append(dict(kind=kind,step=t,probe=probe,state_restored=True,repeat_equal=True,zero_dose_equal=True,gradients_preserved=True,rng_preserved=True,mask_integrity=True))
          save(D/'progress.json',dict(kind=kind,step=t,probe=probe,unix=time.time()))
          print(kind,t,probe,'verified',flush=True)
        previous={name(e):weights[name(e)].clone() for e in EXPERTS}
        del m,weights
        torch.cuda.empty_cache()
    # Freeze one shared reference clustering; never fit to damage or probe B.
    from sklearn.cluster import KMeans
    A=baselines[('unpruned',STEPS[0],'A')]['features'].numpy()
    mean=A.mean(0);scale=np.maximum(A.std(0),1e-6)
    km=KMeans(n_clusters=16,random_state=SEED,n_init=1,max_iter=100,algorithm='lloyd').fit((A-mean)/scale)
    labels={p:km.predict((baselines[('unpruned',STEPS[0],p)]['features'].numpy()-mean)/scale) for p in batches}
    tensor_save(D/'fixed_context_groups.pt',dict(mean=torch.from_numpy(mean),scale=torch.from_numpy(scale),centers=torch.from_numpy(km.cluster_centers_),labels={p:torch.from_numpy(l) for p,l in labels.items()}))
    table(D/'cluster_counts.csv',[dict(probe=p,cluster=k,count=int((l==k).sum()),small=int((l==k).sum())<32) for p,l in labels.items() for k in range(16)])
    table(D/'routing_role_metrics.csv',role_rows(baselines,labels));table(D/'deletion_damage.csv',rows)
    save(D/'numerical_checks.json',dict(profile=read(D/'numerical_checks.json'),production_checks=checks,training_updates=0))
    accounting=read(D/'compute_accounting.json');accounting.update(measure_wall_seconds=time.monotonic()-started,cache_files=len(list((D/'cache').glob('*.pt'))),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated());save(D/'compute_accounting.json',accounting)
    forecasts,decisions=forecast(rows,groups);table(D/'prospective_forecasts.csv',forecasts)
    combined(js,batches,groups,decisions,rows)
    print('Measurement and prospective evaluation complete',flush=True)


def forecast(rows,groups):
    lookup={(r['kind'],r['step'],r['probe'],r['group']):r['damage'] for r in rows if r['subset']=='whole'}
    output=[];decisions=[]
    for kind in ('sparse','unpruned'):
      for i,t in enumerate(STEPS[:-1]):
       for tol in TOLERANCES:
        for rule in ('single','persistent'):
         if rule=='persistent' and i==0:continue
         selected=[]
         for key in groups:
            current=lookup[(kind,t,'A',key)];past=lookup[(kind,STEPS[i-1],'A',key)] if i else None
            chosen=current<=tol and (rule=='single' or past<=tol)
            future=lookup[(kind,STEPS[i+1],'B',key)]
            output.append(dict(seed=17,kind=kind,expert=groups[key]['expert'],group=key,decision_step=t,future_step=STEPS[i+1],tolerance=tol,rule=rule,selected=chosen,current_A_damage=current,previous_A_damage=past,future_B_damage=future,false_safe=(future>tol) if chosen else None,current_B_damage=lookup[(kind,t,'B',key)],cross_probe_low_agreement=(current<=tol)==(lookup[(kind,t,'B',key)]<=tol),coordinates=len(groups[key]['indices'])))
            if chosen:selected.append(key)
         if rule=='persistent':decisions.append(dict(kind=kind,step=t,future=STEPS[i+1],tolerance=tol,selected=selected))
    return output,decisions


def combined(js,batches,groups,decisions,rows):
    masks=load_masks(js['sparse']['mask']);eligible=sum(int(x.sum()) for x in masks.values());cap=eligible//100
    lookup={(r['kind'],r['step'],r['probe'],r['group']):r['damage'] for r in rows if r['subset']=='whole'}
    output=[]
    for decision in decisions:
        chosen=[];count=0
        for key in decision['selected']:
            size=len(groups[key]['indices'])
            if count+size<=cap:chosen.append(key);count+=size
        common=dict(seed=17,kind=decision['kind'],decision_step=decision['step'],future_step=decision['future'],tolerance=decision['tolerance'],groups=';'.join(chosen),coordinates=count,cap=cap,all_eligible_coordinates=eligible,selection='persistent_A_previous_and_current',rule='canonical group order, whole groups within cap')
        if not chosen:
            output.append(dict(**common,status='undefined_no_selected_groups',step=None,probe=None,union_damage=None,sum_individual_damage=None,nonadditivity=None));continue
        indices={}
        for key in chosen:
            g=groups[key];indices.setdefault(g['parameter'],[]).append(g['indices'])
        indices={n:torch.cat(v) for n,v in indices.items()}
        for t in (decision['step'],decision['future']):
            m=model_for(js[decision['kind']],t);before=state_dict_hash(m.state_dict())
            for probe,bs in batches.items():
                base=load(cache_path(decision['kind'],t,probe))['baseline']
                with deletion(m,indices):out=evaluate(m,bs)
                actual=float((out['loss'].double()-base['loss'].double()).mean());summed=sum(lookup[(decision['kind'],t,probe,k)] for k in chosen)
                output.append(dict(**common,status='measured',step=t,probe=probe,union_damage=actual,sum_individual_damage=summed,nonadditivity=actual-summed))
            assert state_dict_hash(m.state_dict())==before
            del m;torch.cuda.empty_cache()
    table(D/'combined_deletion_checks.csv',output)


@contextlib.contextmanager
def worker():
    D.mkdir(parents=True,exist_ok=True);handles=[];stop=threading.Event()
    for label in ('study',socket.gethostname()):
        p=D/'locks'/(label+'.lock');p.parent.mkdir(exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
    def beat():
        while not stop.is_set():
            save(D/'heartbeat.json',dict(host=socket.gethostname(),pid=os.getpid(),unix=time.time()));stop.wait(15)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:yield
    finally:
        stop.set();thread.join()
        for f in handles:f.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','profile','measure']);a=ap.parse_args()
    torch.set_num_threads(2)
    with worker():
        with protected():globals()[a.stage]()
if __name__=='__main__':main()
