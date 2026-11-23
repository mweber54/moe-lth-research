"""Exposure audit then one prespecified bounded matching pass; no model operations."""
from common import *
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp
import csv

def smd(x,y):
 den=np.sqrt((x.var()+y.var())/2)
 return float((x.mean()-y.mean())/den) if den else float(0 if x.mean()==y.mean() else 1e99)
def balance(x,y):return dict(smd=smd(x,y),ks=float(ks_2samp(x,y).statistic),mean_L=float(x.mean()),mean_H=float(y.mean()))
def candidates(w,s):
 mag=np.abs(w);small=np.argsort(mag,kind='stable')[:w.size//2];out=[]
 for ids in np.array_split(small,50):out.extend(ids[np.lexsort((ids,s[ids]))[:math.ceil(len(ids)/4)]])
 return np.array(out,dtype=np.int64)
def match(wl,wh,al,ah,seed):
 il=candidates(wl,al);ih=candidates(wh,ah);rl=np.sqrt(np.mean(wl**2));rh=np.sqrt(np.mean(wh**2))
 assert min(rl,rh)>1e-12 and np.all(al[il]>0) and np.all(ah[ih]>0)
 # Distances in registered log-caliper units. At most16 neighbors per L candidate, single pass.
 fl=np.stack((np.log(np.abs(wl[il]))/.005,np.log(al[il])/.025),1);fh=np.stack((np.log(np.abs(wh[ih]))/.005,np.log(ah[ih])/.025),1)
 dist,near=cKDTree(fh).query(fl,k=16,workers=1);order=np.random.default_rng(seed).permutation(len(il));used=set();left=[];right=[]
 for j in order:
  for k in near[j]:
   if int(k) in used:continue
   if np.any(np.abs(fl[j]-fh[k])>1):continue
   if abs(math.log(abs(wl[il[j]])/rl)-math.log(abs(wh[ih[k]])/rh))>.1:continue
   left.append(il[j]);right.append(ih[k]);used.add(int(k));break
  if len(left)==2621:break
 return np.array(left,dtype=np.int64),np.array(right,dtype=np.int64),len(il),len(ih)
def main():
 torch.set_num_threads(2);assert not (D/'design.json').exists(),'Do not overwrite or retune a registered design'
 start=time.monotonic();coverage=list(csv.DictReader((ROOT/'results/expert_exposure_weight_development_v1/exposure_coverage.csv').open()));inventory=[];groups={};separation=[];inputs={}
 for seed in SEEDS:
  row=next(r for r in coverage if r['trajectory']=='original_source' and int(r['seed'])==seed);p=Path(row['log_path']);assert sha(p)==row['sha256'];inputs[str(p.relative_to(ROOT))]=sha(p)
  logs=[json.loads(x) for x in p.open()];assert [r['step'] for r in logs]==list(range(1,2501));logs=logs[:1000];groups[str(seed)]={}
  for layer in range(4):
   cumulative=np.array([sum(r['accepted'][f'layer_{layer}_expert_{e}'] for r in logs) for e in range(32)]);order=np.lexsort((np.arange(32),cumulative));lo=order[:8].tolist();hi=order[-8:].tolist();groups[str(seed)][str(layer)]={'L':lo,'H':hi}
   separation.append(dict(seed=seed,layer=layer,cv=float(cumulative.std()/cumulative.mean()),low_mean=float(cumulative[lo].mean()),high_mean=float(cumulative[hi].mean()),ratio=float(cumulative[hi].mean()/cumulative[lo].mean()),minimum=int(cumulative.min()),maximum=int(cumulative.max())))
   for e in range(32):
    key=f'layer_{layer}_expert_{e}';r=dict(seed=seed,layer=layer,expert=e,group='L' if e in lo else 'H' if e in hi else 'middle',historical_gate_weighted='unavailable',historical_input_diversity='unavailable')
    for label,z in [('cumulative_1_1000',logs),('recent_751_1000',logs[750:])]:
     a=sum(t['accepted'][key] for t in z);pp=sum(t['proposed'][key] for t in z)
     r.update({label+'_accepted':a,label+'_proposed':pp,label+'_dropped':pp-a,label+'_updates_with_accepted':sum(t['accepted'][key]>0 for t in z),label+'_applied_accepted':sum(t['accepted'][key] for t in z if t['optimizer_step_applied']),label+'_amp_skips':sum(not t['optimizer_step_applied'] for t in z)})
    inventory.append(r)
 table(D/'exposure_inventory.csv',inventory);table(D/'exposure_separation.csv',separation)
 # These quantities have been inspected; all following choices precede matching and all pruning outcomes.
 design=dict(source_seeds=list(SEEDS),checkpoint=1000,exposure='cumulative accepted tokens on attempts1..1000, within layer',recent_window=[751,1000],groups='ascending cumulative accepted, expert-ID tie; first8 L,last8 H; pair equally ranked positions within each group',groups_by_seed_layer=groups,minimum_upper_lower_mean_ratio=1.10,score='existing native FP32 TF32-off task-only mean batchwise abs(w*g),32 sealed past training batches; raw actual-traffic A, no exposure normalization',candidate='bottom half abs magnitude within tensor;50 equal ordinal magnitude strata; bottom ceil25% A per stratum, coordinate ties',matcher='Single fixed random permutation of L candidates, seed 1701000+10000*seed+100*layer+10*projection_index+pair_index. KD nearest16 H in log-magnitude/.005 and log-A/.025; first unused neighbor satisfying coordinate calipers. Stop at2621, no repair/optimization or dose reduction.',pair_calipers=dict(log_raw_magnitude=.005,log_A=.025,log_RMS_normalized_magnitude=.1),dose_per_targeted_tensor=2621,targeted_experts_per_layer_branch=8,whole_expert_dose=2621*64/67108864,balance_gates=dict(raw_magnitude_absolute_smd=.05,raw_magnitude_ks=.05,A_absolute_smd=.05,A_ks=.05,normalized_magnitude_absolute_smd=.2,normalized_magnitude_ks=.1,removed_energy_relative_difference=.01),coverage='All64 paired tensors in every seed must reach2621 and pass every gate; all20 layer exposure ratios >=1.10; otherwise STOP before any losses. No adaptive relaxation.',structural_matching='none; audit occupancy and neuron concentration as residuals',diagnostics='fixed first2 sealed training calibration batches at evalgrid; native FP32 TF32-off assignment/acceptance changes vs unpruned source1000; baseline expert output and gate-weighted output norm; per25-attempt exposure counts and gate sums; cumulative parameter displacement L2 from masked branch initialization, grouped expert/router/shared at evalgrid',production=dict(branches=list(BRANCHES),attempts=1500,source_steps=[1001,2500],validation_steps=list(EVAL_STEPS),optimizer='new identical AdamW/scaler/RNG reset in C,L,H; all parameters trainable; native FP16/aux/LR/clip/capacity',primary='final CE_L - CE_H at1500',secondary='immediate and registered trapezoidal excess-CE AUC vs new C; AUC/1500',pilot='seed17 retained; dispatch gate numerical/resource only; one worker/host',replication='all5 seed effects before descriptive mean/median/range/sample SD/SE; n5, no equivalence inference'))
 save(D/'design.json',design);save(D/'design_seal.json',dict(design_sha256=sha(D/'design.json'),prepare_sha256=sha(HERE/'prepare.py'),exposure_inputs=inputs,created=time.time(),no_model_operations=True))
 (D/'protocol.md').write_text('''# Exposure-stratified pruning intervention v1\n\nThe five original unpruned 32E seeds7,17,29,43,61 branch at1000. Historical exposure quantity is measured directly from every attempted update, including separate applied-update counts; it does not measure relevant or diverse input coverage. Gate-weighted exposure and historical composition are unavailable. Upper/lower exposure quartiles are within layer. Layer0 has a stronger contrast than later layers; all layer contrasts and exact IDs are retained. Choices in design.json were frozen after inventory and before matching/outcomes.\n\nReuse the verified training-only native FP32 task-gradient arrays and their split-half reliability/traffic. A is mean batchwise absolute weight-times-gradient, distinct from absolute signed mean or per-token sensitivity. No exposure-normalized importance is used. No-traffic sensitivity would be unmeasured. Validation never enters selection. Currently-small is bottom half in each fc1/fc2 tensor; low A is bottom quarter within50 magnitude strata. Pair eight low/high experts by exposure rank. A bounded nearest-neighbor pass matches magnitudes and raw A with equal dose and layer/projection allocation. All five sources must pass fixed gates or stop before losses; no exact row/column constraint, threshold sweep or expensive optimizer. Structural/history/function residuals remain confounds. Full definitions/tolerances are in immutable design.json.\n\nC is a newly run unpruned reset control. L/H permanently remove selected coordinates; all surviving experts, routers and shared parameters train. Source checkpoints lack optimizer/scaler/RNG, so use identical fresh AdamW/scaler/RNG after construction. Native LR0.0003, decay0.01, clipping1, auxiliary0.1, top1 capacity1.25, FP16+GradScaler, original subsequent batches1001..2500, exactly1500 attempted updates. Gradient masks and post-update zeroing persist. Full original validation, actual-token-weighted task CE, FP32 TF32 disabled, grid0,25,100,250,500,1000,1500. Applied/skipped updates are explicit.\n\nPrimary final L-H CE; immediate damage and integrated recovery are secondary. AUC is fixed-grid trapezoidal branch-minus-new-C over0..1500, also divided by1500. Show all seeds before aggregates; seed is independent n5. No coordinate/batch pseudoreplication or equivalence claim. Exposure trajectories are responses, never group redefinitions or primary adjustment. Assignment diagnostics are current checkpoint measurements on fixed training inputs, never historical diversity. Parameter displacement norms use each branch masked initialization, exclude direct mask subtraction, include per-expert and router/shared groups at registered snapshots.\n\nOne worker per Viper/Adder, independent locks, atomic status, persistent heartbeat, preserved failed attempts, complete optimizer/scaler/RNG resume state and log commit offsets. Hash-verify inputs on both hosts. Numerical/masking checks precede production; seed17 engineering pilot stays in cohort and gates correctness/resources only. Stop numerical failures without killing unrelated jobs. Production plan22500 attempted updates; diagnostics/engineering separate.\n\nPrior growth predicted small immediate sensitivity but mixed final damage. Reliable gradients have not yet established pruning utility. Exact-column matching was infeasible; this design does not recreate it. Router adaptation benefits sparse recovery without proving an exposure mechanism. The intervention tests a compound group-selection rule; historical exposure is observational and expert function/structure/history may confound it. Even consistent final L-H>0 does not prove waiting makes weights useful or saves compute. Mixed effects do not justify exposure-aware protection here. No pruning algorithm, classifier, winner selection or schedule is implemented.\n''')
 if not all(x['ratio']>=1.10 for x in separation):raise RuntimeError('Exposure contrast failed before matching')
 oldreg=read(ROOT/'results/magnitude_growth_pruning_v1/registration.json');reg=dict(source_configs=oldreg['source_configs'],checkpoints=oldreg['checkpoints'],data_identity=oldreg['data_identity'],mask_files={},source_seeds=list(SEEDS),branches=list(BRANCHES),design_sha256=sha(D/'design.json'),reuse_scores=True)
 score_manifest={};rows=[];history=[];structure=[];checks=[]
 reliability={(int(r['seed']),r['parameter']):r for r in csv.DictReader((OLD/'tensor_reliability_summary.csv').open())}
 for seed in SEEDS:
  states={st:torch.load(source(seed)/f'checkpoints/step_{st}.pt',map_location='cpu',weights_only=True,mmap=True)['model'] for st in (250,500,1000)};masks={'L':{},'H':{}}
  for st in states:assert sha(source(seed)/f'checkpoints/step_{st}.pt')==reg['checkpoints'][str(seed)][str(st)]['sha256']
  for layer in range(4):
   for pi,proj in enumerate(('fc1','fc2')):
    for pair,(el,eh) in enumerate(zip(groups[str(seed)][str(layer)]['L'],groups[str(seed)][str(layer)]['H'])):
     nl=f'blocks.{layer}.moe.experts.{el}.{proj}.weight';nh=f'blocks.{layer}.moe.experts.{eh}.{proj}.weight';meta=dict(seed=seed,layer=layer,projection=proj,pair=pair,expert_L=el,expert_H=eh)
     score=[];ww=[]
     for n in (nl,nh):
      p=OLD/'scores'/f'seed{seed}'/f'{n}.pt';score_manifest[str(p.relative_to(ROOT))]=sha(p);s=load(p);assert reliability[(seed,n)]['eligible']=='True';score.append(s);ww.append(states[1000][n].numpy().astype(np.float64).reshape(-1))
     al,ah=[s['A'].numpy().reshape(-1) for s in score];wl,wh=ww;il,ih,ncl,nch=match(wl,wh,al,ah,1701000+10000*seed+100*layer+10*pi+pair);k=len(il);r=dict(**meta,removed_per_tensor=k,candidates_L=ncl,candidates_H=nch)
     passed=k==2621
     if k:
      for label,x,y in [('raw_magnitude',abs(wl[il]),abs(wh[ih])),('A',al[il],ah[ih]),('normalized_magnitude',abs(wl[il])/np.sqrt(np.mean(wl**2)),abs(wh[ih])/np.sqrt(np.mean(wh**2)))]:
       b=balance(x,y);r.update({label+'_'+key:v for key,v in b.items()});passed &= abs(b['smd'])<=(.2 if label=='normalized_magnitude' else .05) and b['ks']<=(.1 if label=='normalized_magnitude' else .05)
      en=[float(np.sum(w[idx]**2)) for w,idx in ((wl,il),(wh,ih))];r.update(energy_L=en[0],energy_H=en[1],energy_relative_difference=abs(en[0]-en[1])/max(en));passed &= r['energy_relative_difference']<=.01
     r['passed']=bool(passed);rows.append(r)
     for branch,n,w,idx,s in zip(('L','H'),(nl,nh),ww,(il,ih),score):
      masks[branch][n]=torch.from_numpy(idx);rms=np.sqrt(np.mean(w*w));values=np.stack([states[t][n].numpy().astype(np.float64).reshape(-1)[idx] for t in (250,500,1000)]);t=np.array([250,500,1000.]);t-=t.mean();slope=np.sum(abs(values)*t[:,None],0)/sum(t*t)*750/rms;variation=np.sum(abs(np.diff(abs(values),axis=0)),axis=0)/rms
      history.append(dict(**meta,branch=branch,parameter=n,mean_growth_rms=float(slope.mean()) if k else None,mean_variation_rms=float(variation.mean()) if k else None,negative_fraction=float(np.mean(w[idx]<0)) if k else None,sign_change_fraction=float(np.mean(np.any(np.diff(np.sign(values),axis=0)!=0,axis=0))) if k else None,signed_mean_wg=float((-s['S'].numpy().reshape(-1)[idx]).mean()) if k else None,mean_gradient_variance=float(s['gradient_sample_variance'].numpy().reshape(-1)[idx].mean()) if k else None,split_half_median_stratum_rho=float(reliability[(seed,n)]['median_stratum_rho']),calibration_accepted=int(reliability[(seed,n)]['accepted_total'])))
      for axis,bins,ids in [('row',states[1000][n].shape[0],idx//states[1000][n].shape[1]),('column',states[1000][n].shape[1],idx%states[1000][n].shape[1])]:
       counts=np.bincount(ids,minlength=bins);structure.append(dict(**meta,branch=branch,parameter=n,axis=axis,occupied=int(np.count_nonzero(counts)),bins=bins,max_count=int(counts.max()),maximum_share=float(counts.max()/k) if k else None,hhi=float(np.sum((counts/k)**2)) if k else None))
  p=D/'masks'/f'seed{seed}.pt';atomic_tensor(p,masks);p.chmod(0o444);reg['mask_files'][str(seed)]=dict(path=str(p),sha256=sha(p),removed_coordinates_per_branch=sum(len(x) for x in masks['L'].values()))
  print('MATCHED',seed,'passes',sum(r['passed'] for r in rows if r['seed']==seed),'/64',flush=True)
 table(D/'selection_and_balance.csv',rows);table(D/'history_and_score_residuals.csv',history);table(D/'structural_balance.csv',structure);save(D/'reused_score_manifest.json',score_manifest)
 for p in ('registration.json','checkpoint_and_calibration_provenance.json','native_configuration_audit.json','score_algebra_audit.json','reliability_review.json','tensor_reliability_summary.csv','score_reliability.csv','exposure_summary.csv','analysis_seal.json','final_artifact_audit.json'):
  inputs[str((OLD/p).relative_to(ROOT))]=sha(OLD/p)
 for seed in SEEDS:
  p=OLD/'calibration'/f'seed{seed}.pt';inputs[str(p.relative_to(ROOT))]=sha(p)
 save(D/'prior_artifact_hashes.json',inputs);save(D/'registration.json',reg)
 passed=all(r['passed'] for r in rows) and len(rows)==320
 save(D/'preparation_complete.json',dict(status='COMPLETE' if passed else 'STOPPED_BALANCE_GATE',all_five_seeds=passed,registration_sha256=sha(D/'registration.json'),design_sha256=sha(D/'design.json'),pairs=320,passed_pairs=sum(r['passed'] for r in rows),no_model_operations=True,wall_seconds=time.monotonic()-start))
 for seed in SEEDS:
  for b in BRANCHES:save(D/'status'/f'seed{seed}_{b}.json',dict(status='PENDING' if passed else 'NOT_RUN_BALANCE_GATE',seed=seed,branch=b))
 print('PREPARATION',passed,flush=True)
if __name__=='__main__':main()
