"""Independent pre-loss coordinate, provenance and stop-policy audit."""
from common import *
import csv,importlib.util

def main():
 torch.set_num_threads(2);start=time.monotonic();design=read(D/'design.json');seal=read(D/'design_seal.json');assert sha(D/'design.json')==seal['design_sha256'] and sha(HERE/'prepare.py')==seal['prepare_sha256'];prep=read(D/'preparation_complete.json');reg=registry();assert sha(D/'registration.json')==prep['registration_sha256']
 spec=importlib.util.spec_from_file_location('verified_old_gradient_common',ROOT/'scripts/gradient_sensitivity_pruning_v1/common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.frozen()
 provenance=read(OLD/'checkpoint_and_calibration_provenance.json')['seeds'];manifest=read(D/'reused_score_manifest.json');oldinputs=read(D/'prior_artifact_hashes.json');assert all(sha(ROOT/p)==h for p,h in oldinputs.items());coordinate_hashes={};checks=[];verified=0;pair_support=[];balance_rows=list(csv.DictReader((D/'selection_and_balance.csv').open()))
 for seed in SEEDS:
  assert provenance[str(seed)]['checkpoints']['1000']['sha256']==reg['checkpoints'][str(seed)]['1000']['sha256'];assert sha(OLD/'calibration'/f'seed{seed}.pt')==provenance[str(seed)]['calibration_sha256'];status=read(OLD/'status'/f'measure_seed{seed}.json');assert status['status']=='COMPLETE';measurement=read(OLD/status['attempt']/'measurement_checks.json');assert all(measurement[k] for k in ('weight_hash_preserved','RNG_preserved','module_modes_preserved','existing_gradients_preserved','TF32_flags_preserved','calibration_batch_token_target_hashes_verified','coordinate_schema_verified'))
  cp=torch.load(source(seed)/'checkpoints/step_1000.pt',map_location='cpu',weights_only=True,mmap=True);assert cp['step']==1000;assert cp['config']==reg['source_configs'][str(seed)];assert state_dict_hash(cp['model'])==reg['checkpoints'][str(seed)]['1000']['model_hash'];p=D/'masks'/f'seed{seed}.pt';assert sha(p)==reg['mask_files'][str(seed)]['sha256'];masks=load(p);coordinate_hashes[str(p.relative_to(ROOT))]=sha(p);payload={}
  for branch in ('L','H'):
   assert len(masks[branch])==64
   counts=[]
   for n,idx in masks[branch].items():
    bits=n.split('.');layer=int(bits[1]);expert=int(bits[4]);assert expert in design['groups_by_seed_layer'][str(seed)][str(layer)][branch];assert idx.dtype==torch.int64 and idx.ndim==1;z=idx.numpy();assert len(np.unique(z))==len(z);w=cp['model'][n].numpy().reshape(-1);assert np.all((z>=0)&(z<len(w)))
    sp=OLD/'scores'/f'seed{seed}'/f'{n}.pt';key=str(sp.relative_to(ROOT));assert sha(sp)==manifest[key]==measurement['score_files'][str(sp.relative_to(OLD))];verified+=1;A=load(sp)['A'].numpy().reshape(-1);small=np.argsort(abs(w),kind='stable')[:len(w)//2];candidate=[]
    for ids in np.array_split(small,50):candidate.extend(ids[np.lexsort((ids,A[ids]))[:math.ceil(len(ids)/4)]])
    assert np.isin(z,np.array(candidate)).all();assert np.all(A[z]>0);rms=float(np.sqrt(np.mean(w.astype(np.float64)**2)));payload[(branch,n)]=dict(mag=abs(w[z].astype(np.float64)),A=A[z],rms=rms,candidate_A_min=float(A[candidate].min()),candidate_A_max=float(A[candidate].max()));counts.append(len(z));coordinate_hashes[f'seed{seed}/{branch}/{n}']=hashlib.sha256(z.tobytes()).hexdigest()
   checks.append(dict(seed=seed,branch=branch,targeted_tensors=64,targeted_experts=32,coordinates=sum(counts),minimum_tensor_count=min(counts),maximum_tensor_count=max(counts),dose=sum(counts)/67108864,candidate_membership=True,unique_in_bounds=True,score_hashes_original=True))
  from scipy.stats import ks_2samp
  for r in [x for x in balance_rows if int(x['seed'])==seed]:
   nl=f"blocks.{r['layer']}.moe.experts.{r['expert_L']}.{r['projection']}.weight";nh=f"blocks.{r['layer']}.moe.experts.{r['expert_H']}.{r['projection']}.weight";a=payload[('L',nl)];b=payload[('H',nh)];k=len(a['mag']);assert k==len(b['mag'])==int(r['removed_per_tensor'])
   delta=abs(math.log(a['rms']/b['rms']));pair_support.append(dict(seed=seed,layer=r['layer'],projection=r['projection'],pair=r['pair'],dose=k,RMS_L=a['rms'],RMS_H=b['rms'],abs_log_RMS_ratio=delta,joint_magnitude_calipers_incompatible=delta>.105,L_candidate_A_min=a['candidate_A_min'],L_candidate_A_max=a['candidate_A_max'],H_candidate_A_min=b['candidate_A_min'],H_candidate_A_max=b['candidate_A_max']))
   if not k:continue
   assert np.max(abs(np.log(a['mag']/b['mag'])))<=.005+1e-12
   assert np.max(abs(np.log(a['A']/b['A'])))<=.025+1e-12
   assert np.max(abs(np.log((a['mag']/a['rms'])/(b['mag']/b['rms']))))<=.1+1e-12
   for label,x,y in [('raw_magnitude',a['mag'],b['mag']),('A',a['A'],b['A']),('normalized_magnitude',a['mag']/a['rms'],b['mag']/b['rms'])]:
    den=np.sqrt((x.var()+y.var())/2);v=float((x.mean()-y.mean())/den) if den else float(0 if x.mean()==y.mean() else 1e99);xx=np.sort(x);yy=np.sort(y);support_values=np.sort(np.concatenate((xx,yy)));ks=float(np.max(abs(np.searchsorted(xx,support_values,side='right')/len(xx)-np.searchsorted(yy,support_values,side='right')/len(yy))))
    assert np.isclose(v,float(r[label+'_smd']),rtol=1e-8,atol=1e-10) and np.isclose(ks,float(r[label+'_ks']),atol=1e-12),(seed,r['layer'],r['projection'],r['pair'],label,v,r[label+'_smd'],ks,r[label+'_ks'])
   el=float(sum(a['mag']**2));eh=float(sum(b['mag']**2));assert np.isclose(abs(el-eh)/max(el,eh),float(r['energy_relative_difference']),atol=1e-12)
 table(D/'pair_support_diagnostics.csv',pair_support)
 save(D/'coordinate_seal.json',dict(approved_for_intervention=prep['status']=='COMPLETE',candidate_coordinate_hashes=coordinate_hashes,registration_sha256=sha(D/'registration.json'),selection_and_balance_sha256=sha(D/'selection_and_balance.csv'),design_sha256=sha(D/'design.json'),score_manifest_sha256=sha(D/'reused_score_manifest.json'),no_model_operations=True))
 rows=list(csv.DictReader((D/'selection_and_balance.csv').open()));failures=[]
 for r in rows:
  why=[]
  if int(r['removed_per_tensor'])!=2621:why.append('dose')
  for label in ('raw_magnitude','A','normalized_magnitude'):
   if r.get(label+'_smd') and abs(float(r[label+'_smd']))>(.2 if label=='normalized_magnitude' else .05):why.append(label+'_smd')
   if r.get(label+'_ks') and float(r[label+'_ks'])>(.1 if label=='normalized_magnitude' else .05):why.append(label+'_ks')
  if r.get('energy_relative_difference') and float(r['energy_relative_difference'])>.01:why.append('energy')
  assert (not why)==(r['passed']=='True')
  if why:failures.append(dict(seed=r['seed'],layer=r['layer'],projection=r['projection'],pair=r['pair'],expert_L=r['expert_L'],expert_H=r['expert_H'],failed_gates=';'.join(why),removed_per_tensor=r['removed_per_tensor']))
 table(D/'failed_balance_gates.csv',failures or [dict(status='NONE')]);table(D/'coordinate_checks.csv',checks)
 stopped=prep['status']!='COMPLETE'
 save(D/'numerical_checks.json',dict(array_and_provenance_checks_passed=True,selection_gate_passed=not stopped,stop_policy_respected=stopped,verified_used_score_files=verified,source_and_calibration_identity_verified=True,original_measurement_state_RNG_precision_checks_reused=True,original_gradient_implementation_seal_verified=True,candidate_membership_and_uniqueness=True,all_pair_gate_decisions_recomputed=True,all_five_sources_preserved=True,model_forward_backward_or_optimizer_calls=0,new_masking_training_resume_checks='NOT_RUN: pre-loss selection gate failed' if stopped else 'REQUIRED_BEFORE_PRODUCTION',production_allowed=False if stopped else 'ONLY_AFTER_ENGINEERING_CHECKS',wall_seconds=time.monotonic()-start))
 assert read(OLD/'completion.json')['status']=='STOPPED_MATCHING_GATE';assert not (D/'launch.json').exists();assert not (D/'attempts').exists() or not list((D/'attempts').glob('*'))
 print('VERIFIED',verified,'score files; failed paired selections',len(failures),flush=True)
if __name__=='__main__':main()
