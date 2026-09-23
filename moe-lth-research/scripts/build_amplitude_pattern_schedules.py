"""Compile seven immutable dense-source transformations only after all 18 captures validate."""
from pathlib import Path
import gc,json,hashlib
import numpy as np
import torch
import pandas as pd
from moe_lth.experiments.fixed_schedule_intervention import file_hash
from moe_lth.experiments.optimization_geometry_diagnostics import append,digest
from moe_lth.experiments.amplitude_pattern_transforms import block_permutation,token_permutation,accepted_means,materialize_static,source_coefficient_dtype
ROOT=Path('results/gate_amplitude_pattern_decomposition')
CONDITIONS=['global_mean','layer_mean','expert_layer_mean','token_conditioned','temporal','temporal_shuffle','token_shuffle']

def coefficient_hash(layers):
 h=hashlib.sha256()
 for x in layers:
  x=np.ascontiguousarray(x);h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.tobytes())
 return h.hexdigest()

def build_source(row):
 source=Path(row.path);manifest=json.loads((source/'manifest.json').read_text());assert manifest['status']=='validated'
 e,seed,age=int(row.experts),int(row.seed),int(row.age);identity=dict(experts=e,seed=seed,age=age)
 out=ROOT/f'transformed/e{e}/seed{seed}/R{age}'
 if (out/'manifest.json').exists():
  old=json.loads((out/'manifest.json').read_text());assert old['status']=='validated';return
 out.mkdir(parents=True,exist_ok=False)
 records={}
 for key,sha in manifest['entries'].items():
  assert file_hash(source/key)==sha;v=torch.load(source/key,map_location='cpu',weights_only=False)
  assert (v['experts'],v['seed'],v['age'],v['kind'])==(e,seed,age,'dense')
  v['key']=key;v['amp']=[x['amplitudes'].numpy() for x in v['layers']];v['ids']=[x['expert_ids'].numpy() for x in v['layers']];v['ok']=[x['accepted'].numpy().reshape(x['expert_ids'].shape) for x in v['layers']];records[key]=v
 train=sorted((v for v in records.values() if v['phase']=='train'),key=lambda v:v['step']);val=sorted((v for v in records.values() if v['phase']=='validation'),key=lambda v:(v['step'],v['batch_index']))
 assert len(train)==2500 and len(val)==1581
 layers=len(train[0]['layers']);assert layers==4 and all(v['amp'][0].shape[-1]==1 for v in records.values())
 sums=[];counts=[];squares=[];flat=[];token_sums={};token_counts={}
 # Exact recurring batch+position+expert identities, separately for training/validation.
 for phase,seq in [('train',train),('validation',val)]:
  for v in seq:
   su,co=accepted_means(v['amp'],v['ids'],v['ok'],e);v['sums']=su;v['counts']=co
   key=(phase,v['input_hash']);n=int(np.prod(v['amp'][0].shape))
   if key not in token_sums:
    token_sums[key]=np.zeros((layers,n,e),np.float64);token_counts[key]=np.zeros((layers,n,e),np.uint8)
   for l,(a,ids,ok) in enumerate(zip(v['amp'],v['ids'],v['ok'])):
    pos=np.flatnonzero(ok.reshape(-1));ex=ids.reshape(-1)[pos].astype(int);values=a.reshape(-1)[pos].astype(np.float64)
    token_sums[key][l,pos,ex]+=values;token_counts[key][l,pos,ex]+=1
   if phase=='train':
    sums.append(su);counts.append(co);sq,_=accepted_means([a.astype(np.float64)**2 for a in v['amp']],v['ids'],v['ok'],e);squares.append(sq)
    flat.extend(a[ok] for a,ok in zip(v['amp'],v['ok']))
 sums=np.array(sums);counts=np.array(counts);squares=np.array(squares);total_s=sums.sum(0);total_n=counts.sum(0)
 repeat_weight=sum(int(c[c>=2].sum()) for (phase,_),c in token_counts.items() if phase=='train')/int(counts.sum())
 exact_tokens=repeat_weight>=.8
 token_description='exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged'
 token_features={};feature_means={}
 if exact_tokens:
  means={k:np.divide(token_sums[k],token_counts[k],out=np.zeros_like(token_sums[k]),where=token_counts[k]>0) for k in token_sums}
 else:
  # Existing tokenizer ID is a dataset-defined categorical feature, not interpolation.
  from moe_lth.config import load_config
  from moe_lth.data import build_dataloaders
  from moe_lth.experiments import run_router_age_recovery as recovery
  from moe_lth.utils import resolve_data_seed,seed_everything
  run=json.loads((ROOT/'capture_run_manifest.json').read_text());cfg=load_config(next(x['config'] for x in run['conditions'] if (x['experts'],x['seed'],x['age'])==(e,seed,age)))
  seed_everything(seed);ta,va=build_dataloaders(cfg['data'],int(cfg['training']['batch_size']),resolve_data_seed(cfg));batches={'train':recovery._materialize_batches(ta,2500),'validation':recovery._materialize_validation_batches(va,32)}
  vocab=int(cfg['model']['vocab_size']);means={}
  for phase,seq in [('train',train),('validation',val)]:
   su=np.zeros((layers,vocab,e),np.float64);co=np.zeros((layers,vocab,e),np.int64)
   for v in seq:
    tok=batches[phase][v['step']-1 if phase=='train' else v['batch_index']][0];assert digest(tok)==v['input_hash'];tokens=tok.numpy().reshape(-1).astype(int);token_features[(phase,v['input_hash'])]=tokens
    for l,(a,i,ok) in enumerate(zip(v['amp'],v['ids'],v['ok'])):
     mask=ok.reshape(-1);index=(tokens[mask],i.reshape(-1)[mask].astype(int));np.add.at(su[l],index,a.reshape(-1)[mask].astype(np.float64));np.add.at(co[l],index,1)
   feature_means[phase]=np.divide(su,co,out=np.zeros_like(su),where=co>0)
  del batches
  token_description='documented substitution: tokenizer-ID categorical bins within expert/layer; training and validation estimated separately; exact recurrent-edge coverage below 80%; no interpolation'
  print(f'[token identity substitution] e{e} seed{seed} R{age}: {repeat_weight:.2%} exact recurrent coverage; using existing tokenizer-ID bins',flush=True)
 token_identity_second_moment=sum(float(np.divide(su**2,token_counts[k],out=np.zeros_like(su),where=token_counts[k]>0).sum()) for k,su in token_sums.items() if k[0]=='train')/float(counts.sum())
 del token_sums
 permutation=block_permutation(counts,seed);np.save(out/'temporal_permutation_train.npy',permutation)
 val_by_batch={b:sorted([v for v in val if v['batch_index']==b],key=lambda v:v['step']) for b in sorted({v['batch_index'] for v in val})}
 val_perm={b:block_permutation(np.array([v['counts'] for v in seq]),seed) for b,seq in val_by_batch.items()}
 time_s={};time_n={}
 for v in val:
  time_s.setdefault(v['step'],np.zeros((layers,e)));time_n.setdefault(v['step'],np.zeros((layers,e),np.int64));time_s[v['step']]+=v['sums'];time_n[v['step']]+=v['counts']
 result={**identity,'status':'building','source_manifest_hash':file_hash(source/'manifest.json'),'canonical_source':str(source),'conditions':CONDITIONS,'recurring_training_edge_fraction':repeat_weight,'token_conditioned_identity':token_description,'temporal_shuffle':'equal-count expert/layer block permutation, singleton strata fixed; ordered accepted edges transferred intact; token mapping can change','phase_entries':{},'files':{},'amplitude_hashes':{},'shuffle_audit':{},'coefficient_dtypes':{}}
 for phase,seq in [('train',train),('validation',val)]:
  dtype=source_coefficient_dtype(a for v in seq for a in v['amp']);result['coefficient_dtypes'][phase]=str(dtype)
  max_b=max(v['shape'][0] for v in seq);max_s=max(v['shape'][1] for v in seq)
  arrays={c:np.lib.format.open_memmap(out/f'{c}_{phase}.npy',mode='w+',dtype=dtype,shape=(len(seq),layers,max_b,max_s,1)) for c in CONDITIONS}
  entries={};hashes={c:[] for c in CONDITIONS};perturb=dict(accepted_edges=0,temporal_values_changed=0,token_values_changed=0,temporal_step_moved_edges=0,token_index_moved_edges=0)
  for index,v in enumerate(seq):
   ids,ok=v['ids'],v['ok'];batch,seq_len=v['shape'];all_values={c:materialize_static(ids,ok,total_s,total_n,c,dtype) for c in CONDITIONS[:3]}
   key=(phase,v['input_hash']);token=[];shuffled=[];token_shuffled=[]
   for l,(a,i,accepted) in enumerate(zip(v['amp'],ids,ok)):
    pos=np.arange(i.size);values=(means[key][l,pos,i.reshape(-1).astype(int)] if exact_tokens else feature_means[phase][l,token_features[key],i.reshape(-1).astype(int)]).reshape(i.shape);token.append(np.where(accepted,values,0).astype(dtype))
    time_values=np.zeros(a.shape,dtype=dtype);token_values=np.zeros(a.shape,dtype=dtype)
    for expert in range(e):
     target_pos=np.flatnonzero((accepted&(i==expert)).reshape(-1));n=len(target_pos)
     if not n:continue
     if phase=='train':source_v=train[int(permutation[index,l,expert])]
     else:
      step_index=v['step']//50;source_v=val_by_batch[v['batch_index']][int(val_perm[v['batch_index']][step_index,l,expert])]
     source_pos=np.flatnonzero((source_v['ok'][l]&(source_v['ids'][l]==expert)).reshape(-1));assert len(source_pos)==n
     src=source_v['amp'][l].reshape(-1)[source_pos];time_values.reshape(-1)[target_pos]=src
     assert np.array_equal(src,time_values.reshape(-1)[target_pos]),f'Temporal coefficient precision mismatch: {v["key"]}, layer {l}, expert {expert}'
     own=a.reshape(-1)[target_pos];perm=token_permutation(n,seed,v['step'],l,expert);token_values.reshape(-1)[target_pos]=own[perm]
     perturb['accepted_edges']+=n;perturb['temporal_values_changed']+=int(np.count_nonzero(src!=own));perturb['token_values_changed']+=int(np.count_nonzero(own[perm]!=own));perturb['temporal_step_moved_edges']+=n if source_v['step']!=v['step'] else 0;perturb['token_index_moved_edges']+=int(np.count_nonzero(perm!=np.arange(n)))
     assert np.array_equal(np.sort(own),np.sort(token_values.reshape(-1)[target_pos])),f'Token coefficient precision mismatch: {v["key"]}, layer {l}, expert {expert}; source={a.dtype}, destination={dtype}'
    shuffled.append(time_values);token_shuffled.append(token_values)
   all_values['token_conditioned']=token
   all_values['temporal']=materialize_static(ids,ok,v['sums'] if phase=='train' else time_s[v['step']],v['counts'] if phase=='train' else time_n[v['step']],'expert_layer_mean',dtype)
   all_values['temporal_shuffle']=shuffled;all_values['token_shuffle']=token_shuffled
   for c,values in all_values.items():
    assert all(np.isfinite(a).all() and (a>=0).all() and (a<=1).all() for a in values)
    arrays[c][index]=0
    for l,a in enumerate(values):arrays[c][index,l,:batch,:seq_len]=a
    hashes[c].append(coefficient_hash(values))
   entries[v['key']]=dict(index=index,shape=v['shape'],input_hash=v['input_hash'],donor_file_hash=v['source_donor_file_hash'])
  result['phase_entries'][phase]=entries;result['amplitude_hashes'][phase]=hashes
  perturb.update({k+'_fraction':v/perturb['accepted_edges'] for k,v in list(perturb.items()) if k!='accepted_edges'});result['shuffle_audit'][phase]=perturb;append(ROOT/'shuffle_effective_perturbation.csv',[{**identity,'phase':phase,**perturb}])
  for c,arr in arrays.items():
   arr.flush();path=out/f'{c}_{phase}.npy';result['files'][f'{c}_{phase}']=dict(path=str(path.resolve()),sha256=file_hash(path))
  del arrays
 # Marginal preservation follows from per-expert bijections of equal-length blocks; verify maps explicitly.
 for l in range(layers):
  for expert in range(e):
   assert np.array_equal(np.sort(permutation[:,l,expert]),np.arange(2500))
   assert np.array_equal(counts[:,l,expert],counts[permutation[:,l,expert],l,expert])
 n=float(total_n.sum());mean=float(total_s.sum()/n);variance=float(squares.sum()/n-mean**2)
 lm=total_s.sum(1)/total_n.sum(1);em=np.divide(total_s,total_n,out=np.zeros_like(total_s),where=total_n>0);tm=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
 vl=float((total_n.sum(1)*(lm-mean)**2).sum()/n);ve=float((total_n*(em-lm[:,None])**2).sum()/n);vt=float((counts*(tm-em[None])**2).sum()/n);vr=variance-vl-ve-vt;assert vr>=-1e-10
 append(ROOT/'amplitude_variance_decomposition.csv',[{**identity,'component':k,'variance':v,'fraction':v/variance if variance>0 else None,'decomposition':'nested weighted layer -> expert within layer -> time within expert/layer -> token residual'} for k,v in [('layer',vl),('expert_within_layer',ve),('time_within_expert_layer',vt),('token_residual',max(0.,vr))]])
 all_a=np.concatenate(flat).astype(np.float32);q=np.quantile(all_a,[.05,.25,.5,.75,.95]);step_mean=sums.sum((1,2))/counts.sum((1,2));ac=float(np.corrcoef(step_mean[:-1],step_mean[1:])[0,1])
 append(ROOT/'amplitude_statistics.csv',[{**identity,'mean':mean,'sd':variance**.5,'cv':variance**.5/mean,'q05':q[0],'q25':q[1],'q50':q[2],'q75':q[3],'q95':q[4],'lag1_step_mean_autocorrelation':ac,'early_mean':float(sums[:250].sum()/counts[:250].sum()),'late_mean':float(sums[-250:].sum()/counts[-250:].sum()),'recurring_edge_fraction':repeat_weight,'token_conditioned_definition':token_description,'token_identity_conditional_variance':token_identity_second_moment-mean**2,'between_layer_variance':vl,'between_expert_within_layer_variance':ve,'between_step_variance':float((counts.sum((1,2))*(step_mean-mean)**2).sum()/n),'within_expert_step_token_variance':max(0.,vr)}])
 append(ROOT/'amplitude_layer_expert_means.csv',[{**identity,'layer':l,'expert':expert,'mean':em[l,expert],'count':int(total_n[l,expert])} for l in range(layers) for expert in range(e)])
 append(ROOT/'amplitude_step_means.csv',[{**identity,'step':step+1,'mean':v} for step,v in enumerate(step_mean)])
 result['status']='validated';(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(f'[transforms validated] e{e} seed{seed} R{age}; recurring edges {repeat_weight:.2%}',flush=True)
 del records,train,val,means,token_counts,flat,all_a;gc.collect()

def main():
 m=json.loads((ROOT/'capture_run_manifest.json').read_text());assert m['status']=='all_sources_validated' and m['validated_sources']==18,'All 18 sources must validate first'
 sources=pd.read_csv(ROOT/'validated_amplitude_sources.csv');assert len(sources)==18
 for row in sources.itertuples():assert file_hash(Path(row.path)/'manifest.json')==row.manifest_sha256
 for row in sources.itertuples():build_source(row)
 result={'status':'all_transforms_validated','canonical_source':'age/seed dense Level-2 trajectory; identical for sparse/dense','conditions':CONDITIONS,'sources':18,'recipient_recoveries':252,'primary_readout':'primary_r100_common_readout','retention_denominator_tolerance':1e-5,'minimum_recurrent_edge_fraction_for_exact_token_means':.8,'permutation_policy':'one deterministic seed-level rank permutation stratified by exact block length; token permutations are seed/step/layer/expert indexed; both model kinds and ages share indexing','source_manifests':[str((ROOT/f'transformed/e{int(r.experts)}/seed{int(r.seed)}/R{int(r.age)}/manifest.json').resolve()) for r in sources.itertuples()]}
 (ROOT/'amplitude_transform_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
