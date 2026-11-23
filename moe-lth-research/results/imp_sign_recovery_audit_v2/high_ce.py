from common import *
import torch.nn.functional as F
from torch import nn
manifest=rows(OLD/'variant_manifest.csv');logged=pd.read_csv(OLD/'trajectory_losses.csv');refs=rows(OLD/'dense_reference_checks.csv')
results=[];lnrows=[];scales=[];code=[]
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
for file,lo,hi,topic in [('src/moe_lth/models/transformer.py',78,88,'embedding_and_tied_head_initialization'),('src/moe_lth/models/transformer.py',103,103,'unscaled_input_embedding'),('src/moe_lth/models/transformer.py',126,126,'unscaled_output_head'),('src/moe_lth/models/experts.py',7,16,'expert_linear_initialization_and_no_mask_compensation')]:
 lines=(ROOT/file).read_text().splitlines();code.append(dict(topic=topic,file=file,start_line=lo,end_line=hi,quote='\n'.join(f'{i}: {lines[i-1]}' for i in range(lo,hi+1))))
write('initialization_code.csv',code)
for seed in SEEDS:
 entry=p.native._seed_entry(seed);config=entry['config'];p.native.recovery.seed_everything(seed)
 _,loader=p.native.build_dataloaders(config['data'],int(config['training']['batch_size']),seed)
 val=p.native.recovery._materialize_validation_batches(loader,int(config['data']['validation_blocks']))
 vh=p.native.recovery._batch_sequence_hash(val);assert vh==entry['validation_batch_sequence_hash']
 base=next(r for r in manifest if int(r['seed'])==seed and r['variant']=='imp_original' and r['order']=='A')
 init=p.native._read_model_payload(Path(base['initial_checkpoint']))
 for arm in ('dense','imp_original','random1_original','random2_original'):
  if arm=='dense':state=init['model'];masks={}
  else:
   row=next(r for r in manifest if int(r['seed'])==seed and r['variant']==arm and r['order']=='A');state,masks=p.variant_state(row)
  model=p.native.TinyMoELanguageModel(config['model']).cuda();model.load_state_dict(state);model.eval()
  assert not any(m.training for m in model.modules())
  assert all(m.p==0 for m in model.modules() if isinstance(m,nn.Dropout))
  for name,t in state.items():
   if name in masks or name in ('token_embedding.weight','position_embedding.weight','final_norm.weight','final_norm.bias'):
    v=t.double();orig=init['model'][name].double();mask=masks.get(name)
    if mask is not None:assert torch.equal(t[mask],init['model'][name][mask]) and torch.count_nonzero(t[~mask])==0
    scales.append(dict(seed=seed,arm=arm,tensor=name,std=float(v.std()),rms=float(v.square().mean().sqrt()),original_std=float(orig.std()),retained_count=int(mask.sum()) if mask is not None else t.numel(),total_count=t.numel(),survivor_values_unchanged=bool(torch.equal(t[mask],init['model'][name][mask])) if mask is not None else None,survivor_value_check_applicable=mask is not None,variance_compensation_factor=1.0))
  layeracc={}
  def hook(name):
   def fn(mod,inputs,output):
    z=output.detach().float();dim=z.shape[-1];s=layeracc.setdefault(name,[0.,0.,0.,0.,0])
    s[0]+=float(z.mean(-1).double().sum());s[1]+=float(z.var(-1,unbiased=False).double().sum());s[2]+=float(z.double().square().sum());s[3]+=float(z.abs().max());s[4]+=z.numel()//dim
   return fn
  handles=[m.register_forward_hook(hook(n)) for n,m in model.named_modules() if isinstance(m,nn.LayerNorm)]
  sums=dict(loss=0.,independent_loss=0.,logit_sum=0.,logit_sq=0.,max_logit_sum=0.,target_logit_sum=0.,logz_sum=0.,entropy_sum=0.,max_prob_sum=0.,correct=0,input_copies=0,self_margin_sum=0.,self_logit_sum=0.,input_target_equal=0)
  count=0;vcount=0;maxabs=0.;minlogit=float('inf');maxlogit=-float('inf')
  with torch.inference_mode():
   for x,y in val:
    x=x.cuda();y=y.cuda();z=model(x).logits.float();flat=z.reshape(-1,z.shape[-1]);target=y.reshape(-1);current=x.reshape(-1)
    loss=F.cross_entropy(flat,target,reduction='sum');logz=flat.logsumexp(-1);tl=flat.gather(1,target[:,None]).squeeze(1);sl=flat.gather(1,current[:,None]).squeeze(1)
    independent=logz-tl;probs=flat.softmax(-1);ent=-(probs*flat.log_softmax(-1)).sum(-1);pred=flat.argmax(-1)
    for key,value in [('loss',loss),('independent_loss',independent.sum()),('logit_sum',flat.double().sum()),('logit_sq',flat.double().square().sum()),('max_logit_sum',flat.max(-1).values.sum()),('target_logit_sum',tl.sum()),('logz_sum',logz.sum()),('entropy_sum',ent.sum()),('max_prob_sum',probs.max(-1).values.sum()),('correct',(pred==target).sum()),('input_copies',(pred==current).sum()),('self_margin_sum',(sl-tl).sum()),('self_logit_sum',sl.sum()),('input_target_equal',(current==target).sum())]:sums[key]+=float(value.double().cpu())
    count+=target.numel();vcount+=flat.numel();maxabs=max(maxabs,float(flat.abs().max()));minlogit=min(minlogit,float(flat.min()));maxlogit=max(maxlogit,float(flat.max()))
  for h in handles:h.remove()
  if arm=='dense':
   ref=next(r for r in refs if int(r['seed'])==seed and r['order']=='A');old=next(readrow for readrow in [json.loads(line) for line in Path(ref['common_readout_path']).read_text().splitlines()] if readrow['step']==0);expected=old['cross_entropy']
  else:expected=float(logged[(logged.seed==seed)&(logged.order=='A')&(logged.variant==arm)&(logged.step==0)].cross_entropy.iloc[0])
  ce=sums['loss']/count;difference=ce-expected;assert abs(difference)<=1e-3,(seed,arm,difference)
  independent_ce=sums['independent_loss']/count;assert abs(independent_ce-ce)<1e-3
  results.append(dict(seed=seed,arm=arm,checkpoint_step=init['step'],num_experts=config['model']['num_experts'],nominal_sparsity=0 if arm=='dense' else .8,CE=ce,logged_CE=expected,baseline_difference=difference,independent_logsumexp_CE=independent_ce,independent_CE_difference=independent_ce-ce,chance_CE=math.log(config['model']['vocab_size']),target_tokens=count,validation_hash=vh,eval_mode=True,configured_dropout=config['model']['dropout'],tf32=False,logit_mean=sums['logit_sum']/vcount,logit_std=math.sqrt(sums['logit_sq']/vcount-(sums['logit_sum']/vcount)**2),logit_abs_max=maxabs,logit_min=minlogit,logit_max=maxlogit,mean_max_logit=sums['max_logit_sum']/count,mean_target_logit=sums['target_logit_sum']/count,mean_input_token_logit=sums['self_logit_sum']/count,mean_logsumexp=sums['logz_sum']/count,mean_input_minus_target_logit=sums['self_margin_sum']/count,entropy=sums['entropy_sum']/count,mean_max_probability=sums['max_prob_sum']/count,accuracy=sums['correct']/count,input_copy_fraction=sums['input_copies']/count,input_equals_target_fraction=sums['input_target_equal']/count,embedding_std=float(init['model']['token_embedding.weight'].std()),head_tied=model.lm_head.weight.data_ptr()==model.token_embedding.weight.data_ptr(),input_scale_factor=1.0,output_scale_factor=1.0,regime='extremely overconfident untrained model; not near chance'))
  for n,s in layeracc.items():lnrows.append(dict(seed=seed,arm=arm,layernorm=n,token_count=s[4],mean_output_mean=s[0]/s[4],mean_output_variance=s[1]/s[4]))
  write('high_ce_audit.csv',results);write('layernorm_audit.csv',lnrows);write('initialization_scaling.csv',scales)
  print(seed,arm,'CE',ce,'difference',difference,'copy_fraction',results[-1]['input_copy_fraction'],flush=True)
  del model,state;torch.cuda.empty_cache()
write('high_ce_status.csv',[dict(status='complete',evaluation_broken=False,baselines=len(results),all_baseline_differences_below_1e_3=True,phase2_started=False)])
