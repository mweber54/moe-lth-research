"""Fixed native-input routing, task/aux gradients and actual interval updates."""
from af_common import *
import math

def model_state(model):return {n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
def build(state,config,condition,device):
 model=TinyMoELanguageModel(config['model']).to(device);model.load_state_dict(state);return configure(model,condition)
def routing(output):
 rows=[];assign=[]
 for layer,tr in enumerate(output.routes):
  selected=tr.selected_expert_indices.detach();accept=tr.accepted_mask.detach();k=tr.probabilities.shape[-1]
  proposed=torch.bincount(selected.flatten(),minlength=k).cpu().tolist();accepted=torch.bincount(selected.flatten()[accept.flatten()],minlength=k).cpu().tolist()
  rows.append(dict(layer=layer,proposed=proposed,accepted=accepted,dropped_assignment_fraction=1-sum(accepted)/sum(proposed),mean_selected_confidence=float(tr.selected_probabilities.mean()),mean_full_softmax_entropy=float(tr.entropy.mean()),assignment_count=sum(proposed)))
  assign.append(selected.cpu().flatten())
 return rows,torch.stack(assign)
def grads(model,batch,masks,weight):
 ids,targets=batch;params=[(n,p) for n,p in model.named_parameters() if p.requires_grad];device=next(model.parameters()).device
 with torch.autocast(device_type=device.type,enabled=False):
  out=model(ids.to(device));task=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),targets.to(device).flatten());aux=weight*out.auxiliary_loss
  taskg=torch.autograd.grad(task,[p for _,p in params],retain_graph=True,allow_unused=True)
  auxg=torch.autograd.grad(aux,[p for _,p in params],allow_unused=True) if aux.requires_grad else [None]*len(params)
 rows=[]
 for (n,p),gt,ga in zip(params,taskg,auxg):
  count=int(masks[n].sum()) if n in masks else p.numel()
  def norm(g):
   if g is None:return 0.
   q=g[masks[n].to(device)] if n in masks else g
   return float(q.detach().double().norm())
  nt,na=norm(gt),norm(ga)
  rows.append(dict(parameter=n,group=group(n),trainable=True,coordinate_count=count,denominator='surviving expert-weight coordinates' if n in masks else 'all unique parameter coordinates',task_total_l2=nt,weighted_aux_total_l2=na,native_aux_unweighted_total_l2=na/weight if weight else None,task_rms=nt/math.sqrt(count) if count else None,weighted_aux_rms=na/math.sqrt(count) if count else None,task_l2_per_coordinate=nt/count if count else None,weighted_aux_l2_per_coordinate=na/count if count else None,task_over_weighted_aux=nt/na if na>0 else None))
 totals=[]
 for g in ('E','R','S'):
  rr=[r for r in rows if r['group']==g];n=sum(r['coordinate_count'] for r in rr);nt=math.sqrt(sum(r['task_total_l2']**2 for r in rr));na=math.sqrt(sum(r['weighted_aux_total_l2']**2 for r in rr))
  totals.append(dict(group=g,trainable=bool(rr),coordinate_count=n,task_total_l2=nt if rr else None,weighted_aux_total_l2=na if rr else None,native_aux_unweighted_total_l2=na/weight if rr and weight else None,task_rms=nt/math.sqrt(n) if n else None,weighted_aux_rms=na/math.sqrt(n) if n else None,task_over_weighted_aux=nt/na if na else None))
 return dict(per_parameter=rows,groups=totals,task_ce=float(task.detach()),weighted_auxiliary_loss=float(aux.detach()),precision='FP32; production TF32 settings retained',aggregation='CE mean over all probe target tokens; auxiliary=model native mean across layers times configured weight')
def displacement(model,current,previous,initial,start,step):
 rows=[]
 for g in ('E','R','S'):
  names=[n for n,_ in model.named_parameters() if group(n)==g];delta=base=drift=initialnorm=0.
  for n in names:
   a=current[n].double();b=previous[n].double();z=initial[n].double();delta+=float((a-b).square().sum());base+=float(b.square().sum());drift+=float((a-z).square().sum());initialnorm+=float(z.square().sum())
  rows.append(dict(group=g,interval_start=start,step=step,actual_net_interval_update_l2=math.sqrt(delta),normalized_interval_update=math.sqrt(delta/base) if base>0 else None,cumulative_from_initial_l2=math.sqrt(drift),normalized_cumulative_drift=math.sqrt(drift/initialnorm) if initialnorm>0 else None,definition='net parameter displacement, not path length or gradient proxy',unique_parameter_count=sum(p.numel() for n,p in model.named_parameters() if group(n)==g)))
 return rows

def diagnose(model,batch,masks,condition,config,step,dest,full_path,previous_path=None,initial_path=None):
 with protected(model):
  model.eval();device=next(model.parameters()).device;current=model_state(model)
  with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
   rows,assign=routing(model(batch[0].to(device)))
   full=build(load(full_path)['model'],config,'full',device).eval();_,full_assign=routing(full(batch[0].to(device)));del full
  if step==0:init_assign=assign
  else:init_assign=load(dest/'assignments/step_0000.pt')['assignments']
  agree=[dict(layer=i,agreement_with_initial=float((assign[i]==init_assign[i]).float().mean()),agreement_with_corresponding_full=float((assign[i]==full_assign[i]).float().mean())) for i in range(len(rows))]
  previous=load(previous_path)['model'] if previous_path else current;initial=load(initial_path)['model'] if initial_path else current
  start=max(s for s in STEPS if s<step) if step else 0
  update=displacement(model,current,previous,initial,start,step)
  gradient=grads(model,batch,masks,float(config['routing']['aux_loss_weight']))
  for row in update:
   row['frozen']=row['group'] not in TRAINABLE[condition]
   if row['frozen'] and row['cumulative_from_initial_l2']!=0:raise RuntimeError('Frozen-group diagnostic drift')
  path=dest/f'assignments/step_{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(assignments=assign,probe_hash=R._batch_sequence_hash([batch])),path)
  save(dest/f'diagnostics/step_{step:04d}.json',dict(step=step,condition=condition,probe_hash=R._batch_sequence_hash([batch]),routing=rows,assignment_agreement=agree,updates=update,gradients=gradient,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,diagnostic_rng_modes_gradients_preserved=True))
 return current
