"""Exact SVD in FP32; fixed ranks, explicit undefined values, basis invariance."""
from gcommon import *
import math
from collections import OrderedDict
_CACHE=OrderedDict()
RANKS=(8,16,32)
def basis(w):
 key=(str(w.device),w.data_ptr(),w._version,tuple(w.shape))
 if key in _CACHE:
  value=_CACHE.pop(key);_CACHE[key]=value;return value[1:]
 # FP64 Gram eigensystem yields right singular directions; leading ranks are
 # checked against direct SVD. CPU avoids slow sparse-matrix CUDA Jacobi solves.
 x=w.detach().to(device='cpu',dtype=torch.float64)
 eig,v=torch.linalg.eigh(x.T@x)
 s=eig.flip(0).clamp_min(0).sqrt().float().to(w.device)
 vh=v.flip(1).T.float().to(w.device)
 _CACHE[key]=(w,s,vh.T)
 while len(_CACHE)>768:_CACHE.popitem(last=False)
 return s,vh.T

def alignment(v,r):
 n=float(r.square().sum())
 return None if n==0 else float((v.T@r).square().sum()/n)
def overlap(a,b):return float((a.T@b).square().sum()/a.shape[1])
def centered(r):return r-r.mean(0,keepdim=True)
def geometry(state,source=None,initial=None):
 records=[]
 for name,w in state.items():
  if not name.endswith('.fc1.weight') or '.moe.experts.' not in name:continue
  prefix,rest=name.split('.experts.');expert=int(rest.split('.')[0]); rn=prefix+'.router.projection.weight'
  router=state[rn].float();cr=centered(router);s,v=basis(w)
  refs={}
  for label,ref in [('source',source),('initial',initial)]:
   if ref is not None:refs[label]=basis(ref[name])[1]
  gen=torch.Generator(device=w.device).manual_seed(918273+expert)
  dirs=torch.randn((32,w.shape[1]),generator=gen,device=w.device);dirs=dirs/dirs.norm(dim=1,keepdim=True)
  for k in RANKS:
   if k>min(w.shape):continue
   vk=v[:,:k]; sh=[alignment(vk,r) for j,r in enumerate(cr) if j!=expert];valid=[x for x in sh if x is not None]
   cutoff=float(s[k-1]);nextval=float(s[k]) if k<len(s) else 0.;total=float(s.square().sum())
   row=dict(layer=prefix.split('.')[1],expert=expert,rank=k,alignment_centered=alignment(vk,cr[expert]),alignment_raw=alignment(vk,router[expert]),isotropic_baseline=k/w.shape[1],shuffled_mean=sum(valid)/len(valid) if valid else None,shuffled_sd=float(torch.tensor(valid).std(unbiased=False)) if valid else None,random_direction_mean=sum(alignment(vk,r) for r in dirs)/len(dirs),retained_energy=None if total==0 else float(s[:k].square().sum())/total,singular_max=float(s[0]),singular_cutoff=cutoff,relative_cutoff_gap=(cutoff-nextval)/max(cutoff,1e-30),degenerate_cutoff=(cutoff-nextval)/max(cutoff,1e-30)<1e-3,zero_matrix=total==0,rank_deficient=cutoff<=float(s[0])*1e-6,zero_centered_router=float(cr[expert].square().sum())==0,zero_raw_router=float(router[expert].square().sum())==0)
   if total==0:
    for key in ('alignment_centered','alignment_raw','shuffled_mean','shuffled_sd','random_direction_mean','source_subspace_overlap','initial_subspace_overlap','initial_router_current_expert','current_router_initial_expert','source_router_current_expert'):
     if key in row:row[key]=None
   row['singular_frobenius_energy']=total
   row['singular_median']=float(s.median())
   row['shuffled_alignment_values']=json.dumps(sh)
   row['random_direction_alignment_values']=json.dumps([alignment(vk,r) for r in dirs])
   row['alignment_above_shuffled']=None if row['alignment_centered'] is None or row['shuffled_mean'] is None else row['alignment_centered']-row['shuffled_mean']
   for label,b in refs.items():row[label+'_subspace_overlap']=overlap(vk,b[:,:k])
   if initial is not None:
    row['initial_router_current_expert']=alignment(vk,centered(initial[rn].float())[expert])
    row['current_router_initial_expert']=alignment(refs['initial'][:,:k],cr[expert])
   if source is not None:
    row['source_router_current_expert']=alignment(vk,centered(source[rn].float())[expert])
   if total==0:
    for key in list(row):
     if 'alignment' in key or 'subspace_overlap' in key:row[key]=None
   records.append(row)
 return records

def responses(model,inputs,reference=None,initial_assignments=None,native_traces=None,positions=None,expert_bases=None):
 """Fixed input vectors; all experts on first 128 vectors, actual capacity traffic on all."""
 data=[];raw={}
 for layer,(block,x) in enumerate(zip(model.blocks,inputs)):
  x=x.to(next(model.parameters()).device).float();moe=block.moe
  _,_,trace=moe(x.reshape(1,-1,x.shape[-1]));r=moe.router(x);sel=r.selected_experts;accepted=trace.accepted_mask
  if native_traces is not None:
   from types import SimpleNamespace
   nt=native_traces[layer];pos=positions.to(x.device);sel=nt.selected_expert_indices.reshape(-1,moe.top_k)[pos];accepted=nt.accepted_mask[pos]
   r=SimpleNamespace(probabilities=nt.probabilities.reshape(-1,moe.num_experts)[pos],entropy=nt.entropy.reshape(-1)[pos],margin=nt.margin.reshape(-1)[pos])
  top=r.probabilities.topk(2,dim=-1).indices;rr=centered(moe.router.projection.weight.float());win=rr[top[:,0]]-rr[top[:,1]]
  ref=None if reference is None else reference[layer]
  raw[str(layer)]={'selected':sel.cpu(),'probabilities':r.probabilities.cpu(),'accepted':accepted.cpu()}
  for e,expert in enumerate(moe.experts):
   subset=x[:128];pre=expert.fc1(subset);act=expert.activation(pre);out=expert(subset)
   source_out=None if ref is None else ref['outputs'][e].to(out.device)
   v=basis(expert.fc1.weight)[1] if expert_bases is None else expert_bases[layer,e].to(x.device)
   wn=win.square().sum(1);wa=(win@v[:,:16]).square().sum(1)/wn.clamp_min(1e-30)
   valid_winners=(sel[:,0]==e)&(wn>0)
   valid_outputs=None if source_out is None else (out.norm(dim=-1)>0)&(source_out.norm(dim=-1)>0)
   data.append(dict(layer=layer,expert=e,entropy_normalized=float(r.entropy.mean()/math.log(moe.num_experts)),margin_mean=float(r.margin.mean()),proposed=int((sel==e).sum()),accepted=int(((sel==e)&accepted).sum()),source_assignment_agreement=None if ref is None else float((sel.cpu()==ref['selected']).float().mean()),initial_assignment_agreement=None if initial_assignments is None else float((sel.cpu()==initial_assignments[str(layer)]['selected']).float().mean()),preactivation_mean=float(pre.mean()),preactivation_sd=float(pre.std(unbiased=False)),gelu_mean=float(act.mean()),gelu_rms=float(act.square().mean().sqrt()),output_rms=float(out.square().mean().sqrt()),source_output_mse=None if source_out is None else float((out-source_out).square().mean()),output_zero_vectors=int((out.norm(dim=-1)==0).sum()),source_output_zero_vectors=None if source_out is None else int((source_out.norm(dim=-1)==0).sum()),source_output_cosine=None if source_out is None or not bool(valid_outputs.any()) else float(torch.nn.functional.cosine_similarity(out[valid_outputs],source_out[valid_outputs],dim=-1).mean()),zero_winning_runner_directions=int((wn==0).sum()),winning_runner_alignment16=float(wa[valid_winners].mean()) if bool(valid_winners.any()) else None))
 return data,raw

def reference_responses(model,inputs):
 result=[]
 for b,x in zip(model.blocks,inputs):
  x=x.to(next(model.parameters()).device).float()
  result.append({'selected':b.moe.router(x).selected_experts.cpu(),'outputs':[e(x[:128]).cpu() for e in b.moe.experts]})
 return result
