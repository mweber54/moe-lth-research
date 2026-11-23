"""Bounded reproducible checkpoint diagnostics on the frozen validation probe."""
from common import *
from scipy.stats import rankdata
import torch.nn.functional as F

def corr(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
 if len(x)<8:return None,None,'low_count'
 if not np.isfinite(x).all() or not np.isfinite(y).all():return None,None,'nonfinite'
 if x.std()<=1e-12 or y.std()<=1e-12:return None,None,'zero_variance'
 p=float(np.mean((x-x.mean())/x.std()*(y-y.mean())/y.std()));a,b=rankdata(x),rankdata(y)
 return p,float(np.corrcoef(a,b)[0,1]),'valid'
def pearson_only(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
 if len(x)<8 or not np.isfinite(x).all() or not np.isfinite(y).all() or x.std()<=1e-12 or y.std()<=1e-12:return None
 return float(np.mean((x-x.mean())/x.std()*(y-y.mean())/y.std()))
def spectrum(w):
 g=w.double().T@w.double();v,u=torch.linalg.eigh(g);return g,v.flip(0).clamp_min(0),u.flip(1)
def erank(v):
 v=np.maximum(np.asarray(v,dtype=float),0);total=v.sum()
 if not total:return None
 p=v[v>0]/total;return float(np.exp(-(p*np.log(p)).sum()))
def representation(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);v=np.linalg.eigvalsh((x-x.mean(0)).T@(x-x.mean(0))).clip(0,None)[::-1];norm=np.linalg.norm(x,axis=1)
 return dict(cka=cka(x,y),relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(y)) if np.linalg.norm(y) else None,mean_cosine=float(np.mean(np.sum(x*y,axis=1)/np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1),1e-30))),effective_rank=erank(v),top8_energy=float(v[:8].sum()/v.sum()) if v.sum() else None,norm_mean=float(norm.mean()),norm_q05=float(np.quantile(norm,.05)),norm_q95=float(np.quantile(norm,.95)))
def capture(m,tokens,positions):
 """Capture boundaries in original batch order, then select deterministic positions."""
 dev=next(m.parameters()).device;tokens=tokens.to(dev);positions=positions.to(dev);cache={};handles=[]
 def put(k,t):cache[k]=t.detach().reshape(-1,t.shape[-1])[positions].float().cpu().numpy()
 for i,b in enumerate(m.blocks):
  handles.append(b.register_forward_pre_hook(lambda mod,args,i=i:put(f'residual_before_L{i}',args[0])))
  handles.append(b.moe_norm.register_forward_pre_hook(lambda mod,args,i=i:put(f'post_attention_residual_L{i}',args[0])))
  handles.append(b.register_forward_hook(lambda mod,args,out,i=i:put(f'residual_after_L{i}',out[0])))
 try:
  put('token_embedding',m.token_embedding(tokens));pos=torch.arange(tokens.shape[1],device=dev);put('embedding_plus_position',m.token_embedding(tokens)+m.position_embedding(pos)[None]);o=m(tokens)
  for i,h in enumerate(o.pre_router_hidden_states):put(f'router_input_L{i}',h)
 finally:
  for h in handles:h.remove()
 return o,cache

def associations(scores,acts,selections,layer,mode):
 rows=[];pools=[];within=[];n,e=scores.shape
 for label,pairmask in selections.items():
  coefficients=np.full((e,e),np.nan);expert_coefficients=np.full((e,e),np.nan);counts=pairmask.sum(0)
  for j in range(e):
   use=pairmask[:,j];x,y=scores[use,j],acts[use,j];p,s,status=corr(x,y)
   for k in range(e):
    a=pearson_only(scores[use,k],y)
    if a is not None:coefficients[j,k]=a
    b=pearson_only(x,acts[use,k])
    if b is not None:expert_coefficients[j,k]=b
   rows.append(dict(layer=layer,mode=mode,selection=label,expert=j,n=int(use.sum()),pearson=p,spearman=s,status=status,router_pair_pearsons=json.dumps([float(v) if np.isfinite(v) else None for v in coefficients[j]]),expert_pair_pearsons=json.dumps([float(v) if np.isfinite(v) else None for v in expert_coefficients[j]])))
  ids=np.where(np.isfinite(coefficients).all(1))[0];valid=np.where(np.isfinite(coefficients.diagonal()))[0]
  def mean(v,c):return float(np.average(v,weights=c)) if len(v) and c.sum() else None
  actual=coefficients.diagonal()[valid];matched=coefficients.diagonal()[ids];shuffled=np.array([np.mean(np.delete(coefficients[j],j)) for j in ids]);weights=counts[ids]
  expert_ids=np.where(np.isfinite(expert_coefficients).all(1))[0];ew=counts[expert_ids];identity_experts=expert_coefficients.diagonal()[expert_ids];wrong_experts=np.array([np.mean(np.delete(expert_coefficients[j],j)) for j in expert_ids])
  pools.append(dict(matched_expert_control_valid_experts=json.dumps(expert_ids.tolist()),matched_expert_control_n_pairs=int(ew.sum()),matched_expert_control_identity=mean(identity_experts,ew),matched_expert_control_shuffle=mean(wrong_experts,ew),matched_expert_control_above_shuffle=mean(identity_experts-wrong_experts,ew),layer=layer,mode=mode,selection=label,paper_pooled_standardized_pearson=mean(actual,counts[valid]),equal_expert_pearson=float(actual.mean()) if len(actual) else None,n_valid=len(valid),n_excluded=e-len(valid),n_pairs=int(counts[valid].sum()),matched_n_pairs=int(weights.sum()),n_matched_experts=len(ids),matched_valid_experts=json.dumps(ids.tolist()),matched_identity=mean(matched,weights),matched_shuffle=mean(shuffled,weights),matched_above_shuffle=mean(matched-shuffled,weights)))
 for t in range(min(128,n)):
  p,s,status=corr(scores[t],acts[t]);within.append(dict(layer=layer,mode=mode,position_index=t,pearson=p,spearman=s,status=status))
 return rows,pools,within

def probe(m,probe,source_cache,initial_cache,source_assign,initial_assign,dense_cache=None,dense_state=None):
 dev=next(m.parameters()).device;pos=probe['positions'].to(dev);raw={};groups=[];pools=[];within=[];reps=[];function=[];routes=[]
 o,cache=capture(m,probe['tokens'],probe['positions'])
 for ref,refcache in [('source',source_cache),('initial',initial_cache),('same_step_dense',dense_cache)]:
  if refcache is None:continue
  for name,x in cache.items():reps.append(dict(boundary=name,reference=ref,**representation(x,refcache[name])))
 for name,x in cache.items():raw['representation_'+name]=x
 raw['logits']=o.logits.reshape(-1,o.logits.shape[-1])[pos].float().cpu().numpy();raw['probe_ce']=F.cross_entropy(o.logits.reshape(-1,o.logits.shape[-1]),probe['targets'].to(dev).reshape(-1),reduction='none')[pos].float().cpu().numpy()
 for mode in ('fixed_source','native'):
  for l,b in enumerate(m.blocks):
   if mode=='native':trace=o.routes[l];x=torch.from_numpy(cache[f'router_input_L{l}']).to(dev)
   else:
    # Full source hidden batch preserves original capacity identities/order.
    full=probe['source_full_inputs'][l].to(dev);_,_,trace=b.moe(full);x=full.reshape(-1,full.shape[-1])[pos]
   selected=trace.selected_expert_indices.reshape(-1,b.moe.top_k)[pos];accepted=trace.accepted_mask[pos]
   sel=selected.cpu().numpy();acc=accepted.cpu().numpy();scores=b.moe.router.projection(x).float().cpu().numpy();acts=[]
   for j,e in enumerate(b.moe.experts):
    z=e.activation(e.fc1(x));acts.append(z.mean(-1).cpu().numpy())
    if dense_state is not None:
     xx=x[:128];a=e(xx);prefix=f'blocks.{l}.moe.experts.{j}.';ds={k:dense_state[prefix+k].to(dev) for k in ('fc1.weight','fc1.bias','fc2.weight','fc2.bias')};bb=F.linear(F.gelu(F.linear(xx,ds['fc1.weight'],ds['fc1.bias'])),ds['fc2.weight'],ds['fc2.bias']);an=a.norm(dim=-1);bn=bb.norm(dim=-1);valid=(an>0)&(bn>0)
     function.append(dict(mode=mode,layer=l,expert=j,n=128,n_valid=int(valid.sum()),zero_exclusions=int((~valid).sum()),relative_error=float((a-bb).norm()/bb.norm()) if float(bb.norm()) else None,cosine=float(F.cosine_similarity(a[valid],bb[valid]).mean()) if valid.any() else None))
   acts=np.stack(acts,1);ids=np.arange(b.moe.num_experts)[None,:];proposed=(sel[:,:,None]==ids[:,None,:]).any(1);accepted_pairs=((sel[:,:,None]==ids[:,None,:])&acc[:,:,None]).any(1);allpairs=np.zeros_like(proposed);allpairs[:128]=True
   choices=dict(all_expert=allpairs,current_proposed=proposed,current_accepted=accepted_pairs,fixed_initial=initial_assign[mode][l],fixed_source=source_assign[mode][l])
   gs,ps,ws=associations(scores,acts,choices,l,mode);groups+=gs;pools+=ps;within+=ws
   key=f'{mode}_L{l}';raw[key+'_scores']=scores;raw[key+'_activations']=acts;raw[key+'_proposed']=proposed;raw[key+'_accepted']=accepted_pairs
   for k,pairs in choices.items():raw[key+'_selection_'+k]=pairs
   routes.append(dict(mode=mode,layer=l,proposed_load=json.dumps(proposed.sum(0).tolist()),accepted_load=json.dumps(accepted_pairs.sum(0).tolist()),initial_assignment_agreement=float((proposed==initial_assign[mode][l]).all(1).mean()),source_assignment_agreement=float((proposed==source_assign[mode][l]).all(1).mean()),margin_mean=float(trace.margin.mean()),entropy_mean=float(trace.entropy.mean()),capacity_full_batch_tokens=int(probe['tokens'].numel()),selected_probe_positions=len(pos)))
 return dict(groups=groups,pools=pools,within=within,reps=reps,function=function,routes=routes),raw,cache

def assignments(m,probe):
 dev=next(m.parameters()).device;pos=probe['positions'].to(dev);o=m(probe['tokens'].to(dev));out={}
 for mode in ('fixed_source','native'):
  out[mode]=[]
  for l,b in enumerate(m.blocks):
   trace=o.routes[l] if mode=='native' else b.moe(probe['source_full_inputs'][l].to(dev))[2]
   sel=trace.selected_expert_indices.reshape(-1,b.moe.top_k)[pos].cpu().numpy();out[mode].append((sel[:,:,None]==np.arange(b.moe.num_experts)[None,None,:]).any(1))
 return out

def weight_geometry(s,refs,device):
 records=[];energy=[]
 names=[n for n in s if n.endswith('.fc1.weight') and '.moe.experts.' in n]
 for n in names:
  l=int(n.split('.')[1]);e=int(n.split('.experts.')[1].split('.')[0]);g,v,u=spectrum(s[n].to(device));tr=float(v.sum());gn=float(g.norm());r=s[f'blocks.{l}.moe.router.projection.weight'].double().to(device)
  for kind,rr in [('raw',r),('centered',r-r.mean(0))]:
   den=rr.square().sum(1)*tr;q=torch.einsum('ed,df,ef->e',rr,g,rr)/den.clamp_min(1e-300);valid=den>0;other=valid.clone();other[e]=False
   energy.append(dict(layer=l,expert=e,router=kind,q=float(q[e]) if valid[e] else None,unmatched_q_mean=float(q[other].mean()) if other.any() else None,isotropic=1/g.shape[0],trace=tr,all_pair_q=json.dumps([float(q[j]) if valid[j] else None for j in range(len(q))])))
  for ref,refstate in refs.items():
   h,hv,hu=spectrum(refstate[n].to(device));hn=float(h.norm());ht=float(hv.sum());dist=float((g/gn-h/hn).norm()) if gn and hn else None
   for k in (8,16,32):
    sv=v.sqrt();sh=hv.sqrt();rd=bool(sv[k-1]<=sv[0]*1e-6);deg=bool((sv[k-1]-sv[k])/sv[k-1].clamp_min(1e-300)<.001);hrd=bool(sh[k-1]<=sh[0]*1e-6);hd=bool((sh[k-1]-sh[k])/sh[k-1].clamp_min(1e-300)<.001)
    records.append(dict(layer=l,expert=e,reference=ref,rank=k,subspace_overlap=float((u[:,:k].T@hu[:,:k]).square().sum()/k) if not(rd or deg or hrd or hd) and tr and ht else None,rank_deficient=rd,degenerate_cutoff=deg,reference_rank_deficient=hrd,reference_degenerate_cutoff=hd,alignment_raw=float((u[:,:k].T@r[e]).square().sum()/r[e].square().sum()) if r[e].norm() and not(rd or deg) else None,alignment_centered=float((u[:,:k].T@(r[e]-r.mean(0))).square().sum()/(r[e]-r.mean(0)).square().sum()) if (r[e]-r.mean(0)).norm() and not(rd or deg) else None,retained_energy=float(v[:k].sum()/tr) if tr else None,reference_retained_energy=float(hv[:k].sum()/ht) if ht else None,normalized_gram_distance=dist,gram_cosine=float((g*h).sum()/(gn*hn)) if gn and hn else None,trace=tr,reference_trace=ht,unnormalized_difference=float((g-h).norm()),relative_unnormalized_difference=float((g-h).norm()/hn) if hn else None))
 return records,energy

def embedding(s,refs):
 e=s['token_embedding.weight'].double().numpy();g=e@e.T;norm=np.linalg.norm(e,axis=1);v=np.linalg.svd(e,compute_uv=False)**2;rows=[];rowdata=[]
 assert torch.equal(s['token_embedding.weight'],s['lm_head.weight'])
 for label,ref in refs.items():
  b=ref['token_embedding.weight'].double().numpy();bn=np.linalg.norm(b,axis=1);bg=b@b.T;sim=g/np.maximum(norm[:,None]*norm[None,:],1e-30);bs=bg/np.maximum(bn[:,None]*bn[None,:],1e-30)
  rows.append(dict(reference=label,**representation(e,b),embedding_effective_rank=erank(v),top8_spectral_energy=float(v[:8].sum()/v.sum()),token_similarity_relative_distance=float(np.linalg.norm(sim-bs)/np.linalg.norm(bs)),singular_energy=json.dumps(v.tolist()),tied=True,tokenizer='byte',weighting='unweighted'))
  for j in range(len(e)):rowdata.append(dict(reference=label,byte_id=j,norm=float(norm[j]),reference_norm=float(bn[j]),drift=float(np.linalg.norm(e[j]-b[j]))))
 return rows,rowdata
