"""Isolated per-token local derivatives from actual model task backpropagation."""
from common import *
import torch.nn.functional as F

def cosine(a,b):
 den=a.norm()*b.norm();return float((a*b).sum()/den) if den>0 else None

def diagnostic(m,probe,masks):
 dev=next(m.parameters()).device;outputs={};inputs={};hooks=[]
 for l,b in enumerate(m.blocks):
  def hook(mod,args,out,l=l):inputs[l]=args[0];outputs[l]=out[0]
  hooks.append(b.moe.register_forward_hook(hook))
 try:
  with torch.enable_grad():
   o=m(probe['tokens'].to(dev));ce=F.cross_entropy(o.logits.reshape(-1,o.logits.shape[-1]),probe['targets'].to(dev).reshape(-1));up=torch.autograd.grad(ce,list(outputs.values()))
 finally:
  for h in hooks:h.remove()
 records=[]
 for l,b in enumerate(m.blocks):
  # Use an accepted real probe token and its actual CE upstream derivative.
  accepted=o.routes[l].accepted_mask.any(1);t=int(accepted.nonzero()[0]);x=inputs[l].detach().reshape(-1,m.config['d_model'])[t:t+1];upstream=up[l].detach().reshape_as(inputs[l].reshape(-1,m.config['d_model']))[t:t+1];expert=int(o.routes[l].selected_expert_indices.reshape(-1,1)[t,0]);module=b.moe
  caches={};hs=[]
  hs.append(module.router.projection.register_forward_hook(lambda mod,args,out:caches.update(z=out)))
  hs.append(module.experts[expert].fc1.register_forward_hook(lambda mod,args,out:caches.update(pre=out)))
  name=f'blocks.{l}.moe.experts.{expert}.fc1.weight';mask=masks.get(name,torch.ones_like(module.experts[expert].fc1.weight,dtype=torch.bool,device=dev)).to(dev)
  # Fixed masked parameterization: numerical gradients project onto surviving coordinates.
  hs.append(module.experts[expert].fc1.weight.register_hook(lambda grad,mask=mask:grad*mask))
  try:
   with torch.enable_grad():
    y,_,_=module(x.reshape(1,1,-1));aux=module.num_experts*(o.routes[l].usage.detach()*torch.softmax(caches['z']/module.router.temperature,dim=-1)[0]).sum()/probe['tokens'].numel()/len(m.blocks);task=(y.reshape_as(x)*upstream).sum();w=module.experts[expert].fc1.weight;r=module.router.projection.weight
    task_g=torch.autograd.grad(task,(caches['z'],r,caches['pre'],w),retain_graph=True,allow_unused=True);aux_g=torch.autograd.grad(aux,(caches['z'],r),retain_graph=True);combined=torch.autograd.grad(task+.1*aux,r)[0]
   gz,gr,delta,gw=task_g;pred=delta.reshape(-1,1)*x*mask;router_pred=gz.reshape(-1,1)*x
   p=torch.softmax(caches['z'].detach()/module.router.temperature,dim=-1);base=(task.detach()/p[0,expert]).item();expected=-base*p[0,expert]*p[0];expected[expert]+=base*p[0,expert]
   unselected=torch.arange(len(gr),device=dev)!=expert
   for row in range(len(gw)):
    mx=x[0]*mask[row];c=cosine(x[0],mx);signed=cosine(gr[expert],gw[row]);records.append(dict(layer=l,expert=expert,row=row,token_full_batch_index=t,masked_identity_max_error=float((gw[row]-pred[row]).abs().max()),router_identity_max_error=float((gr-router_pred).abs().max()),task_aux_decomposition_error=float((combined-gr-.1*aux_g[1]).abs().max()),softmax_derivative_error=float((gz[0]-expected).abs().max()),unselected_nonzero_task_logit_gradients=int((gz[0,unselected]!=0).sum()),retained_input_norm=float(mx.norm()/x.norm()),input_masked_cosine=c,input_masked_angle_degrees=float(np.degrees(np.arccos(np.clip(c,-1,1)))) if c is not None else None,signed_router_expert_gradient_cosine=signed,absolute_router_expert_gradient_cosine=abs(signed) if signed is not None else None,task_logit_gradient=float(gz[0,expert]),aux_logit_gradient=float(aux_g[0][0,expert]),preactivation_gradient=float(delta.reshape(-1)[row]),zero_gradient=bool(gw[row].norm()==0 or gr[expert].norm()==0),scope='single_token_local_CE_upstream; direct auxiliary contribution with original full-batch load, N and layer averaging'))
  finally:
   for h in hs:h.remove()
 return records
