"""Same accepted-slot dispatch, immutable payload checks once on CPU per batch."""
import math,types,hashlib
import torch
from moe_lth.models.moe_layer import RouteTrace

def install(model):
    for block in model.blocks:
        moe=block.moe
        if moe.top_k!=1:raise ValueError('Only registered top-1 cohort supported')
        moe._native_forward=moe.forward;moe._replay_required=False;moe._slot=None;moe._validated_slot_reference=None;moe._prepared_slot=None
        moe.forward=types.MethodType(forward,moe)

def prepare(slot,count,experts,capacity,device,dtype):
    ids=slot['ids'];gates=slot['gates'];accepted=slot['accepted']
    if ids.device.type!='cpu' or gates.device.type!='cpu' or accepted.device.type!='cpu':raise ValueError('Captured payload must be CPU')
    if ids.shape!=(count,1) or gates.shape!=ids.shape or accepted.shape!=ids.shape:raise ValueError('Slot shape mismatch')
    if ids.dtype!=torch.int64 or accepted.dtype!=torch.bool or slot['capacity']!=capacity:raise ValueError('Slot dtype/capacity mismatch')
    if not torch.equal(slot['token_positions'].sort().values,torch.arange(count)):raise ValueError('Token positions not bijective')
    if ids.min()<0 or ids.max()>=experts or not torch.isfinite(gates).all() or (gates<0).any() or (gates>1).any():raise ValueError('Invalid expert/gate')
    if len(slot['order'])!=experts:raise ValueError('Wrong expert count')
    seen=torch.zeros_like(accepted)
    for eid,pos in enumerate(slot['order']):
        if pos.device.type!='cpu' or pos.dtype!=torch.int64 or len(pos)>capacity or len(pos.unique())!=len(pos):raise ValueError('Invalid expert positions')
        if len(pos):
            if pos.min()<0 or pos.max()>=count or not (ids[pos]==eid).all() or not accepted[pos].all():raise ValueError('Expert/acceptance mismatch')
            seen[pos]=True
    if not torch.equal(seen,accepted):raise ValueError('Missing accepted positions')
    return dict(ids=ids.to(device),gates=gates.to(device=device,dtype=dtype).detach(),accepted=accepted.to(device),order=[p.to(device) for p in slot['order']],count=count,capacity=capacity,device=device,dtype=dtype,usage=torch.bincount(ids.flatten(),minlength=experts).to(device).float()/count)

def fingerprint(slot):
    digest=hashlib.sha256(str(slot['capacity']).encode())
    for value in [slot['ids'],slot['gates'],slot['accepted'],slot['token_positions']]+slot['order']:
        if value.device.type!='cpu':raise ValueError('Immutable slot must be CPU')
        v=value.detach().contiguous();digest.update(str(v.dtype).encode());digest.update(str(tuple(v.shape)).encode());digest.update(v.numpy().tobytes())
    return digest.hexdigest()

def forward(self,inputs,override_ids=None,*,capacity_enabled=True,expert_chunk_size=None):
    if not self._replay_required:return self._native_forward(inputs,override_ids,capacity_enabled=capacity_enabled,expert_chunk_size=expert_chunk_size)
    if self._slot is None or override_ids is not None:raise RuntimeError('Explicit replay payload required')
    if not capacity_enabled or expert_chunk_size is not None:raise ValueError('Native capacity, unchunked replay required')
    slot=self._slot;self._slot=None;batch,length,width=inputs.shape;flat=inputs.reshape(-1,width);capacity=max(1,math.ceil(self.capacity_factor*len(flat)/self.num_experts))
    native=self.router(flat)
    signature=fingerprint(slot)
    if slot is not self._validated_slot_reference:
        self._prepared_slot=prepare(slot,len(flat),self.num_experts,capacity,flat.device,native.probabilities.dtype);self._validated_slot_reference=slot;self._prepared_slot['fingerprint']=signature
    elif self._prepared_slot['fingerprint']!=signature:raise ValueError('Previously validated immutable payload changed')
    p=self._prepared_slot
    if (p['count'],p['capacity'],p['device'],p['dtype'])!=(len(flat),capacity,flat.device,native.probabilities.dtype):raise ValueError('Prepared slot context changed')
    ids,gates,accepted=p['ids'],p['gates'],p['accepted'];output=torch.zeros_like(flat)
    for eid,expert in enumerate(self.experts):
        positions=p['order'][eid]
        if len(positions):output=output.index_add(0,positions,expert(flat[positions])*gates[positions])
    native_usage=torch.bincount(native.selected_experts.flatten(),minlength=self.num_experts).float()/native.selected_experts.numel();aux=self.num_experts*torch.sum(native_usage.detach()*native.probabilities.mean(0))
    trace=RouteTrace(ids[:,0].reshape(batch,length),gates[:,0].reshape(batch,length),native.probabilities.reshape(batch,length,self.num_experts),ids.reshape(batch,length,1),gates.reshape(batch,length,1),native.entropy.reshape(batch,length),native.margin.reshape(batch,length),p['usage'],1-accepted.float().mean(),accepted)
    return output.reshape_as(inputs),aux,trace
