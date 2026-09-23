"""Complete recovery-trajectory routing capture and exact accepted-token replay."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from moe_lth.models.router import RouteOverride
from moe_lth.pruning.router_age import set_router_temperature
from moe_lth.experiments.optimization_geometry_diagnostics import digest


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class Schedule:
    def __init__(self,path,mode,*,audit_path=None):
        if mode not in ('capture','replay'):raise ValueError(mode)
        self.path=Path(path);self.mode=mode;self.audit_path=Path(audit_path) if audit_path else None
        self.counts={'train':0,'validation':0};self.rolling={'train':hashlib.sha256(),'validation':hashlib.sha256()};self.entries={}
        if mode=='capture':
            self.path.mkdir(parents=True,exist_ok=False)
        else:
            self.manifest=json.loads((self.path/'manifest.json').read_text());self.entries=self.manifest['entries']
    def _key(self,phase,step,batch):return f'{phase}/step_{step:04d}_batch_{batch:03d}.pt'
    def forward(self,model,x,step,batch,phase):
        key=self._key(phase,step,batch);path=self.path/key
        set_router_temperature(model,1.)
        if self.mode=='capture':
            if key in self.entries:raise RuntimeError(f'Duplicate captured schedule occurrence: {key}')
            output=model(x)
            payload={'phase':phase,'step':step,'batch_index':batch,'input_hash':digest(x),'shape':list(x.shape),'layers':[{'ids':t.selected_expert_indices.detach().to('cpu',torch.uint8),'capacity_scores':t.selected_probabilities.detach().cpu(),'accepted':t.accepted_mask.detach().cpu()} for t in output.routes]}
            path.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,path)
            sha=file_hash(path);self.entries[key]=sha
        else:
            if key not in self.entries:raise RuntimeError(f'Missing schedule occurrence: {key}')
            sha=file_hash(path)
            if sha!=self.entries[key]:raise RuntimeError(f'Schedule file changed: {key}')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            if payload['input_hash']!=digest(x) or payload['shape']!=list(x.shape):raise RuntimeError(f'Input sequence mismatch: {key}')
            overrides=[RouteOverride(v['ids'].to(x.device,torch.long),capacity_scores=v['capacity_scores'].to(x.device)) for v in payload['layers']]
            output=model(x,overrides)
            for t,v in zip(output.routes,payload['layers']):
                if not torch.equal(t.selected_expert_indices,v['ids'].to(x.device,torch.long)):raise RuntimeError(f'Assignment mismatch: {key}')
                if not torch.equal(t.accepted_mask,v['accepted'].to(x.device)):raise RuntimeError(f'Accepted-token mismatch: {key}')
        self.counts[phase]+=1;self.rolling[phase].update(key.encode());self.rolling[phase].update(sha.encode())
        return output
    def train(self,model,x,step):
        if step!=self.counts['train']+1:raise RuntimeError(f'Training schedule out of order: {step}')
        return self.forward(model,x,step,0,'train')
    def validation(self,model,x,step,batch):return self.forward(model,x,step,batch,'validation')
    def finish_capture(self,metadata):
        if self.mode!='capture':raise RuntimeError('Not capturing')
        result={**metadata,'entries':self.entries,'counts':self.counts,'sequence_hashes':{k:v.hexdigest() for k,v in self.rolling.items()},'gate_amplitudes':'not forced; recipient router probabilities remain differentiable','capacity_policy':'exact donor scores and accepted-token masks'}
        (self.path/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result
    def verify_complete_replay(self):
        result={'counts':self.counts,'sequence_hashes':{k:v.hexdigest() for k,v in self.rolling.items()},'all_assignment_and_acceptance_checks_passed':True}
        if self.mode!='replay' or result['counts']!=self.manifest['counts'] or result['sequence_hashes']!=self.manifest['sequence_hashes']:raise RuntimeError('Incomplete or reordered schedule replay')
        if self.audit_path:self.audit_path.write_text(json.dumps(result,indent=2)+'\n')
        return result
