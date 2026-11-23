"""Tiny synthetic CUDA invariance check; not an experimental replicate."""
import copy,json,tempfile
from pathlib import Path
import torch
from moe_lth.config import load_config
from moe_lth.training.train import train_from_config
from moe_lth.pruning.router_age import state_dict_hash

def test_passive_exposure_preserves_cuda_trajectory():
 c=load_config('configs/multidomain_gpu.yaml');c['device']='cuda'
 c['data'].update(train_path=None,validation_path=None,path=None,seq_len=8,validation_blocks=1)
 c['model'].update(d_model=8,expert_hidden_size=16,num_layers=1,num_heads=2,num_experts=4,max_seq_len=8)
 c['training'].update(steps=4,batch_size=2,precision='fp16',checkpoint_steps=[0,4],eval_interval=4,log_interval=1)
 with tempfile.TemporaryDirectory() as tmp:
  results=[]
  for enabled in (False,True):
   cfg=copy.deepcopy(c);cfg['output_dir']=str(Path(tmp)/str(enabled));cfg['training']['record_exposure']=enabled
   train_from_config(cfg);p=Path(cfg['output_dir']);payload=torch.load(p/'checkpoints/step_4.pt',weights_only=False)
   results.append((state_dict_hash(payload['model']),(p/'logs/train_metrics.jsonl').read_text(),(p/'logs/validation_metrics.jsonl').read_text()))
   if enabled:
    rows=[json.loads(x) for x in (p/'logs/training_exposure.jsonl').read_text().splitlines()];assert len(rows)==4
    assert all(sum(x['proposed'].values())==16 and sum(x['accepted'].values())<=16 for x in rows)
  assert results[0]==results[1]
if __name__=='__main__':
 test_passive_exposure_preserves_cuda_trajectory();print('PASS: identical checkpoint, training losses and validation losses with source counters enabled')
