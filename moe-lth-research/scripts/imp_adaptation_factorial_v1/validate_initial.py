"""Exact full-model and native initial CE equality across four trainability flags."""
from af_common import *
from moe_lth.data import build_dataloaders
import socket

def main():
 torch.set_num_threads(4);device=torch.device('cuda');R.configure_device(device)
 v=next(c for c in read(HIST/'cohort.json')['conditions'] if c['seed']==7 and c['sparsity']==.8);cfg=v['config'];m=load_model_from_checkpoint(cfg['model'],v['initial'],device);mask=load_masks(v['mask']);apply_masks_(m,mask)
 R.seed_everything(7);_,vl=build_dataloaders(cfg['data'],128,7);val=R._materialize_validation_batches(vl,32);basehash=state_dict_hash(m.state_dict());losses=[];output=None
 for c in CONDITIONS:
  configure(m,c);m.eval()
  if state_dict_hash(m.state_dict())!=basehash:raise RuntimeError('Trainability flags changed initial model')
  with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16):y=m(val[0][0].to(device)).logits
  if output is None:output=y.detach().clone()
  elif not torch.equal(output,y):raise RuntimeError('Initial native forward differs across conditions')
  loss=R._evaluate_condition_loss(m,val,device,False,1.)
  losses.append(dict(condition=c,initial_validation_ce=loss))
 if any(r['initial_validation_ce']!=losses[0]['initial_validation_ce'] for r in losses):raise RuntimeError('Initial full-validation CE differs')
 save(OUT/'validation'/f'initial_{socket.getfqdn()}.json',dict(passed=True,identical_initial_model_hash=basehash,identical_forward=True,losses=losses,tying_preserved=m.token_embedding.weight is m.lm_head.weight,tf32_matmul=torch.backends.cuda.matmul.allow_tf32))
if __name__=='__main__':main()
