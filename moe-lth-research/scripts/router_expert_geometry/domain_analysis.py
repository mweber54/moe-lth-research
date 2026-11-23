"""Descriptive routing by verified source corpus, never extra replication units."""
from gcommon import *
import math

def analyze():
 definitions=read(OUT/'probes/domain_labels.json')
 if not definitions['available']:return
 records=[]
 for sid,definition in definitions['seeds'].items():
  labels=definition['sampled_position_domain_labels']
  seed=int(sid.split('seed')[1].split('_')[0]);paths=list((OUT/'existing'/sid).glob('*/*_routing.pt'))
  for j in rows(OUT/'run_manifest.csv'):
   if int(j['seed'])==seed and j.get('attempt'):paths+=list((Path(j['attempt'])/'geometry').glob('*/*_routing.pt'))
  for path in paths:
   raw=torch.load(path,map_location='cpu',weights_only=True)
   for layer,values in raw.items():
    probs=values['probabilities'];selected=values['selected'];accepted=values['accepted']
    for domain in sorted({x for x in labels if x is not None}):
     take=torch.tensor([x==domain for x in labels]);p=probs[take];sel=selected[take];acc=accepted[take];entropy=-(p*p.clamp_min(1e-9).log()).sum(1)/math.log(p.shape[1]);top=p.topk(2,dim=1).values
     for e in range(p.shape[1]):records.append(dict(seed=seed,state=str(path.parent.relative_to(OUT)),probe_mode=path.stem.split('_')[0],layer=layer,domain=domain,expert=e,positions=int(take.sum()),probability_mean=float(p[:,e].mean()),entropy_mean=float(entropy.mean()),margin_mean=float((top[:,0]-top[:,1]).mean()),proposed=int((sel==e).sum()),accepted=int(((sel==e)&acc).sum())))
 table(OUT/'analysis/domain_routing.csv',records)
if __name__=='__main__':torch.set_num_threads(2);analyze()
