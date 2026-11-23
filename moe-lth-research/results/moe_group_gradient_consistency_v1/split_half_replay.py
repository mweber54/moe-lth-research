"""Run one split-half replay, reusing the archived gradient implementation.

One process handles one half and one labeling. Original replay provenance, state,
and deterministic checks apply. Permutations shuffle eligible target labels in each
batch using an independent local generator, without consuming global RNG state.
"""
import argparse, os
from pathlib import Path
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('seed',type=int,choices=(7,17,29)); ap.add_argument('order',choices=('A','B')); ap.add_argument('step',type=int,choices=(250,500)); ap.add_argument('half',type=int,choices=(0,1,2))
ap.add_argument('--permutation',type=int,default=None); ap.add_argument('--suffix',default=''); ap.add_argument('--all-retained',action='store_true')
a=ap.parse_args()
if a.permutation is not None and a.permutation < 0: ap.error('permutation must be nonnegative')
src=(P/'replay.py').read_text()
old="data_path=ROOT/'data/multidomain_balanced/multidomain_train.txt'"
new="""if len(selected) < 180: raise RuntimeError('fewer than 180 retained batches')
 selected=selected if ALL_RETAINED else selected[:180]
 midpoint=len(selected)//2
 if HALF == 1: selected=selected[:midpoint]
 elif HALF == 2: selected=selected[midpoint:]
 data_path=ROOT/'data/multidomain_balanced/multidomain_train.txt'"""
assert src.count(old)==1; src=src.replace(old,new)
old="out=model(x); losses=F.cross_entropy"
new="""if PERM is not None:
   flat=domain.view(-1); eligible=(flat==1)|(flat==2); original=flat[eligible]
   generator=torch.Generator(device='cpu'); generator.manual_seed(230112 + PERM*1000003 + batch_id)
   permutation=torch.randperm(original.numel(),generator=generator).to(domain.device)
   flat[eligible]=original[permutation]
  out=model(x); losses=F.cross_entropy"""
assert src.count(old)==1;src=src.replace(old,new)
old="dest=OUT/f'gradients_{cell}_{step}.pt'"
new="""result.update(half=HALF, all_retained=ALL_RETAINED, permutation=PERM, permutation_seed_base=230112 if PERM is not None else None)
 dest=OUT/f'split_half_gradients_{cell}_{step}_h{HALF}_{"original" if PERM is None else "perm"+str(PERM)}{'_all_retained' if ALL_RETAINED else ''}{SUFFIX}.pt'"""
assert src.count(old)==1;src=src.replace(old,new)
ns={'__file__':str(P/'replay.py'),'__name__':'split_half_replay_impl','HALF':a.half,'PERM':a.permutation,'SUFFIX':a.suffix,'ALL_RETAINED':a.all_retained}
exec(compile(src,str(P/'replay.py'),'exec'),ns)
ns['main'](a.seed,a.order,a.step)
