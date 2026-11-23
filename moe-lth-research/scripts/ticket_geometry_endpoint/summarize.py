"""Endpoint deltas on identical valid experts; descriptive seed summaries only."""
from pathlib import Path
import csv,gzip,json,collections,math,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'results/ticket_geometry_endpoint_v1'
METRICS={'activation':['pearson','spearman','above_shuffle'],'function':['dense_relative_output_error','dense_output_cosine'],'geometry':['router_energy','router_energy_above_shuffle','source_gram_distance','dense_gram_distance','source_overlap8','source_overlap16','source_overlap32','dense_overlap8','dense_overlap16','dense_overlap32','router_alignment8','router_alignment16','router_alignment32']}
rows=[];complete=[]
for marker in sorted(OUT.glob('*/*/complete.json')):
 done=json.loads(marker.read_text())
 for name,expected in done['outputs'].items():
  with open(marker.parent/name,'rb') as handle: actual=hashlib.file_digest(handle,'sha256').hexdigest()
  if actual!=expected:raise RuntimeError('Output changed: '+str(marker.parent/name))
 c=done['condition'];complete.append(c['id']);records={}
 with gzip.open(marker.parent/'experts.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);key=(r['kind'],r['layer'],r.get('mode',''),r.get('selection',''),r['expert']);records.setdefault(key,{})[r['step']]=r
 groups=collections.defaultdict(list)
 for (kind,layer,mode,selection,e),pair in records.items():
  if set(pair)!={0,2500}:continue
  for metric in METRICS[kind]:
   a,b=pair[0].get(metric),pair[2500].get(metric)
   if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):groups[kind,layer,mode,selection,metric].append((a,b))
 for (kind,layer,mode,selection,metric),vs in groups.items():
  a=sum(v[0] for v in vs)/len(vs);b=sum(v[1] for v in vs)/len(vs)
  rows.append(dict(condition=c['id'],num_experts=done['num_experts'],seed=done['seed'],router_age=done['age'],sparsity=c['sparsity'],strict_ticket=c['ticket'],gap=c['gap'],kind=kind,layer=layer,mode=mode,selection=selection,metric=metric,initial=a,final=b,change=b-a,n_matched_experts=len(vs)))
OUT.mkdir(parents=True,exist_ok=True)
if rows:
 with open(OUT/'paired_endpoint_metrics.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plan=json.loads((HERE/'plan.json').read_text());expected=[g['id'] for g in plan['groups']]+[c['id'] for g in plan['groups'] for c in g['sparse']]
status=dict(complete=len(set(complete)),expected=len(expected),missing=sorted(set(expected)-set(complete)),note='Descriptive, post-hoc endpoint cohort. Missing metrics are undefined, not zeros. Rows are not independent experimental replicates. No recovery timecourse is observed.')
(OUT/'summary_coverage.json').write_text(json.dumps(status,indent=2));print(f'Completed {status["complete"]}/{status["expected"]}; table: {OUT}/paired_endpoint_metrics.csv')
