"""Use existing corpus provenance only after exact byte reconstruction."""
from gcommon import *
import bisect

def build():
 meta=read(ROOT/'data/multidomain_balanced/metadata.json')['train'];pieces=[]
 tiny=(ROOT/meta['tiny_path']).read_text();wiki=(ROOT/meta['wiki_path']).read_text();size=meta['chunk_chars']
 from moe_lth.experiments.run_long_validation_extension import _sample_evenly
 target=min(len(tiny),len(wiki));tiny=_sample_evenly(tiny,target,size);wiki=_sample_evenly(wiki,target,size)
 assert len(tiny)==meta['tiny_chars'] and len(wiki)==meta['wiki_chars']
 for start in range(0,max(len(tiny),len(wiki)),size):
  for label,text in [('tinystories',tiny),('wikitext103',wiki)]:
   if start<len(text):pieces.append((label,text[start:start+size]))
 reconstructed='\n\n'.join(text for label,text in pieces).encode();data=ROOT/meta['output_path']
 if hashlib.sha256(reconstructed).hexdigest()!=sha(data):
  save(OUT/'probes/domain_labels.json',dict(available=False,reason='Existing source files do not reconstruct the exact training bytes; no labels invented'));return
 ends=[];labels=[];offset=0
 for i,(label,text) in enumerate(pieces):
  if i:offset+=2;ends.append(offset);labels.append(None)
  offset+=len(text.encode());ends.append(offset);labels.append(label)
 definitions=read(OUT/'probes/global_sequence_indices.json');result={}
 for sid,rec in definitions.items():
  probe=torch.load(OUT/'probes'/f'{sid}.pt',weights_only=True);sequence_labels=[[labels[bisect.bisect_right(ends,start+pos)] for pos in range(128)] for start in rec['global_token_offsets']];flat=[label for seq in sequence_labels for label in seq];result[sid]=dict(sequence_domain_labels=sequence_labels,sampled_position_domain_labels=[flat[i] for i in probe['positions'].tolist()],probe_sha256=sha(OUT/'probes'/f'{sid}.pt'))
 save(OUT/'probes/domain_labels.json',dict(available=True,definition='source-corpus provenance verified by exact byte reconstruction; separators null',train_sha256=sha(data),tiny_source_sha256=sha(ROOT/meta['tiny_path']),wiki_source_sha256=sha(ROOT/meta['wiki_path']),metadata_sha256=sha(ROOT/'data/multidomain_balanced/metadata.json'),seeds=result))
 print('VERIFIED EXISTING CORPUS LABELS',flush=True)
if __name__=='__main__':build()
