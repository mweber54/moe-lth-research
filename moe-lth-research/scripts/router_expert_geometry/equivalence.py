from gcommon import *
from metrics import geometry
import numpy as np

def compare():
 enabled=OUT/'validation/enabled_v10';disabled=OUT/'validation/disabled_v1'
 assert all(read(enabled/f'preservation_step{s}.json')['passed'] for s in (0,25))
 a=torch.load(enabled/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)['model'];b=torch.load(disabled/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)['model']
 diffs={k:float((a[k]-b[k]).abs().max()) for k in a};maximum=max(diffs.values());passed=all(torch.allclose(a[k],b[k],atol=1e-6,rtol=1e-5) for k in a)
 meta_a=read(enabled/'metadata.json');meta_b=read(disabled/'metadata.json');assert all(meta_a[k]==meta_b[k] for k in ('optimizer_steps_applied','amp_overflow_step_count','loss_scale_final','training_batch_sequence_hash','validation_batch_sequence_hash'))
 ca=[json.loads(l) for l in (enabled/'metrics.jsonl').read_text().splitlines()];cb=[json.loads(l) for l in (disabled/'metrics.jsonl').read_text().splitlines()];assert [r['step'] for r in ca]==[r['step'] for r in cb]
 loss_difference=max(abs(x['loss']-y['loss']) for x,y in zip(ca,cb));passed &= loss_difference<1e-6
 # Worker step zero vs standalone analysis on the exact actual recovery state.
 info=read(OUT/'runtime_sources.json')['source_e32_seed7_T2500'];initial=torch.load(info['checkpoints']['0']['path'],map_location='cpu',weights_only=False)['model'];mask=torch.load(info['masks']['0.5']['learned_path'],weights_only=True);base={k:v*mask[k] if k in mask else v for k,v in initial.items()};source=torch.load(info['checkpoints']['100']['path'],map_location='cpu',weights_only=False)['model']
 standalone=geometry(base,source,base);worker=rows(enabled/'geometry/step_0000/weights.csv');delta=max(abs(x['alignment_centered']-float(y['alignment_centered'])) for x,y in zip(standalone,worker));passed &= delta<1e-6
 ga=[json.loads(l) for l in (enabled/'gradient_stats/gradient_stats.jsonl').read_text().splitlines()];gb=[json.loads(l) for l in (disabled/'gradient_stats/gradient_stats.jsonl').read_text().splitlines()];assert [x['optimizer_step_applied'] for x in ga]==[x['optimizer_step_applied'] for x in gb]
 metadata=[read(enabled/'geometry'/f'step_{s:04d}'/'metadata.json') for s in (0,25)];peak=max(x['peak_reserved_bytes'] for x in metadata);seconds=[x['diagnostic_seconds'] for x in metadata];reg=read(OUT/'registration.json')
 save(OUT/'validation/equivalence.json',dict(passed=bool(passed),steps=25,condition='seed7/R0/learned/.50',max_parameter_abs_difference=maximum,validation_loss_difference=loss_difference,worker_standalone_step0_max_alignment_difference=delta,tolerance=dict(atol=1e-6,rtol=1e-5),diagnostic_seconds=seconds,peak_reserved_bytes=peak,estimated_snapshot_storage_bytes=reg['source_state_bytes']*8*75,estimated_total_diagnostic_seconds=sum(seconds)/2*7*75,estimated_total_recovery_framework_seconds=metadata[-1]['training_and_native_framework_seconds']/25*2500*75,note='Matched fixed batches and initialization; all parameters compared. Precision of training unchanged. CPU FP64 Gram geometry validated versus direct SVD.'))
 if not passed:raise RuntimeError('GPU trajectory or standalone equivalence failed')
 print('GPU EQUIVALENCE PASS',maximum,delta,seconds,flush=True)
if __name__=='__main__':torch.set_num_threads(2);compare()
