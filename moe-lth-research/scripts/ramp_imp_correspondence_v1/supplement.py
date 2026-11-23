"""Read-only provenance inventory and final sparse retained-weight statistics.

Run after main completion. This adds explicitly timed recovered-state measurements;
none are fed into initial/early H1/H2 features.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from moe_lth.analysis.ramp import neuron_mask_statistics
from moe_lth.data import build_dataloaders
from itertools import islice, cycle


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main(out):
    torch.set_num_threads(1)
    completion=json.loads((out/'completion.json').read_text())
    if completion['historical_recoveries']!=30:
        raise RuntimeError('Supplement requires main validated completion')
    docs=[]
    paths=set(ROOT.glob('*.md'))|set((ROOT/'docs').rglob('*.md'))|set((ROOT/'experiments').glob('*.yaml'))|set((ROOT/'reports').glob('*.md'))|set((ROOT/'src/moe_lth/analysis').glob('*.py'))
    paths.update((ROOT/'scripts/ramp_imp_correspondence_v1').glob('*'))
    for p in sorted(paths):
        if not p.is_file():continue
        text=p.read_text()
        docs.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,
            headings=' | '.join(line for line in text.splitlines() if line.startswith(('# ','## ','def ','class ')))[:2000]))
    pd.DataFrame(docs).to_csv(out/'documentation_and_analysis_inventory.csv',index=False)
    rows=pd.read_csv(out/'verified_recovery_pairs.csv')
    canonical=pd.read_csv(ROOT/'results/final_dataset_a/canonical/dataset_a_canonical.csv')
    dest=out/'recovered_neuron_statistics';dest.mkdir(exist_ok=True)
    allpaths=[]
    datachecks=[]
    for (arch,seed),group in rows.groupby(['architecture','seed']):
        payload=torch.load(group.iloc[0].initial_checkpoint,map_location='cpu',weights_only=True)
        config=payload['config']
        tr,va=build_dataloaders(config['data'],int(config['training']['batch_size']),int(seed))
        def batchhash(batches):
            digest=hashlib.sha256()
            for batch_id,(tokens,targets) in enumerate(batches):
                digest.update(str(batch_id).encode())
                for label,tensor in [('inputs',tokens),('targets',targets)]:
                    cpu=tensor.detach().cpu().contiguous()
                    digest.update(label.encode());digest.update(str(tuple(cpu.shape)).encode())
                    digest.update(str(cpu.dtype).encode());digest.update(cpu.numpy().tobytes())
            return digest.hexdigest()
        trainhash=batchhash(islice(cycle(tr),2500))
        valhash=batchhash(islice(va,int(config['data']['validation_blocks'])))
        if not (group.train_hash==trainhash).all() or not (group.validation_hash==valhash).all():
            raise RuntimeError(f'Reconstructed canonical token stream mismatch: {arch}/{seed}')
        folder=out/f'{arch.lower()}_seed{seed}'
        checked=[]
        for manifest_path in sorted(folder.glob('*/checkpoint.json')):
            meta=json.loads(manifest_path.read_text())
            checkpoint=torch.load(meta['path'],map_location='cpu',weights_only=True)
            if checkpoint['step']!=meta['step'] or checkpoint['config']['model']!=config['model'] or checkpoint['config']['seed']!=int(seed):
                raise RuntimeError(f'Profile checkpoint payload/config mismatch: {meta["path"]}')
            if sha(Path(meta['path']))!=meta['sha256']:
                raise RuntimeError('Profile checkpoint changed')
            checked.append(meta['path'])
        datachecks.append(dict(architecture=arch,seed=int(seed),train_hash=trainhash,validation_hash=valhash,
            reconstructed_data_match=True,profile_payloads_verified=checked))
        print('verified reconstructed streams and profile payloads',arch,seed,flush=True)
    (out/'reconstruction_validation.json').write_text(json.dumps(datachecks,indent=2))
    for (arch,seed),group in rows.groupby(['architecture','seed']):
        initial=torch.load(group.iloc[0].initial_checkpoint,map_location='cpu',weights_only=True)
        for row in group.itertuples():
            reference=canonical[(canonical.num_experts==int(arch[1:]))&(canonical.seed==seed)&(canonical.sparsity==row.sparsity)&(canonical.router_age==0)]
            if len(reference)!=1:
                raise RuntimeError('No unique canonical paired dense reference')
            reference=reference.iloc[0]
            if sha(Path(row.paired_dense_checkpoint))!=reference.dense_final_checkpoint_sha256:
                raise RuntimeError('Canonical paired dense file hash mismatch')
            dense=torch.load(row.paired_dense_checkpoint,map_location='cpu',weights_only=True)
            if dense['step']!=2500 or abs(dense['validation_loss']-row.dense_loss)>1e-8:
                raise RuntimeError('Paired dense payload endpoint mismatch')
            recovered=torch.load(row.sparse_checkpoint,map_location='cpu',weights_only=True)
            if recovered['step']!=2500 or abs(recovered['validation_loss']-row.sparse_loss)>1e-8:
                raise RuntimeError('Sparse checkpoint payload endpoint mismatch')
            mask=torch.load(row.mask_path,map_location='cpu',weights_only=True)
            if sha(Path(row.sparse_checkpoint))!=row.sparse_file_sha256:
                raise RuntimeError('Recovered checkpoint identity mismatch')
            for name,m in mask.items():
                if torch.count_nonzero(recovered['model'][name][~m]):
                    raise RuntimeError('Recovered model violates historical fixed support')
            stats=neuron_mask_statistics(mask,initial['model'],recovered['model'])
            frame=pd.concat([pd.DataFrame(dict(layer=l,expert=e,neuron=np.arange(len(s['retention'])),**{k:v.numpy() for k,v in s.items()})) for (l,e),s in stats.items()],ignore_index=True)
            frame=frame.rename(columns={c:c.replace('final_','recovered_') for c in frame if c.startswith('final_')})
            frame['architecture']=arch;frame['seed']=seed;frame['mask_id']=row.mask_id;frame['initialization_id']=row.initialization_id
            frame['sparsity']=row.sparsity;frame['recovery_run']=row.recovery_run;frame['recovered_checkpoint']=row.sparse_checkpoint
            frame['measurement_timing']='after_2500_step_sparse_recovery; excluded_from_all_early_predictors'
            path=dest/f'{arch.lower()}_seed{seed}_s{row.sparsity:.2f}.parquet';frame.to_parquet(path,index=False)
            allpaths.append(dict(path=str(path),rows=len(frame),sha256=sha(path),checkpoint=row.sparse_checkpoint))
        print('verified recovered support',arch,seed,flush=True)
    source_files=[p for p in (ROOT/'scripts/ramp_imp_correspondence_v1').glob('*.py')]+[ROOT/'src/moe_lth/analysis/ramp.py',ROOT/'tests/test_ramp_imp_correspondence.py',ROOT/'tests/test_ramp_imp_pipeline.py']
    (out/'supplement_manifest.json').write_text(json.dumps(dict(recovered_artifacts=allpaths,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files},
        early_predictor_leakage=False,extra_training=False,join_keys=['architecture','seed','sparsity','mask_id','layer','expert','neuron']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/ramp_imp_correspondence_v1');main(p.parse_args().output.resolve())
