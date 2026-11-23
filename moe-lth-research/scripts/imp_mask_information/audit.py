"""Read-only source audit; output is confined to the new experiment directory."""
import csv
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.imp import validate_masks

OUT = ROOT / 'results/imp_mask_information_v1'
IMP = ROOT / 'results/imp_80_85'


def rows(path):
    return list(csv.DictReader(path.open())) if path.is_file() else []


def write_csv(path, data, fields=None):
    fields = fields or list(dict.fromkeys(k for r in data for k in r))
    with path.open('w') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(data)


@lru_cache(None)
def sha(path):
    p = Path(path)
    if not p.is_file():
        return ''
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


@lru_cache(None)
def config(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    return payload['config'], payload['step']


def screen(ce, dense):
    if ce is None or dense is None:
        return 'unclassified_missing_matched_comparator'
    if ce < dense:
        return 'outperforms_matched_dense_descriptive'
    if ce == dense:
        return 'matches_dense_endpoint_descriptive'
    return 'near_dense_within_5pct_descriptive' if ce <= dense * 1.05 else 'nonrecovering_above_5pct_descriptive'


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    manifests = rows(IMP / 'imp_mask_manifest.csv')
    mask_map = {r['mask_hash']: r for r in manifests}
    finals = rows(IMP / 'final_recovery_metrics.csv')
    dense = {(r['num_experts'], r['seed']): r for r in finals}
    inventory = []
    mask_checks = {}
    # Validate every stored IMP mask against the checkpoint architecture, including
    # masks awaiting final recovery. Content hashes use the repository convention.
    for i, r in enumerate(manifests):
        path = r['mask_path']; init = r['original_initialization_checkpoint']
        try:
            cfg, step = config(init)
            model = TinyMoELanguageModel(cfg['model'])
            m = torch.load(path, map_location='cpu', weights_only=True)
            validate_masks(model, m)
            from moe_lth.experiments.run_router_age_recovery import _mask_hash
            digest = _mask_hash(m)
            keep = sum(int(v.sum()) for v in m.values()); count = sum(v.numel() for v in m.values())
            ok = digest == r['mask_hash'] and keep == int(r['keep_count']) and count == int(r['eligible_count'])
            mask_checks[path] = dict(verified=ok, sha256=sha(path), actual_sparsity=1-keep/count, error='')
        except Exception as e:
            mask_checks[path] = dict(verified=False, error=str(e))
        if (i+1) % 25 == 0:
            print(f'verified {i+1}/{len(manifests)} IMP masks', flush=True)
    stages = sorted((IMP / 'trajectories').glob('e*/seed_*/*/stage.json'))
    for path in stages:
        r = json.loads(path.read_text()); ident = r['identity']
        if float(ident['sparsity']) == 0:
            continue
        e, seed = str(ident['num_experts']), str(ident['seed'])
        init = r['original_initialization_checkpoint']; cfg, step = config(init)
        m = mask_map.get(r['mask_hash']); mp = m['mask_path'] if m else ''
        checkpoint = path.parent / 'checkpoints/final_recovered.pt'
        d = dense.get((e, seed)); dp = d['matched_dense_checkpoint'] if d else ''
        errors = []
        if not m or not mask_checks[mp]['verified']: errors.append('mask_missing_or_invalid')
        if step != 0 or sha(init) != ident['initial_sha256']: errors.append('rewind_identity')
        if sha(str(checkpoint)) != r['checkpoint_sha256']: errors.append('recovery_identity')
        if not dp or not Path(dp).is_file(): errors.append('dense_missing')
        if not Path(r['router_checkpoint']).is_file(): errors.append('router_missing')
        if not (path.parent / 'metrics.jsonl').is_file(): errors.append('curves_missing')
        if not r['integrity_checks_passed']: errors.append('source_integrity')
        if d and (r['training_batch_sequence_hash'] != d['training_batch_sequence_hash'] or r['validation_batch_sequence_hash'] != d['validation_batch_sequence_hash']): errors.append('protocol_data_mismatch')
        dense_ce = float(d['dense_validation_ce']) if d else None
        inventory.append(dict(configuration_id=f"imp_e{e}_seed{seed}_{path.parent.name}", family='IMP', architecture=json.dumps(cfg['model'], sort_keys=True), num_experts=e, seed=seed, sparsity=r['actual_sparsity'], mask_source=mp, mask_hash=r['mask_hash'], rewind_source=init, rewind_step=step, rewind_sha256=sha(init), recovery_checkpoint=str(checkpoint), recovery_sha256=sha(str(checkpoint)), recovery_protocol=json.dumps(dict(steps=r['recovery_steps'], optimizer=r['optimizer'], scheduler=r['scheduler'], training=cfg['training'], data=cfg['data']),sort_keys=True), router_protocol=f"R{r['router_age_percent']}; {r['router_mode']}; shared trainable; router={r['router_checkpoint']}", original_recovery_ce=r['final_validation_loss'], dense_comparison_ce=dense_ce, dense_checkpoint=dp, dense_sha256=sha(dp) if dp else '', ticket_status=screen(r['final_validation_loss'],dense_ce), availability='verified' if not errors else ';'.join(errors), eligible=not errors, stage_metadata=str(path), curves_path=str(path.parent/'metrics.jsonl'), duration_seconds=r['duration_seconds'], training_batch_hash=r['training_batch_sequence_hash'], validation_batch_hash=r['validation_batch_sequence_hash']))
    write_csv(OUT/'checkpoint_inventory.csv', inventory)
    # Preserve all existing controls and router ages, but never relabel OMP as IMP.
    catalogs = []
    for source in ['ticket_collection_80plus/canonical_recoveries.csv','dataset_a_full_lower_sparsity/analysis_final/canonical_combined.csv']:
        for r in rows(ROOT/'results'/source):
            catalogs.append(dict(source=source, family='non_IMP_catalog', condition_id=r.get('condition_id'), num_experts=r.get('num_experts'), seed=r.get('seed'), sparsity=r.get('actual_sparsity'), router_age=r.get('router_age',r.get('router_age_percent')), mask_source=r.get('mask_path',r.get('mask_artifact_path')), rewind_source=r.get('initialization_checkpoint',r.get('source_checkpoint_path')), recovery_path=r.get('final_checkpoint',r.get('sparse_artifact_path')), ce=r.get('sparse_loss',r.get('sparse_final_validation_loss')), dense_ce=r.get('dense_loss',r.get('dense_final_validation_loss')), exclusion='not an IMP trajectory; retained as secondary catalog only', recorded_valid=r.get('validated',r.get('condition_valid'))))
    write_csv(OUT/'secondary_catalog.csv', catalogs)
    extensions = rows(ROOT/'results/imp_extended_recovery/sparse_dense_comparison.csv')
    write_csv(OUT/'extended_recovery_inventory.csv', extensions)
    candidates = {}
    for name in ['imp_80_85','ticket_collection_80plus','imp_extended_recovery','dataset_a_full_lower_sparsity','router_age_cross_architecture_map']:
        p=ROOT/'results'/name
        candidates[name]=dict(exists=p.is_dir(), files=[str(x.relative_to(ROOT)) for x in p.rglob('*') if x.is_file() and x.suffix in {'.json','.csv','.md'}])
    (OUT/'audit_details.json').write_text(json.dumps(dict(mask_checks=mask_checks, candidate_directories=candidates, eligible=sum(r['eligible'] for r in inventory), rows=len(inventory), note='5pct is an established descriptive screen, not an equivalence test or formal ticket claim.'),indent=2))
    print(json.dumps(dict(inventory_rows=len(inventory),eligible=sum(r['eligible'] for r in inventory),secondary_rows=len(catalogs),extensions=len(extensions))),flush=True)

if __name__ == '__main__':
    audit()
