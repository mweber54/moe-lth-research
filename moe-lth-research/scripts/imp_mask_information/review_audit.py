"""Review completed inventory against canonical comparator file identities."""
import collections
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from audit import OUT,IMP,ROOT,rows,sha,write_csv


def review():
    inventory=[r for r in rows(OUT/'checkpoint_inventory.csv') if r['family']=='IMP']
    if not inventory: raise RuntimeError('Audit not complete')
    expected={r['path']:r['sha256'] for r in rows(IMP/'checkpoint_manifest.csv')}
    checks=[]
    for r in inventory:
        path=r['dense_checkpoint']
        ok=bool(expected.get(path)) and sha(path)==expected[path]
        checks.append(dict(configuration=r['configuration_id'],dense_checkpoint=path,recorded_sha256=expected.get(path,''),actual_sha256=sha(path),verified=ok))
        if not ok:
            r['eligible']='False';r['availability']+=';dense_canonical_hash_mismatch';r['ticket_status']='unclassified_comparator_unverified'
    extensions=[]
    originals={(r['num_experts'],r['seed'],r['mask_hash']):r for r in inventory}
    for path in sorted((ROOT/'results/imp_extended_recovery').glob('e*/seed_*/final_*/extension_manifest.json')):
        ext=json.loads(path.read_text())
        if 'num_experts' not in ext:
            # Legacy diagnostic explicitly reports no matched dense continuation.
            matches=[r for r in inventory if r['mask_hash']==ext['mask_hash']]
            if not matches: continue
            row=dict(matches[0]);row.update(configuration_id=matches[0]['configuration_id']+'_legacy_extension',family='IMP_5000_unpaired_legacy',recovery_checkpoint=str(path.parent/'checkpoints/step_5000.pt'),recovery_sha256='',recovery_protocol=json.dumps(dict(protocol=ext['protocol'],source_audit=ext['source_audit'],continuation=ext['continuation_type'])),original_recovery_ce=ext['extension_final_validation_cross_entropy'],dense_comparison_ce='',dense_checkpoint='',dense_sha256='',ticket_status='unclassified_no_recorded_matched_dense_extension',availability='legacy_metadata_recorded_content_not_reverified',eligible=False,stage_metadata=str(path),curves_path=str(path.parent/'metrics.jsonl'),duration_seconds=ext['duration_seconds'],training_batch_hash=ext['continued_training_batch_hash'],validation_batch_hash=ext['validation_batch_hash'])
            extensions.append(row)
            continue
        base=originals.get((str(ext['num_experts']),str(ext['seed']),ext['mask_hash']))
        if base is None: continue
        # Final names are explicit in the extension writer and checked on disk;
        # do not replace a missing file with another run's checkpoint.
        checkpoint=path.parent/'checkpoints/step_5000.pt'
        dense_path=path.parent.parent/'dense/extension_manifest.json'
        dense_ext=json.loads(dense_path.read_text()) if dense_path.is_file() else None
        row=dict(base)
        row.update(configuration_id=f"imp_e{ext['num_experts']}_seed{ext['seed']}_{path.parent.name}_5000_extension",family='IMP_5000_extension',recovery_checkpoint=str(checkpoint),recovery_sha256=ext['step_5000_checkpoint_sha256'],recovery_protocol=json.dumps(dict(total_attempts=5000,first_phase=2500,second_phase=2500,optimizer='fresh AdamW/scaler at both phases',scheduler='none',second_phase_start=ext['source_checkpoint'],second_phase_start_sha256=ext['source_checkpoint_sha256'],training_batch_hash=ext['training_batch_hash'])),original_recovery_ce=ext['extension_final_validation_cross_entropy'],dense_comparison_ce=dense_ext['extension_final_validation_cross_entropy'] if dense_ext else '',dense_checkpoint=str(dense_path.parent/'checkpoints/step_5000.pt') if dense_ext else '',dense_sha256=dense_ext['step_5000_checkpoint_sha256'] if dense_ext else '',ticket_status='unclassified_extension_content_not_reverified',availability='files_present_content_hashes_recorded_not_reverified' if checkpoint.is_file() and dense_ext else 'missing_extension_artifact',eligible=False,stage_metadata=str(path),curves_path=str(path.parent/'metrics.jsonl'),duration_seconds=ext['duration_seconds'],training_batch_hash=ext['training_batch_hash'],validation_batch_hash=ext['validation_batch_hash'])
        extensions.append(row)
    write_csv(OUT/'checkpoint_inventory.csv',inventory+extensions)
    write_csv(OUT/'dense_identity_verification.csv',checks)
    summaries=[]
    for e in sorted({r['num_experts'] for r in inventory},key=int):
        rr=[r for r in inventory if r['num_experts']==e]
        summaries.append(dict(experts=e,eligible=sum(r['eligible']=='True' for r in rr),training_seeds=sorted({int(r['seed']) for r in rr}),sparsities=sorted({float(r['sparsity']) for r in rr}),statuses=dict(collections.Counter(r['ticket_status'] for r in rr))))
    details=json.loads((OUT/'audit_details.json').read_text())
    (OUT/'audit_review.json').write_text(json.dumps(dict(architecture_summary=summaries,dense_identity_checks_passed=all(x['verified'] for x in checks),all_masks_valid=all(x['verified'] for x in details['mask_checks'].values()),review='No checkpoint substitution. Committed IMP stage metadata overrides stale README. Planned E12/E24 IMP runs excluded. Broader ticket catalog is OMP; 5000-step extensions remain a separate protocol.'),indent=2))
    text=['# Phase 0 audit review','','Committed IMP stages cover E8/E16/E32 and seeds 7,17,29,43,61. The original README is stale: it says not launched, while committed artifacts and hashes demonstrate completed runs. E12/E24 have no committed IMP masks and cannot enter the pilot.','',f"Verified {sum(x['verified'] for x in details['mask_checks'].values())}/{len(details['mask_checks'])} masks against exact checkpoint architecture, stored mask hash and keep counts. Dense checkpoint identities verified: {sum(x['verified'] for x in checks)}/{len(checks)} recovery rows. Source initialization and sparse final checkpoint hashes were independently checked in the primary audit.",'','All primary IMP recoveries rewind the full model to its exact step-0 checkpoint. Router and shared weights are trainable. Fresh AdamW/scaler and constant LR restart each recovery; no scheduler. Canonical Dataset A uses torch 2.7.0, FP16, 2500 attempted updates, fixed seed-specific training batch hashes, and 32 validation batches. Current canonical recovery code remains available; numerical reproduction is pending and is a hard pilot gate.','','Masks are separate bool tensor dictionaries. The stage selecting mask k is not recovery evidence for mask k; its subsequent training stage is. Discovery stages with zero sparsity are excluded from sparse eligibility. Existing 80/85 finals and lower-stage recoveries are preserved, including negative controls.','','The ticket catalog and lower-sparsity sweep use source-final one-shot magnitude masks, not IMP, and are inventoried separately without IMP eligibility. Router-age mapping records distinct router donors and trainability conditions; these must not be silently substituted for original IMP R0. Extended IMP uses 2500 original recovery attempts plus 2500 fresh-optimizer attempts on the next batch stream; matched 5000-step dense comparisons are separate from the primary protocol.','','Near-dense and sparse-better screens are descriptive; no formal winning-ticket or statistical equivalence conclusion is justified from endpoints alone. All candidate directory JSON/CSV/Markdown artifacts are indexed in audit_details.json. secondary_catalog.csv retains source-recorded validity rather than asserting new byte-level validation of non-IMP artifacts.','','Architecture summary:','']
    for s in summaries: text.append(f"- E{s['experts']}: {s['eligible']} eligible sparse recoveries; seeds {s['training_seeds']}; status counts {s['statuses']}")
    (OUT/'audit_review.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':review()
