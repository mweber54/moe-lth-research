"""Independent artifact-completeness audit of the registered stopped study."""
from common import *
import csv,stat


def csv_rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def main():
    frozen();c=read(D/'completion.json');assert c['status']=='STOPPED_MATCHING_GATE'
    assert c['production_continuations']==c['control_reuses']==c['intervention_loss_evaluations']==0
    assert not c['production_experiment_complete'] and c['study_execution_complete']
    required=['protocol.md','registration.json','checkpoint_and_calibration_provenance.json','score_reliability.csv','exposure_summary.csv','matched_group_summary.csv','magnitude_growth_balance.csv','structural_balance.csv','masks/manifest.json','first_order_prediction_checks.csv','numerical_checks.json','per_branch_results.csv','learning_curves.csv','paired_seed_effects.csv','findings.md','score_algebra_audit.json','native_configuration_audit.json','matching_review.json','reliability_review.json']
    for name in required:assert (D/name).is_file(),name
    for name,h in c['output_sha256'].items():assert sha(D/name)==h,'Artifact changed: '+name
    for name,h in c.get('implementation_sha256',{}).items():assert sha(ROOT/name)==h,'Implementation changed: '+name
    assert read(D/'native_configuration_audit.json')['passed']
    br=csv_rows(D/'per_branch_results.csv');assert len(br)==15
    assert {(int(r['seed']),r['branch']) for r in br}=={(s,b) for s in SEEDS for b in ('C','H','L')}
    assert all(int(r['attempted_updates'])==int(r['applied_updates'])==int(r['AMP_skips'])==0 and r['immediate_validation_CE']==r['final_validation_CE']=='' for r in br)
    effects=csv_rows(D/'paired_seed_effects.csv');assert len(effects)==5
    assert all(all(r[k]=='' for k in ('immediate_H_minus_L','integrated_H_minus_L','final_H_minus_L','H_minus_C','L_minus_C')) for r in effects)
    curves=csv_rows(D/'learning_curves.csv');assert len(curves)==105 and all(r['validation_CE']==r['excess_CE']=='' for r in curves)
    reliability=read(D/'reliability_review.json');assert reliability['status']=='PASS' and len(reliability['coverage'])==5
    matching=read(D/'matching_review.json');assert matching['status']=='STOP_MATCHING' and len(matching['coverage'])==5
    assert not (D/'mask_seal.json').exists(),'A stopped match must not expose intervention-ready masks'
    for path,h in matching['candidate_coordinate_file_sha256'].items():
        assert sha(D/path)==h and not ((D/path).stat().st_mode & 0o222)
    algebra=read(D/'score_algebra_audit.json');assert algebra['passed'] and len(algebra['seeds'])==5
    assert all(r['coordinates']==67108864 for r in algebra['seeds'])
    for s in SEEDS:
        st=read(D/'status'/f'measure_seed{s}.json');assert st['status']=='COMPLETE'
        checks=read(D/st['attempt']/'measurement_checks.json');assert len(checks['score_files'])==256
        assert all(not ((D/p).stat().st_mode & 0o222) for p in checks['score_files'])
    numerical=read(D/'numerical_checks.json');assert numerical['no_intervention_or_validation_or_production']
    assert all(str(v).startswith('NOT_RUN') for v in numerical['downstream_checks'].values())
    figures=['sensitivity_reliability.png','magnitude_growth_structural_balance.png','matching_feasibility.png','immediate_final_paired_effects.png','continuation_CE_excess_loss.png']
    for name in figures:assert (D/'figures'/name).stat().st_size>1000
    preserved=[D/'attempts'/f'measure_seed{s}'/'attempt_0001/failure.txt' for s in (7,17)]
    preserved.append(D/'matching_attempts/attempt_0001/stage_worker_attempt1.log')
    assert all(p.is_file() for p in preserved)
    report={'passed':True,'required_deliverables_present':True,'declared_output_hashes_verified':True,
      'source_seeds_measured_and_matched':list(SEEDS),'all_15_registered_conditions_reported':True,
      'unavailable_outcomes_are_empty_not_zero':True,'score_coordinates_audited':335544320,
      'immutable_score_and_candidate_files':True,'no_intervention_ready_mask_seal':True,
      'downstream_tests_explicitly_not_run':True,'preserved_failed_engineering_attempts':3,
      'production_continuations':0,'intervention_losses':0,'replication_unit':'source seed; no pruning effect replications obtained',
      'registration_sha256':sha(D/'registration.json'),'implementation_sha256':sha(Path(__file__))}
    save(D/'final_artifact_audit.json',report)
    c['output_sha256']['final_artifact_audit.json']=sha(D/'final_artifact_audit.json');c['final_artifact_audit_passed']=True;c['preserved_failed_engineering_attempts']=3
    c['implementation_sha256']={str(p.relative_to(ROOT)):sha(p) for p in list(HERE.glob('*.py'))+[HERE/'README.md']}
    save(D/'completion.json',c)
    print('FINAL ARTIFACT AUDIT PASSED',flush=True)
if __name__=='__main__':main()
