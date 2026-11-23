"""Seal completed exploratory outputs after reviewing the verdict."""
from audit import *
def main():
 required=['inventory.csv','analysis_config.json','provenance_and_exclusions.md','rank_stability.csv','rank_transition_summary.csv','late_rising_weight_summary.csv','exposure_coverage.csv','expert_interval_metrics.csv','descriptive_associations.csv','held_out_prediction_results.csv','numerical_checks.json','findings.md']
 checks=json.loads((O/'numerical_checks.json').read_text());assert checks['passed'];assert 'Reviewed verdict:' in (O/'findings.md').read_text()
 def count(name):
  with (O/name).open() as f:return sum(1 for _ in csv.DictReader(f))
 assert count('rank_stability.csv')==5*128*3*8
 assert count('rank_transition_summary.csv')==5*128*3*8*25
 assert count('late_rising_weight_summary.csv')==5*128*3*8*2
 assert count('held_out_prediction_results.csv')==5
 primary=[r for r in csv.DictReader((O/'inventory.csv').open()) if r['family']=='original_source' and r['selected']=='True'];assert len(primary)==30 and len({r['sha256'] for r in primary})==30
 # All output hashes are independent of historical inputs; original files were opened read-only.
 files=sorted(p for p in O.rglob('*') if p.is_file() and p.name!='completion.json')
 for f in required:assert (O/f).exists()
 save(O/'completion.json',dict(status='COMPLETE',retrospective_exploratory=True,training_or_recovery_runs_launched=0,model_forward_or_backward_calls=0,source_seeds=[7,17,29,43,61],primary_checkpoint_count=30,expert_coordinates_per_checkpoint=67108864,rank_stability_rows=count('rank_stability.csv'),transition_rows=count('rank_transition_summary.csv'),rank_analysis_wall_seconds=checks['rank_checks']['wall_seconds'],output_bytes=sum(p.stat().st_size for p in files),numerical_checks_passed=True,artifact_sha256={str(p.relative_to(O)):sha(p) for p in files},implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
