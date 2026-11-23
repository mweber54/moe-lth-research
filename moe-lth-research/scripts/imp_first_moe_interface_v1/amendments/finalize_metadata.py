"""Engineering-only aggregate-label correction; fixed estimands and numbers unchanged."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_interface_v1 import analysis as a
c=a.c;OUT=a.OUT

def numeric_rows(name):
    rs=c._read_csv(OUT/(name+'.csv'))
    numeric={'V_start','V_end','Q','S_start','S_end','diagonal_zero_damage_change','A','T','sparse_minus_unpruned','native_minus_replay','start_rms','end_rms','relative_l2','norm_ratio','cosine_global'}
    for r in rs:
        r['seed']=int(r['seed'])
        for k in numeric:
            if k in r:r[k]=float(r[k]) if r[k]!='' else None
    return rs

def main():
    c._need(c._json_read(OUT/'completion.json')['scientifically_complete'],'Cohort must be complete')
    folder=OUT/'attempts/aggregate_label_metadata_v1';folder.mkdir(parents=True,exist_ok=True)
    path=OUT/'aggregate_summaries.csv';backup=folder/'aggregate_summaries_before_correction.csv'
    if not backup.exists():shutil.copy2(path,backup)
    sub=numeric_rows('contribution_substitution_effects');atten=numeric_rows('attenuation_effects');sensitivity=numeric_rows('recipient_context_sensitivity')
    sparse=[dict(r,contrast=r['metric']) for r in numeric_rows('sparse_unpruned_contrasts')]
    routing=[dict(r,contrast=r['metric']) for r in numeric_rows('routing_mode_contrasts')]
    pooled=[r for r in numeric_rows('interface_statistics') if r['kind']=='validation_native_pair']
    aggregate=a.summaries([(sub,['model_type','mode'],['V_start','V_end','Q','S_start','S_end','diagonal_zero_damage_change']),(atten,['model_type','mode','recipient','donor','alpha'],['A']),(sensitivity,['model_type','mode','donor','alpha'],['T']),(sparse,['mode','contrast','donor','alpha'],['sparse_minus_unpruned']),(routing,['model_type','contrast','recipient','donor','alpha'],['native_minus_replay']),(pooled,['model_type','kind','quantity','population'],['start_rms','end_rms','relative_l2','norm_ratio','cosine_global'])])
    original=c._read_csv(backup);c._need(len(original)==len(aggregate),'Summary correction changed row count')
    fields=('n','mean','median','minimum','maximum','sample_sd')
    for old,new in zip(original,aggregate):
        for field in fields:c._need(float(old[field])==float(new[field]),f'Summary numeric value changed: {field}')
    c._csv_write(path,aggregate)
    checked=a.verify_exported()
    amendment=dict(status='PASS',reason='Grouping key metric was overwritten by the summary measure name; preserve the original contrast in a separate contrast column.',numeric_values_unchanged=True,condition_grid_unchanged=True,source_seal_unchanged=True,caches_reused_without_new_evaluation=True,source_sha256=c._sha256(Path(__file__)),original_summary_sha256=c._sha256(backup),corrected_summary_sha256=c._sha256(path),independent_export_checks=checked)
    c._json_write(folder/'correction.json',amendment)
    checks=c._json_read(OUT/'numerical_checks.json');checks['aggregate_metadata_correction']=amendment;c._json_write(OUT/'numerical_checks.json',checks)
    completion=c._json_read(OUT/'completion.json');completion['aggregate_metadata_correction']='attempts/aggregate_label_metadata_v1/correction.json';c._json_write(OUT/'completion.json',completion)
    findings=OUT/'findings.md';text=findings.read_text();note='\nEngineering note: the aggregate contrast label initially collided with the summary measure field. The original summary is preserved under attempts/aggregate_label_metadata_v1/. Final summaries retain contrast and measure in separate columns; all numerical values, cached evaluations and prespecified conditions are unchanged.\n'
    if note not in text:findings.write_text(text+note)
    a.bundle();print('Aggregate labels corrected; all summary numeric values unchanged; independent checks PASS.')
if __name__=='__main__':main()
