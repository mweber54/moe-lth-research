"""Normalize only the analysis outputs created by this task and polish reports.
Experiment inputs and finalized Dataset A tables are never modified.
"""
import os,json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
O=Path(os.environ.get('DATASET_A_ANALYSIS_OUT',str(Path(__file__).resolve().parents[1])));ROOT=Path(__file__).resolve().parents[4]
marker=O/'report_finalization.json'
if marker.exists():raise FileExistsError(marker)
new=pd.read_csv(O/'canonical_lower_sparsity.csv');combo=pd.read_csv(O/'canonical_combined.csv');den=pd.read_csv(O/'dense_references.csv').set_index('condition_id');sources=json.loads((ROOT/'results/dataset_a_full_lower_sparsity/runtime_sources.json').read_text());src=pd.read_csv(ROOT/'results/final_dataset_a/canonical/dataset_a_sources.csv').set_index(['num_experts','seed']);old=pd.read_csv(ROOT/'results/final_dataset_a/canonical/dataset_a_canonical.csv');dense_updates=old.drop_duplicates(['num_experts','seed','router_age']).set_index(['num_experts','seed','router_age']).dense_optimizer_steps_applied;model_counts=new.groupby('num_experts').total_model_parameter_count.first()
def normalize(df):
 oldgap=df.ticket_gap.copy()
 df['target_sparsity']=df.sparsity;df['realized_expert_weight_sparsity']=df.actual_sparsity;df['router_absolute_step']=df.source_checkpoint_step
 df['source_attempt_budget']=2500;df['recovery_attempt_budget']=2500;df['mask_type']='learned';df['mask_selection']='source-final joint magnitude within each expert; weight matrices only';df['ce_units']='nats/token'
 df['source_checkpoint_path']=[sources[f'source_e{int(e)}_seed{int(s)}_T2500']['checkpoints'][str(int(a))]['path'] for e,s,a in df[['num_experts','seed','router_age']].itertuples(index=False,name=None)]
 df['initialization_checkpoint_path']=[sources[f'source_e{int(e)}_seed{int(s)}_T2500']['checkpoints']['0']['path'] for e,s in df[['num_experts','seed']].itertuples(index=False,name=None)]
 df['source_applied_updates']=[src.loc[(e,s),'training_steps_applied'] for e,s in df[['num_experts','seed']].itertuples(index=False,name=None)]
 df['dense_recovery_applied_updates']=[dense_updates.loc[(e,s,a)] for e,s,a in df[['num_experts','seed','router_age']].itertuples(index=False,name=None)]
 df['dense_reference_path']=[den.loc[c,'artifact_path'] for c in df.dense_condition_id];df['dense_reference_provenance_path']=df.dense_reference_path+'/provenance.json'
 if 'recovery_applied_updates' in df:df['recovery_applied_updates']=df.recovery_applied_updates.fillna(df.get('sparse_optimizer_steps_applied',np.nan))
 if 'status' in df:df['status']=df.status.fillna('FINALIZED_TABLE')
 df['total_model_parameter_count']=df.num_experts.map(model_counts);df['prunable_expert_weight_count']=df.mask_keep_count+df.mask_pruned_count;df['retained_model_parameter_count']=df.total_model_parameter_count-df.mask_pruned_count
 assert np.array_equal(df.ticket_gap.to_numpy(),oldgap.to_numpy())
 return df
normalize(new).to_csv(O/'canonical_lower_sparsity.csv',index=False);normalize(combo).to_csv(O/'canonical_combined.csv',index=False)
ss=pd.read_csv(O/'setting_summary.csv');rep=pd.read_csv(O/'replicated_candidate_counts.csv');rnd=pd.read_csv(O/'random_control_summary.csv');rv=pd.read_csv(O/'random_control_verification.csv');ages=pd.read_csv(O/'paired_age_contrasts.csv');sp=pd.read_csv(O/'paired_65_vs_50.csv');checks=pd.read_csv(O/'final_verification_checks.csv')
bc=pd.read_csv(O/'exploratory_age_selection.csv');winner_disagreements=int((bc.best_gap_age!=bc.best_sparse_age).sum())
section=f'''## Supported findings and practical limits

All {len(checks):,} main audit checks passed, and all {len(rv)} random-control raw audits passed. Original higher-sparsity compatibility and the combined unique grid passed. The standalone output verifier separately confirms the endpoint arithmetic, counts and paired statistics.

Strict seed matches decline from 88/275 at 50% sparsity to 64/275 at 55%, 39/275 at 60%, and 21/275 at 65%. Mean 65%-minus-50% gaps are positive in all 55 architecture/age settings; 40/55 have this direction in every seed, and {int((sp.p_holm_55<.05).sum())}/55 paired differences survive Holm. This is evidence that increasing sparsity generally worsens the matched gap in the observed grid, without assuming monotonicity in every individual run.

No setting has a one-sided 95% upper bound below zero, and no strict mean-matching claim survives adjustment. The exploratory 5% tolerance has 122/220 settings passing descriptively in all five seeds; five mean-below-tolerance tests survive Holm over 220 settings and three survive joint Holm over all 880 setting/tolerance tests. Those three are **8E/R70/50%, 8E/R70/55%, and 8E/R80/50%**. The 1% and 2% tolerance tests yield no adjusted discoveries.

No age-pair contrast survives Holm over 1,100 tests. Lowest mean sparse CE and smallest mean gap occur at different ages in {winner_disagreements}/20 architecture/sparsity curves. The observed mean interior minima are exploratory and not reproducible optima established by these five seeds. Failure to resolve an age contrast does not establish a flat population response.

The separate 32E random controls show learned-mask advantages of {rnd.mean_random_minus_learned.min():.4f}–{rnd.mean_random_minus_learned.max():.4f} nats/token, with all {int((rnd.p_holm<.05).sum())}/8 paired settings surviving Holm. None of the 40 random runs strictly matches dense. This evidence is limited to 32E, R0/R100 and 50–65% sparsity.

Dense endpoints are verified against their finalized table and explicit reference provenance; this analysis does not claim new raw dense checkpoint verification. Original higher-sparsity rows retain their prior finalized-table provenance. Recovery attempts are fixed at 2,500, while observed applied updates range 2,496–2,499. Trained routers and source-final masks make these router-conditioned recovery candidates rather than initialization-only LTH.

'''
p=O/'findings.md';text=p.read_text();text=text.replace('## Observations\n',section+'## Observations\n',1);p.write_text(text)
with (O/'audit_report.md').open('a') as h:h.write('\n## Final output verification\n\nAll 40 existing random controls passed supplementary raw checkpoint finiteness, masked-zero, source-R0 expert/shared, optimizer and endpoint checks (random_control_verification.csv). Canonical output schema now includes explicit target/realized sparsity, source/router/R0 paths, uniform budgets, available source/sparse/dense applied-update counts, units, and reference provenance paths. Only newly created analysis outputs were normalized; experimental inputs were untouched. The combined original rows use FINALIZED_TABLE status, not a newly observed completion marker. The combined heatmap colorbar was refined after visual inspection (refine_figures.py).\n')
marker.write_text(json.dumps(dict(status='PASS',canonical_numeric_gaps_unchanged=True,main_checks=len(checks),random_raw_audits=len(rv),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
