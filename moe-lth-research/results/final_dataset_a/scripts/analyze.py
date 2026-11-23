"""Descriptive Dataset A reporting; no training, interpolation or significance tests."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dataset_a_final_mpl')
import sys,json,itertools,datetime,hashlib,ast
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build import ROOT,DATA,OUT,ES,SEEDS,AGES,SP,atomic,jsave,csv
# Reuse the interim numerical-winner definitions verbatim.
p=DATA/'interim_analysis/analyze.py';tree=ast.parse(p.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['win','agesstr']],type_ignores=[]),str(p),'exec'))
METRICS=['ticket_gap','sparse_final_validation_loss','dense_final_validation_loss','pruning_shock','matched_early_auc_gap','dense_recovery_improvement','sparse_recovery_improvement']
def main():
 f=pd.read_csv(OUT/'canonical/dataset_a_canonical.csv');audit=pd.read_csv(OUT/'audits/dataset_a_completion_audit.csv');gap=pd.read_csv(OUT/'audits/dataset_a_ticket_gap_audit.csv');src=pd.read_csv(OUT/'canonical/dataset_a_sources.csv');dense=pd.read_csv(OUT/'canonical/dataset_a_dense.csv');dev=pd.read_csv(OUT/'canonical/dataset_a_router_development.csv')
 summary=[]
 for e,a,sp in itertools.product(ES,AGES,SP):
  g=f[(f.num_experts==e)&(f.router_age==a)&(f.sparsity==sp)];row=dict(architecture=f'{e}E',num_experts=e,router_age=a,sparsity=sp,n_seeds=g.seed.nunique(),complete_five_seeds=set(g.seed)==set(SEEDS),seeds=json.dumps(sorted(g.seed.tolist())))
  for k in METRICS:
   v=g[k];label={'sparse_final_validation_loss':'sparse_final_loss','dense_final_validation_loss':'dense_final_loss','matched_early_auc_gap':'early_auc_gap'}.get(k,k)
   for stat,val in [('mean',v.mean()),('sd',v.std(ddof=1)),('median',v.median()),('min',v.min()),('max',v.max())]:row[stat+'_'+label]=val
  summary.append(row)
 summary=csv(OUT/'summaries/dataset_a_summary_by_age_sparsity.csv',summary)
 master=[];seedwins=[];metricwins=[];distances=[];paired=[];balanced=[]
 for e,sp in itertools.product(ES,SP):
  g=f[(f.num_experts==e)&(f.sparsity==sp)];base=dict(architecture=f'{e}E',num_experts=e,sparsity=sp);z=g.groupby('router_age').ticket_gap.agg(['mean','std','count']);complete=len(g)==55 and set(g.seed)==set(SEEDS);best=win(g,'ticket_gap') if len(g) else [];b=best[0] if best else None;rank=z.sort_values('mean');inter=g[g.router_age.between(10,90)];ib=win(inter,'ticket_gap') if len(inter) else [];sw={};piv=g.pivot(index='seed',columns='router_age',values='ticket_gap')
  for s in SEEDS:
   h=g[g.seed==s];wa=win(h,'ticket_gap') if len(h) else [];sw[s]=wa;hz=h.set_index('router_age').ticket_gap;hs=hz.sort_values();hi=h[h.router_age.between(10,90)];hmin=hi.ticket_gap.min();row={**base,'seed':s,'winning_ages':agesstr(wa) or None,'n_ages':len(h),'full_age_grid':set(h.router_age)==set(AGES),'agreement_with_mean_winner':bool(set(wa)&set(best)) if wa else None,'winner_r0':0 in wa if wa else None,'winner_intermediate':any(0<x<100 for x in wa) if wa else None,'winner_r100':100 in wa if wa else None,'best_second_gap':hs.iloc[1]-hs.iloc[0] if len(hs)>1 else None,'paired_r0_vs_best_intermediate':hz.get(0,np.nan)-hmin,'paired_r100_vs_best_intermediate':hz.get(100,np.nan)-hmin};seedwins.append(row)
  if b is None:master.append({**base,'complete_five_seed_curve':False,'n_valid_cells':0});continue
  near=[]
  for a in z.index:
   diff=piv[a]-piv[b];flag=bool(abs(diff.mean())<=diff.std(ddof=1));distances.append({**base,'router_age':a,'reference_age':b,'paired_mean_difference':diff.mean(),'paired_seed_sd':diff.std(ddof=1),'n_paired_seeds':diff.count(),'within_one_paired_sd_descriptive':flag})
   if flag:near.append(a)
  basin=[b]
  for direction in [-1,1]:
   a=b+10*direction
   while a in near:basin.append(a);a+=10*direction
  second=int(rank.index[1]) if len(rank)>1 else None;im=z.loc[ib[0],'mean'] if ib else np.nan
  row={**base,'complete_five_seed_curve':complete,'n_valid_cells':len(g),'n_ages':len(z),'min_seeds_per_age':int(z['count'].min()),'max_seeds_per_age':int(z['count'].max()),'mean_minimizing_router_age':b if len(best)==1 else None,'mean_winning_ages':agesstr(best),'minimum_mean_ticket_gap':z.loc[b,'mean'],'r0_mean_ticket_gap':z['mean'].get(0),'r100_mean_ticket_gap':z['mean'].get(100),'best_intermediate_router_age':ib[0] if len(ib)==1 else None,'best_intermediate_winning_ages':agesstr(ib),'best_intermediate_mean_gap':im,'intermediate_advantage_vs_r0':z['mean'].get(0,np.nan)-im,'intermediate_advantage_vs_r100':z['mean'].get(100,np.nan)-im,'second_lowest_age':second,'gap_to_second_lowest':z.loc[second,'mean']-z.loc[b,'mean'] if second is not None else None,'mean_winner_seed_sd':z.loc[b,'std'],'seed_specific_winning_ages':json.dumps(sw),'num_seed_winners_r0':sum(0 in v for v in sw.values()),'num_seed_winners_intermediate':sum(any(0<x<100 for x in v) for v in sw.values()),'num_seed_winners_r100':sum(100 in v for v in sw.values()),'seeds_agreeing_with_mean_winner':sum(bool(set(v)&set(best)) for v in sw.values()),'descriptive_basin':agesstr(sorted(basin)),'basin_width':max(basin)-min(basin),'curve_range':z['mean'].max()-z['mean'].min(),'median_seed_sd':z['std'].median()};master.append(row)
  mw={**base,'complete_five_seed_curve':complete}
  for k in METRICS[:5]+['sparse_early_auc']:mw[k+'_winning_ages']=agesstr(win(g,k));mw[k+'_disagrees']=set(win(g,k)).isdisjoint(best)
  metricwins.append(mw)
  for boundary in [0,100]:
   for age in best:
    h=g[g.router_age==age].set_index('seed');bh=g[g.router_age==boundary].set_index('seed')
    for seed in h.index.intersection(bh.index):paired.append({**base,'seed':seed,'candidate_age':age,'boundary_age':boundary,**{k+'_difference':h.loc[seed,k]-bh.loc[seed,k] for k in ['ticket_gap','sparse_final_validation_loss','dense_final_validation_loss']}})
  full=[s for s,h in g.groupby('seed') if set(h.router_age)==set(AGES)];h=g[g.seed.isin(full)]
  if len(h):balanced.append({**base,'n_full_grid_seeds':len(full),'seeds':json.dumps(full),'winning_ages':agesstr(win(h,'ticket_gap'))})
 master=csv(OUT/'summaries/dataset_a_master_results.csv',master);seedwins=csv(OUT/'summaries/dataset_a_seed_winners.csv',seedwins);metricwins=csv(OUT/'summaries/dataset_a_metric_disagreement.csv',metricwins);csv(OUT/'summaries/dataset_a_age_distance_from_minimum.csv',distances);csv(OUT/'summaries/dataset_a_paired_boundary_contrasts.csv',paired);csv(OUT/'summaries/dataset_a_balanced_seed_sensitivity.csv',balanced)
 # Recovery dynamics: matched raw validation points, dense stored only once.
 t=pd.read_parquet(OUT/'trajectories/dataset_a_recovery_trajectories.parquet');tv=t[t.validation_loss.notna()].copy();dt=tv[tv.model_type=='dense'][['architecture','seed','router_age','recovery_step','validation_loss']].rename(columns={'validation_loss':'dense_validation_loss'});st=tv[tv.model_type=='sparse'].merge(dt,on=['architecture','seed','router_age','recovery_step'],validate='many_to_one');st['validation_gap']=st.validation_loss-st.dense_validation_loss
 csv(OUT/'summaries/dataset_a_recovery_dynamics.csv',st.groupby(['architecture','sparsity','router_age','recovery_step']).validation_gap.agg(['mean','std','count']).reset_index())
 auc=[]
 for cid,h in st.groupby('condition_id'):
  z=h.set_index('recovery_step').validation_gap;r=f[f.condition_id==cid].iloc[0];first=25*(z[0]+z[50]);auc.append({'condition_id':cid,'architecture':r.architecture,'seed':r.seed,'router_age':r.router_age,'sparsity':r.sparsity,'early_auc_gap_steps_0_50':first,'early_auc_gap_steps_50_1250':r.matched_early_auc_gap-first,'initial_gap':z[0],'gap_step_50':z[50],'gap_step_1250':z[1250],'final_gap':z[2500]})
 auc=csv(OUT/'summaries/dataset_a_early_auc_decomposition.csv',auc)
 expo=src.groupby('num_experts').mean_accepted_per_expert.agg(['mean','std','count']).reset_index();csv(OUT/'summaries/dataset_a_source_exposure.csv',expo)
 old=DATA/'interim_analysis/20260921T133833582668Z';iw=pd.read_csv(old/'interim_architecture_sparsity_winner_map.csv');comp=master.merge(iw[['num_experts','sparsity','winning_ages']],on=['num_experts','sparsity'],how='left').rename(columns={'winning_ages':'september_21_winning_ages'});comp['winner_changed']=comp.mean_winning_ages!=comp.september_21_winning_ages;csv(OUT/'summaries/dataset_a_interim_comparison.csv',comp)
 plots(f,summary,master,seedwins,dense,dev,expo,tv,st)
 integrity(f,dense,src,audit,gap,t,summary,master,seedwins)
 report(f,master,seedwins,metricwins,audit,gap,comp,expo,dev,auc)
 dictionary(f,t)
 from supplement import augment
 augment()
 # Code and complete outputs hashed after every report and test result is committed atomically.
 outputs={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='dataset_a_output_hashes.json' and 'cache' not in p.relative_to(OUT).parts and not p.name.endswith('.tmp')}
 jsave(OUT/'dataset_a_output_hashes.json',outputs)
 print((OUT/'reports/dataset_a_terminal_report.txt').read_text(),flush=True)
def plots(f,summary,master,seedwins,dense,dev,expo,tv,st):
 plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 index=[]
 def save(fig,name):
  for ext in ['pdf','png']:
   dest=OUT/'figures'/f'{name}.{ext}';tmp=dest.with_suffix('.'+ext+'.tmp');fig.savefig(tmp,format=ext,dpi=220,bbox_inches='tight');tmp.replace(dest)
  plt.close(fig);index.append(name)
 def lines(ax,g,k,seeds=True):
  if seeds:
   for seed,h in g.groupby('seed'):ax.plot(AGES,h.set_index('router_age')[k].reindex(AGES),'.-',lw=.55,alpha=.45,label=f'Seed {seed}')
  z=g.groupby('router_age')[k].agg(['mean','std']).reindex(AGES);ax.plot(AGES,z['mean'],'ko-',ms=2,lw=1,label='Mean');ax.fill_between(AGES,z['mean']-z['std'],z['mean']+z['std'],color='gray',alpha=.18);ax.set_xlabel('Router age (%)');ax.set_xticks([0,20,40,60,80,100]);ax.grid(alpha=.15)
 labels={'ticket_gap':'Sparse − matched dense CE (nats/token)','sparse_final_validation_loss':'Sparse final CE (nats/token)','dense_final_validation_loss':'Dense final CE (nats/token)','pruning_shock':'Initial sparse − dense CE (nats/token)','matched_early_auc_gap':'Early AUC gap (nats/token × attempts)'}
 for metric in ['ticket_gap','sparse_final_validation_loss','pruning_shock','matched_early_auc_gap']:
  for e in ES:
   fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
   for ax,sp in zip(axs.flat,SP):
    g=f[(f.num_experts==e)&(f.sparsity==sp)];lines(ax,g,metric);ns=g.groupby('router_age').seed.nunique();ax.set_title(f'{sp:.0%} pruning; n={ns.min() if len(ns) else 0}–{ns.max() if len(ns) else 0} seeds');ax.set_ylabel(labels[metric])
   axs.flat[0].legend(fontsize=6,ncol=2);fig.suptitle(f'{e}E — available validated data; finalization gate in status manifest');save(fig,f'{metric}_{e}E')
 fig,axs=plt.subplots(1,5,figsize=(21,4),layout='constrained');lo=summary.mean_ticket_gap.min();hi=summary.mean_ticket_gap.max()
 for ax,e in zip(axs,ES):
  g=summary[summary.num_experts==e];z=g.pivot(index='sparsity',columns='router_age',values='mean_ticket_gap').reindex(index=SP,columns=AGES);im=ax.imshow(z,aspect='auto',interpolation='none',vmin=lo,vmax=hi);ax.set_xticks(range(11),AGES,rotation=90);ax.set_yticks(range(6),[f'{sp:.0%}' for sp in SP]);ax.set_title(f'{e}E');ax.set_xlabel('Router age (%)');ax.set_ylabel('Pruned fraction')
 fig.colorbar(im,ax=axs,label=labels['ticket_gap'],shrink=.8);save(fig,'ticket_gap_heatmaps')
 fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
 for ax,sp in zip(axs.flat,SP):
  for e in ES:
   g=summary[(summary.num_experts==e)&(summary.sparsity==sp)].set_index('router_age');ax.plot(AGES,g.mean_ticket_gap.reindex(AGES),'.-',label=f'{e}E')
  ax.set_title(f'{sp:.0%} pruning');ax.set_xlabel('Router age (%)');ax.set_ylabel(labels['ticket_gap'])
 axs.flat[0].legend();save(fig,'cross_architecture_ticket_gap')
 fig,axs=plt.subplots(1,5,figsize=(20,4),layout='constrained')
 for ax,e in zip(axs,ES):
  g=dense[dense.architecture==e].rename(columns={'final':'dense_final_validation_loss'});lines(ax,g,'dense_final_validation_loss');ax.set_title(f'{e}E');ax.set_ylabel(labels['dense_final_validation_loss'])
 save(fig,'dense_final_loss')
 for e in ES:
  for sp in SP:
   fig,axs=plt.subplots(3,4,figsize=(14,9),layout='constrained');g=tv[(tv.architecture==f'{e}E')&((tv.sparsity==sp)|(tv.model_type=='dense'))]
   for ax,a in zip(axs.flat,AGES):
    for kind,ls,color in [('sparse','-','tab:blue'),('dense','--','black')]:
     h=g[(g.router_age==a)&(g.model_type==kind)];z=h.groupby('recovery_step').validation_loss.agg(['mean','std']);ax.plot(z.index,z['mean'],ls,color=color,label=f'{kind} n={h.seed.nunique()}');ax.fill_between(z.index,z['mean']-z['std'],z['mean']+z['std'],color=color,alpha=.14)
    ax.set_yscale('log');ax.set_title(f'R{a}');ax.set_xlabel('Recovery attempts');ax.set_ylabel('Validation CE, nats/token (log axis)');ax.legend(fontsize=6)
   axs.flat[-1].axis('off');fig.suptitle(f'{e}E / {sp:.0%} pruning — available seed means ± sample SD');save(fig,f'recovery_trajectories_{e}E_s{sp:.2f}')
 rc=[c for c in dev if c.startswith('router_') and c!='router_age']
 for e in ES:
  fig,axs=plt.subplots(2,4,figsize=(14,6),layout='constrained')
  for ax,k in zip(axs.flat,rc):
   lines(ax,dev[dev.num_experts==e],k);units={'router_entropy':'nats','router_margin':'probability difference','router_load_cv':'SD / mean','router_token_id_specialization':'MI, nats','router_assignment_agreement_to_r100':'fraction','router_js_distance_to_r100':'JS divergence, nats','router_cumulative_sampled_exposure':'proposed assignments / expert'};ax.set_ylabel(k.replace('router_','').replace('_',' ')+' ('+units[k]+')')
  axs.flat[-1].axis('off');fig.suptitle(f'{e}E source development — existing probes only; missing points unfilled');save(fig,f'router_development_{e}E')
 fig,ax=plt.subplots(figsize=(6,4),layout='constrained');ax.errorbar(expo.num_experts,expo['mean'],yerr=expo['std'],fmt='o-');ax.set_xlabel('Expert count per layer');ax.set_ylabel('Accepted source assignments / layer-expert');ax.set_xticks(ES)
 for x in expo.itertuples():ax.annotate(f'n={x.count}',(x.num_experts,x.mean))
 save(fig,'accepted_source_exposure')
 fig,axs=plt.subplots(1,5,figsize=(19,4),layout='constrained')
 for ax,e in zip(axs,ES):
  g=master[master.num_experts==e];ax.plot(g.sparsity*100,g.mean_minimizing_router_age,'ko-',label='Mean winner')
  for seed,h in seedwins[seedwins.num_experts==e].groupby('seed'):
   for r in h.itertuples():
    if isinstance(r.winning_ages,str):
     for a in r.winning_ages.split(';'):ax.scatter(r.sparsity*100+(SEEDS.index(seed)-2)*.25,int(a[1:]),s=12,alpha=.6,color=plt.get_cmap('tab10')(SEEDS.index(seed)),label=f'Seed {seed}' if r.sparsity==SP[0] else None)
  ax.set_title(f'{e}E');ax.set_xlabel('Pruned fraction (%)');ax.set_ylabel('Winning router age (%)');ax.set_yticks(AGES)
 axs[0].legend(fontsize=6);save(fig,'winning_age_seed_heterogeneity')
 fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for ax,b in zip(axs,[0,100]):
  for e,g in master.groupby('num_experts'):ax.plot(g.sparsity*100,g[f'intermediate_advantage_vs_r{b}'],'.-',label=f'{e}E')
  ax.axhline(0,color='gray',lw=.7);ax.set_xlabel('Pruned fraction (%)');ax.set_ylabel(f'R{b} gap − best intermediate gap');ax.legend(fontsize=7)
 save(fig,'intermediate_advantages')
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');matrix=np.full((5,6),np.nan)
 for i,e in enumerate(ES):
  for j,sp in enumerate(SP):
   r=master[(master.num_experts==e)&(master.sparsity==sp)].iloc[0];a=r.get('mean_minimizing_router_age');matrix[i,j]=0 if a==0 else 2 if a==100 else 1 if pd.notna(a) else np.nan;ax.text(j,i,str(r.get('mean_winning_ages','NA'))+('' if r.complete_five_seed_curve else '\npartial'),ha='center',va='center')
 ax.imshow(matrix,vmin=0,vmax=2,cmap=matplotlib.colors.ListedColormap(['#b7d7ee','#cce6c1','#efd0a5']),aspect='auto');ax.set_xticks(range(6),[f'{s:.0%}' for s in SP]);ax.set_yticks(range(5),[f'{e}E' for e in ES]);ax.set_xlabel('Pruned fraction');ax.set_title('Descriptive mean minima: blue R0 / green intermediate / tan R100');save(fig,'master_winner_categories')
 atomic(OUT/'figures/README.md','Available validated observations only. Missing cells remain absent; partial panels are not final five-seed results. Shading/error bars are sample SD, not confidence intervals. Dense trajectories are stored once.\n\n'+'\n'.join(f'- [{x}]({x}.pdf)' for x in index))
def integrity(f,dense,src,audit,gap,t,summary,master,seedwins):
 checks=[]
 def test(name,ok,details=''):checks.append(dict(test=name,passed=bool(ok),details=details))
 keys=['num_experts','seed','router_age','sparsity'];expected=set(itertools.product(ES,SEEDS,AGES,SP));actual=set(f[keys].itertuples(index=False,name=None));de=set(itertools.product(ES,SEEDS,AGES));da=set(dense[['architecture','seed','router_age']].itertuples(index=False,name=None));se=set(itertools.product(ES,SEEDS));sa=set(src[['num_experts','seed']].itertuples(index=False,name=None))
 test('exact_1650_sparse_cartesian_product',actual==expected and len(f)==1650,f'{len(actual)} / 1650');test('exact_275_dense_cartesian_product',da==de and len(dense)==275,f'{len(da)} / 275');test('exact_25_source_trajectories',sa==se and len(src)==25 and src.valid.all());test('exactly_five_seeds_per_architecture',all(set(f[f.num_experts==e].seed)==set(SEEDS) for e in ES));test('eleven_ages_per_architecture_seed',all(set(f[(f.num_experts==e)&(f.seed==s)].router_age)==set(AGES) for e,s in se));test('six_sparsities_per_architecture_seed_age',all(set(f[(f.num_experts==e)&(f.seed==s)&(f.router_age==a)].sparsity)==set(SP) for e,s,a in de));test('unique_sparse_keys',not f.duplicated(keys).any());test('no_unexpected_ages_or_sparsities',actual<=expected);test('exact_one_matched_comparator',f.groupby(['num_experts','seed','router_age']).dense_condition_id.nunique().eq(1).all() and f.dense_condition_id.isin(dense.condition_id).all());test('finite_selected_endpoints',np.isfinite(f[['dense_final_validation_loss','sparse_final_validation_loss','ticket_gap']]).all().all());test('independent_ticket_gap_arithmetic',np.allclose(f.ticket_gap,f.sparse_final_validation_loss-f.dense_final_validation_loss,atol=1e-10,rtol=0));test('historical_stored_gap_comparisons',not gap.comparison_status.eq('MISMATCH').any());test('selected_provenance_checks',f.condition_valid.all() and audit[audit.condition_id.isin(f.condition_id)].condition_valid.all());test('no_incompatible_historical_artifacts',f[['source_valid','dense_valid','sparse_valid','mask_valid','comparator_valid']].all().all());test('no_missing_primary_measurement_substitution',f[['ticket_gap','dense_final_validation_loss','sparse_final_validation_loss']].notna().all().all());test('trajectory_unique_monotonic_steps',not t.duplicated(['condition_id','recovery_step']).any() and t.groupby('condition_id',sort=False).recovery_step.apply(lambda x:x.is_monotonic_increasing).all());test('all_validation_curve_points',t[t.validation_loss.notna()].groupby('condition_id').recovery_step.apply(lambda x:list(x)==list(range(0,2501,50))).all());test('summary_shapes',len(summary)==330 and len(master)==30 and len(seedwins)==150);test('full_finalization_gate',audit.condition_valid.all() and actual==expected and da==de and sa==se)
 last_updates=t[t.recovery_step==2500].set_index('condition_id').optimizer_steps_applied
 update_ok=True
 for kind in ['dense','sparse']:
  for row in f.drop_duplicates(kind+'_artifact_path').to_dict('records'):
   cid=row['dense_condition_id'] if kind=='dense' else row['condition_id'];observed=last_updates.get(cid);expected_updates=row.get(kind+'_optimizer_steps_applied')
   if pd.notna(observed) and pd.notna(expected_updates) and observed!=expected_updates:update_ok=False
 test('trajectory_final_optimizer_counts_match_metadata',update_ok)
 mask_checks=pd.read_csv(OUT/'audits/dataset_a_mask_magnitude_verification.csv')
 test('stored_masks_match_r100_local_magnitude_order',len(mask_checks)==150 and mask_checks.passed.all() and set(mask_checks[['num_experts','seed','sparsity']].itertuples(index=False,name=None))==set(itertools.product(ES,SEEDS,SP)))
 hist=pd.read_csv(OUT/'inputs/historical_recovery_audit.csv')
 invalid_paths=set(hist.loc[~hist.valid,'path'])
 selected_paths={str(x).replace('router_conditioned_fixed_budget_dataset.HOME_BACKUP','router_conditioned_fixed_budget_dataset') for x in pd.concat([f.sparse_artifact_path,f.dense_artifact_path])}
 test('excluded_historical_paths_never_selected',not bool(invalid_paths & selected_paths))
 csv(OUT/'audits/dataset_a_excluded_historical_artifacts.csv',hist[~hist.valid])
 prov=json.loads((OUT/'dataset_a_provenance.json').read_text())
 data_ok=all(json.loads(r.dataset_hashes)==prov['data_file_sha256'] for r in src.itertuples())
 test('source_recorded_data_file_hashes_match_current_files',data_ok)
 inputs=json.loads((OUT/'dataset_a_input_hashes.json').read_text());stable=True
 for name,info in inputs.items():
  p=Path(name)
  if not p.exists() or p.stat().st_size!=info['bytes'] or p.stat().st_mtime_ns!=info['mtime_ns']:stable=False
 test('all_read_inputs_stable_through_analysis',stable)
 checks=csv(OUT/'audits/dataset_a_integrity_tests.csv',checks);passed=checks.passed.all();status=dict(status='PASS' if passed else 'BLOCKED',frozen=bool(passed),source_count=int(((audit.job_type=='source')&audit.condition_valid).sum()),dense_count=len(dense),sparse_count=len(f),missing_cells=int((~audit.artifact_exists).sum()),invalid_cells=int((audit.artifact_exists&~audit.condition_valid).sum()),failed_tests=checks.loc[~checks.passed,'test'].tolist(),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),canonical_rows=len(f),note='Canonical-named table is an explicitly partial diagnostic candidate while BLOCKED; no missing rows fabricated.')
 jsave(OUT/'dataset_a_finalization_status.json',status)
 if passed:
  prov=json.loads((OUT/'dataset_a_provenance.json').read_text());jsave(OUT/'DATASET_A_FROZEN.json',dict(utc_finalization_timestamp=status['timestamp_utc'],protocol_version='historical-native-fixed-budget-v1',source_count=25,dense_condition_count=275,sparse_condition_count=1650,seeds=SEEDS,architectures=ES,age_grid=AGES,sparsity_grid=SP,canonical_dataset_sha256=hashlib.sha256((OUT/'canonical/dataset_a_canonical.csv').read_bytes()).hexdigest(),analysis_git_commit=prov['git_commit'],input_provenance_references=['dataset_a_input_hashes.json','dataset_a_provenance.json'],validation_status='PASS'))
def mdtable(df):
 def fmt(v):
  if pd.isna(v):return 'NA'
  if isinstance(v,(float,np.floating)):return f'{v:.5f}'
  return str(v).replace('|','/').replace('\n',' ')
 return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def report(f,master,seedwins,metricwins,audit,gap,comp,expo,dev,auc):
 status=json.loads((OUT/'dataset_a_finalization_status.json').read_text());complete=master[master.complete_five_seed_curve];inter=complete[(complete.intermediate_advantage_vs_r0>0)&(complete.intermediate_advantage_vs_r100>0)];r0=complete[complete.mean_minimizing_router_age==0];r100=complete[complete.mean_minimizing_router_age==100];unanimous=int(complete.seeds_agreeing_with_mean_winner.eq(5).sum());mwc=metricwins[metricwins.complete_five_seed_curve];dis={k:int(mwc[k+'_disagrees'].sum()) for k in METRICS[1:5]+['sparse_early_auc']};missing=audit[~audit.condition_valid];counts=audit.groupby(['architecture','job_type']).condition_valid.sum().unstack();findings=[f'{len(complete)}/30 architecture/sparsity curves have all five seeds and eleven ages.',f'Among complete curves, {len(inter)} have intermediate mean winners, {len(r0)} R0 winners and {len(r100)} R100 winners.',f'All five seed winner sets overlap the mean winner in {unanimous}/{len(complete)} complete curves.',f'At 95% pruning, complete-curve mean winners are '+', '.join(f'{r.architecture}: {r.mean_winning_ages}' for r in complete[complete.sparsity==.95].itertuples())+'.',f'Ticket gaps independently recomputed for {len(f)} rows; {int(gap.comparison_status.eq("MATCH").sum())} stored comparisons match and {int(gap.comparison_status.eq("MISMATCH").sum())} mismatch.']
 matrix=master.pivot(index='architecture',columns='sparsity',values='mean_winning_ages').reindex([f'{e}E' for e in ES]);matrix.columns=[f'{x:.0%}' for x in matrix.columns]
 cols=['architecture','sparsity','mean_winning_ages','complete_five_seed_curve','min_seeds_per_age','max_seeds_per_age','minimum_mean_ticket_gap','intermediate_advantage_vs_r0','intermediate_advantage_vs_r100','descriptive_basin']
 text=f'''# Dataset A final analysis attempt — {status['status']}

## 1. Executive summary

**FINALIZATION {status['status']}.** This is an existing-artifact diagnostic analysis. The canonical-named table is a partial candidate unless the finalization gate passes; it must not be cited as a complete frozen dataset.

'''+ '\n'.join(f'{i+1}. {v}' for i,v in enumerate(findings))+f'''

## 2. Dataset completeness

{mdtable(counts.reset_index())}

Expected 25 source trajectories, 275 dense and 1650 sparse recoveries. Observed validated counts: {status['source_count']}, {status['dense_count']}, {status['sparse_count']}. Exact Cartesian products, not manifest counts, drive the gate. {len(missing)} expected conditions fail. See `../audits/dataset_a_completion_audit.csv` for every cell, original status and reason; `dataset_a_candidate_audit.csv` retains relocated copies and rejected candidates. Source artifacts were moved: the active directory is a host-local /data symlink, and earlier recoveries survive under HOME_BACKUP. Selection uses original manifest paths first and then stable candidate path order, never endpoint quality. Byte-identical backup copies are provenance duplicates, not extra replicates. Conflicting valid checkpoints block admission.

The local manifest has stale PENDING/RUNNING statuses; completion logs do not substitute for absent checkpoint artifacts. The adder and viper logs report shard completion but the adder-owned groups are not all present on this host. Remote access was not established (SSH host-key verification failed). No statuses or raw artifacts were changed.

## 3. Protocol validation

Canonical protocol: source and recovery each use 2500 attempted updates, batch 128, sequence length 128; source global input budget is 40,960,000 token occurrences. Model vocabulary 256, width 256, four transformer layers, four heads, expert width 1024, native top-1 learned routing, capacity factor 1.25, dropout zero, auxiliary coefficient 0.1. Recovery creates fresh AdamW with zero initial optimizer state, LR 0.0003, weight decay 0.01, clip norm 1, fp16 GradScaler, no scheduler. Applied optimizer steps and AMP skips are retained separately when recorded.

Source YAMLs use `common.protocol_errors`; recovery checkpoint configs are loaded directly and checked with that same protocol validator. Source checkpoint SHA-256 values are checked against original checkpoint manifests; recovery checkpoint SHA-256 values against validation attestations or the accepted historical audit. Every admitted recovery checkpoint is checked for finite tensors and every sparse checkpoint for masked zeros. Initial router, shared rewind, sparse rewound expert and mask component hashes are checked. Sparse rewind hashes are recomputed from R0 and the existing stored mask; masks are never regenerated. All 150 stored masks are independently checked against their source-R100 weights: within each expert, the smallest retained magnitude is at least the largest pruned magnitude, the rounded keep count matches the existing pruning definition, and biases are excluded. Existing origin attestations are preserved; no masks are regenerated and no new pruning is performed. Data files and per-seed batch-sequence hashes are preserved and metadata-linked. Historical missing native-router labels are allowed only under the original default trainable mode, changing router hashes and accepted provenance; unavailable historical update counters remain NA.

Source evaluation cadence is 250 attempts; checkpoint-only saves do not change it. Recovery curves must contain exactly steps 0,50,…,2500 in order with finite validation losses and an endpoint matching the summary. The final endpoint is step 2500, never best-validation selection. Validation is token-weighted cross entropy excluding the auxiliary term. Training loss in gradient logs includes the weighted auxiliary term; the separate auxiliary scalar was not logged and remains NA.

Protocol audit and candidate rejection tables expose all checks and exceptions. Historical incompatible runs remain excluded; a raw label match alone never admits a run. Hash validation supports artifact integrity and linkage; it does not independently replay training or prove the historical data generator's entire execution.

## 4. Canonical metric definitions

Ticket gap = raw sparse step-2500 validation CE − same-architecture/seed/age raw dense step-2500 CE. Dense is reused across six sparsities. Gap comparison tolerance is absolute 1e-10. Missing historical stored gaps remain NA with NO_STORED_GAP, not failures. Pruning shock = sparse step-zero CE − matched dense step-zero CE; the preprune field denotes that equivalent unmasked rewound model. Early AUC uses the existing finalizer's trapezoidal integral over 0–1250 at 50-step spacing, without normalization. Recovery improvement = initial − final CE. Positive intermediate advantage = boundary mean gap − best intermediate mean gap. Numerical ties use absolute 1e-12 and preserve all tied ages. Seeds are independent replicates; SD uses ddof=1.

## 5. Full router-age results

OBSERVATION: the following are available-data mean winners. Partial architecture rows are provisional, with changing seed coverage; use the full summary table and balanced-seed sensitivity before interpreting them.

{mdtable(matrix.reset_index())}

{mdtable(master[cols])}

The connected descriptive basin reuses September 21's rule: adjacent measured ages belong while their paired difference from the representative mean winner is within one paired seed SD. Missing ages break connectivity. This descriptive measure is not a confidence interval, equivalence test, or established optimum. Full differences, runner-up gaps and seed SD are retained.

## 6. Sparsity dependence

OBSERVATION: {len(inter)}/{len(complete)} complete architecture/sparsity curves have an intermediate mean below both boundaries. The best intermediate is computed separately even when R0 or R100 wins overall. Extreme 95% observations are listed in the executive summary. Seed-specific boundary contrasts remain paired and descriptive. No monotone shift in age or universal intermediate preference is assumed.

## 7. Architecture dependence

OBSERVATION: inspect the architecture × sparsity table for varying mean winners, gaps and basin widths. Comparisons include equal global source/recovery attempts, not equal exposure or FLOPs. Partial 24E and 32E results cannot resolve five-seed architecture effects. Mean curves never pool grid cells as independent seeds.

## 8. Five-seed reproducibility

OBSERVATION: {unanimous}/{len(complete)} complete curves have all five seeds' numerical winner sets overlapping the mean winner. Individual curves and the 150-row seed table preserve heterogeneous winners and absent seed/age coverage. Per-seed best-intermediate comparisons are selected on that same seed and are not independent confirmation. Mean-winner boundary contrasts are paired by seed; selection remains descriptive.

{mdtable(complete[['architecture','sparsity','mean_winning_ages','seeds_agreeing_with_mean_winner','seed_specific_winning_ages','gap_to_second_lowest','median_seed_sd']])}

## 9. Recovery-trajectory analysis

OBSERVATION: the long table preserves every logged training attempt and validation point, with dense trajectories stored once per dense condition. Missing training or auxiliary measurements remain NA. Validation endpoints and cadence are checked independently. Thirty PDF/PNG panel sets show all eleven ages for each architecture/sparsity, sparse and dense means with seed SD; log CE axes keep large initial losses visible.

`dataset_a_early_auc_decomposition.csv` separates the 0–50 interval from 50–1250, since a large step-zero shock can dominate early AUC. Across available rows, the median initial matched loss gap is {auc.initial_gap.median():.5f}, at step 50 {auc.gap_step_50.median():.5f}, step 1250 {auc.gap_step_1250.median():.5f}, and step 2500 {auc.final_gap.median():.5f}. These medians describe grid coverage, not independent replication or inferential estimates. Per-age/seed traces, rather than these pooled medians, support timing comparisons.

## 10. Metric disagreement

OBSERVATION: among {len(complete)} complete curves, disjoint winner-set counts versus ticket gap are {dis}. Raw sparse CE and matched ticket gap have different denominators/comparators; pruning shock and early AUC measure different recovery phases. See the disagreement table and paired sparse/dense boundary decomposition rather than treating every minimum as the same outcome.

## 11. Router developmental statistics

OBSERVATION: only existing identity-validated source probes are used. Availability across {len(dev)} source seed/age rows: {json.dumps({k:int(dev[k].notna().sum()) for k in dev if k.startswith('router_') and k!='router_age'})}. Entropy, margin, load CV, token-ID specialization, R100 agreement and historical functional JS are never inferred from recovery diagnostics. Missing probes remain gaps in figures. Sampled cumulative proposed exposure is distinct from complete accepted-assignment counters. Historical JS is retained under its existing functional-distance definition, not silently square-root transformed.

INTERPRETATION: normalized age does not guarantee equal developmental state across architectures. Probe coverage is too sparse to establish a full five-seed developmental equivalence; age-matched plots show only the measured subset. No interpolation or new probes were run.

## 12. Exposure scaling

OBSERVATION: source accepted exposure is computed over the complete 2500-attempt log when available and divided by four layers × expert count. It is not multiplied by router age. Available-seed statistics are:

{mdtable(expo)}

Missing historical source counters remain NA. Counts and SD refer to source seeds once each, never to repetitions of the source value across 66 sparse cells. These measurements describe exposure decline or heterogeneity; no scaling law is fitted. Applied expert steps and attempted active steps remain separate.

## 13. Comparison against September 21 interim observations

{mdtable(comp[['architecture','sparsity','september_21_winning_ages','mean_winning_ages','winner_changed','complete_five_seed_curve']])}

OBSERVATION: completed 8E/12E/16E results can be compared directly; 24E/32E still require full artifact coverage before earlier provisional findings become final. The 32E/85% R0 preference must be read with its current coverage flag, not advertised as five-seed confirmation. Balanced full-age seed sensitivity is saved separately. No interim conclusion is silently promoted to final.

## 14. Negative and contradictory results

Boundary winners, disagreement between metric minima, and heterogeneous seed curves are retained equally. Intermediate wins do not imply all seeds favor the same age. Missing remote artifacts are an integrity limitation, not unfavorable outcomes to omit. A selected exact mean minimum is not a statistically established optimum.

## 15. Limitations

The full-grid finalization gate is {status['status']}. Five seeds limit ranking precision. Same data select and describe minima; no p-values or significance claims are made. Equal global budgets do not establish equal per-expert exposure, equal development, equal convergence, identical FLOPs, or causal router mechanisms. Historical missing diagnostics are explicit. Source/mask history relies partly on pre-existing generation attestations; raw training is not replayed.

## 16. Implications for Dataset B

REQUIRES FURTHER EXPERIMENT: a separately specified study could assess whether observed associations persist after controlling expert exposure or changing source development budgets. Dataset B was not started and its protocol was not changed or generated here.

## 17. Candidate mechanism hypotheses

HYPOTHESIS: router development may covary with recoverability through assignment specialization, accepted-load balance, or dense-baseline changes. These competing explanations are not identified by Dataset A. INTERPRETATION: paired endpoint decompositions help locate which measured loss moves; they do not isolate a causal mechanism. REQUIRES FURTHER EXPERIMENT: controlled interventions and an independently specified confirmation design.

## 18. Conclusions supported by Dataset A alone

Only the validated observed cells support descriptive associations under equal global budgets. Full Dataset A completion and freezing require every required source, dense and sparse cell to pass integrity checks. The status manifest and automated tests are authoritative; no frozen marker is written when a required check fails.

No new training, recovery, masks, or Dataset B runs were started. Raw experimental artifacts were opened read-only and were not modified.
'''
 atomic(OUT/'reports/dataset_a_final_analysis.md',text)
 term=f'''DATASET A FINALIZATION REPORT

Source trajectories: {status['source_count']} / 25
Dense recoveries: {status['dense_count']} / 275
Sparse recoveries: {status['sparse_count']} / 1650

Missing cells: {status['missing_cells']}
Invalid cells: {status['invalid_cells']}
Duplicate cells: {int(audit.duplicate_manifest_rows.sum())} manifest duplicates (artifact copies separately audited)
Protocol mismatches: {int((audit.artifact_exists & ~audit.protocol_valid).sum())} existing cells failing validation; see audit reasons
Comparator mismatches: {int((~audit.comparator_valid).sum())} unavailable/invalid matched comparators
Ticket-gap mismatches: {int(gap.comparison_status.eq('MISMATCH').sum())}

Canonical dataset rows: {len(f)} (partial candidate while blocked)
Aggregated summary rows: 330
Master results rows: 30
Seed-winner rows: 150

Main descriptive findings:
'''+ '\n'.join(f'{i+1}. {v}' for i,v in enumerate(findings))+f'''

Previous interim findings confirmed: consult exact comparison table; complete architecture cells remain separately flagged.
Previous interim findings revised: {int(comp.winner_changed.sum())} available-data winner entries changed; partial cells remain provisional.

Finalization status: {status['status']}
Canonical dataset path: {OUT/'canonical/dataset_a_canonical.csv'}
Master results path: {OUT/'summaries/dataset_a_master_results.csv'}
Final report path: {OUT/'reports/dataset_a_final_analysis.md'}
Figures directory: {OUT/'figures'}
Frozen manifest path: {str(OUT/'DATASET_A_FROZEN.json') if status['status']=='PASS' else 'NOT CREATED — finalization gate failed'}

DATASET B WAS NOT STARTED.
NO NEW TRAINING WAS STARTED.
NO RAW EXPERIMENTAL ARTIFACTS WERE MODIFIED.
'''
 atomic(OUT/'reports/dataset_a_terminal_report.txt',term)
def dictionary(f,t):
 explicit={'ticket_gap':('nats/token','sparse and dense metrics.jsonl at step 2500','sparse final minus matched dense final'),'pruning_shock':('nats/token','metrics.jsonl step 0','sparse initial minus dense initial'),'matched_early_auc_gap':('nats/token × attempts','metrics.jsonl steps 0–1250','sparse trapezoidal AUC minus dense AUC'),'router_checkpoint_hash':('SHA-256 component digest','source_checkpoint_manifest.csv','router state_dict_hash; distinct from whole checkpoint file digest'),'source_checkpoint_hash':('SHA-256 file digest','source checkpoint','hash of serialized checkpoint'),'data_hash':('SHA-256 batch-sequence digest','metadata.json','training materialized batch sequence hash, not text-file hash'),'validation_hash':('SHA-256 batch-sequence digest','metadata.json','validation materialized batch sequence hash'),'sparse_preprune_validation_loss':('nats/token','matched dense metrics.jsonl step 0','equivalent unmasked rewound initial model'),'training_loss':('nats/token + weighted auxiliary','gradient_stats.jsonl train_loss','exact logged total optimization loss; not language CE alone'),'auxiliary_loss':('unavailable','not separately recorded in raw gradient logs','NA; never obtained by subtracting unrelated diagnostics'),'learning_rate':('optimizer rate','validated recovery checkpoint config','constant 0.0003; no scheduler'),'optimizer_steps_applied':('updates','gradient_stats.jsonl optimizer_step_applied','cumulative true flags through this attempt; NA if flags absent')}
 lines=['# Dataset A data dictionary','', 'NA means genuinely unavailable diagnostics or an explicitly unavailable derived summary. The candidate table contains validated sparse observations only; the completion audit retains all 1650 expected sparse keys plus dense/source keys. Every excluded key has a reason. A blocked candidate is not frozen.','', 'Source exposure is full-trajectory and repeated by condition; do not treat repetitions as replicates. Developmental sampled exposure is age-cumulative proposed exposure. Entropy and specialization are unnormalized nats, margin is probability difference, load CV dimensionless, agreement a fraction, JS retains the historical functional-JS convention. Summaries use sample SD across seeds.','', '## Canonical condition table','', '| Column | Units | Source | Calculation / interpretation |','|---|---|---|---|']
 for k in f:
  if k in explicit:u,s,c=explicit[k]
  elif k.endswith('_path') or k.endswith('_artifact') or 'artifact_path' in k:u,s,c='path','selected artifact inventory','absolute location; physical backup relocation preserved'
  elif 'hash' in k or k.endswith('_id') and 'config' in k:u,s,c='SHA-256 digest','checkpoint/config/metadata or manifest','file, state-dictionary or canonical sorted-JSON hash as named; provenance records file hashes separately'
  elif k.endswith('_valid'):u,s,c='boolean','completion/protocol/gap audits','all required corresponding checks passed'
  elif 'auc' in k:u,s,c='nats/token × attempts','raw recovery validation curve','existing trapezoidal integral 0–1250, every 50 attempts, no normalization'
  elif 'validation_loss' in k:u,s,c='nats/token','raw recovery metrics.jsonl','step 0 for initial/postprune; step 2500 for final'
  elif 'improvement' in k:u,s,c='nats/token','raw recovery curve','initial minus final loss'
  elif 'assignments' in k or 'accepted_per_expert' in k:u,s,c='layer/expert assignment occurrences','full source or recovery exposure log','sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA'
  elif 'active_' in k:u,s,c='attempts or applied updates per expert','full exposure logs','number with accepted count >0, averaged over 4 × E; applied variant additionally requires optimizer update flag'
  elif 'optimizer_steps' in k or 'amp_skipped' in k:u,s,c='updates or skips','recovery metadata.json','reported separately; applied + AMP skipped = 2500 when both recorded'
  elif k.startswith('router_') and k not in ['router_age','router_age_percent']:u,s,c='see developmental units above','identity-validated existing source-probe table','preserved existing measurement, no recovery-derived substitution or interpolation'
  else:
   defs={'condition_id':('identifier','Cartesian grid','sparse_eE_seedS_RA_sSP; unique condition'), 'architecture':('label','grid','E plus E suffix'),'num_experts':('experts/layer','checkpoint model config','one of 8,12,16,24,32'),'seed':('identifier','config','one of 7,17,29,43,61'),'router_age':('percent','grid','one of 0,10,…,100'),'router_age_percent':('percent','grid','alias of router_age'),'source_checkpoint_step':('attempts','source checkpoint manifest','25 × router age'),'sparsity':('fraction','grid','target removed expert-weight fraction'),'retained_fraction':('fraction','grid','1 − target sparsity'),'actual_sparsity':('fraction','stored mask','pruned count / mask parameter count'),'mask_keep_count':('parameters','mask manifest','number of mask ones'),'mask_pruned_count':('parameters','mask manifest','number of mask zeros'),'source_attempt_budget':('attempts','validated source config','2500'),'recovery_attempt_budget':('attempts','validated recovery config','2500'),'batch_size':('examples','validated config','128'),'sequence_length':('tokens/example','validated config','128'),'top_k':('experts/token/layer','validated config','1'),'capacity_factor':('multiplier','validated config','1.25'),'aux_loss_weight':('coefficient','validated config','0.1'),'protocol_version':('identifier','canonical README','historical-native-fixed-budget-v1'),'dense_condition_id':('identifier','Cartesian grid','one same E/seed/age comparator shared across sparsities')}
   u,s,c=defs.get(k,('fraction or dimensionless','full exposure logs','acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates'))
  lines.append(f'| {k} | {u} | {s} | {c} |')
 lines+=['','## Long trajectories','','One row per actual logged recovery attempt. Dense conditions occur once, sparsity=0. Validation appears at step 0 and every 50 attempts; intermediate validation NAs are unmeasured, not missing scheduled points. Raw float values are retained at binary64 precision in Parquet and round-trip decimal CSV.','', '| Column | Units | Source / calculation |','|---|---|---|']
 for k in t:
  if k in explicit:u,s,c=explicit[k];d=s+'; '+c
  elif k=='validation_loss':u,d='nats/token','metrics.jsonl loss; NA at unevaluated attempts'
  elif k=='recovery_step':u,d='attempts','raw step, not optimizer applied-update count'
  else:u,d='identity / label','raw selected recovery condition identity; model_type=dense or sparse; path is original artifact'
  lines.append(f'| {k} | {u} | {d} |')
 lines+=['','## Summary and selection conventions','','All 330 expected architecture/age/sparsity groups are present with n_seeds, including zero-coverage groups. Master and seed-winner tables preserve all 30 and 150 expected keys; unobserved winners are NA. Summary min/median/max and sample SD use available independent seeds; complete_five_seed_curve is required for complete-grid interpretation. Partial winner comparisons are descriptive only. Positive intermediate advantage means lower intermediate gap. Ties use 1e-12. Basin width is max age minus min age in percentage points using the connected paired-SD rule from the September 21 interim script. No confidence intervals or hypothesis tests are implied.']
 atomic(OUT/'dataset_a_data_dictionary.md','\n'.join(lines)+'\n')
if __name__=='__main__':main()
