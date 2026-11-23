"""Fixed descriptive contrasts and source-seed-held-out ridge prediction."""
from audit import *
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def main():
 cfg=json.loads((O/'analysis_config.json').read_text());df=pd.read_csv(O/'rank_stability.csv');df['unit']=df['ranking']+'/'+df['tensor'];key=['seed','layer','expert','unit'];idx=df.set_index(key+['start_step','end_step']);adj=list(zip(cfg['checkpoint_steps'][:-1],cfg['checkpoint_steps'][1:]));pred=[];assoc=[]
 for row in df.to_dict('records'):
  t=row['start_step'];prior=next(((a,b) for a,b in adj if b==t and a>0),None)
  if not prior:continue
  pr=idx.loc[tuple(row[k] for k in key)+prior]
  before=next(((a,b) for a,b in adj if b==prior[0]),None)
  old=idx.loc[tuple(row[k] for k in key)+before] if before else None
  row.update(prior_start_step=prior[0],prior_end_step=prior[1],prior_growth_rate=pr['mean_absolute_magnitude_change']/pr['interval_attempts'],prior_late_fraction=pr['late_rising_fraction'],prior_top_turnover=pr['top20_turnover'],prior_applied_updates=pr['interval_applied_updates'],prior_log_applied_rate=np.log1p(pr['accepted_rate_applied']),prior_log_cumulative_accepted=np.log1p(row['cumulative_applied_accepted_by_start']),prior_log_rate_change=np.log1p(pr['accepted_rate_applied'])-np.log1p(old['accepted_rate_applied']) if old is not None else 0.,prior_drop_fraction=pr['drop_fraction'],prior_probe_turnover=pr['fixed_probe_proposed_membership_turnover'])
  assert row['prior_end_step']<=row['start_step'];pred.append(row)
 enriched=pd.DataFrame(pred);enriched.to_csv(O/'expert_interval_metrics.csv',index=False)
 # Same-interval description and strictly past-exposure relationships are separate.
 for mode,data,features in [('same_interval_descriptive',df[df.start_step>0],['accepted_rate_applied','drop_fraction','fixed_probe_proposed_membership_turnover']),('past_exposure_subsequent_outcome',enriched,['prior_log_applied_rate','prior_log_rate_change','prior_probe_turnover'])]:
  for keys,g in data.groupby(['seed','layer','start_step','end_step','unit']):
   for feature in features:
    x=g[feature].to_numpy();y=g['late_rising_fraction'].to_numpy();ok=np.isfinite(x)&np.isfinite(y);rho=float(spearmanr(x[ok],y[ok]).statistic) if len(np.unique(x[ok]))>1 and len(np.unique(y[ok]))>1 else None
    assoc.append(dict(mode=mode,seed=int(keys[0]),layer=int(keys[1]),start_step=int(keys[2]),end_step=int(keys[3]),unit=keys[4],feature=feature,experts=int(ok.sum()),spearman_rho=rho,replication_unit='source_seed'))
 table(O/'descriptive_associations.csv',assoc)
 # Prespecified rate-change same-interval feature, compared with previous interval.
 changes=[]
 for r in pred:changes.append(dict(seed=r['seed'],layer=r['layer'],start_step=r['start_step'],end_step=r['end_step'],unit=r['unit'],expert=r['expert'],same_interval_log_rate_change=np.log1p(r['accepted_rate_applied'])-r['prior_log_applied_rate'],late_rising_fraction=r['late_rising_fraction']))
 cg=pd.DataFrame(changes)
 for keys,g in cg.groupby(['seed','layer','start_step','end_step','unit']):
  x=g.same_interval_log_rate_change;y=g.late_rising_fraction
  rho=float(spearmanr(x,y).statistic) if x.nunique()>1 and y.nunique()>1 else None
  assoc.append(dict(mode='same_interval_descriptive',seed=int(keys[0]),layer=int(keys[1]),start_step=int(keys[2]),end_step=int(keys[3]),unit=keys[4],feature='same_interval_log_rate_change',experts=len(g),spearman_rho=rho,replication_unit='source_seed'))
 table(O/'descriptive_associations.csv',assoc)
 pd.DataFrame(assoc).groupby(['seed','mode','feature']).spearman_rho.agg(['mean','median','count']).reset_index().to_csv(O/'seed_association_summaries.csv',index=False)
 # Prediction uses adjacent outcomes only; every observation for a seed stays in its fold.
 p=enriched[[tuple(x) in adj for x in enriched[['start_step','end_step']].to_numpy()]].copy()
 baseline=['mean_absolute_magnitude_start','initial_mean_absolute_magnitude','current_low_half_mean_magnitude','prior_growth_rate','prior_late_fraction','prior_top_turnover','start_step','interval_attempts','prior_applied_updates'];extra=['prior_log_applied_rate','prior_log_cumulative_accepted','prior_log_rate_change','prior_drop_fraction','prior_probe_turnover'];cats=['layer','unit'];results=[];prediction_rows=[]
 assert set(p.seed)==set(cfg['source_seeds']) and p.groupby('seed').start_step.nunique().min()>=2
 for seed in cfg['source_seeds']:
  train=p[p.seed!=seed];test=p[p.seed==seed];y=train.late_rising_fraction;yt=test.late_rising_fraction
  result=dict(held_out_seed=seed,training_seeds=','.join(str(s) for s in cfg['source_seeds'] if s!=seed),observations=len(test),training_outcome_mean=float(y.mean()),held_out_outcome_mean=float(yt.mean()),seed_mean_null_mse=float(np.mean((yt-y.mean())**2)),independent_replication='one held-out source seed')
  for label,cols in [('baseline',baseline),('extension',baseline+extra)]:
   transform=ColumnTransformer([('numeric',StandardScaler(),cols),('identities',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
   model=make_pipeline(transform,Ridge(alpha=1.0));model.fit(train[cols+cats],y);yp=np.clip(model.predict(test[cols+cats]),0,1);result[label+'_mse']=float(mean_squared_error(yt,yp))
   for r,v in zip(test.to_dict('records'),yp):prediction_rows.append({k:r[k] for k in key+['start_step','end_step']}|dict(model=label,outcome=r['late_rising_fraction'],prediction=float(v)))
  result['mse_improvement']=result['baseline_mse']-result['extension_mse'];result['relative_mse_improvement']=result['mse_improvement']/result['baseline_mse'] if result['baseline_mse'] else None;results.append(result)
 table(O/'held_out_prediction_results.csv',results);table(O/'held_out_predictions.csv',prediction_rows)
 # No pooled significance tests. Utilization comparisons retain seed/layer/pair/unit strata.
 strat=[]
 for keys,g in enriched.groupby(['seed','layer','start_step','end_step','unit']):
  med=g.prior_log_applied_rate.median()
  for label,mask in [('low_prior_utilization',g.prior_log_applied_rate<=med),('high_prior_utilization',g.prior_log_applied_rate>med)]:
   z=g[mask];strat.append(dict(seed=int(keys[0]),layer=int(keys[1]),start_step=int(keys[2]),end_step=int(keys[3]),unit=keys[4],stratum=label,experts=len(z),mean_late_rising=float(z.late_rising_fraction.mean()),mean_current_abs_magnitude=float(z.mean_absolute_magnitude_start.mean()),mean_prior_growth=float(z.prior_growth_rate.mean())))
 table(O/'utilization_strata.csv',strat)
 save(O/'prediction_checks.json',dict(passed=True,model='single Ridge alpha1',future_interval_exposure_features_used=False,preprocessing_training_seeds_only=True,seeds_disjoint=True,all_feature_end_steps_at_or_before_outcome_start=True,baseline_columns=baseline,extension_columns=extra,categorical_columns=cats,adjacent_prediction_intervals=adj[2:],outcome='expert low-half to later top20 fraction',n_source_seeds=5,no_model_search=True))
if __name__=='__main__':main()
