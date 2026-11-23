"""Coverage-only preparation: no model forward, gradient, or intervention losses."""
from common import *
import shutil

RULES = {
 'experiment':'gradient_sensitivity_pruning_v1', 'seeds':list(SEEDS), 'source_steps':[250,500,1000],
 'branch_step':1000,'maximum_production_continuations':15,'branches':['C','H','L'],
 'calibration':{'batches':32,'half_batches':16,'selection_rng':'numpy.default_rng(823471 + source_seed)',
   'population':'first occurrence of each distinct complete training-batch hash in absolute steps 1..1000',
   'selection':'uniform without replacement, draw order; first 16 half 0, last 16 half 1',
   'precision':'FP32, TF32 disabled, eval mode, native hard routing/capacity/gates',
   'objective':'mean task CE only; each complete batch equally weighted; no auxiliary loss'},
 'scores':{'A':'mean(abs(w*g))','S':'-w*mean(g)','mean_gradient':'float64 online Welford mean',
   'second_moment':'M2/32 + mean(g)^2','variance':'sample variance across batches M2/31',
   'absolute_mean':'abs(w*mean(g))','sign_consistency':'max(positive,negative)/nonzero(w*g); undefined if zero nonzero products',
   'accumulation':'FP64 Welford plus independent FP64 half sums; no batch-gradient archive',
   'population':'expert fc1/fc2 weights only; no exposure normalization'},
 'reliability':{'candidate_pool':'stable magnitude sort then bottom floor(N/2), flat coordinate tie break',
   'strata':50,'rank':'Spearman average ranks (ties averaged), undefined for constant vectors',
   'groups':'top/bottom ceil(25% of stratum), descending/ascending score then flat index ties',
   'overlap':'intersection size / group size between halves',
   'cross_half':'groups chosen on one half scored on the other, both directions',
   'minimum_accepted_total':1024,'minimum_accepted_per_half':256,'minimum_nonempty_batches_per_half':8,
   'minimum_valid_strata_fraction':0.8,'minimum_median_stratum_rho':0.6,
   'minimum_median_high_overlap':0.5,'minimum_median_low_overlap':0.5,
   'minimum_cross_separated_strata_fraction':0.8,
   'cross_separation':'H>0 and H>=2*L in both directions',
   'minimum_eligible_tensors_fraction_per_seed':0.9,'minimum_eligible_tensors_per_layer_projection':26,
   'failure':'STOP before matching/intervention/training; no calibration expansion or score changes'},
 'matching':{'pool':'bottom half per tensor, 50 equal-count magnitude strata; only reliability-eligible tensors',
   'target_per_branch_per_tensor':2621,'target_fraction':0.01,
   'algorithm':'Within each row, sign and magnitude stratum: visit high scores descending/flat-index tie break; choose unused lower score with H>=2L and H>0 minimizing summed squared normalized feature differences. Ties low flat index. Pair coordinates removed from future consideration. Each pair is directed edge column(H)->column(L). Deterministic DFS column ascending, outgoing edges feature cost then coordinate; extract edge-disjoint directed cycles. Keep whole cycles in discovered order up to target, skipping cycles exceeding remaining budget. No other matching repair.',
   'pair_calipers':{'magnitude_difference_over_tensor_rms':0.005,'OLS_abs_magnitude_slope_times_750_over_rms':0.1,'absolute_magnitude_total_variation_over_rms':0.1},
   'growth':'OLS of abs(w) at 250,500,1000; tensor RMS normalization at 1000',
   'variation':'abs(abs(w500)-abs(w250))+abs(abs(w1000)-abs(w500)), divided by current RMS',
   'sign':'exact paired sign equality','rows':'exact H/L counts at every row','columns':'exact H/L counts at every column',
   'final_checks':{'max_abs_SMD_mag_slope_variation':0.05,'max_KS_mag_slope_variation':0.05,'max_relative_removed_squared_magnitude_difference':0.005,'minimum_mean_A_ratio':2},
   'feasibility':'exclude tensor if fewer than 656 pairs or any final check fails; require >=80% tensors per seed, >=26/32 per layer/projection, and >=0.5% all expert coordinates per branch in every seed; otherwise STOP before intervention losses. No adaptive calipers.',
   'topology_limit':'Equal degrees do not fix exact connectivity; no full topology-control claim',
   'seal':'all five coordinate lists and hashes immutable before any intervention CE'},
 'continuation':{'attempted_updates':1500,'absolute_steps':[1001,2500],'batch_stream':'exact native original complete training batches',
   'optimizer':'fresh AdamW all parameters; canonical source lr, decay, clipping; fresh CUDA GradScaler',
   'precision':'canonical FP16 autocast/GradScaler; native auxiliary task objective/routing/capacity',
   'rng':'seed_everything(source_seed) after construction and all diagnostics, identical per branch',
   'mask':'zero selected coordinates, gradient hooks, post-update zeroing; all surviving parameters trainable',
   'control_reuse':'Only after exact source/data/code/precision/evaluation/reset/accounting audit and three-update state equality; else new C. Historical uninterrupted endpoint forbidden',
   'schedule':'pilot seed17 numerical/execution gate, retain pilot; then seeds7,29,43,61 regardless of effect sign',
   'execution':'Viper/Adder unique attempts, lock/heartbeat, atomic status, failures preserved; no automatic retries; abort nonfinite loss or >100 AMP skips'},
 'endpoints':{'primary':'per-source-seed CE_H(1500)-CE_L(1500)',
   'secondary':['H-C','L-C','immediate changes','trapezoid excess AUC','AUC/1500','applied/AMP skip counts'],
   'validation_grid':[0,25,100,250,500,1000,1500],
   'validation':'identical original full validation batch structure (including original final partial batch), task-only FP32 TF32-off, actual-token weighted',
   'first_order':'sum S removed versus actual mean calibration CE change; validation change separate; sum A never a signed prediction'},
 'numerics':{'finite_difference':'After all masks sealed: first two calibration batches, expert0 fc1/fc2 in each layer, max abs task gradient among bottom-half coordinates; FP64 native clone central differences at eps=[0.01,0.001]*tensor RMS; primary scoring remains FP32. Record FP32/FP64 and +/- routing identities. Boundary cases undefined, never relaxed. Require >=8 away-boundary informative comparisons and every layer represented, rtol=.05 atol=1e-7; abs analytical derivative<1e-7 uninformative.',
  'tests':['meanabs vs absmean','full vs streamed Welford and scores','coordinate identity','past training-only batch hashes','preserve weights/RNG/modes/existing grads','mask counts and disjointness and balance','surviving equality','fresh optimizer/scaler/RNG','persistent zeros and surviving gradient flow','exact continuation data hashes','resume equality','sealed hashes'],
  'stage_policy':'No masking, finite differences, validation or training on reliability failure; downstream tests explicitly NOT_RUN'},
 'analysis':{'replication_unit':'source seed, n=5 intended; no weight/expert/batch pseudoreplication',
  'summaries':['each seed','mean','median','range','sample SD','SE=SD/sqrt(5) descriptive'],
  'constraints':'No novelty claim, causality/routing mechanism claim, equivalence claim, reversed rule, extra scores, sweeps or full pruning policy'},
}

def main():
    assert not (D/'analysis_seal.json').exists(), 'Registration already sealed'
    assert shutil.disk_usage(ROOT).free>120*1024**3
    D.mkdir(parents=True,exist_ok=True)
    old=read(ref.D/'execution_seal.json')
    for p,h in old['implementation'].items(): assert sha(ROOT/p)==h, p
    provenance={'seeds':{},'prior_core_implementation_unchanged':True,'prior_experiments_preserved':True}
    for seed in SEEDS:
        cps={}; states=[]
        for step in (250,500,1000):
            p=ref.source(seed)/f'checkpoints/step_{step}.pt'; cp=load(p)
            assert cp['step']==step
            assert not any(k in cp for k in ('optimizer','scaler','rng','torch_rng'))
            st=cp['model']; schema={n:(list(st[n].shape),str(st[n].dtype)) for n in names(st)}
            assert len(schema)==256 and all(v[1]=='torch.float32' for v in schema.values())
            states.append(schema)
            cps[str(step)]={'path':str(p.relative_to(ROOT)),'sha256':sha(p),'model_sha256':state_dict_hash(st),'expert_schema':schema}
        assert states[0]==states[1]==states[2]
        train,val=ref.data(seed); prefix=train[:1000]
        assert all(tuple(x.shape)==(128,128) and x.shape==y.shape for x,y in prefix)
        hashes=[batch_hash(b) for b in prefix]; first={}
        for i,h in enumerate(hashes): first.setdefault(h,i)
        unique=np.array(list(first.values()),dtype=np.int64); assert len(unique)>=32
        chosen=np.random.default_rng(823471+seed).choice(unique,32,replace=False).tolist()
        cal=[prefix[i] for i in chosen]; vh={batch_hash(b) for b in val}
        assert len({hashes[i] for i in chosen})==32 and not any(hashes[i] in vh for i in chosen)
        p=D/'calibration'/f'seed{seed}.pt';atomic_tensor(p,{'batches':cal,'absolute_steps':[i+1 for i in chosen],'halves':[i//16 for i in range(32)]});p.chmod(0o444)
        provenance['seeds'][str(seed)]={'checkpoints':cps,'unique_prefix_batches':len(unique),
          'train_2500_sha256':R._batch_sequence_hash(train),'continuation_sha256':R._batch_sequence_hash(train[1000:]),
          'validation_sha256':R._batch_sequence_hash(val),'validation_batches':len(val),'validation_target_tokens':sum(y.numel() for _,y in val),
          'calibration_path':str(p.relative_to(ROOT)),'calibration_sha256':sha(p),
          'calibration':[{'index':j,'absolute_step':i+1,'half':j//16,'batch_sha256':hashes[i],'tokens_sha256':tensor_hash(prefix[i][0]),'targets_sha256':tensor_hash(prefix[i][1]),'shape':list(prefix[i][0].shape)} for j,i in enumerate(chosen)]}
        print('Prepared provenance',seed,flush=True)
        del cp,st,train,val,prefix,cal
    save(D/'checkpoint_and_calibration_provenance.json',provenance)
    RULES['provenance_sha256']=sha(D/'checkpoint_and_calibration_provenance.json')
    save(D/'registration.json',RULES)
    (D/'protocol.md').write_text('''# Gradient sensitivity pruning v1\n\nProspective intervention motivated by exploratory work, not preregistration of the whole project. The machine-readable registration is authoritative and is sealed before gradients or intervention losses.\n\nThe fixed source model is measured on 32 complete past training batches, disjoint 16-batch halves. A is mean absolute weight-times-task-gradient; S is its negative signed mean. Batch variance is neither token gradient variance nor curvature. Native hard routing, capacity, gates, and task-only FP32 scoring are preserved. No exposure normalization.\n\nReliability is assessed within each tensor and 50 current-magnitude strata. Each tensor must meet all registered rank, overlap, cross-half separation and traffic thresholds; every seed and layer/projection must meet coverage. Failure stops the experiment without changing batches, scores, or thresholds. Full-32 selection is forbidden before a passing review.\n\nIf reliability passes, pair within row/sign/magnitude stratum under fixed growth/variation calipers, then select directed column cycles to equalize both row and column degrees. Target 1% per branch; only the registered minimum-dose/coverage reductions apply. All five masks are sealed before any intervention losses. Equal row/column degrees do not determine exact connectivity.\n\nIf both gates pass, use a seed17 engineering pilot and all five source seeds, H/L/C shared fresh resets, original steps1001..2500, 1500 attempted FP16 updates, task-only token-weighted FP32 validation on the fixed grid. Audit existing controls before reuse, otherwise run new controls. Failures and skips remain visible. Mask zeros persist via hooks and post-update masking.\n\nPrimary endpoint is final validation CE H minus L; immediate calibration Taylor checks and validation changes are separate diagnostics. Report every source seed before descriptive mean/median/range/SD/SE. No independent replication is attributed to weights, experts or batches. No novelty, mechanism, equivalence or general-policy claim is authorized.\n\nOn a measurement/matching stop, downstream artifacts explicitly record NOT_RUN, not zero effects. No production result is inferred.\n''')
    files=[D/'registration.json',D/'protocol.md',D/'checkpoint_and_calibration_provenance.json']+list(HERE.glob('*.py'))
    files += list((ROOT/'src/moe_lth').rglob('*.py')) + [ROOT/'scripts/magnitude_growth_pruning_v1/common.py',ROOT/'scripts/imp_adaptation_factorial_v1/af_common.py',ROOT/'scripts/imp_adaptation_factorial_v1/af_native.py']
    save(D/'analysis_seal.json',{'stage':'BEFORE_GRADIENTS_AND_INTERVENTION_LOSSES','files':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    for p in (D/'registration.json',D/'protocol.md',D/'checkpoint_and_calibration_provenance.json',D/'analysis_seal.json'):p.chmod(0o444)
    print('SEALED',sha(D/'registration.json'),flush=True)
if __name__=='__main__':main()
