"""Writes the metric inventory, frozen analysis configuration and checkpoint/cache provenance table."""
import hashlib
from fge_common import *

INV = [
 # metric, status, formula, aggregation, reference_state, probe, conditions_steps, exclusions, provenance
 ('coupling: router score-activation Pearson/Spearman (fixed_source/native x all_expert/accepted/fixed_initial)', 'already_computed_valid',
  'corr over tokens of router logit_j and mean post-GELU activation of expert j', 'equal-expert within layer then equal layer; seed = unit', 'source/initial/same-step dense',
  'registered 2048 positions of validation batch 0', '15 conditions x 16 steps (rerun track; historical shares 0/2500/3000/4000/5000)', 'zero variance, n<8; all-time accepted intersection 11-14/128 units',
  'high_sparsity_coupling_v1/report/coupling_trajectories.csv, paired_seed_coupling_changes.csv'),
 ('router-weight q (raw/centered)', 'already_computed_valid', 'router row energy captured by expert fc1 Gram spectrum', 'per layer-expert', 'unmatched-router control', 'weights only', '16 steps', '', 'report/q_trajectories.csv'),
 ('weight-subspace overlap rank 8/16/32, Gram cosine, trace ratio vs fixed dense final', 'already_computed_valid', 'principal-angle overlap of fc1 weight SVD subspaces', 'per layer-expert; valid counts reported', 'fixed dense final (F)', 'weights only', '16 steps', 'rank-deficient/degenerate cutoffs excluded', 'report/geometry_trajectories.csv'),
 ('hidden representation CKA / relative L2 / cosine at residual boundaries', 'already_computed_valid', 'linear CKA, relative L2, mean cosine vs references', 'per boundary', 'source, initial, same-step dense', '2048 probe positions', '16 steps', '', 'report/representation_trajectories.csv'),
 ('expert-output cosine and relative error (n=128 first probe inputs, sparse-native or fixed-source inputs)', 'already_computed_valid', '||f_s-f_d||/||f_d||, cosine', 'per layer', 'same-step dense', 'first 128 of 2048 positions; native = sparse model inputs', '16 steps', 'zero outputs', 'report/expert_function_trajectories.csv'),
 ('CE, probe CE, prediction KL; ticket labels', 'already_computed_valid', 'validation CE; strict success = sparse CE <= dense CE at 5000', 'per condition', 'matched dense', 'validation 32 batches; probe batch', 'historical 5 states, rerun 16', 'TF32 settings differ historical vs rerun', 'report/model_function_trajectories.csv, rerun_tickets.csv, ticket_label_agreement.csv'),
 ('raw activations (mean post-GELU per expert), router scores, routing/capacity masks, probe positions', 'derivable_from_saved_arrays', 'analysis/*/step_*/raw.npz', 'n/a', 'n/a', '2048 positions', '16 steps', 'full 1024-dim hidden activations and expert outputs are NOT saved', 'analysis/rerun/*/step_*/raw.npz; probes/seed_*.pt'),
 ('post-GELU fc1 hidden activations: CKA, cosine, rel. error, feature/sample subspace overlap r8/16/32', 'requires_checkpoint_forward', 'centered linear CKA; mean cos^2 principal angles of top-k right (feature, R^1024) and left (sample, R^2048) singular subspaces of centered activations', 'per layer-expert, then equal-expert layer, then seed', 'same-step dense', 'registered 2048 positions; fixed source inputs (primary), dense-native inputs (secondary)', '10 sparse x 16 steps', 'N-1 and numerical-rank limits, zero centered norm = undefined', 'activation_geometry/'),
 ('complete expert outputs (fc2 incl. biases): same metrics, feature space R^256', 'requires_checkpoint_forward', 'as above', 'as above', 'same-step dense', 'as above', 'as above', 'as above', 'activation_geometry/'),
 ('fixed-gate functional distortion D, relative squared error, accepted-only/full-probe', 'requires_checkpoint_forward', 'D = mean_x ||sum_e a_e(x)(f_s,e - f_d,e)||^2; a_e = full-softmax gate x selected x capacity-accepted (top-1)', 'per layer; token arrays saved', 'same-step dense gates; fixed source-final gates; A gates (step 0); C gates (selection)', 'registered positions; capacity from full 16384-token batch', '10 sparse x 16 steps', 'undefined denominators reported', 'activation_geometry/, fixed_gate_distortion/'),
 ('pruning distortion C -> D (mask applied to selection state) and CE/KL', 'requires_checkpoint_forward', 'as D; CE, KL(C||D) on probe', 'per layer / whole model / single-layer masking', 'C selection state', 'probe', '10 masks', '', 'fixed_gate_distortion/selection_state_*'),
 ('gradient energy retention, projected cosine, norms, subspace overlap r1/2/4/8 (task and aux separate)', 'requires_diagnostic_backward', '||M*g_d||^2/||g_d||^2; cos(M*g_d, g_s|M); Gram-based uncentered principal angles', 'per minibatch, layer, matrix', 'same-step dense', '16 fixed training minibatches per seed', 'steps 0/2500/5000', 'zero gradients undefined', 'gradient_geometry/'),
 ('mask structure: per-tensor sparsity, unit in/out degree, dead units, nesting, coordinate overlap', 'derivable_from_saved_arrays', 'from mask_08/mask_09', 'per expert', 'exact-count random controls', 'masks', '10 masks', '', 'mask_structure/'),
 ('older 8E/12E gradient alignment; one-shot-mask experiments', 'context_only_not_joined', 'n/a', 'n/a', 'n/a', 'n/a', 'different cohorts', 'not interchangeable with 32E IMP cohort', 'results/cross_arch_optimization_geometry, router_expert_geometry_v1'),
 ('fitted nonlinear mode connectivity; linear interpolation between distinct endpoints', 'unavailable_without_new_training', 'n/a', 'n/a', 'n/a', 'n/a', 'no distinct same-mask endpoints', 'see connectivity/eligibility_report.md', 'connectivity/'),
 ('full 1024-dim expert activations from earlier steps; sparse-native expert-function inputs beyond n=128', 'requires_checkpoint_forward', 'recomputed here for fixed-source and dense-native modes only', '', '', '', '', 'sparse-native mode of the new geometry not evaluated (historical n=128 retained, labeled)', ''),
]

CONFIG = dict(
    version=1, written_before_new_outcomes=True, exploratory_post_hoc=True,
    cohort=dict(experts=32, seeds=list(SEEDS), sparsities=list(SPARSITIES), dense_controls='five unique matched dense (one per seed)', steps=list(STEPS)),
    states={'A': 'original initialization', 'B': 'masked initialization actually used for recovery (A * mask)', 'C': 'direct trained IMP state that selected the mask (.80: round_07, .85: round_08 final_recovered; round_08 is bit-identical to final_s0.80 endpoint)',
            'D': 'diagnostic copy of C with selected mask applied (immediate pruning distortion)', 'E': 'recovered sparse and same-step dense snapshots', 'F': 'source final dense (step_2500 reference)'},
    probes=dict(registered='high_sparsity_coupling_v1/probes/seed_S.pt: validation batch 0 (128x128 tokens), 2048 registered positions, full source hidden inputs for capacity', additional_probes='gradient minibatches only (separate from the validation probe)'),
    activation_geometry=dict(quantities=['post-GELU fc1 hidden', 'complete expert output'], modes={'fixed_source': 'primary', 'dense_native': 'secondary (same dense-native vectors to both experts)'},
                             cka='centered linear CKA, undefined if centered norm zero', subspace='feature-space R^d primary (centered), sample-space R^N secondary; ranks 8/16/32 each only if <= numerical rank and <= N-1; degenerate cutoff flagged',
                             precision='fp32 forward, float64 reductions, TF32 off', aggregation='per layer-expert -> equal-expert mean per layer -> seed; seed is the replication unit'),
    fixed_gate_distortion=dict(formula='D = mean_x ||sum_e a_e(x)[f_sparse,e(x)-f_dense,e(x)]||^2', gates='softmax prob of selected top-1 expert, capacity-accepted, capacity priority = selected probability over the full batch',
                               references=['same-step dense gates (fixed_source and dense_native inputs)', 'fixed source-final gates (fixed_source inputs)', 'A gates at step 0 == same-step dense at step 0', 'C gates for C->D'],
                               normalization='mean squared norm of reference MoE output under the same frozen gates and reference experts', populations=['accepted-only', 'full 2048 probe (dropped = 0)']),
    gradient_geometry=dict(steps=[0, 2500, 5000], minibatches='16 per seed: training-loader batches 0,20,...,300 of epoch 1 (batch size 128), same for dense and sparse; selected by index only',
                           loss='task = mean token CE; aux = 0.1 * mean load-balance auxiliary loss, separate backward', precision='fp32, TF32 off, eval mode, native forward, no optimizer step, no scaler',
                           tensors='expert fc1/fc2 weights per layer', quantities=['energy retention ||M*g_d||^2/||g_d||^2', 'cosine(M*g_d, g_s|M)', 'norms', 'uncentered Gram-based subspace overlap r=1,2,4,8 subject to numerical rank'],
                           random_mask_controls=dict(draws=10, exact_per_tensor_keep_counts=True, seeds='seed*1000+sparsity*10+draw+7*layer+(matrix==fc2)')),
    mask_structure=dict(random_controls_draws=5, note='nesting overlap .75 is structural; cross-seed overlap is coordinate-based only'),
    connectivity='inventory first; execute only if distinct same-mask same-init endpoints with independent recovery randomness exist; otherwise unavailable',
    primary=['expert-output/hidden activation geometry (fixed_source)', 'fixed-gate distortion same-step dense gates', 'gradient energy retention'],
    exploratory=['dense_native mode', 'sample-space subspaces', 'fixed-source-gate trajectory', 'gradient subspace overlap', 'mask structure'],
    exclusions='no condition selected on recovery outcomes; all ten sparse conditions retained; historical/rerun identical states not independent',
    expected_coverage=dict(activation_states=5 * 16, activation_pairs_per_state=2, expert_units=128, gradient_states=5 * 3, gradient_pairs=10 * 3),
    compute_estimate='activation ~1-2 min/step/seed on H100; gradient ~3 min/seed; storage ~0.5 GB',
    precision_note='historical continuation had TF32 off, rerun TF32 on; these diagnostics use TF32 off for all',
)


def main():
    cols = ['metric', 'status', 'formula', 'aggregation', 'reference_state', 'probe_population', 'available_conditions_steps', 'exclusions', 'provenance']
    table(OUT / 'metric_inventory.csv', [dict(zip(cols, r)) for r in INV])
    save(OUT / 'analysis_config.json', CONFIG)
    cov = {r['path']: r for r in __import__('csv').DictReader(open(HSC / 'checkpoint_coverage.csv'))}
    rows = []
    for seed in SEEDS:
        def add(state, path, **kw):
            p = Path(path)
            known = cov.get(str(p))
            rows.append(dict(seed=seed, state=state, path=str(p), sha256=(known['sha256'] if known else sha(p)), bytes=p.stat().st_size, **kw))
        add('A_original_initialization', initial_path(seed))
        add('F_source_final_dense', source_final_path(seed))
        for s in SPARSITIES:
            add('C_selection_state', selection_state_path(seed, s), sparsity=s, mask=str(mask_for(seed, s)), mask_sha256=sha(mask_for(seed, s)))
        for cid in [cond_id(seed)] + [cond_id(seed, s) for s in SPARSITIES]:
            for step in STEPS:
                p = snapshot_path(cid, step)
                j = read(p.with_suffix('.json'))
                rows.append(dict(seed=seed, state='E_snapshot', condition=cid, step=step, path=str(p), sha256=j['snapshot_sha256'], model_hash=j['model_hash'],
                                 expert_hash=j['expert_hash'], router_hash=j['router_hash'], bytes=p.stat().st_size))
        pj = read(HSC / f'probes/seed_{seed}.json')
        rows.append(dict(seed=seed, state='probe', path=str(HSC / f'probes/seed_{seed}.pt'), sha256=sha(HSC / f'probes/seed_{seed}.pt'), probe_json_sha256=pj['sha256']))
    for n in ('fge_common.py', 'fge_metrics.py', 'fge_function.py', 'activation_worker.py', 'gradient_worker.py', 'selection_distortion.py', 'mask_structure.py'):
        rows.append(dict(state='implementation', path=str(SCRIPT_DIR / n), sha256=sha(SCRIPT_DIR / n)))
    table(OUT / 'checkpoint_and_cache_provenance.csv', rows)


if __name__ == '__main__':
    main()
