"""Freeze a post hoc, measurement-only temporal analysis before associations."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'results/expert_deletion_temporal_prediction_v1'
SOURCE = ROOT / 'results/expert_deletion_reliability_v1'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def main():
    D.mkdir(parents=True, exist_ok=True)
    if (D / 'freeze.json').exists():
        raise RuntimeError('Protocol already frozen; refusing to overwrite')
    original = json.loads((SOURCE / 'configuration.json').read_text())
    config = dict(post_hoc=True, all_checkpoints_previously_inspected=True,
        seed=17, source_case=original['source_case'], trajectories=['sparse','unpruned'],
        experts=[0,8,16,24], layer=1, steps=[25,250,1000,2500], step2500='pre-reset',
        discovery_batches=list(range(16)), confirmation_batches=list(range(16,32)),
        discovery_blocks=[1,2,3,4], confirmation_blocks=[5,6,7,8],
        transitions=[dict(early=25,future=250,previous=None,priority='secondary'),
                     dict(early=250,future=1000,previous=25,priority='primary'),
                     dict(early=1000,future=2500,previous=250,priority='primary')],
        rules=['current','history_mean','history_max'],
        subsets=['whole','baseline_accepted'], doses=[0,1,2,4,8],
        controls='output-zero retained separately; not a weight-deletion dose',
        minimum_block_accepted=32, tie_floor=1e-12, minimum_rank_candidates=3,
        tie_break='lexicographic intervention ID; low first, high last; extreme ties unidentified',
        bootstrap_resamples=2000, discovery_bootstrap_seed=1742000,
        confirmation_bootstrap_seed=1742001,
        bootstrap_unit='complete batches, jointly across interventions; same discovery draws across history checkpoints',
        reference_tolerances=[0.0001,0.001],
        magnitude_baseline='unavailable: exact per-union checkpoint-specific measurements not recorded; no new tensor measurements',
        training_updates=0, model_executions=0,
        original_configuration_sha256=sha(SOURCE/'configuration.json'),
        original_output_seal_sha256=sha(SOURCE/'output_seal.json'))
    save(D/'analysis_config.json', config)
    (D/'protocol.md').write_text('''# Frozen exploratory temporal protocol

This is post hoc exploration: all checkpoint measurements and both populations
have already been inspected. Confirmation blocks are previously examined contexts,
not independent generalization data. Freeze precedes computing temporal associations.

Use only existing signed paired batch damage sums divided by actual target counts.
Discovery blocks 1–4 predict confirmation blocks 5–8. Primary transitions are
250→1000 and 1000→2500; 25→250 is secondary with early-traffic limitations.
Retain seed17 NONMATCHING engineering case, layer1 experts0/8/16/24, both trajectories,
all fixed coordinate unions and controls, and pre-reset2500. Deduplicate endpoint aliases.
Join by expert, parameter and verified coordinate digest; keep output-zero distinct.
Equal doses only: dose1 has3 unions, doses2/4 have4, doses0/8 and output-zero are
singletons and support trajectories only. Overlapping candidates are dependent.

Current score is early discovery damage. History mean and max use25/250 to predict1000
and250/1000 to predict2500. Lower predicts lower future damage. No fitting, winner
selection, future/confirmation leakage or expert selection. Exact per-union magnitude
is unavailable; do not reconstruct it from new tensor measurements. Expert load is
constant within panels and is context only. Native accepted populations change over
time; conditional damage is not a same-population comparison.

Report all rules, equal-dose Spearman with average ranks, tied-pair counts and scores.
Undefined if fewer than3 finite candidates or range≤1e-12. Select extremes by exact
score then lexical ID; any extreme alternatives within1e-12 flag an unidentified
selection. Singletons have no rank, high-low or selection-policy advantage claims.

Future high-minus-low and panel-uniform-minus-selected contrasts are paired. Compare
current-selected minus history-selected future damage (positive favors history);
same IDs give exactly zero. Hindsight regret is selected minus future panel minimum,
an optimistic diagnostic. No claim about arbitrary pruning or safe joint deletion.

Bootstrap2000 complete batches with seeds1742000(discovery),1742001(confirmation).
All interventions share draws; history checkpoints share discovery batch IDs.
A: retain original selected IDs and bootstrap confirmation for fixed-selection CIs.
B: independently bootstrap discovery and confirmation, recompute selection and report
frequencies and policy contrasts. Conditional replicates form ratios of sums/counts;
zero denominators yield undefined, with frequencies. Report future block contrasts
and four leave-one-confirmation-block-out estimates. Percentile95% intervals are
descriptive context uncertainty, not seed replication or multiplicity adjustment.

Coverage requires≥32 baseline-accepted positions in each four-batch block. Report
current discovery, history discovery and future confirmation separately, concentration,
all panels, and coverage-qualified descriptive flags. Whole effects remain despite
coverage failure; never replace experts or change criterion. Qualified history results
require all used early discovery blocks and all future confirmation blocks to pass.

Secondary tolerances only1e-4/1e-3 nats/token. Current qualifies if early current≤tol;
persistent qualifies if both early discoveries≤tol. Empty false-safe undefined; all
qualify means ceiling/no discrimination. Future mean>tol is point false-safe; classify
future intervals above/below/crossing. Compute selected coordinate union divided by
all sampled panel coordinates (and active matrix coordinates), never sum overlaps.
No threshold tuning; zero false-safe is not evidence of safety.

Verify hashes, temporal identities, signs/count weighting, old within-checkpoint
summaries, leakage invariance, ties/singletons/empty/zero-denominator behavior and
exact-zero identical-policy comparisons using numeric tests only. No model execution,
new probes/seeds, tensor magnitude measurement, updates or continuation experiment.
Damage changes can reflect weights, routing, gates, populations and downstream
computation; no specialization, winning-ticket mechanism or permanent-deletion claim.
''')
    save(D/'freeze.json',dict(unix=time.time(), configuration_sha256=sha(D/'analysis_config.json'),
        protocol_sha256=sha(D/'protocol.md'), preparation_sha256=sha(Path(__file__))))
    print('Temporal protocol frozen before associations')

if __name__ == '__main__': main()
