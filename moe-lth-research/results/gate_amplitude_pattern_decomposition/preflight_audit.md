# Gate-amplitude pattern decomposition preflight

Level 1 and Level 2 have 36 completed recipients each. Their manifests retain `analysis_ready_for_review`; the requested numerical Level-2 attenuation values are present in the completed outcome tables. No prior results are changed.

## Amplitude availability

The six exact R100 assignment/acceptance/gate schedules are in stable storage. Level-2 recipient proposals were reduced to per-layer/per-expert statistics at 51 training occurrences. Nine model/optimizer snapshots exist per recipient. The full native proposal tensor at every intervening training occurrence was NOT saved. Exact reconstruction requires deterministic capture reruns of the chosen source trajectories, with unchanged Level-2 forward/backward semantics and exact final-state checks. No decomposition recipients have been launched.

`amplitude_source_manifest.csv` records the 18 chosen dense source trajectories. The user selected matched dense sources: exactly 18 capture reruns. Native sparse/dense tensors cannot both be used independently while also satisfying the requirement that paired recipients receive identical amplitudes.

## Readout choice

The approved primary is the original common R100 assignment-plus-gate readout, with condition-matched and native readouts reported separately. Using age-specific transformed validation amplitudes without a common readout confounds training and evaluation effects.

## Identity and shuffle feasibility

All six training schedules contain 357 distinct batch hashes, each recurring 7–8 times over 2500 steps. Exact batch-plus-position identity is available. Recurrence of the same accepted expert edge still needs checking before interpreting token-conditioned means.

Expert/layer accepted block lengths vary over time. A deterministic equal-length-stratified permutation preserves expert/layer identity and the exact amplitude multiset without padding, dropping, duplication, or interpolation. The measurable movable-edge fraction is approximately 89–94%; singleton strata remain fixed. This is a restricted temporal shuffle, not destruction of all temporal information. Moving a block onto another batch also changes its token mapping; any temporal-specific necessity claim must acknowledge that limitation.

Source-independent unit tests passed for edge-weighted global/layer/expert means, deterministic shared sparse/dense permutation indices, equal-length bijective block permutations, and exact token-shuffle marginals. No transformed scientific schedule has yet been constructed from missing amplitudes.

## Scope and interpretation

The full specified grid comprises seven new conditions × 18 matched pairs = 252 recipient recoveries, in addition to the necessary capture-only reconstruction runs. No conditions have been silently dropped, and no broader sweep is planned. Source selection and the primary readout are resolved. All 18 captures must validate before transformation compilation and all compiled checks must pass before decomposition training.

The source proposals requested here come from Level-2 states; they are not the original Level-1 online amplitude trajectory. Level 1 also retains task gradients through gates, whereas every new condition is detached. Consequently failure of all reduced conditions cannot establish HIGH-ORDER GATE PATTERN REQUIRED without a successful full detached recipient-source control. No extra control has been silently added. Conclusions will be limited to information sufficient in the specified detached interventions.

## Approved canonical source and readout

The user selected the canonical age/seed dense amplitude source tensor: exactly 18 deterministic dense Level-2 capture reruns, not statistical replicates. Every transformation will be shared unchanged by sparse and dense recipients. The primary is primary_r100_common_readout for every condition and reference; secondary_condition_matched_readout and secondary_native_readout remain separate diagnostics. No decomposition recovery may launch until all 18 sources pass integrity checks. The mechanistic question concerns age-conditioned dense amplitude information, not sparse models own native proposals.

## Prespecified transformation and analysis rules

Exact token-conditioned means require at least 80% of training accepted-edge observations to belong to recurring identical batch/position/expert keys. Otherwise the already available tokenizer-ID categorical feature is used within expert/layer and documented as a substitution; no interpolation is used. Training and secondary validation estimates remain separate. Retention denominators must exceed 1e-5. Descriptive full retention is 0.8–1.2 with preserved contrast sign in all three seeds; every raw contrast, SD, and sign is also reported. Restricted time-shuffle singleton blocks remain unchanged and moved-edge coverage is reported.

## Coefficient-precision repair

Actual captured training proposals are FP32 despite fp16/autocast training. The original compiler incorrectly allocated FP16 training buffers. The exact token-marginal invariant stopped compilation before any decomposition training. The compiler now infers and preserves source precision, with exact copy and permutation assertions unchanged. All 18 source captures are reused; the failed partial build is archived in failed_builds.
