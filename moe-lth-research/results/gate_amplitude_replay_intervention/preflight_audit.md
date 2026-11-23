# Gate-amplitude replay preflight

The completed six donor captures retain the exact selected branch coefficients as `capacity_scores`. In this top-1 implementation, `selected_probabilities` directly multiplies `expert_output` inside `MoEFeedForward.forward`; there is no top-k renormalization or post-capacity rescaling. The capture stored that tensor without changing its dtype. Accepted masks and ordered tensor indices identify layer, sample/token position, and selected edge; batch hashes and chronological keys identify occurrences. Full logits and all-expert probabilities were not retained and are not necessary to replay the actual branch coefficient.

Full archive verification is recorded in donor_amplitude_manifest.csv. No donor rerun is needed if all six pass.

## Gradient semantics and self-replay criterion

Detached gate replay removes the task-loss gradient through gate amplitudes into router parameters and upstream hidden states. It preserves the native auxiliary loss `num_experts * sum(detached_usage * mean_native_probability)` with its historical coefficient. This auxiliary loss remains differentiable into the router and preceding shared/expert parameters. The auxiliary usage is based on the replayed expert IDs, as in Level 1. Routers remain trainable; no surrogate or straight-through estimator is introduced.

A deterministic unit check produced identical same-state logits, accepted masks, and branch coefficients. Router task-gradient norm changed from 0.2357963 to exactly zero, while the native auxiliary router-gradient norm remained 0.0737528. Shared and expert task gradients also changed because upstream gates no longer carry gradient. These numerical values are from a small validation fixture, not experimental measurements.

Consequently exact native-donor final-model and final-loss reproduction is not a valid expectation for detached training self-replay. Determinism does not make different backward passes equivalent. Restoring that equivalence would violate the requested stop-gradient intervention. A clarification is pending on replacing final-model identity with exact same-state forward/amplitude/acceptance checks and explicit gradient checks, while reporting trajectory divergence separately. Recipient training has not been launched pending that criterion.

The historical-results filesystem now has approximately 1.3 TB free, so stable repository storage is available for the new experiment. Historical artifacts are unchanged. No broader sweep is planned.

## User-approved resolution

The user explicitly approved exact same-state forward/amplitude/acceptance and gradient checks and instructed not to require final donor-model identity under detached gate replay. All six real final donor states passed exact forward replay on validation batch 0. Complete archive hashes passed for all 24,486 train/validation occurrences. The isolated runner will use stable storage, no donor reruns, and exactly 36 matched recipients.

All six donor final states also reproduce the full 31-batch validation loss exactly under detached assignment-plus-gate readout. Results are in donor_self_replay.csv. This tests the forward path at fixed donor state, not a detached training trajectory.

## Existing ordering is not uniformly target-favorable

Under the completed Level-1 assignment-only readout, 12E R60-minus-R20 contrasts are +0.051494, +0.004692, and +0.016119 for seeds 7, 17, and 29: R60 was already worse than R20 in these matched ticket gaps. The new primary readout additionally fixes gates and may alter baseline contrasts. All three training levels will therefore be reevaluated under that common policy. Attenuation of an already unfavorable contrast is not evidence that a favorable target advantage collapsed.
