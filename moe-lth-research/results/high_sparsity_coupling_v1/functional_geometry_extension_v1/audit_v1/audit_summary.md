# Functional geometry audit v1 — summary

## Scope and handling

This is a targeted audit of the coupling, recovery-geometry, gradient, mask-count, selection-state-pruning, and performance-label findings. No training, recovery, mask search, optimizer update, fitted connectivity analysis, or diagnostic forward/backward pass was run. Existing measurements were reused. Checkpoint tensors and masks were read on CPU for count and hash checks only. Original reports, checkpoints, masks, and caches remain unchanged.

The cohort retains all ten sparse conditions and five matched dense controls. Source seed is the replication unit (`n=5`); the historical and rerun tracks share weights and are not independent replications.

## Main results

1. **Coupling is a reporting-label problem, not a distinct sparse-native result.** The quoted changes reproduce the paired `fixed_source` / `all_expert` result: sparse changes are `+0.1396` at `.80` and `+0.1712` at `.85`; matched dense change is `-0.3652`; difference-in-changes are `+0.5048` and `+0.5363`. Every sparse condition increases and every dense control decreases. All 128 layer-expert units are valid in the common four-endpoint population. An independent recomputation of per-unit Pearson correlations from saved `raw.npz` score, activation, and selection arrays reproduces all ten primary paired rows; the maximum absolute discrepancy is `1.11e-16`. Here `n=128` refers to token positions per expert in the all-expert counterfactual, not to the layer-expert population. See [coupling_reconciliation.csv](./coupling_reconciliation.csv).

2. **The coupling gap widens on genuinely matched inputs.** `fixed_source` uses the same source-final dense hidden vectors for sparse and dense models. Native-input and accepted-route results are separate probes: their hidden inputs and route populations differ. Do not merge the native results into the fixed-source contrast.

3. **Universal geometry divergence is unsupported.** Centered output CKA on fixed-source inputs decreases from step 0 to 5000 in every seed at both sparsities, but the trajectory is not monotonic. In contrast, with dense-native inputs output CKA increases from step 0 to 5000 in every seed, while step 2500 is generally a local high point. Means are `.570 -> .523` (`.80`) and `.544 -> .487` (`.85`) for fixed-source; `.442 -> .489` (`.80`) and `.413 -> .456` (`.85`) for dense-native. Feature/sample overlaps, separate ranks, errors, cosines, intermediate snapshots, and weight-geometry references are in [geometry_interval_changes.csv](./geometry_interval_changes.csv).

4. **Fixed-gate distortion is squared and denominator-sensitive.** For fixed-source inputs and same-step dense gates, the equal-layer mean of per-layer full-probe ratios increases from 2500 to 5000 for every seed: `1.045 -> 4.108` (`.80`) and `1.191 -> 4.978` (`.85`). The average numerator `D` falls (`475.2 -> 338.0` at `.80; `490.2 -> 341.2` at `.85), while reference-output energy falls (`597.7 -> 306.0`). The ratio of aggregate energies differs from the mean of layer ratios. These are squared quantities against a moving dense reference, not unsquared errors or evidence of movement away from fixed geometry.

5. **Gradient-energy claims need narrowing.** At step 5000, task-gradient actual-minus-random-mean ranges are:
   - `fc1/.80`: `-0.00243` to `+0.00085` (near the null).
   - `fc1/.85`: `+0.00277` to `+0.00370`.
   - `fc2/.80`: `+0.00305` to `+0.00533`.
   - `fc2/.85`: `+0.00173` to `+0.00330`.

   All five seeds in the latter three groups exceed the ten stored exact-count random draws. Auxiliary gradients have no random controls. This is not an equivalence test and does not establish useful alignment. See [gradient_audit.csv](./gradient_audit.csv).

6. **The `.328` `fc1` keep fraction is explained by the pruning implementation.** The nominal budget is allocated jointly over each expert's concatenated `fc1+fc2` weights, not separately by tensor or globally across all experts. The combined expert-weight keep fractions are `0.2000008` at `.80` and `0.1499996` at `.85`. Biases and other model parameters remain unpruned. See [mask_count_reconciliation.csv](./mask_count_reconciliation.csv).

7. **C-to-D evidence is transition-specific.** For the direct IMP selection states, `.7902848 -> .80` removes `4.6321%` of remaining weights; `.80 -> .85` removes `25.0005%`. Probe CE changes range from `-0.000408` to `+0.000907` for `.80`, and rise by `+0.041234` to `+0.162638` for `.85`. These are unequal pruning increments, not a controlled comparison of target sparsity alone. See [pruning_transition_audit.csv](./pruning_transition_audit.csv).

8. **Four performance matches remain.** Under the strict same-seed CE criterion at 5000 attempted rerun steps, matches are seed 7 at `.80` and `.85`, seed 43 at `.80`, and seed 61 at `.80`. The candidate conclusion is supportable only as descriptive cohort-level co-occurrence, with numerical functional and gradient measures rather than the undefined word “substantial.” See [claim_evidence_table.csv](./claim_evidence_table.csv).

## Reporting corrections and supersession

- Corrected the consolidation export name to the mode-neutral `expert_function_trajectories_n128`; corrected the coupling figure title to identify raw router-score/post-GELU-activation Pearson on fixed-source/all-expert positions.
- Added a focused regression test for the mode-neutral export name.
- Original reports and all raw artifacts were preserved. No raw measurement output was invalidated. This audit supersedes only the affected labels and interpretations.
- No causal, unique-winner, universal-divergence, beyond-cohort, or novelty claim is supported.

## Deliverables

- [Metric definitions and provenance](./metric_definition_and_provenance.csv)
- [Coupling reconciliation](./coupling_reconciliation.csv)
- [Geometry interval changes](./geometry_interval_changes.csv)
- [Mask count reconciliation](./mask_count_reconciliation.csv)
- [Gradient audit](./gradient_audit.csv)
- [Selection-state pruning audit](./pruning_transition_audit.csv)
- [Claim evidence table](./claim_evidence_table.csv)
- [Corrected findings](./corrected_findings.md)
- [Machine-readable checks](./audit_checks.json)
