# Ranked candidate findings

Scores are reviewer judgments from 1 (weak) to 5 (strong), not p-values or calibrated probabilities. Repeated ages, sparsities and checkpoints do not count as independent replications. Novelty scores are provisional after the targeted literature screen.

| Candidate finding | Evidence strength | Effect size | Reproducibility | Novelty potential | Scientific significance | Feasibility |
|---|---:|---:|---:|---:|---:|---:|
| **1. Router-age dependence collapses under combined gate-value/derivative replay, beyond fixing assignments** | 4 | 5 | 3 | 3 | 4 | 5 |
| **2. Coordinate pruning dissociates signed fixed-probe coupling from dense recovery and unsigned geometry** | 4 | 4 | 4 | 3 | 3 | 4 |
| **3. Learned masks recover while moving away from source leading subspaces; random masks move toward them** | 3 | 3 | 4 | 2 | 3 | 3 |

## 1. Selected: a gate/backward-path bottleneck in router-age effects

**Precise observed claim.** In these 8E/12E, 85%-coordinate-pruned expert recovery protocols, all 12 target/control age contrasts in sparse-minus-dense validation CE have much smaller magnitude after replacing native selected gate coefficients with one stored R100 donor schedule **and detaching gate task gradients**, even when expert assignments and capacity acceptances were already clamped. Group-mean attenuation is 94.69%, 98.67%, 99.81%, 99.54%; the maximum residual absolute contrast is 0.007014210 nats/token. This is a claim about this combined intervention and common readout, not about unique forward-value mediation or all natural MoEs.

Supporting source: `results/gate_amplitude_replay_intervention/outcomes.csv`, recipient `recovery/e*/seed*/R*/{dense,sparse}/summary.json`, replay/gradient semantics audits, and `src/moe_lth/experiments/gate_amplitude_replay.py:21`. `gate_contrasts_reproduced.csv` recomputes all 12 effects from the actual condition losses; all 36 Level-2 endpoint summaries match the common-gate outcomes to 2.3e-16. Figure 01 retains all four comparisons and every seed. Three seeds per architecture, reused across ages, are the independent units.

Why the scores: matched training interventions and exact assignment audits make evidence stronger than observational drift (4); the magnitude is large in all comparisons (5); only three reused seeds and one donor cap replication (3); generic gating is established and backward-path isolation is missing (novelty 3); distinguishing discrete assignment from continuous coefficients/derivatives could matter broadly (significance 4); two missing factorial cells can reuse existing artifacts (feasibility 5).

**Attempted falsification / counterevidence:** four Level-1 comparisons already favor the control, so this is not removal of a universal favorable intermediate-age advantage. Assignment-only substitution already attenuates several contrasts substantially. Readout choice can reverse signs, all full-gate sparse endpoints remain worse than their common-readout dense controls, and detached coefficients remove router/upstream task gradients. Once all router-specific forward outputs are fixed, losing age dependence is partly expected from disconnecting the distinguishing pathway. Auxiliary gradients remain a possible residual path. Pattern decomposition does not supply a matched full detached canonical-source control and has severe, heterogeneous secondary losses. These facts rule out a finished forward-gating-mechanism claim.

**Publishable if:** isolated value or Jacobian interventions show a substantial, reproducible sparsity/age interaction on new seeds and a second donor/readout, with preserved controls and a clear mechanistic explanation beyond standard scaling. **Unpublishable if:** ordinary detachment/readout mismatch fully explains it, effects are tiny after isolation, random masks behave identically when learned-mask specificity is claimed, or transfer fails. Strongest candidate for a bounded next test, not a paper-ready discovery.

## 2. Signed coupling is not a universal ticket/geometry readout

From actual fixed-source/all-expert cached arrays, sparse changes are +.139590/.171162 versus dense −.365186. Paired differences-in-changes are **+.504776** at 80% (seed range .405341–.611653) and **+.536348** at 85% (.423316–.662443), all five seeds positive. New raw-array arithmetic agrees with the audited report to 2.3e-16. Unmatched-expert identity controls are independently recomputed in coupling_pairing_controls_from_raw.csv. This effect appears in winners and nonwinners and is not an effect of E acquiring zero coordinates.

Evidence/reproducibility scores reflect raw arrays, matched dense controls, five source seeds and sensitivity tables. Effect size 4 reflects a large bounded correlation contrast. Novelty/significance are reduced because correlation does not identify expert usefulness and the compared GELU/full-softmax implementation differs from the nearest paper. Accepted-pair effects reverse under restrictive all-time coverage: only 11–14/128 units remain. Thus the stronger-sounding universal anti-coupling claim fails. Ahrac et al.'s model/readout/selection differs; this is a boundary-condition observation, not a direct refutation. A no-auxiliary or altered gating comparison would be needed for mechanism.

## 3. Recovery need not restore source leading eigenspaces

Reaggregation of `router_expert_geometry_v1/analysis_extended/v1/svd_endpoint_changes.csv` on endpoint-matched expert populations yields learned-mask decreases in **29/30, 28/30, 29/30** conditions at ranks 8/16/32. Random masks and dense controls increase in every condition (0 decreases out of 30 and 15 respectively). Mean rank-16 changes are learned −.069168, random +.128705, dense +.264856. Counts include three ages and two sparsities within each of five reused seeds; they are not 30 independent trials.

This was checked from audited per-condition measurements, not by recomputing every expert-weight SVD here (evidence 3). Reference dependence, small singular gaps, different starting overlap and magnitude-mask selection are simpler competing explanations. Full-Gram shape has mixed signs and does not mirror leading-subspace trends. Recovery away from a source optimum is not itself surprising, nor does this identify ticket usefulness. The original 30/30 rank-16 claim was invalid because of changing valid populations; retain both counterexamples. Its novelty is lower than a clean path intervention.

## Investigated but not promoted

**Embedding orientation with stable spectra:** all 240 rerun E checkpoints measured independently; see embedding_geometry_findings.md. Dense mean initial-relative rotation is 3.0243°, 80% 3.0854°, 85% 3.1909°. The 85% excess is consistent but only .1665°; 80% has two exceptions. Gaussian null spectra match ~155 entropy rank and all E matrices have numerical rank 256. Stable spectra with modest drift are real but unsurprising and do not establish functional relevance or E-mediated recovery. **NO-GO as a standalone paper claim.**

**Generic gradient alignment predicts ticket winners:** falsified in the existing exactly reproduced 8E/12E diagnostic cohort: native/forced gradient, function and update winners match actual best-gap routers in only 2/6 architecture/seed groups, trajectory distance 1/6. Three reused seeds, not six independent replications. This is useful negative evidence, not proof geometry is irrelevant.

**Architecture-dependent optimum / strict lottery tickets:** lower-sparsity Dataset A has 212/1100 strict individual matches but zero all-five settings among 220; no adjusted age-pair contrast establishes a universal optimum. Learned-mask advantages over random controls are real within their specific grid but are broadly anticipated by pruning literature. Older root reports cannot override finalized audits or missing raw artifacts. No architecture scaling law or novel LTH theorem is established.

## Why not GO or NO-GO overall?

GO would require a new mechanistic claim already isolated and replicated, which is absent. A blanket NO-GO would ignore a large, fully reproducible controlled effect with a small discriminating follow-up. **CONDITIONAL GO**, restricted to resolving that confound; no broad additional compute is justified yet.

## Seed-level variability of the selected effect

All contrasts below use the same common donor readout. Increment is full-gate minus assignment-only age contrast; a negative increment shifts toward the target. SD is descriptive spread over three seeds, not an expert/token-level uncertainty estimate.

| Experts | Control age | Assignment contrast mean | Full-gate contrast mean | Increment mean ± seed SD | Mean attenuation |
|---|---:|---:|---:|---:|---:|
| 8 | R0 | -0.065494 | -0.003862 | +0.061632 ± 0.118098 | 94.69% |
| 8 | R100 | -0.318137 | -0.001051 | +0.317086 ± 0.198151 | 98.67% |
| 12 | R20 | +0.930014 | -0.000281 | -0.930294 ± 0.508118 | 99.81% |
| 12 | R100 | -0.421066 | +0.001418 | +0.422484 ± 0.512759 | 99.54% |
