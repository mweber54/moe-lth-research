exec((__import__('pathlib').Path(__file__).parent/'finalize_audit.py').read_text())
# finalize_audit above contains only read-only calculations and new audit outputs.
verdict='24E VALID UNDER FIXED-COMPUTE INTERPRETATION ONLY'
v32='32E CLEARED UNDER CURRENT PROTOCOL'
l80=late[late.start_age==80].groupby('experts')[['absolute_improvement','relative_improvement']].mean()
d100=disp[(disp.router_age==100)&(disp.component=='expert')].groupby('experts')[['absolute_mean','relative_mean']].mean()
pair24=rr.loc[24].sparse_dense_accepted_ratio
ratio_text=', '.join(f'{e}E {source_ratio.loc[e]:.4f}' for e in (8,12,16,24))
confounds=[
 ('H1','Source expert undertraining','Unproven','Sampled accepted exposure falls with E and expert displacement differs, but 24E late validation improvement is comparable to other architectures. Complete source accepted history unavailable; no unique source developmental failure established.'),
 ('H2','Recovery undertraining','Budget dependence measured; inadequacy unproven','Complete accepted exposure falls with E, while active applied-step coverage remains high. All endpoints are fixed-budget recoverability measurements; no longer-budget counterfactual is available.'),
 ('H3','Router scaling issue','Architecture dependence present; no normalization/capacity bug found','E-normalized load loss has balanced baseline 1. Full-softmax gate amplitudes and expert competition change with E. Capacity drops measured separately.'),
 ('H4','Comparator effect','Descriptive possibility quantified','Matched sparse and dense final losses, within-seed covariance and ranges are reported in comparator_covariation.csv. Ticket gap is a difference, with no denominator. Covariation cannot by itself establish why the curve is flat.'),
 ('H5','Genuine seed-conditioned phenomenon','Supported as a fixed-budget observation','Raw endpoints reproduce all 27 gaps and published seed winners; no protocol error explains the observed flat mean curve. Mechanistic or budget-invariant interpretation is not established.'),
 ('H6','Implementation/configuration error','No 24E-specific error found','Checkpoint, masks, rewind, dense comparator and final hashes pass. Expert count and save/refinement grids intentionally differ. Current code and historical metadata do not establish bitwise hardware/runtime equivalence.'),
 ('H7','Logging/analysis error','Ticket calculation passes; exposure interpretation corrected','Published gaps and sample SD agree with raw final losses. Historical cumulative exposure is sampled proposed counts, not complete accepted training exposure.')]
save('confound_assessment.csv',pd.DataFrame(confounds,columns=['hypothesis','explanation','assessment','evidence']))
protocols=f'''# Rerun recommendation

{verdict}

No training rerun is required by the evidence for the existing fixed-global-token/attempt-budget estimand. No historical artifacts are invalidated. The audit does not establish exposure-matched or convergence-matched development, nor a budget-independent explanation of the flat 24E curve. Fewer accepted tokens alone do not meet the requested rerun threshold. Source and recovery remain separate limitations.

The actual manuscript (`paper.tex`, architecture appendix, near line 1660) describes fixed-budget comparisons. The localization suite protocol prescribes the historical 2500-step recovery and prohibits automatic scaling-law/causal conclusions. The 24E final report explicitly confines claims to its budget and treats developmental similarity as secondary. Those sources support retaining the fixed-budget interpretation. A stronger claim about equally developed experts would need a separate preregistered comparison across all four architectures.

Protocol A — fixed global token budget: keep source and recovery at 2500 attempts, 40,960,000 input-token occurrences each. This preserves the existing scientific question and results. Report accepted assignments, applied updates, full-softmax gate semantics, nonconvergence and native sparse/dense routing differences. It does not match wall-clock/FLOPs, expert exposure, or maturity; “fixed compute” here means the offered verdict category, with the precise estimand being fixed global tokens/attempts.

Protocol B — approximately accepted-exposure matched: define q_E = accepted assignments per layer-token, and match one physical expert's exposure. With equal batch size, S_E/S_8 = (E/8)*(q_8/q_E). Calibrate source and recovery separately; for recovery estimate q separately for sparse/dense and router age, then preregister a common paired stopping schedule rather than giving the two comparators outcome-selected budgets. `recovery_scaling_common_ages.csv` gives measured inverse exposure ratios from the common R0/R20/R100 grid as candidate multipliers, not validated future budgets. Source sampled acceptance suggests relative exposures {ratio_text}; its inverse ratios are provisional because 96% of source steps lack routing counts. Collect complete counters in any subsequently authorized new experiment before adopting exact matching. Active-step matching is a different target and may require little extension; it cannot simultaneously ensure accepted-token matching. Longer training can change q, so a single linear multiplier is only a starting estimate, never an automatic E/8 rule.

Protocol C — convergence matched: before running outcomes, preregister a held-out validation slope threshold and patience, with expert-displacement and router-statistic stabilization as secondary diagnostics and a maximum budget. Apply the identical criterion and evaluation schedule to every architecture; use a validation set separate from final test reporting. Define source ages relative to that preregistered source endpoint and apply a separately preregistered paired recovery stopping rule. Advantage: asks a maturity-oriented question. Disadvantages: unequal tokens/compute, criterion dependence, possible overfitting, and global loss plateau does not prove every expert is mature. The present curves do not justify a post-hoc cutoff selected to change ticket winners.

Blast radius if the scientific question changes: an exposure- or convergence-matched source protocol requires new source trajectories, new R100-derived masks and new matched dense AND sparse recoveries for each affected architecture/seed/age, including 8E/12E/16E comparators. A recovery-only budget change can retain source checkpoints/masks but requires both dense and sparse recoveries across the affected age grids, not only an inconvenient 24E winner. For 24E the existing grid is R0/R10/R20/R30/R40/R50/R60/R80/R100, seeds 7/17/29. Any expanded grid must be preregistered. Keep original fixed-budget results as a separate estimand rather than overwriting them.

Recommendation: retain Protocol A for the current descriptive architecture extension; use Protocol B or C only as a separately authorized developmental study. No new run was launched.

{v32}

This clearance applies strictly to the existing fixed-global-budget protocol and claims. It does not clear a claim of matched expert maturity or authorize this audit to start 32E.
'''
(O/'rerun_recommendation.md').write_text(protocols)
answers=[
 ('Were total training tokens held constant across expert counts?','Yes: 40,960,000 input-token occurrences per source trajectory (2500 × 128 × 128), across all seeds. Four layers generate 163,840,000 proposed top-1 assignments. Tokens include repeated data; they are not unique examples.'),
 ('Were optimizer steps held constant?','Attempted steps: yes, 2500. Applied source optimizer steps: NA because GradScaler skip history was not persisted. Recovery applied counts are measured separately; fp16 overflows make attempts and successful updates differ.'),
 ('Were accepted expert assignments per expert held approximately constant?','No in sampled source logs and complete recovery logs. Exact complete source totals are NA. Active steps must be considered separately from accepted tokens.'),
 ('What is actual relative per-expert source exposure for 8E/12E/16E/24E?',f'Full historical accepted ratios: NA. Best available sampled accepted ratios averaged across seeds: {ratio_text}. These are explicitly samples, not extrapolated cumulative totals.'),
 ('Does exposure approximately scale as 1/E?','Compare measured sampled source ratios with 1, 2/3, 1/2, 1/3 in source_relative_exposure.csv. Recovery common-age ratios are independently measured below. Proposed source sample means follow 1/E exactly; capacity acceptance need not.'),
 ('What exactly does the historical cumulative-exposure metric measure?','It sums proposed top-k assignment counts over full logged training batches, then averages over 4E physical experts. It logs step 1 and each multiple of 25. It is not first-row-only, accepted counts, optimizer updates, or checkpoint samples. The first-validation-batch caveat concerns a separate maturity probe.'),
 ('Why do 16E R100 ≈103424 and 24E R100 ≈68949 scale inversely?','101 × 16384 = 1,654,784 proposed assignments per layer in the logged batches. Divide by 16 → 103424 exactly; divide by 24 → 68949.333333. The four-layer factor cancels in the mean. This identity follows from logging and cannot demonstrate undertraining.'),
 ('Are those complete accepted exposure or sampled logging exposure?','Sampled logging exposure, before capacity. See sampled_source_exposure.csv for acceptance-corrected counts on exactly those batches.'),
 ('Were active expert update opportunities comparable?','Complete source history: HISTORICAL ACTIVE-STEP EXPOSURE UNAVAILABLE. Recovery per-expert active-attempt and active-applied-update counts are available in recovery_expert_exposure.csv and summarized below. Lower tokens per expert does not automatically imply proportionally fewer active updates.'),
 ('Were source models comparably converged by R100?','No architecture is shown to have converged. All improve materially near R100. Their late slopes are comparable; R100 is a normalized budget label, not evidence of maturity.'),
 ('Was 24E still materially improving near R100?',f'Yes: mean R80→R100 improvement {l80.loc[24,"absolute_improvement"]:.6f} ({100*l80.loc[24,"relative_improvement"]:.2f}%). This is comparable to 8E/12E/16E, not a 24E-specific lag.'),
 ('Did 24E experts move less from initialization?',f'No; 24E mean displacement is larger. Mean per-expert absolute displacement: 8E {d100.loc[8,"absolute_mean"]:.6f}, 24E {d100.loc[24,"absolute_mean"]:.6f}; relative displacement: {d100.loc[8,"relative_mean"]:.6f} versus {d100.loc[24,"relative_mean"]:.6f}. Full age/seed/component distributions are saved. Displacement is descriptive and does not alone diagnose insufficient training.'),
 ('Does capacity behavior change materially with expert count?','Measured sampled source drop rates and full recovery acceptance ratios are tabulated. The formula scales capacity inversely with E with ceil rounding. No 24E-specific capacity bug was found; observed drop differences modify the simple 1/E relationship.'),
 ('Does router auxiliary-loss scaling change with expert count?','Balanced baseline remains 1 per layer before weight 0.1. No accidental E normalization error found. Router dimensionality, full-softmax gate amplitudes and distribution-dependent gradients still change with E; see router_scaling_audit.md.'),
 ('Are there architecture-dependent implementation bugs?','No 24E-specific bug found in inspected paths and artifacts. Same-seed initialization is not cross-architecture identical because RNG consumption changes with E. This is expected, not an accidental seed mismatch.'),
 ('Are checkpoint ages correctly mapped?','Yes for all 27 used 24E seed-age combinations: payload step=25×age, correct seed/E, fresh file and component hashes match historical manifest. Additional saved 5%-grid checkpoints were also checked. No duplicated component hashes within a seed trajectory.'),
 ('Are masks correct?','Yes. All 27 runs match independently regenerated source-R100 expert-local magnitude masks. Actual weight sparsity is approximately 85%, with rounded per-expert keep counts; biases are excluded. One mask per seed is intentionally reused across router ages. R0 rewind hashes and final masked zeros pass.'),
 ('Are matched dense comparators correct?','Yes: all 27 24E sparse runs point to their same-seed/same-age dense recovery endpoint, with matched source router, data hashes, protocol and fresh-optimizer metadata. No universal R100 comparator was substituted.'),
 ('Were 24E ticket gaps recomputed correctly?','Yes: 27 raw metrics.jsonl final losses independently cross-check summaries; sparse minus dense reproduces published means/sample SD to numerical precision, without using the existing summary script.'),
 ('Is the flat 24E curve real according to raw artifacts?','Yes as a descriptive fixed-budget observation. Winners remain seed 7 R20, seed 17 R40, seed 29 R100; mean winner R50. Near ties and three seeds limit precision; “flat” is not a formal equivalence test.'),
 ('Is source training exposure a confound?','For equal-development claims, E and per-expert sampled exposure are confounded, and maturation equality is unestablished. For a fixed-global-token question, reduced expert exposure is part of the architectural change. No material 24E-specific developmental failure is demonstrated.'),
 ('Is recovery exposure a separate confound?','Yes for exposure-matched recoverability claims: full recovery counters directly show different accepted examples per expert. Fixed-step recoverability remains a valid different question. This does not by itself establish that 2500 attempts are inadequate.'),
 ('Are sparse and dense recovery exposures comparable within matched pairs?',f'They share input data/attempt schedules but native routing does not ensure equal acceptance. Across 24E pairs the sparse/dense total accepted ratio ranges {pair24.min():.6f}–{pair24.max():.6f}, mean {pair24.mean():.6f}. Per-expert allocations may differ more; all distributions are retained. Interpret the ticket gap as total native recovery performance, not an exposure-controlled contrast.'),
 ('Is 24E scientifically valid under fixed-global-compute interpretation?','Yes under the precise fixed-global-token/attempt-budget interpretation supported by the manuscript and localization protocol. FLOPs, wall time and exact successful source update counts are not proven matched.'),
 ('Is it valid for a developmental cross-architecture comparison?','Valid for descriptive router-development statistics under the specified budget; insufficient to claim equally developed experts or a budget-invariant developmental mechanism. Neither source nor recovery was accepted-exposure matched.'),
 ('Does 24E need to be rerun?',verdict+'. No mandatory rerun is supported for the current fixed-budget claims. Stronger developmental claims require a separate design, not a 24E-only repair.'),
 ('If yes, exactly what must be rerun?','No rerun mandated. If the estimand changes, rerun scope is specified in rerun_recommendation.md: source changes propagate to source/masks/both recoveries; recovery-only changes require both paired recoveries, across all affected architectures.'),
 ('Does the issue invalidate 8E/12E/16E comparisons?','No fixed-budget results are invalidated. The same unequal-exposure/nonconvergence limits apply to developmental comparisons across all four architectures. Original/expanded 12E model states match at all reused ages.'),
 ('Should 32E proceed under the current protocol?',v32+'. Only for the current explicitly fixed-global-budget descriptive question. No training was started.'),
 ('What protocol should be used going forward?','Protocol A for continuity with existing claims; separately preregister B (accepted exposure matched, calibrated separately for source/recovery) or C (convergence matched) for a stronger developmental question. Do not select a protocol or stopping rule to obtain a preferred ticket curve.')]
report='ARCHITECTURE SCALING AUDIT\n\n'+verdict+'\n\nAudit only: no training, recoveries or mechanism experiments launched; historical artifacts preserved.\n\n'
report+='\n\n'.join(f'{i}. **{q}**\n\n{a}' for i,(q,a) in enumerate(answers,1))
report+='\n\n## Primary source-exposure table\n\nNA is intentional: complete accepted source history is unavailable.\n\n'+table(primary)
report+='\n\n## Theoretical baseline (one physical expert in one layer)\n\n'+table(df('theoretical_expert_exposure.csv').drop_duplicates('experts')[['experts','total_training_tokens','top_k','potential_expert_assignments','ideal_assignments_per_expert','relative_to_8E']])
report+='\n\n## Measured source samples and capacity\n\n'+table(relative.drop(columns='measurement'))+'\n\n'+table(cap.groupby('experts')[['capacity_per_expert','capacity_fraction_batch_tokens','sampled_source_drop_rate']].mean().reset_index())
report+='\n\n## Full recovery exposure, common R0/R20/R100 grid\n\nEqual seed/age weighting. Accepted assignments include overflow forwards; active applied steps exclude skipped updates.\n\n'+table(rg)
report+='\n\n## Source late development\n\nR80→R100 source validation improvement by architecture/seed:\n\n'+table(late[late.start_age==80][['experts','seed','start_loss','end_loss','absolute_improvement','relative_improvement']])
report+='\n\nExpert R100 displacement (equal-seed mean of physical-expert means):\n\n'+table(d100.reset_index())
report+='\n\n## Independent 24E endpoints\n\n'+table(summary)+'\n\n'+table(df('24e_seed_contrasts.csv'))
report+='\n\nMean contrasts: R50−R0 = '+str(float(summary.set_index('router_age').loc[50,'mean_ticket_gap']-summary.set_index('router_age').loc[0,'mean_ticket_gap']))+'; R50−R100 = '+str(float(summary.set_index('router_age').loc[50,'mean_ticket_gap']-summary.set_index('router_age').loc[100,'mean_ticket_gap']))+'.\n\nSparse/dense covariance is descriptive, with ages not independent replicates:\n\n'+table(pd.DataFrame(h4))
report+='\n\n## Competing explanations\n\n'+table(pd.DataFrame(confounds,columns=['Hypothesis','Explanation','Assessment','Evidence']))
report+='\n\nBest-supported interpretation: (1) a real seed-conditioned fixed-budget observation, (2) native sparse/dense recovery and comparator behavior, (3) architecture-dependent accepted exposure and gate weighting as plausible contributors. Historical data do not causally rank those mechanisms. H1/H2 are not established diagnoses.\n\n## Evidence and reproducibility\n\nSee code_path_manifest.md for file/function provenance; architecture_config_diff.csv and architecture_parameter_counts.csv for full configuration/counts; historical_data_availability.md for coverage; router_scaling_audit.md for formulas; rerun_recommendation.md for the three candidate protocols and full blast radius. audit.py performs CPU checkpoint/log arithmetic; finalize_audit.py independently checks raw final-loss logs and historical manifests; write_report.py assembles this report. All CSV missing quantities use NA. No paper/result files outside this audit directory were edited.\n\n'+v32+'\n'
(O/'final_report.md').write_text(report)
terminal=f'''ARCHITECTURE SCALING AUDIT COMPLETE

Protocol integrity:
- checkpoint mapping: PASS; 27 used 24E pairs plus saved intermediate checkpoints
- pruning masks: PASS; recomputed from R100, intentionally shared across ages
- matched dense controls: PASS; all 27
- recovery initialization: PASS; reconstructed initial hashes and fresh-optimizer metadata
- ticket-gap recomputation: PASS; raw final losses and published sample SD agree

Source training:
- global tokens matched?: YES; 40,960,000 each
- optimizer steps matched?: 2500 attempts; applied source steps NA
- per-expert exposure matched?: NO in source samples; complete counts NA
- 8E relative exposure: {source_ratio.loc[8]:.6f} sampled accepted; complete NA
- 12E relative exposure: {source_ratio.loc[12]:.6f} sampled accepted; complete NA
- 16E relative exposure: {source_ratio.loc[16]:.6f} sampled accepted; complete NA
- 24E relative exposure: {source_ratio.loc[24]:.6f} sampled accepted; complete NA

Recovery:
- global recovery steps matched?: YES, 2500 attempts; applied updates separately logged
- per-expert recovery exposure matched?: NO; full accepted counts measured
- sparse/dense exposure matched within pairs?: input schedule yes; accepted distributions no

24E convergence:
- late-training behavior: still improving, comparably to 8E/12E/16E
- expert displacement: mean absolute {d100.loc[24,'absolute_mean']:.6f}, relative {d100.loc[24,'relative_mean']:.6f}; 8E {d100.loc[8,'absolute_mean']:.6f}, {d100.loc[8,'relative_mean']:.6f}
- evidence of underdevelopment?: unequal exposure observed; unique 24E inadequacy unproven

Architecture scaling:
- capacity issue?: no formula bug found; observed drop rates differ
- router/aux-loss scaling issue?: balanced normalization correct; full-softmax gate semantics depend on E
- implementation/config drift?: no 24E-specific error found

Best explanation for flat 24E curve:
1. Real seed-conditioned fixed-budget observation
2. Native recovery/comparator behavior; covariance reported
3. Exposure and gate-weighting differences are plausible, not causally established

24E verdict:
{verdict}

Does 24E need rerun?: No mandatory rerun for fixed-budget claims
If yes, exact rerun scope: Not applicable; alternative developmental study scope in recommendation

Does this affect historical 8E/12E/16E?: Same developmental caveats; fixed-budget results preserved

32E verdict:
{v32}

Recommended protocol going forward: A for existing fixed-budget claims; B/C only as a preregistered separate study

Path to final_report.md:
{O/'final_report.md'}
'''
(O/'terminal_summary.txt').write_text(terminal)
integ=json.loads((O/'artifact_integrity.json').read_text());integ['independent_checks']={'raw_loss_matches':54,'used_checkpoint_manifest_matches':27,'mask_run_checks':27,'dense_pairs_24e':27,'recovery_complete_counter_conditions':len(rec),'source_drop_reconstruction_max_error':float(cap.max_reconstruction_error.max()),'published_summary_matches':True,'12e_reused_states_equal':bool(df('12e_reference_provenance.csv').states_equal.all())};integ['outputs']={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ('artifact_integrity.json','finalize_execution.log')};(O/'artifact_integrity.json').write_text(json.dumps(integ,indent=2));print(terminal)
