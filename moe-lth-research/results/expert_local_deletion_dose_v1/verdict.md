# Concise verdict: expert-local deletion-dose check

The numerical pipeline works, and expert-function dependence is distinguishable.
Task-CE rankings across A/B are not yet consistently reliable at equal dose.
**Improve CE measurement precision and low-traffic coverage before expanding a
CE-based dispensability study.** Cross-seed replication of the functional dose
response is reasonable as a measurement audit; it would not resolve the current
within-seed A/B task-loss ranking problem by itself. No expansion was launched.

- **Measurable and reproducible?** All 192 representative repeat checks were
  bitwise identical. At equal doses of 1/2/4 groups, adequate-coverage within-expert
  A/B CE rank correlations have median 0.2
  for sparse and 0.4 for unpruned, with
  observed reversals (range −1 to 1). Functional relative-output-RMS ranks have
  median 1 in both trajectories (range 0.8–1). Eight-group interventions change
  expert-output RMS by 4.37%–10.77%
  of baseline where coverage exists. This establishes reproducible function changes,
  not equally reliable CE-based importance ordering.
- **Does whole-probe averaging conceal conditional effects?** Yes. Weight-union
  whole-probe damage ranges from -0.000120258 to 0.0030185 nats/token;
  baseline-accepted damage ranges from -0.0661777 to 1.28494.
  The separate output-zero control reaches 0.0360179 whole-probe nats/token.
  Sparse step 25 has zero accepted positions for experts 8/16/24 and only 17 on A,
  16 on B for expert 0. These cases remain explicit and do not demonstrate low
  expert dependence. Conditional subsets are fixed from baseline; they are not
  isolated expert losses and may have very small denominators.
- **Does dependence change over recovery?** Task sensitivity changes, without a
  universal monotone trend. Eight-group unpruned whole-probe damage is much larger
  at step 25 than later; sparse effects are small, signed, and more variable at
  step 2500. Fixed-coordinate temporal contrasts and traffic are stored per expert.
  Changes in task loss, context mix and accepted traffic prevent a specialization
  interpretation. Equal-dose function-output changes remain observable throughout
  covered cases.
- **What next?** Improve the task-CE probe precision/coverage first, using a separately
  declared measurement budget, without choosing a dose or threshold to force a trend.
  The method is ready for engineering replication, but these data do not justify
  stable CE-based group ranking, a specialization predictor or permanent pruning.

Eight-group whole-probe damage, means/ranges across the four experts and A/B:

| Trajectory | Step | Mean damage | Range |
|---|---:|---:|---:|
| sparse | 25 | -6.63144e-06 | -3.41907e-05 to 0 |
| sparse | 250 | 8.82594e-06 | -4.28974e-08 to 2.66255e-05 |
| sparse | 1000 | 7.88074e-06 | -1.0842e-05 to 2.59769e-05 |
| sparse | 2500 | 3.8243e-05 | -0.000120258 to 0.000108577 |
| unpruned | 25 | 0.00106538 | 0 to 0.0030185 |
| unpruned | 250 | 1.01682e-05 | -2.38776e-05 to 5.54427e-05 |
| unpruned | 1000 | 4.35411e-06 | -2.95843e-05 to 2.75016e-05 |
| unpruned | 2500 | -2.70513e-05 | -5.35028e-05 to 3.05999e-05 |

Zero dose, native baseline/cache equivalence, exact restoration, model hashes,
RNG, gradients, masks and native selection/gates passed. Four fixed orderings reduce
to 52 unique weight unions plus four separate output controls, with no global cap.
The 0.001 and 0.0001 reference lines were not used for selection. Both A/B have been
examined previously; this is one seed with nested groups, experts, orderings and
checkpoints, not independent replications or a generalization test.

Measurement/table serialization took 103.09 seconds
on NVIDIA H100 NVL, with 192
single-group CE measurements reused. New cache storage is
351.41 MB; all outputs are approximately
387.21 MB. See final_audit.json for
final measured storage and compute_accounting.json for forward counts and timing.
