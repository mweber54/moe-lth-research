# Bounded E32 adaptation factorial

Read the frozen results protocol, manifest, input hashes, audit_review.md, reuse decisions and validation before execution. The 60 slots share five dense MoE controls across .80/.85; 15 compatible full trajectories are reused, with 45 new 5000-attempt restricted recoveries. Only E/R/S trainability changes.

```bash
.venv-highres/bin/python scripts/imp_adaptation_factorial_v1/af_cluster.py --seal
.venv-highres/bin/python scripts/imp_adaptation_factorial_v1/af_cluster.py --launch
.venv-highres/bin/python scripts/imp_adaptation_factorial_v1/af_report.py
cat results/imp_adaptation_factorial_v1/completion.json
cat results/imp_adaptation_factorial_v1/workers/*.heartbeat.json
tail -n 20 results/imp_adaptation_factorial_v1/logs/<job>.log
```

Seal/launch refuse duplicates. One detached worker per Viper/Adder uses shared exclusive claims, memory admission, fresh attempt_0001 directories and 15-second heartbeats. No unrelated jobs are killed. A technical failure writes STOP.json and preserves all attempts; there are no automatic retries or performance-based stopping/extension.

Raw primary validation curves: attempts/<job>/attempt_0001/curves.json and combined ce_curves.csv. Raw training curves: training_curve.csv plus the original two native gradient_stats logs listed in each result. Per-attempt CE is mean over all 16384 batch target tokens; auxiliary is the native layer mean and its .1-weighted contribution, with combined loss/applied-update/scale fields. Diagnostics live under diagnostics/step_*.json with exact coordinate denominators and native-input routing/assignment semantics. Clipping is exact native for new runs; historical reconstructed coefficients are labelled.

New resumable states are segment_0/interruption.pt and segment_2500/interruption.pt, atomically saved every100 attempts and retained at segment ends. They contain the model, AdamW/scaler, all RNG families, materialized-data position, identity and log sizes. NumPy RNG arrays are encoded as safe tensors; af_native.decode_numpy restores the native state without unsafe loading. Resume through the same af_engine.run inputs only after checking that the old host/PID is dead and selecting the preserved attempt explicitly; the normal dispatcher never overwrites or resumes uncertain attempts automatically. Exact CPU and CUDA/FP16 resumes are tested.

findings.md is live until completion.json.scientifically_complete is true. Endpoint tables, every-seed G/D/decomposition/interaction and descriptive summaries are separate. Primary comparison uses original full validation CE at 5000; diagnostic FP32 probe loss is not substituted. Recovery to the original level uses the fixed original full-dense 5000 CE even at intermediate steps. Two sparsities share seeds and dense controls; n=5 per summary, not n=10. No further experiment is authorized by this workflow.
