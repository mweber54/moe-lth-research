# IMP capacity diagnostic

This workflow evaluates the registered step-1000 and step-1500 sparse and
unpruned shared-only checkpoints. It does not train, update an optimizer, run
mask discovery, or create recovery branches. The 25/100/250 checkpoints are
used only when exact saved model snapshots are found; missing snapshots are not
reconstructed.

Run the artifact/provenance audit without GPU use:

```sh
python3 scripts/imp_capacity_diagnostic_v1/run.py --audit-only
```

Run the checkpoint-only evaluation after GPU admission is available:

```sh
python3 -m pytest tests/test_model.py -q
python3 scripts/imp_capacity_diagnostic_v1/run.py --evaluate
```

The evaluator uses the source worker's exclusive-claim and GPU admission
helpers. It refuses to evaluate above 90% GPU utilization, below 24 GiB free
GPU memory, or below 200 GiB free output storage. It profiles one complete
canonical batch before the cohort and writes an atomic cache per unique
checkpoint-state/validation identity. Re-run the same command after an
interruption to reuse completed cache records.

The audit verifies every file hash in both registered IMP-transition input
manifests, ties each sparse step-1000 snapshot to its mask and reconstructed
common parent, and records historical endpoint candidates without treating
them as additional readout states. Sparse model tensors are compared by exact
numeric value; masked `-0.0` versus `+0.0` encodings are reported separately
because they can differ in raw state hashes without changing model weights.

Layer-local `NO_DROP_Lk` policies disable capacity only at layer `k`; rejected
assignments remain valid in native-capacity layers. The policy-to-layer map is
shared by execution and routing validation, which checks assignment counts at
every layer and requires complete acceptance only in capacity-disabled layers.
The correction attempt and cache-revalidation evidence are retained under
`results/imp_capacity_diagnostic_v1/attempts/`; historical registration hashes
are not changed.

Primary shareable outputs:

```sh
cat results/imp_capacity_diagnostic_v1/summary_bundle/absolute_losses.csv
cat results/imp_capacity_diagnostic_v1/summary_bundle/seed_level_capacity_contrasts.csv
cat results/imp_capacity_diagnostic_v1/summary_bundle/findings.md
```
