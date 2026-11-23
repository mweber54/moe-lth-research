# Checkpoint-only temporal deletion prediction

This directory analyzes existing reliability CSVs. It never imports model code,
executes model forwards, deserializes checkpoint tensors or generates probes.
Only the original coordinate tensors are loaded, to verify identities and true
coordinate-union coverage. Checkpoint files are verified as byte streams.

Results: `results/expert_deletion_temporal_prediction_v1/`.

The protocol/configuration were frozen before temporal associations. The separate
`estimand_clarification.json`, also written before associations, explicitly fixes
whole discovery CE as the predictor for both whole and future-native accepted
outcomes. Original source files remain read-only.

For a fresh output directory, prepare with `prepare.py`; this refuses to overwrite
an existing freeze. Analysis, plotting and auditing can be reproduced with:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/expert_deletion_temporal_prediction_v1/analyze.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/expert_deletion_temporal_prediction_v1/report.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/expert_deletion_temporal_prediction_v1/verify.py
```

The audit invokes only `tests/test_expert_deletion_temporal_prediction.py`, which
uses synthetic arrays; older model-executing tests are deliberately not invoked.
Current-minus-history future damage is positive when history helps; selection
advantage is panel-uniform-minus-selected. Identical policy IDs yield exactly zero.
Future regret uses an optimistic hindsight minimum. Both fixed-ID confirmation
bootstrap intervals and discovery-reselected/independently confirmation-resampled
intervals are reported. None are independent model-seed inference.
