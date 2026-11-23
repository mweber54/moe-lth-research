# Exposure-stratified pruning v1

This is the requested pre-loss exposure audit and frozen feasibility gate for an exposure-stratified pruning intervention. The single registered selection pass covers all five sources. It must not be re-run over the existing directory or retuned to improve coverage.

Run sequence: `prepare.py`, then `verify.py`, then `report.py`, using `.venv-highres/bin/python`. The original artifacts are read-only dependencies. `design.json` and its initial seal precede matching; all candidate coordinates and score/source hashes are preserved in the new results directory.

The frozen all-five-seed gate failed. Therefore there is deliberately no production launcher: no GPU masking, baseline expert-function measurement, engineering continuation, intervention loss or training is authorized by this registration. Candidate files under `masks/` are diagnostic coordinate lists, explicitly unapproved in `coordinate_seal.json`. Per-condition unavailable results and trajectories remain NA, never zero. The desired continuation, diagnostics, numerical validation, pilot and cluster protocol is documented in `protocol.md`/`design.json` but was not executed. A different matching design would require a separately declared question and registration; do not edit thresholds or recycle this gate.
