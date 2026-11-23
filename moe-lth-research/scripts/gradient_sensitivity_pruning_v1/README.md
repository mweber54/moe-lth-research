# Gradient sensitivity pruning v1

This isolated study uses the five original 32-expert source models at step1000. Existing experiment scripts and artifacts are read-only dependencies. The authoritative prospective definitions are in `results/gradient_sensitivity_pruning_v1/registration.json`; the original seal, pre-scoring fixture amendment and numerical/transport engineering amendments are preserved separately.

The stages are `prepare.py` (past-only provenance and registration), `measure.py --seed SEED` (32 fixed complete training batches, streamed scores and traffic), `review.py` (five-source reliability gate), and `match.py` (registered row pairs and directed column cycles, no model forward). Calibration workers use isolated attempt directories, locks, heartbeats and atomic statuses. The matching transport amendment changes only process serialization, not matching calculations.

All five groups must pass the registered measurement and matching gates and be sealed before any intervention-loss diagnostics or production pilot. A failed gate stops the study. It does not authorize another matcher, changed thresholds, extra batches, altered doses, or extra experiments. A stop report keeps all planned conditions and marks pruning endpoints unavailable. Candidate coordinate files are diagnostic candidates, not applied intervention masks.

`audit_scores.py` verifies all retained coordinate statistics without new gradients or interventions. `report_matching_stop.py` writes the authorized matching-stop report and plots. `final_audit.py` checks the deliverable manifest and explicit unavailable results. `completion.json` distinguishes completion of the registered stopping path from completion of a production experiment.

Use the existing `.venv-highres/bin/python` environment with bounded CPU threads. Do not rerun preparation or measurement into completed/sealed output directories. No full pruning-policy implementation is included.
