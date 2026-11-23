# IMP connection placement pilot

Run from repository root. All source experiments are read-only. No discovery or full-matrix launch exists in this workflow.

```bash
.venv-highres/bin/python scripts/imp_mask_information/audit.py
.venv-highres/bin/python scripts/imp_mask_information/review_audit.py
.venv/bin/python -m pytest tests/test_mask_information.py tests/test_imp.py tests/test_pruning.py -q
.venv-highres/bin/python scripts/imp_mask_information/pilot.py
.venv-highres/bin/python scripts/imp_mask_information/pilot.py --execute
.venv/bin/python scripts/imp_mask_information/analyze.py
```

The default pilot command only prepares and validates masks and writes a bounded plan/cost estimate. Execute reproduces IMP first, then runs five each of global, expert and matrix null masks, with an exclusive lock. Failed reproduction halts all null recoveries. Existing committed stages are identity-checked and reused. Exact stage resumption uses the canonical interruption snapshots, subject to safe-loader support; unsafe deserialization is never enabled. Preserve incomplete output if safe resume fails and diagnose before proceeding.

`TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` enforces safe deserialization inside legacy canonical checkpoint utilities. No source files or original results are modified. `pilot_results.csv` contains completed recoveries only. `summary.md` and five figure types are refreshed after each completed comparison. One-checkpoint observations do not support independent-seed confidence intervals. Full expansion and degree-preserving follow-up require a later review.
