# Research investigation v1

Read `research_verdict.md` first. Mandatory deliverables are `research_audit.md`, `candidate_findings.md`, `novelty_review.md`, `decisive_experiment.md`, and `figures/`. `embedding_geometry_findings.md` records the independent falsification of the proposed embedding lead.

Reproduce numerical measurements and figures from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-highres/bin/python results/research_investigation_v1/embedding_geometry.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-highres/bin/python results/research_investigation_v1/reproduce_candidates.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-highres/bin/python results/research_investigation_v1/synthesize.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-highres/bin/python results/research_investigation_v1/finalize_research.py
```

These scripts perform checkpoint/statistical analysis only; there are no optimizer steps or experimental model forwards/backwards. Original-dtype E files in `E_cache/` are content-addressed and rechecked. All statistical computations use CPU float64. Checkpoint file hashes reuse the previous verified audit, while E hashes are freshly checked. All writes stay in this directory. Human-authored audit/literature/experiment-design documents are retained alongside reproducible numerical tables and their input hashes.

`repository_inventory.csv` is metadata coverage, not a count of independently trained models. `report_triage.csv` identifies report versions and superseded claims. `candidate_reproduction_validation.json`, `geometry_validation.json`, `completion.json`, and `artifact_manifest.json` record checks and output identities. `consumed_input_sha256.json` records the scientific inputs; historical duplicates and backup copies never increase seed counts.

Every geometry/performance correlation is exploratory; the fixed-list output includes all investigated metrics and steps, with five seeds separately at each sparsity. It is not a validated predictor search. Principal-subspace tables include cutoff gaps, and byte IDs are numeric without invented semantic grouping. Finite Gaussian matrices are explicitly labeled null controls and never mixed into experimental sample sizes.

The proposed factorial training experiment is described only in `decisive_experiment.md`; it has not been implemented in production training code or launched.
