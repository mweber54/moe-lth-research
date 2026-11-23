# Gradient sensitivity matching audit v1

This is an array-only feasibility audit. It never constructs or runs a model, recomputes gradients, applies a pruning mask to a model, or measures loss. The original experiment remains stopped and unchanged.

`prepare.py` freezes the three designs, independent random40-tensor subset and solver limits. `ranges.cpp` counts eligible edges and coordinate support exactly without enumerating the billions of pre-row pairs. `validate.py` checks it against brute force and checks known-optimum matching fixtures. `descriptive.py` reproduces the original matcher and reliability eligibility, builds complete admissible same-row graphs, and evaluates A and the bounded C LP/floor generator across1280 tensors. Two descriptive workers per host are used on Viper/Adder. All arrays derive from the original sealed scores and steps250/500/1000.

`solve_queue.py` runs B/C binary optimization on all40 selected tensors, at most two subprocesses at once. Each has a30s HiGHS limit,45s parent wall limit,8GiB address-space limit and20000-node ceiling. No unsuccessful case is retried with more resources. Graph restrictions, missing incumbents, solver bounds and timeouts remain explicit. Encoded maximum-cardinality bounds do not certify SMD/KS feasibility; every returned integer candidate is postchecked.

`report.py`, `diagnosis.py` and `final_audit.py` report the results and verify preservation. `enrich.py` fills in row concentration from the same populations; it does not change matching decisions. `bounds_test.py` independently checks the complete-graph SCC coordinate bound by exhaustive enumeration on small graphs.

A pair removes one coordinate per branch; a graph edge count is not a pruning dose. H/L are pair-relative roles, and each coordinate has a single shared capacity across roles. Descriptive quartile populations are not additional matching restrictions. Local per-tensor inclusion does not satisfy whole-source coverage or dose. C's proposed column-TV tolerance0.05 is not demonstrated harmless structure imbalance. No pruning outcomes exist, so neither candidate feasibility nor reliable scores establish pruning utility.

Use the existing `.venv-highres/bin/python` environment and frozen output manifests. Do not rerun preparation into the completed output directory or launch an intervention from this audit.
