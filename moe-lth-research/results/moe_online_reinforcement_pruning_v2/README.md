# Online reinforcement pruning v2

`protocol.md` and `registration.json` define the frozen design. V1 is unchanged; `input_manifest.json` records reused files and hashes. `score_validation.json` records independent sign and eligibility checks. Historical invalid V1 pilot exposure is disclosed in the registration.

Monitor matching with `tail -n 2 results/moe_online_reinforcement_pruning_v2/audit_match_*.log` from the project root. Monitor the gate-controlled supervisor with `tail -f results/moe_online_reinforcement_pruning_v2/supervisor_retry1.log`. Check production branches with `python3 results/moe_online_reinforcement_pruning_v2/status.py`.

`audit_match.py` independently solves the registered five-bin flow LPs and reproduces the random draws. `verify_selections.py` independently checks the saved indices and the archived separate five-bin MILP results before creating the six-cell seal. `engineering.py` reruns all six untouched next-step oracles and separate-process restart checks, with fresh GPU admission, before creating an engineering gate tied to the code and seal hashes. The supervisor then runs the pilot, checks its endpoints and comparability, runs the remaining cells, and analyzes verified endpoints. A failed stage stops the supervisor; see its log and the relevant attempt log. Checkpoint and completion files, rather than process startup, establish results.

The continuation implementation is provisional until its engineering gate passes. Do not interpret preliminary audit logs or a process ID as a completed branch.
