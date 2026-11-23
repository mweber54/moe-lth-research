# Router–expert geometry experiment

Run from repository root. Outputs are exclusively results/router_expert_geometry_v1.
Historical files are imported read-only; no shared code changed. Registration is
initial_manifest.csv + registration.json. Current status is run_manifest.csv.

Preparation (already completed; refuses to overwrite registration):

    OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry/prepare.py
    .venv-highres/bin/python scripts/router_expert_geometry/validate.py

Pre-launch analysis, one process per host, partition seeds 7/29/61 and 17/43:

    OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry/analyze_existing.py --seed 7
    .venv-highres/bin/python scripts/router_expert_geometry/summarize.py

GPU checks (fresh attempt paths required, failed/partial attempts preserved):

    .venv-highres/bin/python scripts/router_expert_geometry/worker.py --condition G_e32_seed7_R0_learned_s0.50 --attempt results/router_expert_geometry_v1/validation/enabled_v10 --short 25 --audit-preservation
    .venv-highres/bin/python scripts/router_expert_geometry/worker.py --condition G_e32_seed7_R0_learned_s0.50 --attempt results/router_expert_geometry_v1/validation/disabled_v1 --short 25 --disabled
    .venv-highres/bin/python scripts/router_expert_geometry/equivalence.py

Smoke-only comparator is the existing finalized dense endpoint; production sparse
runs depend on the newly measured matched dense recovery. No smoke state is reused
for production. The short run updates its batch hash to the exact 25-batch prefix.

Production launch requires launch_seal.json, passing GPU checks and completed initial
analysis. It holds experiment.lock and launches one worker per host via persistent
coordinator screen. Never launch a second coordinator or reset RUNNING/FAILED states
without verifying remote processes. No automatic retries or intermediate resumes.

    screen -dmS router-expert-geometry bash -c 'exec .venv-highres/bin/python -u scripts/router_expert_geometry/cluster.py >> results/router_expert_geometry_v1/coordinator.log 2>&1'

Monitor (both hosts share the output tree):

    cat results/router_expert_geometry_v1/queue_status.json
    tail -f results/router_expert_geometry_v1/coordinator.log
    tail -f results/router_expert_geometry_v1/logs/G_e32_seed7_R0_dense.log
    ssh -i ~/.ssh/dataset_b_cluster -o UserKnownHostsFile=/tmp/geometry_known_hosts viper.cs.kent.edu 'ps -u "$USER" -o pid,etime,args | grep router_expert_geometry'
    ssh -i ~/.ssh/dataset_b_cluster -o UserKnownHostsFile=/tmp/geometry_known_hosts adder.cs.kent.edu 'ps -u "$USER" -o pid,etime,args | grep router_expert_geometry'

worker heartbeat.json records actual remote PID and step; queue PID is the local SSH
supervisor. All failures preserve artifacts. Diagnostic model states are not
resumable checkpoints. Dataset B is not restarted by any script.
