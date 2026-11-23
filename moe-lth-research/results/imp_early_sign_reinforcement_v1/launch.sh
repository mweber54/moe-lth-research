#!/usr/bin/env bash
set -euo pipefail
study=results/imp_early_sign_reinforcement_v1
matcher_pid="$1"
while [[ ! -s "$study/matching_seal.json" ]]; do
    if ! kill -0 "$matcher_pid" 2>/dev/null; then
        echo "Matching exited before producing a seal. Recovery dispatch stopped."
        exit 1
    fi
    sleep 8
done
.venv-highres/bin/python -Bu "$study/matching_audit.py"

project_dir="$PWD"
remote=adder.cs.kent.edu
if ! ssh -i ~/.ssh/dataset_b_cluster -o BatchMode=yes -o ConnectTimeout=15 "$remote" \
    "cd '$project_dir' || exit; nohup setsid .venv-highres/bin/python -Bu '$study/worker.py' --shard 1 --shards 2 >> '$study/adder_worker.log' 2>&1 < /dev/null &"; then
    exec .venv-highres/bin/python -Bu "$study/worker.py"
fi
exec .venv-highres/bin/python -Bu "$study/worker.py" --shard 0 --shards 2
