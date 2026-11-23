#!/usr/bin/env bash
set -euo pipefail
study=results/imp_early_sign_reinforcement_neuron_matched_v2
cd "$(dirname "$(realpath "$0")")/../.."
test -s "$study/matching_seal.json"
test ! -e "$study/stop.request"
remote=adder.cs.kent.edu
ssh -i ~/.ssh/dataset_b_cluster -o BatchMode=yes -o ConnectTimeout=15 "$remote"     "cd '$PWD' || exit; nohup setsid .venv-highres/bin/python -Bu '$study/worker.py' --shard 1 >> '$study/adder_worker.log' 2>&1 < /dev/null &"
exec .venv-highres/bin/python -Bu "$study/worker.py" --shard 0
