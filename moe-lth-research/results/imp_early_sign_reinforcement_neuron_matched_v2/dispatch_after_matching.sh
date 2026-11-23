#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../.."
study=results/imp_early_sign_reinforcement_neuron_matched_v2
pid=$(cat "$study/rectangle_match.pid")
expected=".venv-highres/bin/python -Bu $study/rectangle_match.py"
while [[ ! -s "$study/matching_seal.json" ]]; do
    command_line=$(ps -p "$pid" -o args= || true)
    if [[ "$command_line" != "$expected" ]]; then
        test -s "$study/matching_seal.json" && break
        echo "Matching exited without a seal. No recovery workers launched."
        exit 1
    fi
    sleep 10
done
exec bash "$study/launch.sh"
