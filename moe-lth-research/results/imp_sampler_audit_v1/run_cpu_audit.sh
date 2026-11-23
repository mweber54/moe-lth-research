#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
mkdir -p preserved
cp -p ../imp_early_sign_reinforcement_neuron_matched_v2/rectangle_sampler.cpp preserved/rectangle_sampler.cpp
g++ -O3 -std=c++17 -shared -fPIC preserved/rectangle_sampler.cpp -o rectangle_sampler_original_rebuilt.so
g++ -O3 -std=c++17 -shared -fPIC rectangle_sampler_validated.cpp -o rectangle_sampler_validated.so
../../.venv/bin/python audit.py
