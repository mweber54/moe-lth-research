#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
g++ -O3 -std=c++17 -shared -fPIC exact_rectangle.cpp -o exact_rectangle.so
CUDA_VISIBLE_DEVICES='' ../../.venv/bin/python run.py
