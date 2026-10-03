#!/usr/bin/env bash
set -euo pipefail

# Run from the repository copied to /kaggle/working/ChessAI-cloud.
PROJECT_ROOT="${1:-/kaggle/working/ChessAI-cloud}"
CONFIG="${2:-$PROJECT_ROOT/configs/clean_az.json}"
cd "$PROJECT_ROOT"

python -m pip install --quiet -r requirements.txt pytest
python clean_az.py init --config "$CONFIG"
python -m pytest -q tests_clean
python clean_az.py preflight --config "$CONFIG" --require-two-gpus --require-cpp-engine

# This is intentionally the only command that starts iteration 1. It remains
# blocked until every preceding validation, including the compiled C++ engine,
# succeeds.
python clean_az.py iteration --config "$CONFIG"
