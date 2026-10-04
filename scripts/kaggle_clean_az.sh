#!/usr/bin/env bash
set -euo pipefail

# Run from the repository copied to /kaggle/working/ChessAI-cloud.
PROJECT_ROOT="${1:-/kaggle/working/ChessAI-cloud}"
CONFIG="${2:-$PROJECT_ROOT/configs/clean_az.json}"
cd "$PROJECT_ROOT"

python -m pip install --quiet -r requirements.txt pytest pybind11

# ---------------------------------------------------------------
# 1. Build the C++ MCTS engine (az_cpp_mcts) before anything that
#    imports it.  Output lands in the repo root, which is on sys.path.
# ---------------------------------------------------------------
python -m pip install --quiet cmake ninja
cmake -S cpp -B cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build cpp/build --parallel
python -c "import az_cpp_mcts; print('az_cpp_mcts OK:', az_cpp_mcts.__file__)"

# ---------------------------------------------------------------
# 2. All unit/differential tests (clean gates + C++ engine tests).
# ---------------------------------------------------------------
python -m pytest -q tests_clean tests

# ---------------------------------------------------------------
# 3. Preflight: artifacts, GPU count, and the compiled engine.
# ---------------------------------------------------------------
python clean_az.py init --config "$CONFIG"
python clean_az.py preflight --config "$CONFIG" --require-two-gpus --require-cpp-engine

# ---------------------------------------------------------------
# 4. CUDA smoke: full self-play through the C++ engine, with a
#    reference-engine trajectory parity check.
# ---------------------------------------------------------------
python tools/cpp_selfplay_smoke.py --games 2 --sims 50 --parity

# ---------------------------------------------------------------
# 5. Performance gate: >= 50 games/hour at 400 sims on 2x T4.
#    Fails the run (non-zero exit) if below threshold.
# ---------------------------------------------------------------
python tools/benchmark_selfplay.py --engine cpp --sims 100,200,400,800 \
  --selfplay --selfplay-sims 400 --selfplay-games 8 \
  --require-games-per-hour 50 \
  --json-out "$PROJECT_ROOT/kaggle_cpp_benchmark.json"

# This is intentionally the only command that starts iteration 1. It remains
# blocked until every preceding validation, including the compiled C++ engine
# and the performance gate, succeeds.
python clean_az.py iteration --config "$CONFIG"
