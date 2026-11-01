# ChessAI — Clean AlphaZero Chess Engine

A clean-room AlphaZero implementation for chess, trained from scratch via self-play on 2× NVIDIA Tesla T4 GPUs (Kaggle). No human games required — the network starts from random weights and learns entirely through self-play with MCTS guidance.

## Architecture

### Neural Network (`model/chess_net.py`)
- **Backbone:** 128-channel, 8-block residual CNN (2D convolution over the board)
- **Policy head:** 32-channel conv → 4544-dim action vector
- **Value head:** 32-channel conv → scalar win/draw/loss
- **Total parameters:** ~12.2 M

### Input Representation (`environment/state_encoder.py`)
18 input planes (8×8 each):
- Planes 0–11: piece positions (6 piece types × 2 colours)
- Plane 12: side to move
- Planes 13–16: castling rights (K, Q, k, q)
- Plane 17: en-passant square

### Action Space (`environment/action_encoder.py`)
4544 discrete actions:
- 4032 normal moves (64 from-squares × 63 to-squares)
- 256 white promotions (64 from × 4 pieces)
- 256 black promotions

### MCTS
- **Python GPU:** batched tree search on GPU (`mcts/gpu_mcts.py`)
- **C++ engine:** high-performance batched MCTS via pybind11 (`cpp/`, `mcts/cpp_mcts.py`)
- 400 simulations per move, c_puct = 1.5, Dirichlet noise (α=0.3, ε=0.25) at root

## Training Pipeline

### Iteration Cycle
1. **Self-play:** 256 games using current best model + MCTS (temperature 1.0 for first 30 moves, then 0)
2. **Replay buffer:** completed games stored (capacity 150,000 positions)
3. **Candidate training:** 1,200 steps on batch 256 (AdamW, lr 5e-4, mixed precision, gradient clipping)
4. **Evaluation:** 60 head-to-head games vs. current best
5. **Gating:** promote if Wilson lower bound ≥ 0.55 (≥ 32 completed games)
6. **Checkpoint:** save best/candidate/iteration checkpoints; prune to keep last 5

### Supervised Pretraining (optional)
Bootstrap the network on human games before self-play:
```bash
export PRETRAIN_PGN_GLOB="/kaggle/input/datasets/pearsejim01/pgn-mentor-dataset/**/*.pgn"
ITERATIONS=10 bash scripts/kaggle_clean_az.sh
```
- Parses PGNs, trains on (state, played-move, game-result) for 3 epochs
- Overwrites `best_model.pt` at iteration 0 with provenance-matched weights
- Guard skips if iteration > 0 (runs only once across sessions)

## Repository Layout

```
├── clean_az.py                  # CLI entry point (init/preflight/train-candidate/iteration/status/pretrain)
├── configs/clean_az.json        # All hyperparameters
├── az/                          # Core AlphaZero modules
│   ├── config.py                # Typed config (dataclasses)
│   ├── provenance.py            # Run manifest, artifact integrity
│   ├── checkpoints.py           # Role-based save/load with provenance
│   ├── replay.py                # CPU replay buffer (sparse policy storage)
│   ├── iteration.py             # Iteration lifecycle + DDP launch
│   ├── training.py              # DDP-aware candidate training
│   ├── selfplay.py              # Self-play generation
│   ├── evaluation.py            # Head-to-head evaluation
│   ├── gating.py                # Wilson LB promotion gate
│   ├── model.py                 # Fresh model construction
│   ├── preflight.py             # Artifact / GPU / engine validation
│   └── reproducibility.py       # Seed & CUDA determinism
├── model/chess_net.py           # ChessNet architecture
├── environment/
│   ├── state_encoder.py         # Board → 18×8×8 planes
│   ├── action_encoder.py        # Move → action ID
│   └── gpu_chess.py             # Batched GPU game rules
├── mcts/
│   ├── gpu_mcts.py              # Batched GPU MCTS
│   ├── cpp_mcts.py              # C++ engine Python adapter
│   ├── mcts.py, node.py, policy.py
├── cpp/                         # C++ MCTS engine (pybind11 + CMake)
├── training/
│   ├── loss.py                  # alpha_zero_loss (policy CE + value MSE)
│   ├── pretrain.py              # Supervised pretraining on PGNs
│   └── replay_buffer.py, self_play.py
├── evaluation/evaluate_models.py
├── tools/
│   ├── benchmark_selfplay.py    # Games/hour gate (≥50/hour @ 400 sims)
│   ├── cpp_selfplay_smoke.py    # CUDA smoke + parity check
│   └── upload_checkpoints.py    # Manual checkpoint export
├── scripts/
│   ├── kaggle_clean_az.sh       # Full Kaggle orchestration
│   ├── kaggle_persist.sh        # Whole run-dir save/restore via Kaggle Dataset
│   ├── kaggle_upload_checkpoints.sh
│   └── kaggle_creds.sh
├── tests/                       # Engine, MCTS, geometry, promotion tests
└── tests_clean/                 # Provenance, gating, config, checkpoint tests
```

## Quick Start

### Local (development)
```bash
pip install -r requirements.txt pytest pybind11
pip install cmake ninja
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build cpp/build --parallel
python -m pytest -q tests_clean tests
```

### Kaggle (full training)
```bash
# In a Kaggle GPU notebook (2× T4):
!git clone https://github.com/pundirarjun/ChessAI-cloud.git
%cd ChessAI-cloud
ITERATIONS=10 bash scripts/kaggle_clean_az.sh
```

### CLI Commands
```bash
python clean_az.py init --config configs/clean_az.json       # Create run namespace
python clean_az.py preflight --config configs/clean_az.json  # Validate environment
python clean_az.py pretrain --config configs/clean_az.json \
  --pgn-glob /path/to/pgns/**/*.pgn                          # Supervised pretrain
python clean_az.py iteration --config configs/clean_az.json  # Run one iteration
python clean_az.py status --config configs/clean_az.json     # Show run state
```

## Configuration

All hyperparameters live in `configs/clean_az.json`. Key settings:

| Parameter | Value | Description |
|---|---|---|
| `mcts.simulations` | 400 | MCTS simulations per move |
| `mcts.engine` | `cpp` | C++ batched engine |
| `self_play.num_games` | 256 | Self-play games per iteration |
| `self_play.temperature_moves` | 30 | Moves with sampling before greedy |
| `training.steps` | 1200 | Optimizer steps per iteration |
| `training.batch_size` | 256 | Training batch size |
| `evaluation.games` | 60 | Head-to-head eval games |
| `evaluation.promotion_score` | 0.55 | Wilson LB threshold for promotion |
| `storage.replay_capacity` | 150000 | Max replay buffer positions |
| `runtime.num_gpus` | 2 | DDP GPU count |

## Kaggle Persistence

Kaggle wipes `/kaggle/working` when a session stops. Two datasets persist state:

- **`arjunthakur9999/chessai-clean-az-run`** — `run_state.zip` containing the entire run dir (config, manifest, state, replay, checkpoints). Restored automatically at session start; saved once after all iterations complete.
- **`arjunthakur9999/checkpoints`** — `checkpoints.zip` with weights + provenance for direct download.

Restore verifies the stored config hash matches the current config, so a config change never resurrects an incompatible run.

## Tests

```bash
python -m pytest -q tests_clean tests   # 87 passed, 2 skipped (C++ ext not built on Windows)
```

- `tests/` — engine geometry, promotions, castling, insufficient material, GPU chess, MCTS, C++ adapter
- `tests_clean/` — config provenance, gating logic, replay/checkpoint integrity, evaluation schedule, network loss, reference contract

## Requirements

- Python 3.10+
- PyTorch 2.x (CUDA-enabled on Kaggle)
- python-chess ≥ 1.999
- CMake + Ninja + pybind11 (for C++ engine)
- Kaggle CLI (for persistence scripts)

## License

See repository for license details.
