# ChessAI Clean-Start Audit

Audit date: 2026-10-03  
Scope: every tracked source file. No checkpoint, replay buffer, self-play output,
or training history was opened, loaded, or changed during this audit.

## Result

The repository contains useful foundations, but it is **not safe to start an
AlphaZero training run**. The active RL entry point is hard-coded to continue
RL56, no candidate-vs-best gate exists, training is single-GPU, replay
provenance is insufficient, and there is no C++ implementation.

The migration will preserve the configured network baseline (18x8x8 input,
128 channels, 8 residual blocks, 4544 policy outputs) and reference
implementations, while creating a separate clean-run namespace. Legacy artifacts
will be rejected rather than imported.

## Classifications

| Component | Class | Finding |
|---|:---:|---|
| `model/chess_net.py` | A | Correct configurable baseline: 18 planes, 128 channels, 8 residual blocks, 4544 logits, scalar tanh value. |
| `environment/state_encoder.py` | B | Correct absolute-color 18-plane encoding, but undocumented and untested for round-trip/reference parity. |
| `environment/action_encoder.py` | B | Exact 4544 ordering (4032 ordinary + 512 promotions), but lacks bounds checks and randomized inverse tests. |
| `environment/chess_env.py` | B | Useful python-chess reference wrapper. It should remain reference-only and make draw-claim semantics explicit. |
| `environment/gpu_chess.py` | B | Substantial tensorized Python/PyTorch chess engine with castling, EP, legal filtering, clocks, and a legal-EP repetition hash. It is not C++, has only focused tests, lacks broad differential testing, and needs full dead-position validation. |
| `mcts/node.py`, `mcts/mcts.py`, `mcts/policy.py` | B | Useful slow reference MCTS. Perspective-aware PUCT/backup structure is sound; retain as an oracle, not the hot path. |
| `mcts/gpu_mcts.py` | B | Separates temporary virtual statistics from real statistics and batches leaf evaluation. Selection still has Python work per reserved leaf, has no memory-budget guard, and has no broad reference-comparison suite. |
| `training/self_play.py` | B | GPU-local worker sharding is a good self-play shape and GPU path excludes truncations. It hard-codes late temperature 0.10, lacks game/position provenance, and stores worst-case policy tensors on GPU. |
| `training/self_play_legacy.py` | B | Keep as a python-chess reference fallback, not production self-play. |
| `training/replay_buffer.py` | C | Mutable 3/4-tuple deque accepts legacy/unknown data; it cannot enforce a clean run ID or trace a sample to game and position. |
| `training/loss.py`, `trainer.py`, `train_step.py` | B | AlphaZero policy cross-entropy + value MSE and AMP are a correct basis. Missing DDP, scheduler/scaler state, legal-target assertions, and explicit training accounting. |
| `training/checkpoint.py` | C | Only persists model, optimizer, and iteration. Missing role, run provenance, config, RNG, scheduler/scaler, retention, and legacy protection. |
| `training/iteration.py`, `train_loop.py` | C | Broken loss-return contract and no candidate-vs-best gate. |
| `training/rl_training.py` | D | Hard-coded RL57 predecessor checkpoint/replay paths, old artifact loading, no gating, no DDP. Must not be used. |
| `main.py` | D | Automates an RL46--RL55 chain and copies/deletes old artifacts. Must not be used. |
| `training/evaluate.py` | C | Small evaluator has no reproducible gate configuration or detailed accounting. |
| `evaluation/evaluate_models.py` | B | Explicit A/B identity checks, color balance, and truncated-game exclusion are useful. It is tied to RL55/RL56 globals and needs migration into clean config/checkpoints/statistical gating. |
| `training/pretrain.py`, `pgn_dataset.py` | B | Optional supervised utilities; keep separate from fresh self-play RL. |
| Other training utilities | B/D | Dataset formats conflict (dense policy versus action ID); old replay analyzer executes at import and assumes legacy paths, so it is obsolete. |
| `tools/gpu_smoke_test.py` | B | Useful single-GPU MCTS smoke test only; not a DDP/two-GPU test. |
| `test_mcts_distribution.py` | D | Kaggle-only diagnostic executes at import and breaks normal pytest collection. |
| Current `tests/` | C | Narrow coverage; no broad chess differential, state/action contract, provenance, DDP, or promotion-gate suite. One evaluation test contradicts the configured default seed. |
| `lichess_bot/` | D | Deployment-only, tied to RL23/RL30 artifacts and slow reference MCTS; exclude from the clean training path. |
| C++ engine | C | No C++ source, build configuration, pybind extension, or C++ MCTS exists. |

Legend: A reusable; B correct foundation needing modification/validation; C must
be rewritten/replaced for the target; D obsolete on the active clean-run path.

## 18-plane contract

The canonical state tensor is `float32[18, 8, 8]`. Row zero is rank 8 and
column zero file a. Piece colors are absolute, not rotated to player
perspective:

| Planes | Meaning |
|---|---|
| 0--5 | White pawn, knight, bishop, rook, queen, king |
| 6--11 | Black pawn, knight, bishop, rook, queen, king |
| 12 | All ones iff White is to move |
| 13--16 | White king-side, White queen-side, Black king-side, Black queen-side castling rights |
| 17 | One at the en-passant target; otherwise zero |

Repetition history is terminal-state information, not one of the 18 network
planes. The GPU engine's hash deliberately ignores EP targets when no legal EP
capture exists, which is the correct FIDE-position-identity behavior.

## Baseline verification evidence

- `python -m pytest -q` fails at collection: `test_mcts_distribution.py`
  executes a Kaggle-only script against `/kaggle/working/chess-zero`, and the
  system Python has no PyTorch.
- The checked-in virtual environment has CPU-only PyTorch 2.13.0 and
  python-chess 1.11.2, but no pytest and no CUDA device. GPU and two-GPU gates
  cannot truthfully pass on this Windows workstation.
- Repository status was clean before this audit report was added.

## Migration requirements

1. Create a clean run root with a unique run ID and reject legacy artifact
   names/paths.
2. Use one typed configuration object for model, search, self-play, training,
   evaluation, reproducibility, paths, and retention.
3. Keep python-chess/reference MCTS for differential tests; do not call the
   current PyTorch engine a C++ implementation.
4. Separate `best_model.pt`, `candidate_model.pt`, and
   `iteration_N.pt`; promote only through reproducible, color-balanced
   evaluation with a configurable statistical criterion.
5. Add DDP training and GPU-local self-play worker plans independently.
6. Block train commands behind clean-provenance and correctness preflight gates.
