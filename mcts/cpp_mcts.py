"""C++ MCTS engine adapter: duck-types the GPUMCTS interface used by self-play.

The search core runs in the az_cpp_mcts extension (no Python work per node);
neural-network evaluation is batched through a callback that mirrors
``GPUMCTS._evaluate`` (channels_last + CUDA fp16 autocast).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from environment.gpu_chess import GPUChess

try:
    import az_cpp_mcts
except ImportError as exc:  # pragma: no cover - exercised when not built
    az_cpp_mcts = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class CppMctsAdapter:
    """Duck-types the subset of GPUMCTS used by training.self_play."""

    def __init__(
        self,
        model,
        action_encoder=None,
        device: Optional[torch.device | str] = None,
        c_puct: float = 1.5,
        seed: int = 0,
    ):
        if az_cpp_mcts is None:
            raise ImportError(
                f"az_cpp_mcts extension is not available: {_IMPORT_ERROR}"
            )
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.action_encoder = action_encoder
        if action_encoder is not None and action_encoder.size() != 4544:
            raise ValueError("CppMctsAdapter requires the 4544-action space.")
        self.c_puct = float(c_puct)
        self.mcts = az_cpp_mcts.BatchMcts()
        self.mcts.set_c_puct(self.c_puct)
        self.mcts.set_seed(int(seed))
        self.num_games = 0

    # ------------------------------------------------------------------
    # Neural-network evaluation callback
    # ------------------------------------------------------------------
    def _evaluate(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """planes: float32 [n, 18*8*8] -> (logits [n, 4544], values [n])."""
        n = planes.shape[0]
        x = (
            torch.from_numpy(planes)
            .view(n, 18, 8, 8)
            .to(self.device, dtype=torch.float32)
        )
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, values = self.model(x)
            else:
                logits, values = self.model(x)
        logits_np = logits.float().reshape(n, 4544).cpu().numpy()
        values_np = values.float().reshape(n).cpu().numpy()
        return logits_np, values_np

    # ------------------------------------------------------------------
    # GPUMCTS-compatible API
    # ------------------------------------------------------------------
    def search(
        self,
        root_states: GPUChess,
        num_simulations: int,
        dirichlet_alpha: float | None = None,
        dirichlet_epsilon: float = 0.25,
        batch_size: int = 16,
        repetition_history: Optional[torch.Tensor] = None,
    ) -> None:
        if root_states.pieces.device != self.device:
            raise ValueError("root_states must live on the adapter device")
        games = int(root_states.pieces.shape[0])
        if games <= 0:
            self.num_games = 0
            return
        if num_simulations < 0:
            raise ValueError("num_simulations must be non-negative")

        def _cpu(t: torch.Tensor, dtype) -> np.ndarray:
            return np.ascontiguousarray(
                t.detach().to("cpu").to(dtype).numpy()
            )

        pieces = _cpu(root_states.pieces, torch.int64)
        turn = _cpu(root_states.turn, torch.int8)
        castling = _cpu(root_states.castling, torch.int16)
        ep_square = _cpu(root_states.ep_square, torch.int16)
        halfmove_clock = _cpu(root_states.halfmove_clock, torch.int16)
        fullmove_number = _cpu(root_states.fullmove_number, torch.int16)

        if repetition_history is None:
            history = np.zeros((games, 0), dtype=np.int64)
        else:
            if (
                repetition_history.ndim != 2
                or int(repetition_history.shape[0]) != games
            ):
                raise ValueError(
                    "repetition_history must have shape [games, positions]."
                )
            history = _cpu(repetition_history, torch.int64)

        alpha = -1.0 if dirichlet_alpha is None else float(dirichlet_alpha)
        self.mcts.search(
            pieces,
            turn,
            castling,
            ep_square,
            halfmove_clock,
            fullmove_number,
            history,
            int(num_simulations),
            alpha,
            float(dirichlet_epsilon),
            int(batch_size),
            self._evaluate,
        )
        self.num_games = games

    def root_visit_policy(self) -> torch.Tensor:
        return torch.from_numpy(self.mcts.root_visit_policy()).to(self.device)

    def select_actions(self, temperature: float = 1.0) -> torch.Tensor:
        actions = self.mcts.select_actions(float(temperature))
        return torch.from_numpy(actions).to(
            device=self.device, dtype=torch.int64
        )

    def advance(self, actions: torch.Tensor) -> GPUChess:
        if isinstance(actions, torch.Tensor):
            act = np.ascontiguousarray(
                actions.detach().to("cpu").numpy().astype(np.int64)
            )
        else:
            act = np.ascontiguousarray(np.asarray(actions, dtype=np.int64))
        pieces, turn, castling, ep, half, full = self.mcts.advance(act)
        games = int(pieces.shape[0])
        states = GPUChess(self.device, games)
        states.pieces = torch.from_numpy(pieces).to(self.device)
        states.turn = torch.from_numpy(turn.astype(bool)).to(self.device)
        states.castling = torch.from_numpy(castling).to(self.device)
        states.ep_square = torch.from_numpy(ep).to(self.device)
        states.halfmove_clock = torch.from_numpy(half).to(self.device)
        states.fullmove_number = torch.from_numpy(full).to(self.device)
        return states
