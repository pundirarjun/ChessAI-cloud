"""Tensorized GPU Monte-Carlo Tree Search.

The tree, chess states, move generation, PUCT statistics, and neural-network
leaf evaluations all live on the selected torch device.  No python-chess board
or per-node Python object is used in the hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from environment.gpu_chess import GPUChess, MAX_LEGAL_MOVES


@dataclass
class GPUSearchResult:
    actions: torch.Tensor
    policy: torch.Tensor


class GPUMCTS:
    def __init__(
        self,
        model,
        action_encoder=None,
        device: Optional[torch.device | str] = None,
        c_puct: float = 1.5,
        max_children: int = MAX_LEGAL_MOVES,
    ):
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.action_encoder = action_encoder
        if self.action_encoder is not None and self.action_encoder.size() != 4544:
            raise ValueError("GPUMCTS requires the project's 4544-action space.")
        self.c_puct = float(c_puct)
        self.max_children = int(max_children)
        if self.max_children < 218:
            raise ValueError("max_children must be at least 218 for the full chess action space.")

        self.chess = GPUChess(self.device, 1)
        self._reset_tree()

    def _reset_tree(self):
        self.num_games = 0
        self.max_nodes = 0
        self.max_edges = 0
        self.pieces = None
        self.turn = None
        self.castling = None
        self.ep_square = None
        self.halfmove_clock = None
        self.fullmove_number = None
        self.parent = None
        self.visit_count = None
        self.value_sum = None
        self.expanded = None
        self.terminal = None
        self.edge_start = None
        self.edge_count = None
        self.edge_child = None
        self.edge_action = None
        self.edge_prior = None
        self.edge_valid = None
        self.edge_used = 0

        # Temporary selection-debug state. This is intentionally kept inside
        # GPUMCTS so the diagnostic can observe the ACTUAL selection decision
        # without changing the MCTS scoring or tree update logic.
        self._debug_select_calls = 0
        self._debug_trace_actions = (263, 3186)
        self._debug_trace_limit = 50

        # Reused indexing buffers. These avoid allocating the same arange
        # tensors on every tree traversal / root operation.
        self._child_cols = None
        self._edge_ids = None
        self._root_ids = None
        self._action_ids = None

    def _allocate(self, num_games: int, num_simulations: int):
        # One root plus at most one newly expanded leaf per simulation and game.
        # Each expansion can have at most 218 legal chess moves, but 256 keeps
        # room for the complete 4544 action space's promotion structure.
        # Fixed GPU edge slots: every node owns max_children contiguous slots.
        # This removes the old dynamic edge-pool accounting and its GPU->CPU
        # .item() synchronization from the hot path.
        # Allocate a bounded pool based on the maximum number of expansions.
        # Each simulation can add at most max_children child nodes per active
        # root, so the node/edge pools remain linear in simulations.
        expansions = num_games * (max(1, num_simulations) + 1)
        self.max_nodes = num_games + expansions * self.max_children
        self.max_edges = expansions * self.max_children
        self.num_games = num_games

        dev = self.device
        self.pieces = torch.zeros((self.max_nodes, 12), dtype=torch.int64, device=dev)
        self.turn = torch.zeros((self.max_nodes,), dtype=torch.bool, device=dev)
        self.castling = torch.zeros((self.max_nodes,), dtype=torch.int16, device=dev)
        self.ep_square = torch.full((self.max_nodes,), -1, dtype=torch.int16, device=dev)
        self.halfmove_clock = torch.zeros((self.max_nodes,), dtype=torch.int16, device=dev)
        self.fullmove_number = torch.ones((self.max_nodes,), dtype=torch.int16, device=dev)

        self.parent = torch.full((self.max_nodes,), -1, dtype=torch.int32, device=dev)
        self.visit_count = torch.zeros((self.max_nodes,), dtype=torch.int32, device=dev)
        self.value_sum = torch.zeros((self.max_nodes,), dtype=torch.float32, device=dev)
        self.expanded = torch.zeros((self.max_nodes,), dtype=torch.bool, device=dev)
        self.terminal = torch.zeros((self.max_nodes,), dtype=torch.bool, device=dev)
        self.edge_start = torch.full((self.max_nodes,), -1, dtype=torch.int32, device=dev)
        self.edge_count = torch.zeros((self.max_nodes,), dtype=torch.int16, device=dev)

        self.edge_child = torch.full((self.max_edges,), -1, dtype=torch.int32, device=dev)
        self.edge_action = torch.zeros((self.max_edges,), dtype=torch.int16, device=dev)
        self.edge_prior = torch.zeros((self.max_edges,), dtype=torch.float32, device=dev)
        self.edge_valid = torch.zeros((self.max_edges,), dtype=torch.bool, device=dev)
        self.edge_used = 0
        self._debug_select_calls = 0

        # Cache frequently reused GPU indexing tensors for this search.
        self._child_cols = torch.arange(
            self.max_children, device=dev, dtype=torch.long
        )
        self._edge_ids = torch.arange(
            self.max_edges, device=dev, dtype=torch.long
        )
        self._root_ids = torch.arange(
            self.num_games, device=dev, dtype=torch.long
        )
        self._action_ids = torch.arange(4544, device=dev, dtype=torch.long)

    def _state_view(self, node_ids: torch.Tensor) -> GPUChess:
        """Create a lightweight GPUChess view containing selected tree nodes."""
        g = object.__new__(GPUChess)
        g.__dict__ = self.chess.__dict__.copy()
        g.batch_size = int(node_ids.numel())
        g.pieces = self.pieces[node_ids]
        g.turn = self.turn[node_ids]
        g.castling = self.castling[node_ids]
        g.ep_square = self.ep_square[node_ids]
        g.halfmove_clock = self.halfmove_clock[node_ids]
        g.fullmove_number = self.fullmove_number[node_ids]
        return g

    def _write_states(self, node_ids: torch.Tensor, states: GPUChess):
        self.pieces[node_ids] = states.pieces
        self.turn[node_ids] = states.turn
        self.castling[node_ids] = states.castling
        self.ep_square[node_ids] = states.ep_square
        self.halfmove_clock[node_ids] = states.halfmove_clock
        self.fullmove_number[node_ids] = states.fullmove_number

    def _evaluate(self, node_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = self._state_view(node_ids)
        model_input = states.to_model_input()
        if model_input.is_cuda:
            model_input = model_input.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, values = self.model(model_input)
            else:
                logits, values = self.model(model_input)
        legal = states.legal_move_mask()
        return logits.float(), values.squeeze(-1).float(), legal

    def _expand(self, node_ids: torch.Tensor, logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        """Expand a batch without GPU->CPU synchronization.

        Every expanded node receives a fixed ``max_children`` edge slot block.
        This deliberately avoids ``Tensor.item()``, dynamic prefix sizing and
        CPU decisions in the hot path.  Only the first ``legal_count`` slots
        are considered valid.
        """
        n = node_ids.numel()
        if n == 0:
            return torch.empty((0,), dtype=torch.bool, device=self.device)

        counts = legal.sum(dim=1, dtype=torch.int16)
        # MAX_LEGAL_MOVES is 256 and the engine guarantees <= 218 legal moves.
        # No host-side check is needed here.

        # Fixed-size legal action extraction. topk puts all True entries first
        # because the input is 0/1. This removes the dynamic nonzero()/cumsum()
        # path and therefore removes synchronization caused by computing the
        # total number of edges on the CPU.
        # Rank legal actions by their original 4544-action index. This keeps
        # the project's deterministic action ordering while still producing a
        # fixed-size CUDA tensor with no host synchronization.
        action_grid = self._action_ids[None, :].expand(n, -1)
        ranked = torch.where(legal, action_grid, torch.full_like(action_grid, 4544))
        actions = torch.topk(
            ranked, k=self.max_children, dim=1, largest=False, sorted=True
        ).values
        slots = self._child_cols[None, :]
        valid = actions < 4544

        edge_start = self.edge_used
        reserved = n * self.max_children
        edge_end = edge_start + reserved
        if edge_end > self.max_edges:
            raise RuntimeError("GPU MCTS edge pool exhausted; increase the search capacity.")

        edge_ids = self._edge_ids[edge_start:edge_end].view(n, self.max_children)
        child_nodes = self.num_games + edge_ids

        self.edge_start[node_ids] = edge_ids[:, 0].to(torch.int32)
        self.edge_count[node_ids] = counts

        masked = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        priors = torch.softmax(masked, dim=1)
        priors = priors * legal.to(priors.dtype)
        priors = priors / priors.sum(dim=1, keepdim=True).clamp_min(1e-12)

        flat_valid = valid.reshape(-1)
        flat_actions = actions.reshape(-1)
        flat_edges = edge_ids.reshape(-1)
        flat_children = child_nodes.reshape(-1)
        flat_parents = node_ids[:, None].expand(-1, self.max_children).reshape(-1)

        v_actions = flat_actions[flat_valid]
        v_edges = flat_edges[flat_valid]
        v_children = flat_children[flat_valid]
        v_parents = flat_parents[flat_valid]

        self.parent[v_children] = v_parents.to(torch.int32)
        self.edge_child[v_edges] = v_children.to(torch.int32)
        self.edge_action[v_edges] = v_actions.to(torch.int16)
        prior_actions = actions.clamp_max(4543)
        self.edge_prior[v_edges] = priors.gather(1, prior_actions).reshape(-1)[flat_valid]
        self.edge_valid[edge_ids.reshape(-1)] = valid.reshape(-1)

        # Apply all valid child actions as one GPU batch.
        parent_states = self._state_view(v_parents)
        child_states = parent_states.apply_actions_unchecked(
            torch.arange(v_actions.numel(), device=self.device, dtype=torch.long),
            v_actions,
        )
        self._write_states(v_children, child_states)

        child_view = self._state_view(v_children)
        draw_terminal = (child_view.halfmove_clock >= 100) | child_view.insufficient_material()
        self.terminal[v_children] = draw_terminal

        self.expanded[node_ids] = True
        self.terminal[node_ids] = counts == 0
        self.edge_used = edge_end
        return (counts == 0)


    def _select_leaves(self, root_ids: torch.Tensor, max_depth: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Select one leaf per root in parallel and return leaf ids + paths.

        TEMPORARY DEBUG VERSION:
        Traces the *actual* selection decision for actions 263 and 3186.
        This does not alter PUCT, argmax, tree state, or backup behavior.
        Remove the debug prints after the MCTS issue is identified.
        """
        g = root_ids.numel()
        current = root_ids.clone()
        active = torch.ones((g,), dtype=torch.bool, device=self.device)
        paths = torch.full((g, max_depth + 1), -1, dtype=torch.int32, device=self.device)
        paths[:, 0] = root_ids.to(torch.int32)
        depth = 0

        # One _select_leaves call corresponds to one actual tree traversal.
        # Keep the trace bounded so a long search does not flood the notebook.
        trace_this_call = self._debug_select_calls < self._debug_trace_limit
        self._debug_select_calls += 1

        while depth < max_depth:
            selectable = active & self.expanded[current] & ~self.terminal[current] & (self.edge_count[current] > 0)
            sel_rows = torch.nonzero(selectable, as_tuple=False).flatten()
            if sel_rows.numel() == 0:
                break
            nodes = current[sel_rows]
            starts = self.edge_start[nodes].to(torch.long)
            counts = self.edge_count[nodes].to(torch.long)
            arange_child = self._child_cols[None, :]
            edge_idx = starts[:, None] + arange_child
            edge_idx_safe = edge_idx.clamp(0, self.max_edges - 1)
            valid = self.edge_valid[edge_idx_safe]
            child = self.edge_child[edge_idx_safe].to(torch.long)
            visits = self.visit_count[child].float()
            sums = self.value_sum[child]
            q = torch.where(visits > 0, sums / visits, torch.zeros_like(sums))
            priors = self.edge_prior[edge_idx_safe]
            parent_visits = self.visit_count[nodes].float().clamp_min(1.0)[:, None]
            scores = -q + self.c_puct * priors * torch.sqrt(parent_visits) / (1.0 + visits)
            scores = scores.masked_fill(~valid, -torch.inf)
            best = torch.argmax(scores, dim=1)
            next_nodes = child.gather(1, best[:, None]).squeeze(1)

            # --------------------------------------------------------------
            # TEMPORARY ACTUAL-SELECTION TRACE
            # --------------------------------------------------------------
            if trace_this_call:
                for local_row in range(sel_rows.numel()):
                    node_id = int(nodes[local_row].item())
                    best_col = int(best[local_row].item())
                    selected_action = int(edge_idx_safe[local_row, best_col].item())
                    # edge_idx is an edge id; recover the actual chess action.
                    selected_action = int(self.edge_action[edge_idx_safe[local_row, best_col]].item())

                    target_lines = []
                    for target_action in self._debug_trace_actions:
                        matches = (self.edge_action[edge_idx_safe[local_row]] == target_action) & valid[local_row]
                        if bool(matches.any().item()):
                            pos = int(torch.nonzero(matches, as_tuple=False)[0, 0].item())
                            target_edge = edge_idx_safe[local_row, pos]
                            target_lines.append(
                                f"action={target_action} "
                                f"P={float(priors[local_row, pos].item()):+.6f} "
                                f"N={int(visits[local_row, pos].item())} "
                                f"Q={float(q[local_row, pos].item()):+.6f} "
                                f"U={float((scores[local_row, pos] + q[local_row, pos]).item()):+.6f} "
                                f"PUCT={float(scores[local_row, pos].item()):+.6f} "
                                f"edge={int(target_edge.item())}"
                            )

                    # Only print nodes containing one of the target actions.
                    if target_lines:
                        print("\\n[MCTS DEBUG] ACTUAL SELECTION")
                        print(f"  select_call={self._debug_select_calls} depth={depth} parent_node={node_id}")
                        for line in target_lines:
                            print(f"  TARGET {line}")
                        print(
                            f"  SELECTED action={selected_action} "
                            f"edge={int(edge_idx_safe[local_row, best_col].item())} "
                            f"child={int(next_nodes[local_row].item())} "
                            f"P={float(priors[local_row, best_col].item()):+.6f} "
                            f"N={int(visits[local_row, best_col].item())} "
                            f"Q={float(q[local_row, best_col].item()):+.6f} "
                            f"PUCT={float(scores[local_row, best_col].item()):+.6f}"
                        )

            current[sel_rows] = next_nodes
            depth += 1

            # Record the node only for rows that actually traversed at this
            # depth. Do NOT copy the previous node into inactive rows.
            paths[sel_rows, depth] = next_nodes.to(torch.int32)

            # Rows that reached an unexpanded/terminal node stop. Others keep
            # traversing on the next depth iteration.
            active = active & self.expanded[current] & ~self.terminal[current]

        return current, paths[:, :depth + 1]

    def _backup(self, paths: torch.Tensor, values: torch.Tensor):
        # Back up each root's path from leaf to root.  Paths may have different
        # lengths, so the value sign must be based on the distance from each
        # row's actual leaf rather than on the global padded depth.
        depth = paths.shape[1]
        valid = paths >= 0
        lengths = valid.sum(dim=1)
        d = torch.arange(depth, device=self.device)[None, :]
        distance_from_leaf = lengths[:, None] - 1 - d
        use = valid
        nodes = paths[use].to(torch.long)
        signs = torch.where(
            (distance_from_leaf[use] % 2) == 0,
            torch.ones_like(distance_from_leaf[use], dtype=values.dtype),
            -torch.ones_like(distance_from_leaf[use], dtype=values.dtype),
        )
        vv = values[:, None].expand(-1, depth)[use] * signs
        if nodes.numel():
            self.visit_count.index_add_(
                0, nodes, torch.ones(nodes.shape, dtype=torch.int32, device=self.device)
            )
            self.value_sum.index_add_(0, nodes, vv)

    def _terminal_values(self, node_ids: torch.Tensor) -> torch.Tensor:
        states = self._state_view(node_ids)
        legal = states.legal_move_mask()
        no_moves = ~legal.any(dim=1)
        check = states.is_in_check()
        checkmate = no_moves & check
        return torch.where(checkmate, -torch.ones_like(states.halfmove_clock, dtype=torch.float32), torch.zeros_like(states.halfmove_clock, dtype=torch.float32))

    def _apply_virtual_loss(self, paths: torch.Tensor, loss: float = 1.0):
        """Temporarily reserve selected paths so parallel selections diverge."""
        nodes = paths.reshape(-1).to(torch.long)
        valid = nodes >= 0
        nodes = nodes[valid]
        if nodes.numel() == 0:
            return
        ones = torch.ones((nodes.numel(),), dtype=torch.int32, device=self.device)
        losses = torch.full((nodes.numel(),), float(loss), dtype=torch.float32, device=self.device)
        self.visit_count.index_add_(0, nodes, ones)
        self.value_sum.index_add_(0, nodes, losses)

    def _remove_virtual_loss(self, paths: torch.Tensor, loss: float = 1.0):
        nodes = paths.reshape(-1).to(torch.long)
        valid = nodes >= 0
        nodes = nodes[valid]
        if nodes.numel() == 0:
            return
        ones = torch.ones((nodes.numel(),), dtype=torch.int32, device=self.device)
        losses = torch.full((nodes.numel(),), -float(loss), dtype=torch.float32, device=self.device)
        self.visit_count.index_add_(0, nodes, -ones)
        self.value_sum.index_add_(0, nodes, losses)

    def _backup_batched(self, paths: torch.Tensor, values: torch.Tensor):
        """GPU backup for [batch, games, depth] paths with variable lengths."""
        b, g, depth = paths.shape
        flat_paths = paths.reshape(b * g, depth)
        flat_values = values.reshape(-1)

        valid = flat_paths >= 0
        lengths = valid.sum(dim=1)
        d = torch.arange(depth, device=self.device)[None, :]
        distance_from_leaf = lengths[:, None] - 1 - d

        use = valid
        nodes = flat_paths[use].to(torch.long)
        base_values = flat_values[:, None].expand(-1, depth)[use]
        signs = torch.where(
            (distance_from_leaf[use] % 2) == 0,
            torch.ones_like(base_values),
            -torch.ones_like(base_values),
        )
        vv = base_values * signs

        if nodes.numel():
            self.visit_count.index_add_(
                0, nodes, torch.ones(nodes.shape, dtype=torch.int32, device=self.device)
            )
            self.value_sum.index_add_(0, nodes, vv)

    def search(self,
        root_states: GPUChess,
        num_simulations: int,
        dirichlet_alpha: float | None = None,
        dirichlet_epsilon: float = 0.25,
        batch_size: int = 16,
    ):
        """Run batched GPU MCTS.

        Several simulations are selected with virtual loss before any neural
        network evaluation.  Their leaves are then evaluated in one large CUDA
        batch.  This reduces Python/model-launch overhead dramatically while
        preserving the normal MCTS tree statistics.
        """
        if root_states.device != self.device:
            raise ValueError("root_states must live on the GPUMCTS device")
        games = root_states.pieces.shape[0]
        if games <= 0:
            return
        if num_simulations < 0:
            raise ValueError("num_simulations must be non-negative")
        batch_size = max(1, min(int(batch_size), int(num_simulations) if num_simulations else 1))

        self._allocate(games, max(1, num_simulations))
        roots = self._root_ids
        self.model.eval()
        self._write_states(roots, root_states)

        logits, _, legal = self._evaluate(roots)
        self._expand(roots, logits, legal)

        if dirichlet_alpha is not None:
            self.add_dirichlet_noise(dirichlet_alpha, dirichlet_epsilon)
        if num_simulations <= 0:
            return

        # Chess trees normally reach leaves well below this bound. Keeping it
        # fixed avoids allocating a path tensor proportional to simulations.
        max_depth = 64
        remaining = int(num_simulations)
        while remaining > 0:
            bsz = min(batch_size, remaining)
            selected_paths = []
            selected_leaves = []

            # Selection itself has unavoidable tree dependency. We keep this
            # small loop on the host, but every operation inside it is CUDA.
            for _ in range(bsz):
                leaves, paths = self._select_leaves(roots, max_depth=max_depth)
                selected_leaves.append(leaves)
                selected_paths.append(paths)
                self._apply_virtual_loss(paths)

            # Normalize path depth for stacking without moving anything to CPU.
            depth = max(p.shape[1] for p in selected_paths)
            padded = []
            for p in selected_paths:
                if p.shape[1] < depth:
                    q = torch.full((games, depth), -1, dtype=torch.int32, device=self.device)
                    q[:, :p.shape[1]] = p
                    p = q
                padded.append(p)
            paths = torch.stack(padded, dim=0)
            leaves = torch.stack(selected_leaves, dim=0)

            # Remove temporary virtual losses before applying real backups.
            self._remove_virtual_loss(paths)

            flat_leaves = leaves.reshape(-1)
            terminal = self.terminal[flat_leaves]
            values = torch.zeros((flat_leaves.numel(),), dtype=torch.float32, device=self.device)

            t_ids = flat_leaves[terminal]
            if t_ids.numel():
                values[terminal] = self._terminal_values(t_ids)

            nt_mask = ~terminal
            nt_ids = flat_leaves[nt_mask]
            if nt_ids.numel():
                # One large NN + legal-move batch for the whole simulation batch.
                logits, nn_values, legal = self._evaluate(nt_ids)
                self._expand(nt_ids, logits, legal)
                no_moves = ~legal.any(dim=1)
                nt_view = self._state_view(nt_ids)
                check = nt_view.is_in_check()
                exact = torch.where(no_moves & check,
                                    -torch.ones_like(nn_values),
                                    torch.zeros_like(nn_values))
                newly_terminal = no_moves | (nt_view.halfmove_clock >= 100) | nt_view.insufficient_material()
                values[nt_mask] = torch.where(newly_terminal, exact, nn_values)

            self._backup_batched(paths, values.reshape(bsz, games))
            remaining -= bsz

    def root_policy(self, temperature: float = 1.0) -> torch.Tensor:
        roots = self._root_ids
        policy = torch.zeros((self.num_games, 4544), dtype=torch.float32, device=self.device)
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = self._child_cols[None, :]
        edge_idx = starts[:, None] + cols
        safe = edge_idx.clamp(0, self.max_edges - 1)
        valid = self.edge_valid[safe]
        actions = self.edge_action[safe].to(torch.long)
        visits = self.visit_count[self.edge_child[safe].to(torch.long)].float()
        if temperature <= 0:
            weights = visits
            best = torch.argmax(weights.masked_fill(~valid, -1), dim=1)
            chosen = actions[torch.arange(self.num_games, device=self.device), best]
            policy.scatter_(1, chosen[:, None], torch.ones_like(chosen[:, None], dtype=policy.dtype))
            return policy
        weights = torch.where(valid, visits.clamp_min(0.0).pow(1.0 / temperature), torch.zeros_like(visits))
        denom = weights.sum(dim=1, keepdim=True)
        probs = weights / denom.clamp_min(1e-12)
        rows = torch.arange(self.num_games, device=self.device)[:, None].expand_as(actions)
        policy.scatter_add_(1, actions, torch.where(valid, probs, torch.zeros_like(probs)))
        return policy

    def select_actions(self, temperature: float = 1.0) -> torch.Tensor:
        roots = self._root_ids
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = self._child_cols[None, :]
        edge_idx = starts[:, None] + cols
        safe = edge_idx.clamp(0, self.max_edges - 1)
        valid = self.edge_valid[safe]
        actions = self.edge_action[safe].to(torch.long)
        visits = self.visit_count[self.edge_child[safe].to(torch.long)].float()
        if temperature <= 0:
            return actions.gather(1, torch.argmax(visits.masked_fill(~valid, -1), dim=1, keepdim=True)).squeeze(1)
        weights = torch.where(valid, visits.clamp_min(0).pow(1.0 / temperature), torch.zeros_like(visits))
        probs = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return actions.gather(1, torch.multinomial(probs, 1)).squeeze(1)

    def add_dirichlet_noise(self, alpha: float = 0.3, epsilon: float = 0.25):
        if alpha <= 0 or not 0 <= epsilon <= 1:
            raise ValueError("Invalid Dirichlet parameters")
        roots = self._root_ids
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        maxc = self.max_children
        cols = self._child_cols[None, :]
        edge_idx = starts[:, None] + cols
        e = edge_idx.clamp(0, self.max_edges - 1)
        valid = self.edge_valid[e]
        # A Dirichlet distribution is sampled independently for each root.
        noise = torch.distributions.Dirichlet(torch.full((maxc,), alpha, device=self.device)).sample((self.num_games,))
        # Renormalize only over actual legal children.
        noise = noise * valid.float()
        noise = noise / noise.sum(dim=1, keepdim=True).clamp_min(1e-12)
        old = self.edge_prior[e]
        self.edge_prior[e] = torch.where(valid, (1 - epsilon) * old + epsilon * noise, old)


    def root_visit_policy(self) -> torch.Tensor:
        """Return normalized root visit counts over all 4544 actions."""
        roots = self._root_ids
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = self._child_cols[None, :]
        edge_idx = starts[:, None] + cols
        safe = edge_idx.clamp(0, self.max_edges - 1)
        valid = self.edge_valid[safe]
        actions = self.edge_action[safe].to(torch.long)
        visits = self.visit_count[self.edge_child[safe].to(torch.long)].float()
        visits = torch.where(valid, visits, torch.zeros_like(visits))
        probs = visits / visits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        policy = torch.zeros((self.num_games, 4544), dtype=torch.float32, device=self.device)
        policy.scatter_add_(1, actions, probs)
        return policy

    def advance(self, actions: torch.Tensor) -> GPUChess:
        """Return the GPU states reached by one selected action per root."""
        actions = actions.to(device=self.device, dtype=torch.long)
        roots = self._root_ids
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = self._child_cols[None, :]
        edge_idx = starts[:, None] + cols
        safe = edge_idx.clamp(0, self.max_edges - 1)
        valid = self.edge_valid[safe]
        edge_actions = self.edge_action[safe].to(torch.long)
        matches = (edge_actions == actions[:, None]) & valid
        pos = torch.argmax(matches.to(torch.int8), dim=1)
        child_ids = self.edge_child[safe.gather(1, pos[:, None]).squeeze(1)].to(torch.long)
        return self._state_view(child_ids)

    def get_child_states(self) -> GPUChess:
        roots = self._root_ids
        actions = self.select_actions(temperature=0.0)
        root_starts = self.edge_start[roots].to(torch.long)
        root_counts = self.edge_count[roots].to(torch.long)
        cols = self._child_cols[None, :]
        edge_idx = root_starts[:, None] + cols
        safe = edge_idx.clamp(0, self.max_edges - 1)
        valid = self.edge_valid[safe]
        edge_actions = self.edge_action[safe].to(torch.long)
        chosen_pos = (edge_actions == actions[:, None]) & valid
        chosen_edge = torch.argmax(chosen_pos.to(torch.int8), dim=1)
        child_ids = self.edge_child[safe.gather(1, chosen_edge[:, None]).squeeze(1)].to(torch.long)
        return self._state_view(child_ids)

    def root_children(self):
        roots = self._root_ids
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = self._child_cols[None, :]
        edge_idx = starts[:, None] + cols
        safe = edge_idx.clamp(0, self.max_edges - 1)
        valid = self.edge_valid[safe]
        actions = self.edge_action[safe].to(torch.long)
        visits = self.visit_count[self.edge_child[safe].to(torch.long)].float()
        return actions, visits, valid
