# ============================================================
# RL53 DEEP MCTS CHILD-VALUE / BACKUP DIAGNOSTIC
# ============================================================
#
# Purpose:
#   Trace unstable root moves into their child positions and
#   inspect:
#
#       Root state
#          |
#          +--> selected action
#                    |
#                    +--> child state
#                           |
#                           +--> NN value
#                           +--> legal moves
#                           +--> terminal status
#                           +--> MCTS value
#
# This does NOT train anything.
#
# ============================================================

import os
import sys
import time
import math
import random
import traceback

import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

PROJECT_DIR = "/kaggle/working/chess-zero"

CHECKPOINT_PATH = (
    f"{PROJECT_DIR}/checkpoints/rl_iteration_53.pt"
)

REPLAY_PATH = (
    f"{PROJECT_DIR}/checkpoints/replay_buffer_rl53.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MCTS_SIMULATIONS = [100, 200, 400, 800]

C_PUCT = 1.5

TOP_K = 10

# These are the unstable cases identified from the previous
# RL53 diagnostic.
#
# (label, replay_index, action)
TRACE_CASES = [
    ("P2-A2472", 8331, 2472),
    ("P3-A263", 110785, 263),
    ("P3-A191", 110785, 191),
    ("P4-A1463", 29184, 1463),
    ("P5-A3186", 56443, 3186),
]

# ============================================================
# ENVIRONMENT
# ============================================================

sys.path.insert(0, PROJECT_DIR)

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


# ============================================================
# HELPERS
# ============================================================

def banner(title, char="="):
    print()
    print(char * 100)
    print(title)
    print(char * 100)


def load_checkpoint_model(path, device):
    banner("LOADING RL53 MODEL")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    print("Checkpoint type:", type(checkpoint))

    model = ChessNet().to(device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Some checkpoints may directly contain the state dict.
            state_dict = checkpoint

    else:
        raise RuntimeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if missing:
        print("Missing:", missing)

    if unexpected:
        print("Unexpected:", unexpected)

    model.eval()

    print("RL53 model loaded successfully.")

    return model


def model_predict(model, states):
    """
    Returns:
        policy_logits
        value
    """

    model.eval()

    with torch.inference_mode():

        x = states.to(
            device=DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )

        policy_logits, value = model(x)

        value = value.reshape(-1)

    return policy_logits, value


def get_legal_actions(env):
    """
    Extract legal action IDs from GPUChess.
    """

    mask = env.legal_move_mask()

    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    actions = torch.nonzero(
        mask[0],
        as_tuple=False,
    ).flatten()

    return actions


def print_network_values(
    model,
    env,
    title,
):
    banner(title, "-")

    state = env.to_model_input()

    if state.ndim == 3:
        state = state.unsqueeze(0)

    policy_logits, values = model_predict(
        model,
        state,
    )

    value = float(values[0].item())

    legal_actions = get_legal_actions(env)

    logits = policy_logits[0]

    legal_logits = logits[legal_actions]

    probs = torch.softmax(
        legal_logits,
        dim=0,
    )

    order = torch.argsort(
        probs,
        descending=True,
    )

    print(f"Network value: {value:+.6f}")
    print(f"Legal moves:   {len(legal_actions)}")

    print()
    print(
        "Rank | Action | Network Prob | Logit"
    )
    print("-" * 60)

    for rank, idx in enumerate(
        order[:TOP_K].tolist(),
        start=1,
    ):

        action = int(
            legal_actions[idx].item()
        )

        probability = float(
            probs[idx].item()
        )

        logit = float(
            legal_logits[idx].item()
        )

        print(
            f"{rank:4d} | "
            f"{action:6d} | "
            f"{probability:13.6f} | "
            f"{logit:+.6f}"
        )

    return value


# ============================================================
# REPLAY STATE RECONSTRUCTION
# ============================================================

def replay_to_env(sample, device):
    """
    Reconstruct GPUChess from the stored 18-plane model state.

    This uses the same reconstruction approach as the previous
    successful diagnostic.
    """

    state = sample[0]

    if not torch.is_tensor(state):
        state = torch.tensor(
            state,
            dtype=torch.float32,
        )

    state = state.to(
        device=device,
        dtype=torch.float32,
    )

    if state.ndim == 4:
        state = state[0]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This section assumes the replay sample's first element
    # is the stored 18 x 8 x 8 state.
    #
    # The previous diagnostic successfully reconstructed these
    # exact states with zero difference.
    # --------------------------------------------------------

    env = GPUChess.from_model_input(
        state.unsqueeze(0)
    )

    return env


# ============================================================
# SAFE STATE COPY
# ============================================================

def clone_env(env):
    """
    Try to make an independent GPUChess state.

    Uses clone() if available.
    Otherwise reconstructs through model input.
    """

    if hasattr(env, "clone"):
        try:
            return env.clone()
        except Exception:
            pass

    state = env.to_model_input()

    if state.ndim == 3:
        state = state.unsqueeze(0)

    return GPUChess.from_model_input(
        state
    )


# ============================================================
# APPLY ACTION
# ============================================================

def apply_action(env, action):
    """
    Apply one encoded chess action.

    GPUChess versions may expose different method names,
    so try the common APIs.
    """

    action_tensor = torch.tensor(
        [action],
        dtype=torch.long,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Preferred GPUChess API
    # --------------------------------------------------------

    if hasattr(env, "apply_action"):
        result = env.apply_action(
            action_tensor
        )

        if result is None:
            return env

        return result

    # --------------------------------------------------------
    # Alternative API
    # --------------------------------------------------------

    if hasattr(env, "step"):
        result = env.step(
            action_tensor
        )

        if isinstance(result, tuple):
            return result[0]

        return result

    raise RuntimeError(
        "GPUChess does not expose apply_action() or step()."
    )


# ============================================================
# TERMINAL INFORMATION
# ============================================================

def terminal_info(env):
    """
    Inspect terminal-state methods if available.

    This deliberately does not assume a single exact method
    name because GPUChess has evolved during the project.
    """

    info = {
        "terminal": None,
        "checkmate": None,
        "stalemate": None,
        "fifty_move": None,
        "insufficient_material": None,
    }

    # Generic terminal check
    for name in [
        "is_terminal",
        "terminal",
        "is_game_over",
    ]:
        if hasattr(env, name):

            obj = getattr(env, name)

            try:
                value = obj() if callable(obj) else obj

                if torch.is_tensor(value):
                    value = bool(
                        value.reshape(-1)[0].item()
                    )

                info["terminal"] = bool(value)
                break

            except Exception:
                pass

    # Checkmate
    for name in [
        "is_checkmate",
        "checkmate",
    ]:
        if hasattr(env, name):

            obj = getattr(env, name)

            try:
                value = obj() if callable(obj) else obj

                if torch.is_tensor(value):
                    value = bool(
                        value.reshape(-1)[0].item()
                    )

                info["checkmate"] = bool(value)
                break

            except Exception:
                pass

    # Stalemate
    for name in [
        "is_stalemate",
        "stalemate",
    ]:
        if hasattr(env, name):

            obj = getattr(env, name)

            try:
                value = obj() if callable(obj) else obj

                if torch.is_tensor(value):
                    value = bool(
                        value.reshape(-1)[0].item()
                    )

                info["stalemate"] = bool(value)
                break

            except Exception:
                pass

    return info


# ============================================================
# STATE SUMMARY
# ============================================================

def state_summary(env):

    print("State summary:")

    attrs = [
        "turn",
        "castling",
        "ep_square",
        "halfmove_clock",
        "fullmove_number",
    ]

    for attr in attrs:

        if hasattr(env, attr):

            value = getattr(env, attr)

            try:
                if torch.is_tensor(value):
                    value = value.detach().cpu().tolist()
            except Exception:
                pass

            print(
                f"  {attr}: {value}"
            )


# ============================================================
# DIRECT CHILD INSPECTION
# ============================================================

def inspect_child(
    model,
    root_env,
    action,
    label,
):
    banner(
        f"CHILD STATE: {label} | ACTION {action}",
        "-"
    )

    print(
        "Root action:",
        action
    )

    child = clone_env(root_env)

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    print()
    print("Before applying action:")
    state_summary(child)

    root_terminal = terminal_info(child)

    print(
        "Terminal:",
        root_terminal
    )

    # --------------------------------------------------------
    # APPLY
    # --------------------------------------------------------

    try:

        child = apply_action(
            child,
            action,
        )

    except Exception as e:

        print()
        print(
            "FAILED TO APPLY ACTION"
        )
        print(
            repr(e)
        )
        traceback.print_exc()

        return None

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    print()
    print("After applying action:")
    state_summary(child)

    child_terminal = terminal_info(child)

    print(
        "Terminal:",
        child_terminal
    )

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    state = child.to_model_input()

    if state.ndim == 3:
        state = state.unsqueeze(0)

    policy_logits, values = model_predict(
        model,
        state,
    )

    child_value = float(
        values[0].item()
    )

    legal_actions = get_legal_actions(
        child
    )

    print()
    print(
        f"Child NN value: {child_value:+.6f}"
    )

    print(
        f"Child legal moves: {len(legal_actions)}"
    )

    # --------------------------------------------------------
    # CHILD POLICY
    # --------------------------------------------------------

    logits = policy_logits[0]

    if len(legal_actions) > 0:

        legal_logits = logits[
            legal_actions
        ]

        probs = torch.softmax(
            legal_logits,
            dim=0,
        )

        order = torch.argsort(
            probs,
            descending=True,
        )

        print()
        print(
            "Top child policy moves:"
        )

        print(
            "Rank | Action | Probability"
        )
        print("-" * 45)

        for rank, idx in enumerate(
            order[:TOP_K].tolist(),
            start=1,
        ):

            a = int(
                legal_actions[idx].item()
            )

            p = float(
                probs[idx].item()
            )

            print(
                f"{rank:4d} | "
                f"{a:6d} | "
                f"{p:.6f}"
            )

    return {
        "env": child,
        "value": child_value,
        "legal_actions": legal_actions,
        "terminal": child_terminal,
    }


# ============================================================
# RUN ONE MCTS SEARCH
# ============================================================

def run_mcts(
    model,
    env,
    simulations,
):
    """
    Run a fresh MCTS search.

    IMPORTANT:
    We do NOT pass batch_size here because the current
    GPUMCTS.search() implementation does not require it.
    """

    search = GPUMCTS(
        model=model,
        device=DEVICE,
        c_puct=C_PUCT,
    )

    root_states = clone_env(env)

    start = time.time()

    with torch.inference_mode():

        search.search(
            root_states,
            num_simulations=simulations,
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0,
        )

    elapsed = time.time() - start

    return search, elapsed


# ============================================================
# ROOT STAT EXTRACTION
# ============================================================

def root_stats(
    search,
    top_k=10,
):
    """
    Extract root child statistics.

    Uses the GPU MCTS internal tree arrays.
    """

    root_id = int(
        search._root_ids[0].item()
    )

    start = int(
        search.edge_start[root_id].item()
    )

    count = int(
        search.edge_count[root_id].item()
    )

    end = start + count

    actions = search.edge_action[
        start:end
    ]

    priors = search.edge_prior[
        start:end
    ]

    visits = search.visit_count[
        start:end
    ]

    sums = search.value_sum[
        start:end
    ]

    root_visits = int(
        search.visit_count[root_id].item()
    )

    q = torch.zeros_like(
        sums,
        dtype=torch.float32,
    )

    visited = visits > 0

    q[visited] = (
        sums[visited]
        / visits[visited].float()
    )

    u = (
        C_PUCT
        * priors
        * math.sqrt(
            max(root_visits, 1)
        )
        / (1.0 + visits.float())
    )

    # Selection score used by this MCTS:
    #
    # score = -Q + U
    #
    # Therefore smaller Q is preferred when values are from
    # the child/opponent perspective.
    puct = -q + u

    order = torch.argsort(
        puct,
        descending=True,
    )

    return {
        "actions": actions.detach().cpu(),
        "priors": priors.detach().cpu(),
        "visits": visits.detach().cpu(),
        "q": q.detach().cpu(),
        "u": u.detach().cpu(),
        "puct": puct.detach().cpu(),
        "root_visits": root_visits,
    }, order[:top_k].cpu()


# ============================================================
# PRINT ROOT TOP MOVES
# ============================================================

def print_root_top(
    stats,
    order,
    simulations,
):

    print()
    print(
        f"MCTS-{simulations}"
    )

    print(
        f"Root visits: {stats['root_visits']}"
    )

    print()

    print(
        "Rank | Action | "
        "Prior P | Visits N | "
        "Q | U | PUCT"
    )

    print("-" * 80)

    for rank, idx in enumerate(
        order.tolist(),
        start=1,
    ):

        action = int(
            stats["actions"][idx].item()
        )

        prior = float(
            stats["priors"][idx].item()
        )

        visits = int(
            stats["visits"][idx].item()
        )

        q = float(
            stats["q"][idx].item()
        )

        u = float(
            stats["u"][idx].item()
        )

        puct = float(
            stats["puct"][idx].item()
        )

        print(
            f"{rank:4d} | "
            f"{action:6d} | "
            f"{prior:.6f} | "
            f"{visits:8d} | "
            f"{q:+.6f} | "
            f"{u:+.6f} | "
            f"{puct:+.6f}"
        )


# ============================================================
# TRACE ROOT ACTION
# ============================================================

def trace_case(
    model,
    replay,
    label,
    replay_index,
    action,
):

    banner(
        f"TRACE CASE: {label}",
        "="
    )

    print(
        f"Replay index: {replay_index}"
    )

    print(
        f"Root action:  {action}"
    )

    # --------------------------------------------------------
    # RECONSTRUCT ROOT
    # --------------------------------------------------------

    sample = replay[
        replay_index
    ]

    root_env = replay_to_env(
        sample,
        DEVICE,
    )

    print()
    print("ROOT POSITION")

    root_value = print_network_values(
        model,
        root_env,
        "ROOT NETWORK EVALUATION",
    )

    # --------------------------------------------------------
    # VERIFY ACTION IS LEGAL
    # --------------------------------------------------------

    legal_actions = get_legal_actions(
        root_env
    )

    legal_set = set(
        int(x)
        for x in legal_actions.cpu().tolist()
    )

    print()

    if action not in legal_set:

        print(
            f"ERROR: action {action} "
            "is NOT legal in this position."
        )

        print(
            "Legal actions:"
        )

        print(
            sorted(legal_set)
        )

        return

    print(
        f"Action {action} is LEGAL."
    )

    # --------------------------------------------------------
    # DIRECT CHILD
    # --------------------------------------------------------

    child_info = inspect_child(
        model=model,
        root_env=root_env,
        action=action,
        label=label,
    )

    if child_info is None:
        return

    child_env = child_info["env"]

    # --------------------------------------------------------
    # CHILD NETWORK VALUE
    # --------------------------------------------------------

    child_value = child_info["value"]

    print()
    print(
        "Root NN value:",
        f"{root_value:+.6f}",
    )

    print(
        "Child NN value:",
        f"{child_value:+.6f}",
    )

    print()
    print(
        "If the value is expressed from the side-to-move "
        "perspective, the sign relationship here is important."
    )

    # --------------------------------------------------------
    # RUN MCTS AT DIFFERENT DEPTHS
    # --------------------------------------------------------

    results = {}

    for simulations in MCTS_SIMULATIONS:

        print()
        print(
            "=" * 80
        )

        print(
            f"RUNNING ROOT MCTS: "
            f"{simulations} SIMULATIONS"
        )

        print(
            "=" * 80
        )

        try:

            search, elapsed = run_mcts(
                model,
                root_env,
                simulations,
            )

            stats, order = root_stats(
                search,
                TOP_K,
            )

            results[simulations] = (
                stats,
                order,
            )

            print(
                f"Elapsed: {elapsed:.3f}s"
            )

            print_root_top(
                stats,
                order,
                simulations,
            )

            # ------------------------------------------------
            # Find traced action
            # ------------------------------------------------

            actions = stats[
                "actions"
            ]

            matches = torch.nonzero(
                actions == action,
                as_tuple=False,
            ).flatten()

            if len(matches) > 0:

                idx = int(
                    matches[0].item()
                )

                print()
                print(
                    "TRACED ACTION:"
                )

                print(
                    f"Action: {action}"
                )

                print(
                    f"Prior:  "
                    f"{float(stats['priors'][idx]):.6f}"
                )

                print(
                    f"Visits: "
                    f"{int(stats['visits'][idx])}"
                )

                print(
                    f"Q:      "
                    f"{float(stats['q'][idx]):+.6f}"
                )

                print(
                    f"U:      "
                    f"{float(stats['u'][idx]):+.6f}"
                )

                print(
                    f"PUCT:   "
                    f"{float(stats['puct'][idx]):+.6f}"
                )

            else:

                print()
                print(
                    "Traced action is not present "
                    "in root children."
                )

        except Exception as e:

            print()
            print(
                "MCTS ERROR:"
            )

            print(
                repr(e)
            )

            traceback.print_exc()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    banner(
        f"SUMMARY: {label}",
        "-"
    )

    print(
        "Simulation | Visits | Q | U | PUCT"
    )

    print("-" * 65)

    for simulations in MCTS_SIMULATIONS:

        if simulations not in results:
            continue

        stats, _ = results[
            simulations
        ]

        actions = stats[
            "actions"
        ]

        matches = torch.nonzero(
            actions == action,
            as_tuple=False,
        ).flatten()

        if len(matches) == 0:

            print(
                f"{simulations:10d} | "
                f"NOT FOUND"
            )

            continue

        idx = int(
            matches[0].item()
        )

        visits = int(
            stats["visits"][idx]
        )

        q = float(
            stats["q"][idx]
        )

        u = float(
            stats["u"][idx]
        )

        puct = float(
            stats["puct"][idx]
        )

        print(
            f"{simulations:10d} | "
            f"{visits:6d} | "
            f"{q:+.6f} | "
            f"{u:+.6f} | "
            f"{puct:+.6f}"
        )


# ============================================================
# MAIN
# ============================================================

banner(
    "RL53 DEEP MCTS CHILD-VALUE / BACKUP DIAGNOSTIC"
)

print(
    "Project:",
    PROJECT_DIR,
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH,
)

print(
    "Replay:",
    REPLAY_PATH,
)

print(
    "Device:",
    DEVICE,
)

print(
    "MCTS simulations:",
    MCTS_SIMULATIONS,
)

print(
    "c_puct:",
    C_PUCT,
)

print(
    "Dirichlet:",
    "OFF",
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

# ============================================================
# LOAD MODEL
# ============================================================

model = load_checkpoint_model(
    CHECKPOINT_PATH,
    DEVICE,
)

# ============================================================
# LOAD REPLAY
# ============================================================

banner(
    "LOADING RL53 REPLAY BUFFER"
)

replay = torch.load(
    REPLAY_PATH,
    map_location="cpu",
    weights_only=False,
)

print(
    "Replay type:",
    type(replay),
)

# Handle common replay-buffer formats.
if hasattr(replay, "buffer"):
    replay = list(replay.buffer)

elif isinstance(replay, dict):

    if "buffer" in replay:
        replay = replay["buffer"]

    elif "samples" in replay:
        replay = replay["samples"]

    else:
        raise RuntimeError(
            "Unknown replay dictionary format."
        )

elif not isinstance(replay, list):

    replay = list(replay)

print(
    "Replay size:",
    len(replay),
)

# ============================================================
# RUN TRACE CASES
# ============================================================

for (
    label,
    replay_index,
    action,
) in TRACE_CASES:

    try:

        trace_case(
            model=model,
            replay=replay,
            label=label,
            replay_index=replay_index,
            action=action,
        )

    except Exception as e:

        print()
        print(
            "=" * 100
        )

        print(
            f"FAILED CASE: {label}"
        )

        print(
            repr(e)
        )

        traceback.print_exc()

# ============================================================
# FINAL
# ============================================================

banner(
    "DIAGNOSTIC FINISHED"
)

print(
    """
Do NOT start RL54 from this output alone.

The important things to inspect are:

1. Root NN value
2. Child NN value immediately after the traced move
3. Whether the child is terminal
4. Child legal-move count
5. Root action Q across 100/200/400/800
6. Whether Q changes because the search reaches different
   child states or because backup/sign handling is inconsistent

Especially inspect:

    P2-A2472
    P3-A263
    P3-A191
    P4-A1463
    P5-A3186
"""
)