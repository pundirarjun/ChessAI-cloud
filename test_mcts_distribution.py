# ============================================================
# RL53 DEEP MCTS CHILD-VALUE / BACKUP DIAGNOSTIC
# CORRECTED FOR CURRENT GPUChess API
# ============================================================

import os
import sys
import time
import math
import random
import traceback
import numpy as np
import torch

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/kaggle/working/chess-zero"

CHECKPOINT_PATH = (
    f"{PROJECT_DIR}/checkpoints/rl_iteration_53.pt"
)

REPLAY_PATH = (
    f"{PROJECT_DIR}/checkpoints/replay_buffer_rl53.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MCTS_SIMULATIONS = [100, 200, 400, 800]

C_PUCT = 1.5

TOP_K = 10

# ------------------------------------------------------------
# Unstable root moves found in previous diagnostic
# ------------------------------------------------------------

TRACE_CASES = [

    # Position 2
    ("P2-A2472", 8331, 2472),

    # Position 3
    ("P3-A263", 110785, 263),
    ("P3-A191", 110785, 191),

    # Position 4
    ("P4-A1463", 29184, 1463),

    # Position 5
    ("P5-A3186", 56443, 3186),
]


# ============================================================
# IMPORT PROJECT
# ============================================================

sys.path.insert(
    0,
    PROJECT_DIR
)

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


# ============================================================
# PRINT HELPERS
# ============================================================

def banner(title, char="="):

    print()
    print(char * 100)
    print(title)
    print(char * 100)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    banner(
        "LOADING RL53 MODEL"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    print(
        "Checkpoint type:",
        type(checkpoint)
    )

    model = ChessNet().to(
        DEVICE
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        raise RuntimeError(
            f"Unsupported checkpoint type: "
            f"{type(checkpoint)}"
        )

    # --------------------------------------------------------
    # Remove possible torch.compile prefix
    # --------------------------------------------------------

    clean_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "_orig_mod."
        ):

            key = key[
                len("_orig_mod.") :
            ]

        clean_state_dict[
            key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            clean_state_dict,
            strict=False
        )
    )

    print(
        "Missing keys:",
        len(missing)
    )

    print(
        "Unexpected keys:",
        len(unexpected)
    )

    if missing:

        print(
            "Missing:",
            missing[:10]
        )

    if unexpected:

        print(
            "Unexpected:",
            unexpected[:10]
        )

    model.eval()

    print(
        "RL53 model loaded successfully."
    )

    return model


# ============================================================
# LOAD REPLAY
# ============================================================

def load_replay():

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
        type(replay)
    )

    # Handle possible replay wrappers
    if hasattr(
        replay,
        "buffer"
    ):

        replay = list(
            replay.buffer
        )

    elif isinstance(
        replay,
        dict
    ):

        if "buffer" in replay:

            replay = replay[
                "buffer"
            ]

        elif "samples" in replay:

            replay = replay[
                "samples"
            ]

        else:

            raise RuntimeError(
                "Unknown replay dictionary format."
            )

    elif not isinstance(
        replay,
        list
    ):

        replay = list(
            replay
        )

    print(
        "Replay size:",
        len(replay)
    )

    return replay


# ============================================================
# REPLAY -> GPUChess
# ============================================================

def replay_states_to_gpu_chess(
    samples,
    device
):
    """
    Exact reconstruction used by the previous successful
    diagnostic.

    Replay state:

        [18, 8, 8]

    GPUChess:

        12 piece bitboards
        turn
        castling
        en-passant
        halfmove clock
        fullmove number
    """

    states_np = np.stack(
        [
            np.asarray(
                sample[0],
                dtype=np.float32
            )
            for sample in samples
        ],
        axis=0
    )

    states = torch.from_numpy(
        states_np
    ).to(
        device=device,
        dtype=torch.float32
    )

    B = states.shape[0]

    if states.shape != (
        B,
        18,
        8,
        8
    ):

        raise ValueError(
            f"Unexpected state shape: "
            f"{states.shape}"
        )

    # --------------------------------------------------------
    # Create normal GPUChess object
    # --------------------------------------------------------

    chess = GPUChess(
        device=device,
        batch_size=B
    )

    # --------------------------------------------------------
    # PIECE PLANES
    # --------------------------------------------------------

    # GPUChess:
    #
    # square = rank * 8 + file
    #
    # Model representation:
    #
    # row = 7 - rank
    #
    # Therefore flip model rows back.

    board_planes = (
        states[:, :12]
        .flip(2)
    )

    square_bits = (
        chess.square_bits
    )

    pieces = torch.zeros(
        (
            B,
            12
        ),
        dtype=torch.int64,
        device=device
    )

    for p in range(12):

        occupancy = (
            board_planes[:, p]
            .reshape(B, 64)
            > 0.5
        )

        bb = torch.zeros(
            (B,),
            dtype=torch.int64,
            device=device
        )

        for sq in range(64):

            bb = torch.where(
                occupancy[:, sq],
                bb | square_bits[sq],
                bb
            )

        pieces[:, p] = bb

    chess.pieces = pieces

    # --------------------------------------------------------
    # SIDE TO MOVE
    # --------------------------------------------------------

    # Plane 12:
    #
    #   1 = white to move
    #   0 = black to move

    white_to_move = (
        states[:, 12, 0, 0]
        > 0.5
    )

    # GPUChess:
    #
    #   False = white
    #   True  = black

    chess.turn = ~white_to_move

    # --------------------------------------------------------
    # CASTLING RIGHTS
    # --------------------------------------------------------

    WK = 1
    WQ = 2
    BK = 4
    BQ = 8

    castling = torch.zeros(
        (B,),
        dtype=torch.int16,
        device=device
    )

    castling |= torch.where(
        states[:, 13, 0, 0] > 0.5,
        torch.tensor(
            WK,
            dtype=torch.int16,
            device=device
        ),
        torch.tensor(
            0,
            dtype=torch.int16,
            device=device
        )
    )

    castling |= torch.where(
        states[:, 14, 0, 0] > 0.5,
        torch.tensor(
            WQ,
            dtype=torch.int16,
            device=device
        ),
        torch.tensor(
            0,
            dtype=torch.int16,
            device=device
        )
    )

    castling |= torch.where(
        states[:, 15, 0, 0] > 0.5,
        torch.tensor(
            BK,
            dtype=torch.int16,
            device=device
        ),
        torch.tensor(
            0,
            dtype=torch.int16,
            device=device
        )
    )

    castling |= torch.where(
        states[:, 16, 0, 0] > 0.5,
        torch.tensor(
            BQ,
            dtype=torch.int16,
            device=device
        ),
        torch.tensor(
            0,
            dtype=torch.int16,
            device=device
        )
    )

    chess.castling = castling

    # --------------------------------------------------------
    # EN PASSANT
    # --------------------------------------------------------

    ep_plane = (
        states[:, 17]
        .flip(1)
        .reshape(B, 64)
        > 0.5
    )

    ep_exists = ep_plane.any(
        dim=1
    )

    ep_square = torch.full(
        (B,),
        -1,
        dtype=torch.int16,
        device=device
    )

    if ep_exists.any():

        rows = torch.nonzero(
            ep_exists,
            as_tuple=False
        ).flatten()

        squares = torch.argmax(
            ep_plane[rows].to(
                torch.int8
            ),
            dim=1
        )

        ep_square[rows] = (
            squares.to(
                torch.int16
            )
        )

    chess.ep_square = ep_square

    # --------------------------------------------------------
    # CLOCKS
    # --------------------------------------------------------

    # Replay states don't contain these.

    chess.halfmove_clock = torch.zeros(
        (B,),
        dtype=torch.int16,
        device=device
    )

    chess.fullmove_number = torch.ones(
        (B,),
        dtype=torch.int16,
        device=device
    )

    return chess


# ============================================================
# VERIFY RECONSTRUCTION
# ============================================================

def verify_reconstruction(
    samples,
    env
):

    reconstructed = (
        env.to_model_input()
    )

    original = torch.from_numpy(
        np.stack(
            [
                np.asarray(
                    sample[0],
                    dtype=np.float32
                )
                for sample in samples
            ]
        )
    ).to(
        DEVICE
    )

    max_difference = (
        reconstructed
        - original
    ).abs().max().item()

    mean_difference = (
        reconstructed
        - original
    ).abs().mean().item()

    print()
    print(
        "State reconstruction:"
    )

    print(
        f"Maximum difference: "
        f"{max_difference:.10f}"
    )

    print(
        f"Mean difference:    "
        f"{mean_difference:.10f}"
    )

    if max_difference > 1e-5:

        raise RuntimeError(
            "State reconstruction FAILED."
        )

    print(
        "State reconstruction verified."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_state(
    model,
    env
):

    model_input = (
        env.to_model_input()
    )

    if DEVICE.type == "cuda":

        model_input = (
            model_input.contiguous(
                memory_format=torch.channels_last
            )
        )

    with torch.inference_mode():

        if DEVICE.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits, value = model(
                    model_input
                )

        else:

            logits, value = model(
                model_input
            )

    return (
        logits.float(),
        value.reshape(-1).float()
    )


# ============================================================
# LEGAL ACTIONS
# ============================================================

def get_legal_actions(
    env
):

    legal = (
        env.legal_move_mask()
    )

    actions = torch.nonzero(
        legal[0],
        as_tuple=False
    ).flatten()

    return actions


# ============================================================
# CLONE ENV
# ============================================================

def clone_env(
    env
):

    return env.clone()


# ============================================================
# APPLY ONE ACTION
# ============================================================

def apply_action(
    env,
    action
):

    action_tensor = torch.tensor(
        [action],
        dtype=torch.long,
        device=DEVICE
    )

    # Current GPUChess API:
    #
    # push_actions(actions)

    child = env.push_actions(
        action_tensor
    )

    return child


# ============================================================
# TERMINAL INFORMATION
# ============================================================

def inspect_terminal(
    env
):

    info = {}

    # Current GPUChess has terminal/result helpers.
    # We try several possible names to keep this diagnostic
    # compatible.

    candidates = [
        "is_terminal",
        "terminal",
        "is_game_over",
    ]

    for name in candidates:

        if hasattr(
            env,
            name
        ):

            try:

                obj = getattr(
                    env,
                    name
                )

                value = (
                    obj()
                    if callable(obj)
                    else obj
                )

                if torch.is_tensor(
                    value
                ):

                    value = (
                        value
                        .detach()
                        .cpu()
                        .reshape(-1)
                        .tolist()
                    )

                info[
                    name
                ] = value

            except Exception as e:

                info[
                    name
                ] = f"ERROR: {e}"

    return info


# ============================================================
# STATE SUMMARY
# ============================================================

def print_state_summary(
    env
):

    print(
        "State metadata:"
    )

    attrs = [
        "turn",
        "castling",
        "ep_square",
        "halfmove_clock",
        "fullmove_number",
    ]

    for attr in attrs:

        if hasattr(
            env,
            attr
        ):

            value = getattr(
                env,
                attr
            )

            if torch.is_tensor(
                value
            ):

                value = (
                    value
                    .detach()
                    .cpu()
                    .tolist()
                )

            print(
                f"  {attr}: {value}"
            )


# ============================================================
# INSPECT CHILD POSITION
# ============================================================

def inspect_child(
    model,
    root_env,
    action,
    label
):

    banner(
        f"{label} — CHILD POSITION",
        "-"
    )

    print(
        f"Action: {action}"
    )

    child = clone_env(
        root_env
    )

    print()
    print(
        "ROOT STATE METADATA"
    )

    print_state_summary(
        child
    )

    print()
    print(
        "ROOT TERMINAL INFO:"
    )

    print(
        inspect_terminal(
            child
        )
    )

    # --------------------------------------------------------
    # Apply move
    # --------------------------------------------------------

    try:

        child = apply_action(
            child,
            action
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
    # Child metadata
    # --------------------------------------------------------

    print()
    print(
        "CHILD STATE METADATA"
    )

    print_state_summary(
        child
    )

    print()
    print(
        "CHILD TERMINAL INFO:"
    )

    print(
        inspect_terminal(
            child
        )
    )

    # --------------------------------------------------------
    # Child network
    # --------------------------------------------------------

    logits, values = (
        evaluate_state(
            model,
            child
        )
    )

    child_value = float(
        values[0].item()
    )

    legal_actions = (
        get_legal_actions(
            child
        )
    )

    print()
    print(
        "CHILD NETWORK"
    )

    print(
        f"Child NN value: "
        f"{child_value:+.6f}"
    )

    print(
        f"Child legal moves: "
        f"{len(legal_actions)}"
    )

    # --------------------------------------------------------
    # Child policy
    # --------------------------------------------------------

    if len(
        legal_actions
    ) > 0:

        legal_logits = (
            logits[0][legal_actions]
        )

        probs = torch.softmax(
            legal_logits,
            dim=0
        )

        order = torch.argsort(
            probs,
            descending=True
        )

        print()
        print(
            "Top child policy:"
        )

        print(
            "Rank | Action | Probability | Logit"
        )

        print(
            "-" * 65
        )

        for rank, idx in enumerate(
            order[:TOP_K].tolist(),
            start=1
        ):

            a = int(
                legal_actions[
                    idx
                ].item()
            )

            p = float(
                probs[idx].item()
            )

            l = float(
                legal_logits[
                    idx
                ].item()
            )

            print(
                f"{rank:4d} | "
                f"{a:6d} | "
                f"{p:.6f} | "
                f"{l:+.6f}"
            )

    return {
        "env": child,
        "value": child_value,
        "legal_actions": legal_actions,
    }


# ============================================================
# RUN MCTS
# ============================================================

def run_mcts(
    model,
    env,
    simulations
):

    search = GPUMCTS(
        model=model,
        device=DEVICE,
        c_puct=C_PUCT
    )

    root = clone_env(
        env
    )

    start = time.time()

    with torch.inference_mode():

        # IMPORTANT:
        #
        # Do NOT pass batch_size here.
        #
        # Current GPUMCTS.search() does not take it.

        search.search(
            root,
            num_simulations=simulations,
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0
        )

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    elapsed = (
        time.time()
        - start
    )

    return (
        search,
        elapsed
    )


# ============================================================
# EXTRACT ROOT STATISTICS
# ============================================================

def extract_root_stats(
    search
):

    root_id = int(
        search._root_ids[
            0
        ].item()
    )

    edge_start = int(
        search.edge_start[
            root_id
        ].item()
    )

    edge_count = int(
        search.edge_count[
            root_id
        ].item()
    )

    edge_end = (
        edge_start
        + edge_count
    )

    actions = (
        search.edge_action[
            edge_start:edge_end
        ]
    )

    priors = (
        search.edge_prior[
            edge_start:edge_end
        ]
    )

    visits = (
        search.visit_count[
            edge_start:edge_end
        ]
    )

    sums = (
        search.value_sum[
            edge_start:edge_end
        ]
    )

    root_visits = int(
        search.visit_count[
            root_id
        ].item()
    )

    # --------------------------------------------------------
    # Q
    # --------------------------------------------------------

    q = torch.zeros_like(
        sums,
        dtype=torch.float32
    )

    visited = (
        visits > 0
    )

    q[visited] = (
        sums[visited]
        / visits[
            visited
        ].float()
    )

    # --------------------------------------------------------
    # U
    # --------------------------------------------------------

    u = (
        C_PUCT
        * priors
        * math.sqrt(
            max(
                root_visits,
                1
            )
        )
        /
        (
            1.0
            + visits.float()
        )
    )

    # --------------------------------------------------------
    # PUCT
    # --------------------------------------------------------

    # Current implementation:
    #
    # score = -Q + U

    puct = (
        -q
        + u
    )

    return {
        "actions":
            actions.detach().cpu(),

        "priors":
            priors.detach().cpu(),

        "visits":
            visits.detach().cpu(),

        "q":
            q.detach().cpu(),

        "u":
            u.detach().cpu(),

        "puct":
            puct.detach().cpu(),

        "root_visits":
            root_visits,
    }


# ============================================================
# PRINT ROOT STATS
# ============================================================

def print_root_stats(
    stats,
    simulations,
    traced_action
):

    actions = stats[
        "actions"
    ]

    priors = stats[
        "priors"
    ]

    visits = stats[
        "visits"
    ]

    q = stats[
        "q"
    ]

    u = stats[
        "u"
    ]

    puct = stats[
        "puct"
    ]

    # --------------------------------------------------------
    # Top actions
    # --------------------------------------------------------

    order = torch.argsort(
        puct,
        descending=True
    )

    print()
    print(
        f"MCTS-{simulations}"
    )

    print(
        f"Root visits: "
        f"{stats['root_visits']}"
    )

    print()

    print(
        "Rank | Action | Prior P | "
        "Visits N | Q | U | PUCT"
    )

    print(
        "-" * 85
    )

    for rank, idx in enumerate(
        order[:TOP_K].tolist(),
        start=1
    ):

        print(
            f"{rank:4d} | "
            f"{int(actions[idx]):6d} | "
            f"{float(priors[idx]):.6f} | "
            f"{int(visits[idx]):8d} | "
            f"{float(q[idx]):+.6f} | "
            f"{float(u[idx]):+.6f} | "
            f"{float(puct[idx]):+.6f}"
        )

    # --------------------------------------------------------
    # Traced action
    # --------------------------------------------------------

    matches = torch.nonzero(
        actions == traced_action,
        as_tuple=False
    ).flatten()

    print()

    if len(matches) == 0:

        print(
            f"Traced action {traced_action}: "
            f"NOT FOUND"
        )

        return None

    idx = int(
        matches[0].item()
    )

    print(
        "TRACED ACTION"
    )

    print(
        f"Action: {traced_action}"
    )

    print(
        f"Prior:  "
        f"{float(priors[idx]):.6f}"
    )

    print(
        f"Visits: "
        f"{int(visits[idx])}"
    )

    print(
        f"Q:      "
        f"{float(q[idx]):+.6f}"
    )

    print(
        f"U:      "
        f"{float(u[idx]):+.6f}"
    )

    print(
        f"PUCT:   "
        f"{float(puct[idx]):+.6f}"
    )

    return idx


# ============================================================
# TRACE ONE CASE
# ============================================================

def trace_case(
    model,
    replay,
    label,
    replay_index,
    action
):

    banner(
        f"TRACE CASE: {label}"
    )

    print(
        f"Replay index: {replay_index}"
    )

    print(
        f"Root action:  {action}"
    )

    # --------------------------------------------------------
    # Reconstruct exact root state
    # --------------------------------------------------------

    sample = replay[
        replay_index
    ]

    root_env = replay_states_to_gpu_chess(
        [sample],
        DEVICE
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_reconstruction(
        [sample],
        root_env
    )

    # --------------------------------------------------------
    # Root network
    # --------------------------------------------------------

    root_logits, root_values = (
        evaluate_state(
            model,
            root_env
        )
    )

    root_value = float(
        root_values[0].item()
    )

    legal_actions = (
        get_legal_actions(
            root_env
        )
    )

    legal_set = set(
        int(x)
        for x in
        legal_actions.cpu().tolist()
    )

    print()
    print(
        "ROOT"
    )

    print(
        f"Network value: "
        f"{root_value:+.6f}"
    )

    print(
        f"Legal moves: "
        f"{len(legal_actions)}"
    )

    if action not in legal_set:

        print()
        print(
            "ERROR:"
        )

        print(
            f"Action {action} "
            "is NOT legal."
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
    # Direct child
    # --------------------------------------------------------

    child_info = inspect_child(
        model=model,
        root_env=root_env,
        action=action,
        label=label
    )

    if child_info is None:

        return

    child_value = (
        child_info["value"]
    )

    child_env = (
        child_info["env"]
    )

    print()
    print(
        "VALUE COMPARISON"
    )

    print(
        f"Root NN value:  "
        f"{root_value:+.6f}"
    )

    print(
        f"Child NN value: "
        f"{child_value:+.6f}"
    )

    # --------------------------------------------------------
    # Run MCTS
    # --------------------------------------------------------

    results = {}

    for simulations in (
        MCTS_SIMULATIONS
    ):

        print()
        print(
            "=" * 90
        )

        print(
            f"RUNNING MCTS: "
            f"{simulations} SIMULATIONS"
        )

        print(
            "=" * 90
        )

        try:

            search, elapsed = (
                run_mcts(
                    model,
                    root_env,
                    simulations
                )
            )

            stats = (
                extract_root_stats(
                    search
                )
            )

            results[
                simulations
            ] = stats

            print(
                f"Elapsed: "
                f"{elapsed:.3f}s"
            )

            print_root_stats(
                stats,
                simulations,
                action
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
    # Cross-depth summary
    # --------------------------------------------------------

    banner(
        f"CROSS-DEPTH SUMMARY — {label}",
        "-"
    )

    print(
        "Simulation | Visits | Q | U | PUCT"
    )

    print(
        "-" * 70
    )

    for simulations in (
        MCTS_SIMULATIONS
    ):

        if simulations not in results:
            continue

        stats = results[
            simulations
        ]

        actions = stats[
            "actions"
        ]

        matches = torch.nonzero(
            actions == action,
            as_tuple=False
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

        print(
            f"{simulations:10d} | "
            f"{int(stats['visits'][idx]):6d} | "
            f"{float(stats['q'][idx]):+.6f} | "
            f"{float(stats['u'][idx]):+.6f} | "
            f"{float(stats['puct'][idx]):+.6f}"
        )


# ============================================================
# MAIN
# ============================================================

banner(
    "RL53 DEEP MCTS CHILD-VALUE / BACKUP DIAGNOSTIC"
)

print(
    "Project:",
    PROJECT_DIR
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

print(
    "Replay:",
    REPLAY_PATH
)

print(
    "Device:",
    DEVICE
)

print(
    "MCTS simulations:",
    MCTS_SIMULATIONS
)

print(
    "c_puct:",
    C_PUCT
)

print(
    "Dirichlet: OFF"
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ============================================================
# LOAD
# ============================================================

model = load_model()

replay = load_replay()


# ============================================================
# RUN ALL CASES
# ============================================================

for (
    label,
    replay_index,
    action
) in TRACE_CASES:

    try:

        trace_case(
            model=model,
            replay=replay,
            label=label,
            replay_index=replay_index,
            action=action
        )

    except Exception as e:

        print()
        print(
            "#" * 100
        )

        print(
            f"FAILED CASE: {label}"
        )

        print(
            repr(e)
        )

        traceback.print_exc()


# ============================================================
# FINISHED
# ============================================================

banner(
    "RL53 DEEP MCTS DIAGNOSTIC FINISHED"
)

print(
    """
IMPORTANT:

Do NOT train RL54 yet.

We specifically want to determine whether the instability
comes from:

    1. Neural-network value
    2. Child-state evaluation
    3. Backup sign
    4. Terminal handling
    5. PUCT selection
    6. Tree expansion

The most important cases are:

    P2-A2472
    P3-A263
    P3-A191
    P4-A1463
    P5-A3186
"""
)