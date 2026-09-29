# ============================================================
# RL53 MCTS ROOT STATISTICS DIAGNOSTIC
#
# PURPOSE
# -------
# Investigate WHY certain positions change substantially
# between MCTS-100 / 200 / 400 / 800.
#
# For the same exact replay positions, inspect:
#
#   P = neural-network prior
#   N = visit count
#   Q = backed-up value
#   U = exploration term
#   PUCT = -Q + U
#
# Tested at:
#
#   100 simulations
#   200 simulations
#   400 simulations
#   800 simulations
#
# Dirichlet noise:
#   OFF
#
# Positions:
#   The 5 most unstable positions from the previous
#   RL53 convergence diagnostic.
#
# ============================================================


import os
import sys
import random
import time

import numpy as np
import torch


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = "/kaggle/working/chess-zero"

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "rl_iteration_53.pt"
)

REPLAY_BUFFER = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "replay_buffer_rl53.pt"
)


# ------------------------------------------------------------
# IMPORTANT:
#
# These are the exact replay indices corresponding to the
# unstable positions from the previous diagnostic.
# ------------------------------------------------------------

TARGET_REPLAY_INDICES = [
    147127,   # Previous Position 18
    8331,     # Previous Position 11
    110785,   # Previous Position 27
    29184,    # Previous Position 01
    56443,    # Previous Position 31
]


MCTS_SIMULATIONS = [
    100,
    200,
    400,
    800
]


MCTS_BATCH_SIZE = 16

TOP_K = 10

C_PUCT = 1.5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("RL53 MCTS ROOT STATISTICS DIAGNOSTIC")
print("=" * 100)

print(
    f"Project:       {PROJECT_ROOT}"
)

print(
    f"Checkpoint:    {CHECKPOINT}"
)

print(
    f"Replay:        {REPLAY_BUFFER}"
)

print(
    f"Device:        {DEVICE}"
)

print(
    f"Positions:     {len(TARGET_REPLAY_INDICES)}"
)

print(
    f"MCTS tests:    {MCTS_SIMULATIONS}"
)

print(
    f"Batch size:    {MCTS_BATCH_SIZE}"
)

print(
    f"Top-K:         {TOP_K}"
)

print(
    f"c_puct:        {C_PUCT}"
)

print(
    "Dirichlet:     OFF"
)

print("=" * 100)


# ============================================================
# CUDA
# ============================================================

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

else:

    print(
        "WARNING: CUDA is not available."
    )


torch.set_grad_enabled(False)


# ============================================================
# IMPORT PROJECT
# ============================================================

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 100)
print("LOADING RL53 MODEL")
print("=" * 100)


model = ChessNet().to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)


print(
    "Checkpoint type:",
    type(checkpoint)
)


# ------------------------------------------------------------
# Extract state dict
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint["state_dict"]
        )

    elif (
        "model" in checkpoint
        and isinstance(
            checkpoint["model"],
            dict
        )
    ):

        state_dict = (
            checkpoint["model"]
        )

    else:

        state_dict = checkpoint

else:

    raise RuntimeError(
        f"Unsupported checkpoint format: "
        f"{type(checkpoint)}"
    )


# ------------------------------------------------------------
# Remove torch.compile prefix
# ------------------------------------------------------------

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith(
        "_orig_mod."
    ):

        key = key[
            len("_orig_mod.") :
        ]

    clean_state_dict[key] = value


missing, unexpected = (
    model.load_state_dict(
        clean_state_dict,
        strict=False
    )
)


print(
    f"Missing keys:       {len(missing)}"
)

print(
    f"Unexpected keys:    {len(unexpected)}"
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
    "\nRL53 model loaded successfully."
)


# ============================================================
# LOAD REPLAY
# ============================================================

print("\n" + "=" * 100)
print("LOADING RL53 REPLAY BUFFER")
print("=" * 100)


replay = torch.load(
    REPLAY_BUFFER,
    map_location="cpu",
    weights_only=False
)


print(
    "Replay type:",
    type(replay)
)

print(
    "Replay size:",
    len(replay)
)


# ------------------------------------------------------------
# Verify requested indices
# ------------------------------------------------------------

for idx in TARGET_REPLAY_INDICES:

    if idx < 0 or idx >= len(replay):

        raise IndexError(
            f"Replay index {idx} "
            f"is outside replay buffer."
        )


# ============================================================
# SELECT EXACT POSITIONS
# ============================================================

samples = [
    replay[idx]
    for idx in TARGET_REPLAY_INDICES
]


print("\nSelected positions:")

for i, idx in enumerate(
    TARGET_REPLAY_INDICES
):

    print(
        f"  Position {i + 1}: "
        f"replay index {idx}"
    )


# ============================================================
# RECONSTRUCT GPUChess
# ============================================================

def replay_states_to_gpu_chess(
    samples,
    device
):

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


    chess = GPUChess(
        device=device,
        batch_size=B
    )


    # ========================================================
    # PIECE PLANES
    # ========================================================

    board_planes = (
        states[:, :12]
        .flip(2)
    )


    square_bits = (
        chess.square_bits
    )


    pieces = torch.zeros(
        (B, 12),
        dtype=torch.int64,
        device=device
    )


    for piece_type in range(12):

        occupancy = (
            board_planes[
                :,
                piece_type
            ]
            .reshape(B, 64)
            > 0.5
        )


        bitboard = torch.zeros(
            (B,),
            dtype=torch.int64,
            device=device
        )


        for square in range(64):

            bitboard = torch.where(
                occupancy[:, square],
                bitboard
                | square_bits[square],
                bitboard
            )


        pieces[
            :,
            piece_type
        ] = bitboard


    chess.pieces = pieces


    # ========================================================
    # SIDE TO MOVE
    # ========================================================

    white_to_move = (
        states[:, 12, 0, 0]
        > 0.5
    )


    chess.turn = (
        ~white_to_move
    )


    # ========================================================
    # CASTLING
    # ========================================================

    castling = torch.zeros(
        (B,),
        dtype=torch.int16,
        device=device
    )


    WK = 1
    WQ = 2
    BK = 4
    BQ = 8


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


    # ========================================================
    # EN PASSANT
    # ========================================================

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


    if bool(ep_exists.any()):

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


    # ========================================================
    # CLOCKS
    # ========================================================

    chess.halfmove_clock = (
        torch.zeros(
            (B,),
            dtype=torch.int16,
            device=device
        )
    )


    chess.fullmove_number = (
        torch.ones(
            (B,),
            dtype=torch.int16,
            device=device
        )
    )


    return chess


# ============================================================
# BUILD POSITIONS
# ============================================================

print("\n" + "=" * 100)
print("RECONSTRUCTING POSITIONS")
print("=" * 100)


root_states = (
    replay_states_to_gpu_chess(
        samples,
        DEVICE
    )
)


print(
    "Reconstruction complete."
)


# ============================================================
# VERIFY EXACT STATE
# ============================================================

print(
    "\nVerifying state reconstruction..."
)


reconstructed = (
    root_states.to_model_input()
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
).to(DEVICE)


max_diff = (
    reconstructed
    - original
).abs().max().item()


mean_diff = (
    reconstructed
    - original
).abs().mean().item()


print(
    f"Maximum difference: "
    f"{max_diff:.10f}"
)

print(
    f"Mean difference:    "
    f"{mean_diff:.10f}"
)


if max_diff > 1e-5:

    raise RuntimeError(
        "State reconstruction failed."
    )


print(
    "State reconstruction verified."
)


# ============================================================
# LEGAL MOVE MASK
# ============================================================

print(
    "\nGenerating legal moves..."
)


legal_mask = (
    root_states.legal_move_mask()
)


legal_counts = (
    legal_mask
    .sum(dim=1)
)


for i in range(
    len(TARGET_REPLAY_INDICES)
):

    print(
        f"Position {i + 1}: "
        f"{int(legal_counts[i].item())} "
        f"legal moves"
    )


# ============================================================
# NETWORK ROOT EVALUATION
# ============================================================

print("\n" + "=" * 100)
print("NETWORK ROOT EVALUATION")
print("=" * 100)


with torch.inference_mode():

    model_input = (
        root_states.to_model_input()
    )


    if DEVICE.type == "cuda":

        model_input = (
            model_input.contiguous(
                memory_format=torch.channels_last
            )
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            network_logits, network_values = (
                model(model_input)
            )

    else:

        network_logits, network_values = (
            model(model_input)
        )


network_logits = (
    network_logits.float()
)

network_values = (
    network_values
    .squeeze(-1)
    .float()
)


# ------------------------------------------------------------
# Mask to legal moves
# ------------------------------------------------------------

masked_logits = (
    network_logits.masked_fill(
        ~legal_mask,
        torch.finfo(
            network_logits.dtype
        ).min
    )
)


network_policy = (
    torch.softmax(
        masked_logits,
        dim=1
    )
)


network_policy = (
    network_policy
    * legal_mask.float()
)


network_policy = (
    network_policy
    /
    network_policy.sum(
        dim=1,
        keepdim=True
    ).clamp_min(1e-12)
)


for i in range(
    len(TARGET_REPLAY_INDICES)
):

    print(
        f"\nPosition {i + 1}"
    )

    print(
        f"Replay index: "
        f"{TARGET_REPLAY_INDICES[i]}"
    )

    print(
        f"Network value: "
        f"{network_values[i].item():+.6f}"
    )


# ============================================================
# MCTS ROOT STAT EXTRACTION
# ============================================================

def extract_root_statistics(
    search,
    position_idx,
    top_k=10
):

    """
    Extract root-child statistics directly from the
    GPUMCTS internal tree.

    Returns:

        action
        prior P
        visits N
        Q
        U
        PUCT
    """


    # --------------------------------------------------------
    # Root IDs
    # --------------------------------------------------------

    roots = search._root_ids


    root_id = (
        roots[position_idx]
        .item()
    )


    # --------------------------------------------------------
    # Root edge range
    # --------------------------------------------------------

    start = int(
        search.edge_start[
            root_id
        ].item()
    )


    count = int(
        search.edge_count[
            root_id
        ].item()
    )


    if count <= 0:

        return []


    edge_ids = torch.arange(
        start,
        start + count,
        device=DEVICE,
        dtype=torch.long
    )


    # --------------------------------------------------------
    # Child nodes
    # --------------------------------------------------------

    child_nodes = (
        search.edge_child[
            edge_ids
        ]
        .to(torch.long)
    )


    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    actions = (
        search.edge_action[
            edge_ids
        ]
        .to(torch.long)
    )


    # --------------------------------------------------------
    # Prior P
    # --------------------------------------------------------

    priors = (
        search.edge_prior[
            edge_ids
        ]
        .float()
    )


    # --------------------------------------------------------
    # Visit count N
    # --------------------------------------------------------

    visits = (
        search.visit_count[
            child_nodes
        ]
        .float()
    )


    # --------------------------------------------------------
    # Value sum
    # --------------------------------------------------------

    value_sum = (
        search.value_sum[
            child_nodes
        ]
        .float()
    )


    # --------------------------------------------------------
    # Q value
    #
    # Q = value_sum / visits
    # --------------------------------------------------------

    q_values = torch.where(
        visits > 0,
        value_sum / visits,
        torch.zeros_like(
            value_sum
        )
    )


    # --------------------------------------------------------
    # Parent visit count
    # --------------------------------------------------------

    parent_visits = (
        search.visit_count[
            root_id
        ]
        .float()
        .clamp_min(1.0)
    )


    # --------------------------------------------------------
    # Exploration term U
    #
    # U =
    # c_puct * P * sqrt(N_parent) / (1 + N_child)
    # --------------------------------------------------------

    exploration = (
        C_PUCT
        * priors
        * torch.sqrt(
            parent_visits
        )
        /
        (1.0 + visits)
    )


    # --------------------------------------------------------
    # PUCT score
    #
    # Matches the search implementation:
    #
    # score = -Q + U
    # --------------------------------------------------------

    puct_scores = (
        -q_values
        + exploration
    )


    # --------------------------------------------------------
    # Sort by visit count
    # --------------------------------------------------------

    order = torch.argsort(
        visits,
        descending=True
    )


    order = order[
        :min(
            top_k,
            count
        )
    ]


    results = []


    for j in order.tolist():

        results.append({

            "action":
                int(
                    actions[j].item()
                ),

            "prior":
                float(
                    priors[j].item()
                ),

            "visits":
                int(
                    visits[j].item()
                ),

            "q":
                float(
                    q_values[j].item()
                ),

            "u":
                float(
                    exploration[j].item()
                ),

            "puct":
                float(
                    puct_scores[j].item()
                )
        })


    return results


# ============================================================
# RUN MCTS
# ============================================================

all_searches = {}

all_stats = {}


for simulations in (
    MCTS_SIMULATIONS
):

    print("\n\n" + "=" * 100)

    print(
        f"RUNNING MCTS "
        f"WITH {simulations} SIMULATIONS"
    )

    print("=" * 100)


    search = GPUMCTS(
        model=model,
        device=DEVICE,
        c_puct=C_PUCT
    )


    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        start_event.record()

    else:

        start_time = time.perf_counter()


    with torch.inference_mode():

        search.search(
            root_states.clone(),
            num_simulations=simulations,
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0,
            batch_size=MCTS_BATCH_SIZE
        )


    if DEVICE.type == "cuda":

        end_event.record()

        torch.cuda.synchronize()

        elapsed_ms = (
            start_event.elapsed_time(
                end_event
            )
        )

    else:

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0


    all_searches[
        simulations
    ] = search


    all_stats[
        simulations
    ] = {}


    print(
        f"Elapsed: "
        f"{elapsed_ms / 1000:.2f}s"
    )


    # --------------------------------------------------------
    # Extract root statistics
    # --------------------------------------------------------

    for position_idx in range(
        len(TARGET_REPLAY_INDICES)
    ):

        stats = (
            extract_root_statistics(
                search,
                position_idx,
                TOP_K
            )
        )


        all_stats[
            simulations
        ][position_idx] = stats


# ============================================================
# PRINT ROOT STATISTICS
# ============================================================

for position_idx, replay_idx in enumerate(
    TARGET_REPLAY_INDICES
):

    print("\n\n" + "#" * 100)

    print(
        f"POSITION {position_idx + 1}"
    )

    print(
        f"Replay index: {replay_idx}"
    )

    print(
        f"Legal moves: "
        f"{int(legal_counts[position_idx].item())}"
    )

    print(
        f"Network value: "
        f"{network_values[position_idx].item():+.6f}"
    )

    print("#" * 100)


    for simulations in (
        MCTS_SIMULATIONS
    ):

        search = (
            all_searches[
                simulations
            ]
        )


        stats = (
            all_stats[
                simulations
            ][position_idx]
        )


        root_id = (
            search._root_ids[
                position_idx
            ]
            .item()
        )


        root_visits = int(
            search.visit_count[
                root_id
            ].item()
        )


        root_value_sum = float(
            search.value_sum[
                root_id
            ].item()
        )


        root_q = (
            root_value_sum
            / root_visits
            if root_visits > 0
            else 0.0
        )


        print(
            "\n"
            + "-" * 100
        )

        print(
            f"MCTS-{simulations}"
        )

        print(
            f"Root visits: "
            f"{root_visits}"
        )

        print(
            f"Root value sum: "
            f"{root_value_sum:+.6f}"
        )

        print(
            f"Root Q: "
            f"{root_q:+.6f}"
        )


        print(
            "\n"
            "Rank | Action | Prior P | "
            "Visits N | Q value | "
            "U | PUCT"
        )

        print(
            "-" * 100
        )


        for rank, item in enumerate(
            stats,
            start=1
        ):

            print(
                f"{rank:4d} | "
                f"{item['action']:6d} | "
                f"{item['prior']:.6f} | "
                f"{item['visits']:8d} | "
                f"{item['q']:+.6f} | "
                f"{item['u']:.6f} | "
                f"{item['puct']:+.6f}"
            )


# ============================================================
# CROSS-DEPTH MOVE TRACKING
# ============================================================

print("\n\n" + "=" * 100)
print("CROSS-DEPTH MOVE TRACKING")
print("=" * 100)


for position_idx, replay_idx in enumerate(
    TARGET_REPLAY_INDICES
):

    print(
        "\n"
        + "#" * 100
    )

    print(
        f"POSITION {position_idx + 1} "
        f"(replay {replay_idx})"
    )

    print(
        "#" * 100
    )


    # Collect all actions appearing in
    # top-K at any search depth.

    action_set = set()


    for simulations in (
        MCTS_SIMULATIONS
    ):

        for item in all_stats[
            simulations
        ][position_idx]:

            action_set.add(
                item["action"]
            )


    # --------------------------------------------------------
    # Print each important action across depths
    # --------------------------------------------------------

    for action in sorted(
        action_set
    ):

        print(
            "\nAction:",
            action
        )


        for simulations in (
            MCTS_SIMULATIONS
        ):

            stats = (
                all_stats[
                    simulations
                ][position_idx]
            )


            found = None


            for item in stats:

                if (
                    item["action"]
                    == action
                ):

                    found = item
                    break


            if found is None:

                print(
                    f"  {simulations:3d}: "
                    f"not top-{TOP_K}"
                )

            else:

                print(
                    f"  {simulations:3d}: "
                    f"N={found['visits']:4d}, "
                    f"P={found['prior']:.4f}, "
                    f"Q={found['q']:+.4f}, "
                    f"U={found['u']:.4f}, "
                    f"PUCT={found['puct']:+.4f}"
                )


# ============================================================
# TOP-1 TRACKING
# ============================================================

print("\n\n" + "=" * 100)
print("TOP-1 MOVE TRACKING")
print("=" * 100)


for position_idx, replay_idx in enumerate(
    TARGET_REPLAY_INDICES
):

    print(
        f"\nPosition {position_idx + 1} "
        f"(replay {replay_idx})"
    )


    for simulations in (
        MCTS_SIMULATIONS
    ):

        stats = (
            all_stats[
                simulations
            ][position_idx]
        )


        if not stats:

            print(
                f"  {simulations}: "
                f"NO ROOT CHILDREN"
            )

            continue


        top = stats[0]


        print(
            f"  {simulations:3d}: "
            f"action={top['action']:4d}, "
            f"N={top['visits']:4d}, "
            f"P={top['prior']:.5f}, "
            f"Q={top['q']:+.5f}, "
            f"U={top['u']:.5f}, "
            f"PUCT={top['puct']:+.5f}"
        )


# ============================================================
# TOP-1 CHANGES
# ============================================================

print("\n\n" + "=" * 100)
print("TOP-1 CHANGES")
print("=" * 100)


for position_idx, replay_idx in enumerate(
    TARGET_REPLAY_INDICES
):

    top_actions = []


    for simulations in (
        MCTS_SIMULATIONS
    ):

        stats = (
            all_stats[
                simulations
            ][position_idx]
        )


        if stats:

            top_actions.append(
                stats[0]["action"]
            )

        else:

            top_actions.append(
                None
            )


    print(
        f"\nPosition {position_idx + 1} "
        f"(replay {replay_idx})"
    )


    print(
        f"  100: {top_actions[0]}"
    )

    print(
        f"  200: {top_actions[1]}"
    )

    print(
        f"  400: {top_actions[2]}"
    )

    print(
        f"  800: {top_actions[3]}"
    )


# ============================================================
# ROOT VALUE COMPARISON
# ============================================================

print("\n\n" + "=" * 100)
print("ROOT VALUE COMPARISON")
print("=" * 100)


print(
    f"{'Position':>10} "
    f"{'Replay':>10} "
    f"{'Network':>12} "
    f"{'MCTS100':>12} "
    f"{'MCTS200':>12} "
    f"{'MCTS400':>12} "
    f"{'MCTS800':>12}"
)


print("-" * 90)


for position_idx, replay_idx in enumerate(
    TARGET_REPLAY_INDICES
):

    values = []


    for simulations in (
        MCTS_SIMULATIONS
    ):

        search = (
            all_searches[
                simulations
            ]
        )


        root_id = (
            search._root_ids[
                position_idx
            ]
            .item()
        )


        visits = float(
            search.visit_count[
                root_id
            ].item()
        )


        value_sum = float(
            search.value_sum[
                root_id
            ].item()
        )


        root_q = (
            value_sum / visits
            if visits > 0
            else 0.0
        )


        values.append(
            root_q
        )


    print(
        f"{position_idx + 1:>10} "
        f"{replay_idx:>10} "
        f"{network_values[position_idx].item():>+12.5f} "
        f"{values[0]:>+12.5f} "
        f"{values[1]:>+12.5f} "
        f"{values[2]:>+12.5f} "
        f"{values[3]:>+12.5f}"
    )


# ============================================================
# FINAL DIAGNOSTIC GUIDE
# ============================================================

print("\n\n" + "=" * 100)
print("WHAT WE ARE LOOKING FOR")
print("=" * 100)


print(
"""
For each unstable position, inspect:

    Prior P
    Visits N
    Q
    U
    PUCT


IMPORTANT:

PUCT in this implementation is:

    PUCT = -Q + c_puct * P * sqrt(N_parent) / (1 + N_child)


So:

    P       = neural network prior
    Q       = backed-up value estimate
    U       = exploration pressure
    N       = search visits


------------------------------------------------------------
CASE 1 — HEALTHY SEARCH
------------------------------------------------------------

A move may start with:

    high P
    low N

and then gain visits because its Q value looks good.

As simulations increase, its N should generally become
more concentrated if the search is finding something useful.


------------------------------------------------------------
CASE 2 — VALUE-DRIVEN RE-RANKING
------------------------------------------------------------

If:

    P is relatively low
    but Q becomes substantially better

and that move gains many visits,

then MCTS is using the value network to override the
policy prior.

That is potentially useful search behavior.


------------------------------------------------------------
CASE 3 — VALUE INSTABILITY
------------------------------------------------------------

If the same move's Q changes dramatically:

    MCTS-100
    MCTS-200
    MCTS-400
    MCTS-800

and this causes the top move to change,

then we need to investigate the value estimates.


------------------------------------------------------------
CASE 4 — PRIOR / PUCT DOMINATION
------------------------------------------------------------

If Q values are very similar between moves but one move
continually wins because of P and U,

then the behavior is more strongly driven by the policy
prior and exploration term.


------------------------------------------------------------
CASE 5 — SEARCH INSTABILITY
------------------------------------------------------------

If:

    Q values oscillate
    top move changes
    visit distribution changes strongly
    and the same position does not settle by 800

then we should inspect:

    - value network
    - backup signs
    - PUCT implementation
    - terminal handling
    - tree expansion
    - repeated-state handling


DO NOT CHANGE RL54 YET.

This test is specifically intended to tell us WHICH PART
of the search is responsible for the instability.
"""
)


print(
    "\n" + "=" * 100
)

print(
    "RL53 ROOT STATISTICS DIAGNOSTIC FINISHED"
)

print(
    "=" * 100
)