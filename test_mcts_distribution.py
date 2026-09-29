# ============================================================
# RL53 MCTS CONVERGENCE DIAGNOSTIC
#
# Purpose:
#   Test whether MCTS converges as simulations increase.
#
# Exact same RL53 replay-buffer positions are used for:
#
#   MCTS-100
#   MCTS-200
#   MCTS-400
#   MCTS-800
#
# Dirichlet noise: OFF
#
# Main comparisons:
#
#   MCTS-100 vs MCTS-200
#   MCTS-200 vs MCTS-400
#   MCTS-400 vs MCTS-800
#
# Metrics:
#   - Entropy
#   - Normalized entropy
#   - Max visit probability
#   - KL divergence
#   - Symmetric KL
#   - Pearson correlation
#   - Spearman correlation
#   - Top-1 agreement
#   - Top-3 agreement
#
# ============================================================


import os
import sys
import math
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

NUM_POSITIONS = 32

# The important part of this diagnostic.
MCTS_SIMULATIONS = [100, 200, 400, 800]

MCTS_BATCH_SIZE = 16

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 80)
print("RL53 MCTS CONVERGENCE DIAGNOSTIC")
print("=" * 80)

print(f"Project:       {PROJECT_ROOT}")
print(f"Checkpoint:    {CHECKPOINT}")
print(f"Replay:        {REPLAY_BUFFER}")
print(f"Device:        {DEVICE}")
print(f"Positions:     {NUM_POSITIONS}")
print(f"MCTS tests:    {MCTS_SIMULATIONS}")
print(f"Batch size:    {MCTS_BATCH_SIZE}")
print("Dirichlet:     OFF")
print("=" * 80)


# ============================================================
# CUDA INFORMATION
# ============================================================

if DEVICE.type == "cuda":

    print(
        f"GPU:           "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA:          "
        f"{torch.version.cuda}"
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

else:

    print("WARNING: CUDA is not available.")


torch.set_grad_enabled(False)


# ============================================================
# IMPORT PROJECT
# ============================================================

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 80)
print("LOADING RL53 MODEL")
print("=" * 80)


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
# Handle checkpoint formats
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    elif (
        "model" in checkpoint
        and isinstance(checkpoint["model"], dict)
    ):

        state_dict = checkpoint["model"]

    else:

        state_dict = checkpoint

else:

    raise RuntimeError(
        f"Unsupported checkpoint format: "
        f"{type(checkpoint)}"
    )


# ------------------------------------------------------------
# Remove torch.compile prefix if present
# ------------------------------------------------------------

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("_orig_mod."):

        key = key[len("_orig_mod."):]

    clean_state_dict[key] = value


missing, unexpected = model.load_state_dict(
    clean_state_dict,
    strict=False
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

print("\nRL53 model loaded successfully.")


# ============================================================
# LOAD REPLAY BUFFER
# ============================================================

print("\n" + "=" * 80)
print("LOADING RL53 REPLAY BUFFER")
print("=" * 80)


replay = torch.load(
    REPLAY_BUFFER,
    map_location="cpu",
    weights_only=False
)


print(
    "Replay object type:",
    type(replay)
)

print(
    "Replay size:",
    len(replay)
)


if len(replay) < NUM_POSITIONS:

    raise RuntimeError(
        f"Replay buffer only contains "
        f"{len(replay)} samples."
    )


# ============================================================
# INSPECT SAMPLE
# ============================================================

sample0 = replay[0]

print("\nSample structure:")
print("Type:", type(sample0))
print("Length:", len(sample0))


for i, x in enumerate(sample0):

    if hasattr(x, "shape"):

        print(
            f"[{i}] "
            f"shape={x.shape}, "
            f"dtype={x.dtype}"
        )

    else:

        print(
            f"[{i}] "
            f"type={type(x)}, "
            f"value={x}"
        )


# ============================================================
# REPRODUCIBLE SAMPLING
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


if DEVICE.type == "cuda":

    torch.cuda.manual_seed_all(SEED)


indices = random.sample(
    range(len(replay)),
    NUM_POSITIONS
)


samples = [
    replay[i]
    for i in indices
]


print("\nSelected replay positions:")
print(indices)


# ============================================================
# REPLAY STATE -> GPUChess
# ============================================================

def replay_states_to_gpu_chess(
    samples,
    device
):

    states_np = np.stack(
        [
            np.asarray(
                s[0],
                dtype=np.float32
            )
            for s in samples
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

    #
    # Stored neural-network representation:
    #
    #   [18, 8, 8]
    #
    # GPUChess:
    #
    #   square = rank * 8 + file
    #
    # Model representation has vertically flipped
    # board coordinates, therefore flip row dimension.
    #

    board_planes = states[
        :, :12
    ].flip(2)


    square_bits = chess.square_bits


    pieces = torch.zeros(
        (B, 12),
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


    # ========================================================
    # SIDE TO MOVE
    # ========================================================

    white_to_move = (
        states[:, 12, 0, 0]
        > 0.5
    )


    # GPUChess:
    #
    #   False = white
    #   True  = black
    #

    chess.turn = ~white_to_move


    # ========================================================
    # CASTLING RIGHTS
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


    if ep_exists.any():

        rows = torch.nonzero(
            ep_exists,
            as_tuple=False
        ).flatten()


        squares = torch.argmax(
            ep_plane[rows].to(torch.int8),
            dim=1
        )


        ep_square[rows] = (
            squares.to(torch.int16)
        )


    chess.ep_square = ep_square


    # ========================================================
    # CLOCKS
    # ========================================================

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
# RECONSTRUCT POSITIONS
# ============================================================

print("\n" + "=" * 80)
print("RECONSTRUCTING GPU CHESS POSITIONS")
print("=" * 80)


root_states = replay_states_to_gpu_chess(
    samples,
    DEVICE
)


print("Reconstruction complete.")


# ============================================================
# VERIFY RECONSTRUCTION
# ============================================================

print("\nVerifying reconstruction...")


reconstructed_input = (
    root_states.to_model_input()
)


original_states = torch.from_numpy(
    np.stack(
        [
            np.asarray(
                s[0],
                dtype=np.float32
            )
            for s in samples
        ]
    )
).to(DEVICE)


max_difference = (
    reconstructed_input
    - original_states
).abs().max().item()


mean_difference = (
    reconstructed_input
    - original_states
).abs().mean().item()


print(
    f"Maximum state difference: "
    f"{max_difference:.10f}"
)

print(
    f"Mean state difference:    "
    f"{mean_difference:.10f}"
)


if max_difference > 1e-5:

    raise RuntimeError(
        "State reconstruction is NOT identical. "
        "Do not continue."
    )


print(
    "State reconstruction verified."
)


# ============================================================
# LEGAL MOVES
# ============================================================

print("\nGenerating legal moves...")


legal_mask = (
    root_states.legal_move_mask()
)


legal_counts = (
    legal_mask
    .sum(dim=1)
    .float()
)


print(
    f"Average legal moves: "
    f"{legal_counts.mean().item():.2f}"
)

print(
    f"Minimum legal moves: "
    f"{legal_counts.min().item():.0f}"
)

print(
    f"Maximum legal moves: "
    f"{legal_counts.max().item():.0f}"
)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

EPS = 1e-12


def entropy(policy):

    p = policy.clamp_min(EPS)

    return -(
        p * torch.log(p)
    ).sum(dim=1)


def normalized_entropy(
    policy,
    legal_counts
):

    h = entropy(policy)

    denominator = torch.log(
        legal_counts.clamp_min(2)
    )

    return h / denominator


def max_probability(policy):

    return policy.max(
        dim=1
    ).values


def kl_divergence(
    p,
    q
):

    p = p.clamp_min(EPS)
    q = q.clamp_min(EPS)

    return (
        p * (
            torch.log(p)
            - torch.log(q)
        )
    ).sum(dim=1)


def symmetric_kl(
    p,
    q
):

    return (
        kl_divergence(p, q)
        + kl_divergence(q, p)
    ) * 0.5


def correlation(
    p,
    q
):

    p_mean = p.mean(
        dim=1,
        keepdim=True
    )

    q_mean = q.mean(
        dim=1,
        keepdim=True
    )


    p_centered = (
        p - p_mean
    )

    q_centered = (
        q - q_mean
    )


    numerator = (
        p_centered
        * q_centered
    ).sum(dim=1)


    denominator = torch.sqrt(
        (
            p_centered ** 2
        ).sum(dim=1)
        *
        (
            q_centered ** 2
        ).sum(dim=1)
    ).clamp_min(EPS)


    return (
        numerator
        / denominator
    )


def rankdata_torch(x):

    order = torch.argsort(
        x,
        dim=1
    )


    ranks = torch.zeros_like(
        x,
        dtype=torch.float32
    )


    rank_values = torch.arange(
        x.shape[1],
        device=x.device,
        dtype=torch.float32
    )


    ranks.scatter_(
        1,
        order,
        rank_values.unsqueeze(0)
        .expand_as(x)
    )


    return ranks


def spearman_correlation(
    p,
    q
):

    rp = rankdata_torch(p)
    rq = rankdata_torch(q)

    return correlation(
        rp,
        rq
    )


def top1_agreement(
    p,
    q
):

    return (
        p.argmax(dim=1)
        ==
        q.argmax(dim=1)
    ).float()


def top3_agreement(
    p,
    q
):

    p_top3 = torch.topk(
        p,
        k=3,
        dim=1
    ).indices


    q_top3 = torch.topk(
        q,
        k=3,
        dim=1
    ).indices


    results = []


    for i in range(
        p.shape[0]
    ):

        a = set(
            p_top3[i].tolist()
        )

        b = set(
            q_top3[i].tolist()
        )

        results.append(
            len(a.intersection(b))
            / 3.0
        )


    return torch.tensor(
        results,
        device=p.device,
        dtype=torch.float32
    )


# ============================================================
# RUN MCTS
# ============================================================

def run_mcts(
    root_states,
    simulations
):

    print("\n" + "=" * 80)
    print(
        f"RUNNING MCTS — "
        f"{simulations} SIMULATIONS"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Create a fresh MCTS tree for EVERY simulation count.
    #
    # This guarantees that:
    #
    # MCTS-100
    # MCTS-200
    # MCTS-400
    # MCTS-800
    #
    # are independent searches from exactly the same roots.
    # --------------------------------------------------------

    search = GPUMCTS(
        model=model,
        device=DEVICE,
        c_puct=1.5
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


    # --------------------------------------------------------
    # NO DIRICHLET NOISE
    # --------------------------------------------------------

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


    policy = (
        search.root_visit_policy()
        .float()
    )


    # Safety normalization.

    policy = (
        policy
        / policy.sum(
            dim=1,
            keepdim=True
        ).clamp_min(EPS)
    )


    print(
        f"Elapsed time: "
        f"{elapsed_ms / 1000:.2f}s"
    )


    return (
        policy.detach().clone(),
        elapsed_ms
    )


# ============================================================
# RUN ALL SEARCH DEPTHS
# ============================================================

all_results = {}


for simulations in MCTS_SIMULATIONS:

    policy, elapsed_ms = run_mcts(
        root_states,
        simulations
    )


    all_results[simulations] = {

        "policy":
            policy,

        "elapsed_ms":
            elapsed_ms,

        "entropy":
            entropy(policy),

        "normalized_entropy":
            normalized_entropy(
                policy,
                legal_counts
            ),

        "max_probability":
            max_probability(policy)
    }


    print(
        f"\nMCTS-{simulations} summary:"
    )

    print(
        f"  Entropy:       "
        f"{all_results[simulations]['entropy'].mean().item():.4f}"
    )

    print(
        f"  Norm entropy:  "
        f"{all_results[simulations]['normalized_entropy'].mean().item():.4f}"
    )

    print(
        f"  Max probability:"
        f" {all_results[simulations]['max_probability'].mean().item():.4f}"
    )


# ============================================================
# COMPARE ADJACENT SEARCH DEPTHS
# ============================================================

print("\n\n" + "=" * 80)
print("MCTS CONVERGENCE — ADJACENT DEPTH COMPARISON")
print("=" * 80)


for a, b in zip(
    MCTS_SIMULATIONS[:-1],
    MCTS_SIMULATIONS[1:]
):

    p = all_results[a]["policy"]
    q = all_results[b]["policy"]


    kl_pq = kl_divergence(
        p,
        q
    )

    kl_qp = kl_divergence(
        q,
        p
    )

    skl = symmetric_kl(
        p,
        q
    )

    pearson = correlation(
        p,
        q
    )

    spearman = spearman_correlation(
        p,
        q
    )

    top1 = top1_agreement(
        p,
        q
    )

    top3 = top3_agreement(
        p,
        q
    )


    print("\n" + "-" * 80)

    print(
        f"MCTS-{a}  vs  MCTS-{b}"
    )

    print("-" * 80)

    print(
        f"KL({a} || {b}):        "
        f"{kl_pq.mean().item():.6f}"
    )

    print(
        f"KL({b} || {a}):        "
        f"{kl_qp.mean().item():.6f}"
    )

    print(
        f"Symmetric KL:          "
        f"{skl.mean().item():.6f}"
    )

    print(
        f"Pearson:               "
        f"{pearson.mean().item():.6f}"
    )

    print(
        f"Spearman:              "
        f"{spearman.mean().item():.6f}"
    )

    print(
        f"Top-1 agreement:       "
        f"{top1.mean().item() * 100:.2f}%"
    )

    print(
        f"Top-3 overlap:         "
        f"{top3.mean().item() * 100:.2f}%"
    )


    # Store comparison.

    all_results[
        f"{a}_vs_{b}"
    ] = {

        "kl_pq":
            kl_pq.detach().clone(),

        "kl_qp":
            kl_qp.detach().clone(),

        "symmetric_kl":
            skl.detach().clone(),

        "pearson":
            pearson.detach().clone(),

        "spearman":
            spearman.detach().clone(),

        "top1":
            top1.detach().clone(),

        "top3":
            top3.detach().clone()
    }


# ============================================================
# COMPARE 100 vs 800 DIRECTLY
# ============================================================

print("\n\n" + "=" * 80)
print("MCTS-100 vs MCTS-800")
print("=" * 80)


policy100 = all_results[100]["policy"]
policy800 = all_results[800]["policy"]


kl_100_800 = kl_divergence(
    policy100,
    policy800
)

kl_800_100 = kl_divergence(
    policy800,
    policy100
)

sym_100_800 = symmetric_kl(
    policy100,
    policy800
)

corr_100_800 = correlation(
    policy100,
    policy800
)

spear_100_800 = spearman_correlation(
    policy100,
    policy800
)

top1_100_800 = top1_agreement(
    policy100,
    policy800
)

top3_100_800 = top3_agreement(
    policy100,
    policy800
)


print(
    f"KL(100 || 800):       "
    f"{kl_100_800.mean().item():.6f}"
)

print(
    f"KL(800 || 100):       "
    f"{kl_800_100.mean().item():.6f}"
)

print(
    f"Symmetric KL:         "
    f"{sym_100_800.mean().item():.6f}"
)

print(
    f"Pearson:              "
    f"{corr_100_800.mean().item():.6f}"
)

print(
    f"Spearman:             "
    f"{spear_100_800.mean().item():.6f}"
)

print(
    f"Top-1 agreement:      "
    f"{top1_100_800.mean().item() * 100:.2f}%"
)

print(
    f"Top-3 overlap:        "
    f"{top3_100_800.mean().item() * 100:.2f}%"
)


# ============================================================
# POSITION-BY-POSITION CONVERGENCE
# ============================================================

print("\n\n" + "=" * 80)
print("POSITION-BY-POSITION CONVERGENCE")
print("=" * 80)


for i in range(NUM_POSITIONS):

    print(
        f"\nPosition {i + 1:02d} "
        f"(replay index {indices[i]})"
    )

    print(
        f"Legal moves: "
        f"{int(legal_counts[i].item())}"
    )


    for a, b in zip(
        MCTS_SIMULATIONS[:-1],
        MCTS_SIMULATIONS[1:]
    ):

        comparison = all_results[
            f"{a}_vs_{b}"
        ]


        print(
            f"  {a:3d} -> {b:3d}: "
            f"symKL="
            f"{comparison['symmetric_kl'][i].item():.4f}, "
            f"corr="
            f"{comparison['pearson'][i].item():.4f}, "
            f"top1="
            f"{bool(comparison['top1'][i].item())}"
        )


# ============================================================
# FIND MOST UNSTABLE POSITIONS
# ============================================================

print("\n\n" + "=" * 80)
print("MOST UNSTABLE POSITIONS")
print("=" * 80)


comparison_400_800 = all_results[
    "400_vs_800"
]


sym_kl_400_800 = (
    comparison_400_800[
        "symmetric_kl"
    ]
)


sorted_indices = torch.argsort(
    sym_kl_400_800,
    descending=True
)


print(
    "\nLargest MCTS-400 -> MCTS-800 changes:"
)


for rank in range(
    min(10, NUM_POSITIONS)
):

    i = sorted_indices[
        rank
    ].item()


    print(
        f"{rank + 1:2d}. "
        f"Position {i + 1:02d} "
        f"(replay {indices[i]}): "
        f"symKL="
        f"{sym_kl_400_800[i].item():.6f}, "
        f"Pearson="
        f"{comparison_400_800['pearson'][i].item():.4f}, "
        f"Top1="
        f"{bool(comparison_400_800['top1'][i].item())}"
    )


# ============================================================
# TOP MOVES
# ============================================================

def print_top_moves(
    position_idx,
    policy,
    name,
    k=10
):

    p = policy[position_idx]


    positive_count = int(
        (p > 0).sum().item()
    )


    k = min(
        k,
        positive_count
    )


    values, actions = torch.topk(
        p,
        k=k
    )


    print(
        f"\n{name}"
    )


    for rank, (
        action,
        probability
    ) in enumerate(
        zip(
            actions.tolist(),
            values.tolist()
        ),
        start=1
    ):

        print(
            f"  {rank:2d}. "
            f"action={action:4d} "
            f"prob={probability:.6f}"
        )


# ============================================================
# SHOW TOP MOVES FOR MOST UNSTABLE POSITIONS
# ============================================================

print("\n\n" + "=" * 80)
print("TOP MOVES FOR MOST UNSTABLE POSITIONS")
print("=" * 80)


for rank in range(
    min(5, NUM_POSITIONS)
):

    position_idx = (
        sorted_indices[rank].item()
    )


    print(
        "\n" + "#" * 80
    )

    print(
        f"POSITION {position_idx + 1:02d} "
        f"(replay index {indices[position_idx]})"
    )

    print(
        "#" * 80
    )


    print_top_moves(
        position_idx,
        all_results[100]["policy"],
        "MCTS-100"
    )


    print_top_moves(
        position_idx,
        all_results[200]["policy"],
        "MCTS-200"
    )


    print_top_moves(
        position_idx,
        all_results[400]["policy"],
        "MCTS-400"
    )


    print_top_moves(
        position_idx,
        all_results[800]["policy"],
        "MCTS-800"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n" + "=" * 100)
print("FINAL MCTS CONVERGENCE SUMMARY")
print("=" * 100)


print(
    f"{'Sims':>8} "
    f"{'Entropy':>12} "
    f"{'NormEnt':>12} "
    f"{'MaxProb':>12} "
    f"{'Time(s)':>12}"
)

print("-" * 65)


for simulations in MCTS_SIMULATIONS:

    r = all_results[
        simulations
    ]


    print(
        f"{simulations:>8} "
        f"{r['entropy'].mean().item():>12.4f} "
        f"{r['normalized_entropy'].mean().item():>12.4f} "
        f"{r['max_probability'].mean().item():>12.4f} "
        f"{r['elapsed_ms'] / 1000:>12.2f}"
    )


# ============================================================
# ADJACENT CONVERGENCE SUMMARY
# ============================================================

print("\n\n" + "=" * 100)
print("ADJACENT MCTS CONVERGENCE")
print("=" * 100)


print(
    f"{'Comparison':>18} "
    f"{'SymKL':>12} "
    f"{'Pearson':>12} "
    f"{'Spearman':>12} "
    f"{'Top1 %':>12} "
    f"{'Top3 %':>12}"
)

print("-" * 90)


for a, b in zip(
    MCTS_SIMULATIONS[:-1],
    MCTS_SIMULATIONS[1:]
):

    r = all_results[
        f"{a}_vs_{b}"
    ]


    print(
        f"{a:>7} -> {b:<7} "
        f"{r['symmetric_kl'].mean().item():>12.6f} "
        f"{r['pearson'].mean().item():>12.6f} "
        f"{r['spearman'].mean().item():>12.6f} "
        f"{r['top1'].mean().item() * 100:>12.2f} "
        f"{r['top3'].mean().item() * 100:>12.2f}"
    )


# ============================================================
# DIRECT 100 -> 800
# ============================================================

print("\n\n" + "=" * 100)
print("DIRECT MCTS-100 -> MCTS-800")
print("=" * 100)


print(
    f"Symmetric KL:    "
    f"{sym_100_800.mean().item():.6f}"
)

print(
    f"Pearson:         "
    f"{corr_100_800.mean().item():.6f}"
)

print(
    f"Spearman:        "
    f"{spear_100_800.mean().item():.6f}"
)

print(
    f"Top-1 agreement: "
    f"{top1_100_800.mean().item() * 100:.2f}%"
)

print(
    f"Top-3 overlap:   "
    f"{top3_100_800.mean().item() * 100:.2f}%"
)


# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n" + "=" * 100)
print("HOW TO INTERPRET THE RESULT")
print("=" * 100)


print(
"""
The MOST IMPORTANT comparison is:

    MCTS-100 -> MCTS-200
    MCTS-200 -> MCTS-400
    MCTS-400 -> MCTS-800


If the search is converging, you should generally see:

    Symmetric KL       decreasing
    Pearson            increasing
    Spearman           increasing
    Top-1 agreement    increasing


For example, a pattern like:

    100 -> 200    large change
    200 -> 400    smaller change
    400 -> 800    very small change

would indicate that MCTS is approaching a stable policy.


If instead you see:

    100 -> 200    large change
    200 -> 400    large change
    400 -> 800    large change

then the search is NOT obviously converging.

In that case, we should investigate:

    - value estimates
    - PUCT behavior
    - visit allocation
    - search backup
    - leaf evaluation
    - tree expansion


IMPORTANT:

Do NOT change RL54 yet.

Do NOT change replay-buffer size yet.

Do NOT change the loss yet.

Do NOT automatically increase self-play simulations yet.

First determine whether the search itself converges.


The most useful numbers to send back are:

    100 -> 200:
        SymKL
        Pearson
        Top1 %

    200 -> 400:
        SymKL
        Pearson
        Top1 %

    400 -> 800:
        SymKL
        Pearson
        Top1 %

and:

    MCTS-100 -> MCTS-800:
        SymKL
        Pearson
        Top1 %
"""
)


print("\n" + "=" * 80)
print("DIAGNOSTIC FINISHED")
print("=" * 80)