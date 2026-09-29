# ============================================================
# RL53 NETWORK vs MCTS DIAGNOSTIC
#
# Purpose:
#   Compare the neural-network policy with MCTS on the EXACT
#   same positions from the RL53 replay buffer.
#
# Tests:
#   100 MCTS simulations
#   200 MCTS simulations
#   400 MCTS simulations
#
# Dirichlet noise: OFF
#
# Metrics:
#   - Network entropy
#   - MCTS entropy
#   - Network max probability
#   - MCTS max probability
#   - KL(Network || MCTS)
#   - KL(MCTS || Network)
#   - Pearson correlation
#   - Spearman correlation
#   - Top-1 agreement
#   - Average number of legal moves
# ============================================================

import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

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

MCTS_SIMULATIONS = [100, 200, 400]

MCTS_BATCH_SIZE = 16

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("RL53 NETWORK vs MCTS DIAGNOSTIC")
print("=" * 70)

print(f"Project:     {PROJECT_ROOT}")
print(f"Checkpoint:  {CHECKPOINT}")
print(f"Replay:      {REPLAY_BUFFER}")
print(f"Device:      {DEVICE}")
print(f"Positions:   {NUM_POSITIONS}")
print(f"MCTS tests:  {MCTS_SIMULATIONS}")
print(f"Noise:       OFF")
print("=" * 70)


# ------------------------------------------------------------
# CUDA INFORMATION
# ------------------------------------------------------------

if DEVICE.type == "cuda":
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"CUDA:        {torch.version.cuda}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_grad_enabled(False)


# ------------------------------------------------------------
# IMPORT PROJECT
# ------------------------------------------------------------

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\nLoading RL53 model...")

model = ChessNet().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

print("Checkpoint type:", type(checkpoint))

# Handle common checkpoint formats.
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]

    else:
        # Assume the dictionary itself is the state dict.
        state_dict = checkpoint

else:
    raise RuntimeError(
        f"Unsupported checkpoint format: {type(checkpoint)}"
    )

# Remove possible torch.compile prefix.
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod."):]

    clean_state_dict[key] = value


missing, unexpected = model.load_state_dict(
    clean_state_dict,
    strict=False
)

print(f"Missing keys:    {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

if missing:
    print("Missing:", missing[:10])

if unexpected:
    print("Unexpected:", unexpected[:10])

model.eval()

print("RL53 model loaded successfully.")


# ------------------------------------------------------------
# LOAD REPLAY BUFFER
# ------------------------------------------------------------

print("\nLoading RL53 replay buffer...")

replay = torch.load(
    REPLAY_BUFFER,
    map_location="cpu",
    weights_only=False
)

print("Replay object type:", type(replay))
print("Replay size:", len(replay))


# ------------------------------------------------------------
# INSPECT ONE SAMPLE
# ------------------------------------------------------------

sample0 = replay[0]

print("\nSample structure:")
print("Type:", type(sample0))
print("Length:", len(sample0))

for i, x in enumerate(sample0):
    if hasattr(x, "shape"):
        print(
            f"[{i}] shape={x.shape}, "
            f"dtype={x.dtype}"
        )
    else:
        print(
            f"[{i}] type={type(x)}, "
            f"value={x}"
        )


# ------------------------------------------------------------
# SAMPLE POSITIONS
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

indices = random.sample(
    range(len(replay)),
    NUM_POSITIONS
)

samples = [replay[i] for i in indices]

print("\nSelected replay positions:")
print(indices)


# ------------------------------------------------------------
# CONVERT REPLAY STATE -> GPUChess
# ------------------------------------------------------------
#
# Stored state:
#
#   [18, 8, 8]
#
# GPUChess model input:
#
#   0-11  = piece planes
#   12    = white turn
#   13    = white kingside castling
#   14    = white queenside castling
#   15    = black kingside castling
#   16    = black queenside castling
#   17    = en-passant square
#
# Important:
#
# Replay states do NOT store halfmove/fullmove counters.
# We therefore initialize those to 0/1.
#
# This diagnostic is about policy/MCTS behavior, so that does
# not affect the ordinary positions except the 50-move rule.
# ------------------------------------------------------------

def replay_states_to_gpu_chess(samples, device):

    states_np = np.stack(
        [np.asarray(s[0], dtype=np.float32) for s in samples],
        axis=0
    )

    states = torch.from_numpy(states_np).to(
        device=device,
        dtype=torch.float32
    )

    B = states.shape[0]

    if states.shape != (B, 18, 8, 8):
        raise ValueError(
            f"Unexpected state shape: {states.shape}"
        )

    chess = GPUChess(
        device=device,
        batch_size=B
    )

    # --------------------------------------------------------
    # PIECE PLANES
    # --------------------------------------------------------

    #
    # GPUChess uses:
    #
    # square = rank * 8 + file
    #
    # Model representation uses:
    #
    # row = 7 - rank
    #
    # Therefore flip the row dimension back.
    #

    board_planes = states[:, :12].flip(2)

    # GPUChess square bit values.
    square_bits = chess.square_bits

    pieces = torch.zeros(
        (B, 12),
        dtype=torch.int64,
        device=device
    )

    # Build each bitboard.
    for p in range(12):

        occupancy = board_planes[:, p].reshape(
            B, 64
        ) > 0.5

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

    # Plane 12 = white_to_move
    white_to_move = states[:, 12, 0, 0] > 0.5

    # GPUChess turn:
    #   False = white
    #   True  = black
    chess.turn = ~white_to_move

    # --------------------------------------------------------
    # CASTLING RIGHTS
    # --------------------------------------------------------

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
        torch.tensor(WK, dtype=torch.int16, device=device),
        torch.tensor(0, dtype=torch.int16, device=device)
    )

    castling |= torch.where(
        states[:, 14, 0, 0] > 0.5,
        torch.tensor(WQ, dtype=torch.int16, device=device),
        torch.tensor(0, dtype=torch.int16, device=device)
    )

    castling |= torch.where(
        states[:, 15, 0, 0] > 0.5,
        torch.tensor(BK, dtype=torch.int16, device=device),
        torch.tensor(0, dtype=torch.int16, device=device)
    )

    castling |= torch.where(
        states[:, 16, 0, 0] > 0.5,
        torch.tensor(BQ, dtype=torch.int16, device=device),
        torch.tensor(0, dtype=torch.int16, device=device)
    )

    chess.castling = castling

    # --------------------------------------------------------
    # EN PASSANT
    # --------------------------------------------------------

    ep_plane = states[:, 17].flip(1).reshape(
        B, 64
    ) > 0.5

    ep_exists = ep_plane.any(dim=1)

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

        ep_square[rows] = squares.to(torch.int16)

    chess.ep_square = ep_square

    # --------------------------------------------------------
    # CLOCKS
    # --------------------------------------------------------

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


print("\nReconstructing GPU chess positions...")

root_states = replay_states_to_gpu_chess(
    samples,
    DEVICE
)

print("Reconstruction complete.")


# ------------------------------------------------------------
# VERIFY RECONSTRUCTION
# ------------------------------------------------------------

print("\nVerifying reconstructed states...")

reconstructed_input = root_states.to_model_input()

original_states = torch.from_numpy(
    np.stack(
        [np.asarray(s[0], dtype=np.float32) for s in samples]
    )
).to(DEVICE)

max_difference = (
    reconstructed_input - original_states
).abs().max().item()

mean_difference = (
    reconstructed_input - original_states
).abs().mean().item()

print(f"Maximum state difference: {max_difference:.8f}")
print(f"Mean state difference:    {mean_difference:.8f}")

if max_difference > 1e-5:
    print(
        "\nWARNING: Reconstruction is not identical!"
    )
else:
    print(
        "State reconstruction verified."
    )


# ------------------------------------------------------------
# LEGAL MOVE MASK
# ------------------------------------------------------------

print("\nGenerating legal moves...")

legal_mask = root_states.legal_move_mask()

legal_counts = legal_mask.sum(
    dim=1
).float()

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


# ------------------------------------------------------------
# NETWORK POLICY
# ------------------------------------------------------------

print("\nRunning RL53 network...")

with torch.inference_mode():

    model_input = root_states.to_model_input()

    if DEVICE.type == "cuda":
        model_input = model_input.contiguous(
            memory_format=torch.channels_last
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits, values = model(model_input)

    else:
        logits, values = model(model_input)

    logits = logits.float()
    values = values.squeeze(-1).float()


# ------------------------------------------------------------
# MASK NETWORK POLICY TO TRUE LEGAL MOVES
# ------------------------------------------------------------

masked_logits = logits.masked_fill(
    ~legal_mask,
    torch.finfo(logits.dtype).min
)

network_policy = torch.softmax(
    masked_logits,
    dim=1
)

# Safety normalization.
network_policy = network_policy * legal_mask.float()

network_policy = network_policy / network_policy.sum(
    dim=1,
    keepdim=True
).clamp_min(1e-12)


# ------------------------------------------------------------
# METRIC FUNCTIONS
# ------------------------------------------------------------

EPS = 1e-12


def entropy(policy):
    p = policy.clamp_min(EPS)

    return -(
        p * torch.log(p)
    ).sum(dim=1)


def normalized_entropy(policy, legal_counts):
    h = entropy(policy)

    return h / torch.log(
        legal_counts.clamp_min(2)
    )


def max_probability(policy):
    return policy.max(dim=1).values


def kl_divergence(p, q):

    p = p.clamp_min(EPS)
    q = q.clamp_min(EPS)

    return (
        p * (torch.log(p) - torch.log(q))
    ).sum(dim=1)


def correlation(p, q):

    p_mean = p.mean(dim=1, keepdim=True)
    q_mean = q.mean(dim=1, keepdim=True)

    p_centered = p - p_mean
    q_centered = q - q_mean

    numerator = (
        p_centered * q_centered
    ).sum(dim=1)

    denominator = torch.sqrt(
        (p_centered ** 2).sum(dim=1)
        *
        (q_centered ** 2).sum(dim=1)
    ).clamp_min(EPS)

    return numerator / denominator


def rankdata_torch(x):

    # Simple rank implementation.
    order = torch.argsort(x, dim=1)

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
        rank_values.unsqueeze(0).expand_as(x)
    )

    return ranks


def spearman_correlation(p, q):

    rp = rankdata_torch(p)
    rq = rankdata_torch(q)

    return correlation(rp, rq)


def top1_agreement(p, q):

    p_top = p.argmax(dim=1)
    q_top = q.argmax(dim=1)

    return (
        p_top == q_top
    ).float()


# ------------------------------------------------------------
# NETWORK BASELINE METRICS
# ------------------------------------------------------------

net_entropy = entropy(network_policy)
net_norm_entropy = normalized_entropy(
    network_policy,
    legal_counts
)
net_max = max_probability(network_policy)

print("\n" + "=" * 70)
print("NETWORK BASELINE")
print("=" * 70)

print(
    f"Entropy:              "
    f"{net_entropy.mean().item():.4f}"
)

print(
    f"Normalized entropy:   "
    f"{net_norm_entropy.mean().item():.4f}"
)

print(
    f"Max probability:      "
    f"{net_max.mean().item():.4f}"
)

print(
    f"Value mean:           "
    f"{values.mean().item():.4f}"
)

print(
    f"Value std:            "
    f"{values.std().item():.4f}"
)


# ------------------------------------------------------------
# HELPER: RUN MCTS
# ------------------------------------------------------------

def run_mcts(
    root_states,
    simulations
):

    print(
        f"\nRunning MCTS: "
        f"{simulations} simulations..."
    )

    search = GPUMCTS(
        model=model,
        device=DEVICE,
        c_puct=1.5
    )

    start_event = None
    end_event = None

    if DEVICE.type == "cuda":

        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        torch.cuda.synchronize()

        start_event.record()

    else:
        import time
        start_time = time.perf_counter()

    # IMPORTANT:
    # No Dirichlet noise.
    search.search(
        root_states,
        num_simulations=simulations,
        dirichlet_alpha=None,
        dirichlet_epsilon=0.0,
        batch_size=MCTS_BATCH_SIZE
    )

    if DEVICE.type == "cuda":

        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(
            end_event
        )

    else:

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

    mcts_policy = search.root_visit_policy()

    # Normalize again for safety.
    mcts_policy = mcts_policy / (
        mcts_policy.sum(
            dim=1,
            keepdim=True
        ).clamp_min(EPS)
    )

    print(
        f"Elapsed: {elapsed_ms / 1000:.2f}s"
    )

    return mcts_policy, elapsed_ms


# ------------------------------------------------------------
# STORE RESULTS
# ------------------------------------------------------------

all_results = {}


# ------------------------------------------------------------
# RUN 100 / 200 / 400
# ------------------------------------------------------------

for simulations in MCTS_SIMULATIONS:

    mcts_policy, elapsed_ms = run_mcts(
        root_states,
        simulations
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mcts_entropy = entropy(mcts_policy)

    mcts_norm_entropy = normalized_entropy(
        mcts_policy,
        legal_counts
    )

    mcts_max = max_probability(
        mcts_policy
    )

    kl_net_mcts = kl_divergence(
        network_policy,
        mcts_policy
    )

    kl_mcts_net = kl_divergence(
        mcts_policy,
        network_policy
    )

    pearson = correlation(
        network_policy,
        mcts_policy
    )

    spearman = spearman_correlation(
        network_policy,
        mcts_policy
    )

    top1 = top1_agreement(
        network_policy,
        mcts_policy
    )

    all_results[simulations] = {
        "policy": mcts_policy.detach().clone(),
        "entropy": mcts_entropy.detach().clone(),
        "normalized_entropy": mcts_norm_entropy.detach().clone(),
        "max_probability": mcts_max.detach().clone(),
        "kl_net_mcts": kl_net_mcts.detach().clone(),
        "kl_mcts_net": kl_mcts_net.detach().clone(),
        "pearson": pearson.detach().clone(),
        "spearman": spearman.detach().clone(),
        "top1": top1.detach().clone(),
        "elapsed_ms": elapsed_ms,
    }

    print("\n" + "-" * 70)
    print(f"MCTS = {simulations} SIMULATIONS")
    print("-" * 70)

    print(
        f"MCTS entropy:             "
        f"{mcts_entropy.mean().item():.4f}"
    )

    print(
        f"MCTS normalized entropy:  "
        f"{mcts_norm_entropy.mean().item():.4f}"
    )

    print(
        f"MCTS max probability:     "
        f"{mcts_max.mean().item():.4f}"
    )

    print(
        f"KL(Network || MCTS):      "
        f"{kl_net_mcts.mean().item():.4f}"
    )

    print(
        f"KL(MCTS || Network):      "
        f"{kl_mcts_net.mean().item():.4f}"
    )

    print(
        f"Pearson correlation:      "
        f"{pearson.mean().item():.4f}"
    )

    print(
        f"Spearman correlation:     "
        f"{spearman.mean().item():.4f}"
    )

    print(
        f"Top-1 agreement:           "
        f"{top1.mean().item() * 100:.2f}%"
    )

    print(
        f"Time:                     "
        f"{elapsed_ms / 1000:.2f}s"
    )


# ------------------------------------------------------------
# POSITION-BY-POSITION RESULTS
# ------------------------------------------------------------

print("\n\n" + "=" * 70)
print("POSITION-BY-POSITION RESULTS")
print("=" * 70)

for i in range(NUM_POSITIONS):

    print(
        f"\nPosition {i + 1:02d} "
        f"(replay index {indices[i]})"
    )

    print(
        f"Legal moves: "
        f"{int(legal_counts[i].item())}"
    )

    print(
        f"Network entropy: "
        f"{net_entropy[i].item():.4f}"
    )

    print(
        f"Network max: "
        f"{net_max[i].item():.4f}"
    )

    for simulations in MCTS_SIMULATIONS:

        r = all_results[simulations]

        print(
            f"  MCTS {simulations:3d}: "
            f"entropy={r['entropy'][i].item():.4f}, "
            f"max={r['max_probability'][i].item():.4f}, "
            f"corr={r['pearson'][i].item():.4f}, "
            f"KL(N||M)={r['kl_net_mcts'][i].item():.4f}, "
            f"top1={bool(r['top1'][i].item())}"
        )


# ------------------------------------------------------------
# TOP MOVES COMPARISON
# ------------------------------------------------------------

def print_top_moves(
    position_idx,
    policy,
    name,
    k=10
):

    p = policy[position_idx]

    values, actions = torch.topk(
        p,
        k=min(k, int((p > 0).sum().item()))
    )

    print(f"\n{name}")

    for rank, (action, prob) in enumerate(
        zip(actions.tolist(), values.tolist()),
        start=1
    ):

        print(
            f"  {rank:2d}. "
            f"action={action:4d} "
            f"prob={prob:.6f}"
        )


# ------------------------------------------------------------
# SHOW A FEW POSITIONS
# ------------------------------------------------------------

print("\n\n" + "=" * 70)
print("TOP-MOVE COMPARISON")
print("=" * 70)

for position_idx in range(
    min(5, NUM_POSITIONS)
):

    print(
        "\n" + "#" * 70
    )

    print(
        f"POSITION {position_idx + 1} "
        f"(replay index {indices[position_idx]})"
    )

    print(
        "#" * 70
    )

    print_top_moves(
        position_idx,
        network_policy,
        "RL53 NETWORK"
    )

    for simulations in MCTS_SIMULATIONS:

        print_top_moves(
            position_idx,
            all_results[simulations]["policy"],
            f"MCTS {simulations}"
        )


# ------------------------------------------------------------
# FINAL SUMMARY TABLE
# ------------------------------------------------------------

print("\n\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"{'Sims':>6} "
    f"{'Entropy':>10} "
    f"{'NormEnt':>10} "
    f"{'MaxProb':>10} "
    f"{'KL N||M':>10} "
    f"{'KL M||N':>10} "
    f"{'Pearson':>10} "
    f"{'Spearman':>10} "
    f"{'Top1 %':>10}"
)

print("-" * 100)

for simulations in MCTS_SIMULATIONS:

    r = all_results[simulations]

    print(
        f"{simulations:>6} "
        f"{r['entropy'].mean().item():>10.4f} "
        f"{r['normalized_entropy'].mean().item():>10.4f} "
        f"{r['max_probability'].mean().item():>10.4f} "
        f"{r['kl_net_mcts'].mean().item():>10.4f} "
        f"{r['kl_mcts_net'].mean().item():>10.4f} "
        f"{r['pearson'].mean().item():>10.4f} "
        f"{r['spearman'].mean().item():>10.4f} "
        f"{r['top1'].mean().item()*100:>10.2f}"
    )


# ------------------------------------------------------------
# NETWORK BASELINE
# ------------------------------------------------------------

print("-" * 100)

print(
    f"{'NET':>6} "
    f"{net_entropy.mean().item():>10.4f} "
    f"{net_norm_entropy.mean().item():>10.4f} "
    f"{net_max.mean().item():>10.4f}"
)


# ------------------------------------------------------------
# INTERPRETATION HELP
# ------------------------------------------------------------

print("\n\n" + "=" * 70)
print("HOW TO READ THIS TEST")
print("=" * 70)

print("""
The important numbers are:

1. Pearson / Spearman
   --------------------
   HIGH correlation:
       MCTS is mostly following the neural-network prior.

   LOW correlation:
       MCTS is substantially changing the policy.

2. Top-1 agreement
   ----------------
   HIGH:
       Network and MCTS usually select the same move.

   LOW:
       Search is frequently changing the preferred move.

3. MCTS entropy vs Network entropy
   --------------------------------
   If MCTS entropy drops as simulations increase:

       100 -> high entropy
       200 -> lower entropy
       400 -> lower entropy

   then additional simulations are making the search more selective.

4. MCTS max probability
   ---------------------
   If this increases:

       100 -> 0.10
       200 -> 0.15
       400 -> 0.25

   then additional simulations are concentrating visits.

5. KL divergence
   ---------------
   Small KL:
       MCTS is close to the network.

   Large KL:
       MCTS is substantially changing the policy.

The most important comparison is:

       NETWORK
          vs
       MCTS-100
          vs
       MCTS-200
          vs
       MCTS-400

If MCTS-100 is extremely correlated with the network,
while MCTS-400 becomes substantially different, then
100 simulations may simply be too small for your current
network.

If even MCTS-400 remains extremely correlated with the
network, then the search itself may not be extracting
much additional information from the value network.
""")


print("\nDiagnostic finished.")