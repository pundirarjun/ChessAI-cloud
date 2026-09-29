import torch
import random

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = (
    "/kaggle/working/chess-zero/checkpoints/"
    "rl_iteration_53.pt"
)

REPLAY_BUFFER = (
    "/kaggle/working/chess-zero/checkpoints/"
    "replay_buffer_rl53.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_POSITIONS = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RL53 REPLAY BUFFER POLICY DIAGNOSTIC")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Replay buffer: {REPLAY_BUFFER}")


model = ChessNet().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded.")


# ============================================================
# LOAD REPLAY BUFFER
# ============================================================

print("\nLoading replay buffer...")

buffer = torch.load(
    REPLAY_BUFFER,
    map_location="cpu",
    weights_only=False,
)

print(f"Replay buffer type: {type(buffer)}")


# ============================================================
# EXTRACT SAMPLES
# ============================================================

if hasattr(buffer, "buffer"):
    samples = list(buffer.buffer)

elif isinstance(buffer, (list, tuple)):
    samples = list(buffer)

elif isinstance(buffer, dict):

    if "buffer" in buffer:
        samples = list(buffer["buffer"])

    elif "samples" in buffer:
        samples = list(buffer["samples"])

    else:
        raise RuntimeError(
            f"Unknown replay-buffer dictionary keys: "
            f"{buffer.keys()}"
        )

else:
    raise RuntimeError(
        f"Unknown replay buffer type: {type(buffer)}"
    )


print(f"Total replay samples: {len(samples)}")


if len(samples) < NUM_POSITIONS:
    raise RuntimeError(
        f"Replay buffer contains only {len(samples)} "
        f"samples."
    )


# ============================================================
# INSPECT SAMPLE STRUCTURE
# ============================================================

print("\n")
print("=" * 70)
print("REPLAY SAMPLE STRUCTURE")
print("=" * 70)

example = samples[0]

print(f"Sample type: {type(example)}")

if hasattr(example, "__dict__"):
    print("Sample attributes:")

    for key, value in example.__dict__.items():
        if torch.is_tensor(value):
            print(
                f"  {key}: "
                f"Tensor shape={tuple(value.shape)}, "
                f"dtype={value.dtype}"
            )
        else:
            print(
                f"  {key}: "
                f"{type(value).__name__} = {value}"
            )

elif isinstance(example, dict):

    print("Sample dictionary:")

    for key, value in example.items():

        if torch.is_tensor(value):
            print(
                f"  {key}: "
                f"Tensor shape={tuple(value.shape)}, "
                f"dtype={value.dtype}"
            )

        else:
            print(
                f"  {key}: "
                f"{type(value).__name__}"
            )

elif isinstance(example, (tuple, list)):

    print(
        f"Sample contains {len(example)} elements:"
    )

    for i, value in enumerate(example):

        if torch.is_tensor(value):
            print(
                f"  [{i}]: "
                f"Tensor shape={tuple(value.shape)}, "
                f"dtype={value.dtype}"
            )

        else:
            print(
                f"  [{i}]: "
                f"{type(value).__name__}"
            )


# ============================================================
# GET STATE FROM REPLAY SAMPLE
# ============================================================

def get_state(sample):

    """
    Extract the chess state from a replay sample.

    The diagnostic handles the common formats used
    by the project.
    """

    # --------------------------------------------------------
    # Object-style sample
    # --------------------------------------------------------

    if hasattr(sample, "state"):

        return sample.state

    # --------------------------------------------------------
    # Dictionary-style sample
    # --------------------------------------------------------

    if isinstance(sample, dict):

        for key in [
            "state",
            "board",
            "observation",
            "state_tensor",
        ]:

            if key in sample:
                return sample[key]

    # --------------------------------------------------------
    # Tuple/list style
    # --------------------------------------------------------

    if isinstance(sample, (tuple, list)):

        # Normally state is first element.
        return sample[0]

    raise RuntimeError(
        "Could not determine state from replay sample."
    )


# ============================================================
# SELECT RANDOM REAL POSITIONS
# ============================================================

print("\n")
print("=" * 70)
print("SELECTING REAL REPLAY POSITIONS")
print("=" * 70)

random.seed(42)

selected_samples = random.sample(
    samples,
    NUM_POSITIONS,
)

states_raw = []

for sample in selected_samples:

    state = get_state(sample)

    states_raw.append(state)


# ============================================================
# CONVERT STATES
# ============================================================

print("Converting replay states...")


# ------------------------------------------------------------
# Case 1: already GPUChess
# ------------------------------------------------------------

if all(
    isinstance(x, GPUChess)
    for x in states_raw
):

    # Usually not expected after torch.load,
    # but supported.
    states = states_raw


# ------------------------------------------------------------
# Case 2: state tensors
# ------------------------------------------------------------

elif torch.is_tensor(states_raw[0]):

    state_tensor = torch.stack(
        [
            x.float()
            for x in states_raw
        ]
    )

    print(
        f"State tensor shape: "
        f"{tuple(state_tensor.shape)}"
    )

    raise RuntimeError(
        "\nYour replay buffer stores encoded state tensors "
        "rather than GPUChess board states.\n\n"
        "Send me the 'REPLAY SAMPLE STRUCTURE' output above "
        "and I will adapt the conversion exactly to your "
        "replay-buffer format."
    )


else:

    raise RuntimeError(
        f"Unsupported replay state type: "
        f"{type(states_raw[0])}"
    )


# ============================================================
# RAW NETWORK POLICY
# ============================================================

print("\n")
print("=" * 70)
print("RAW RL53 POLICY ON REAL GAME POSITIONS")
print("=" * 70)


with torch.inference_mode():

    model_input = states.to_model_input()

    if model_input.is_cuda:

        model_input = model_input.contiguous(
            memory_format=torch.channels_last
        )

    if DEVICE.type == "cuda":

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            logits, values = model(
                model_input
            )

    else:

        logits, values = model(
            model_input
        )

    # --------------------------------------------------------
    # Legal moves
    # --------------------------------------------------------

    legal = states.legal_move_mask()

    # --------------------------------------------------------
    # Mask illegal actions
    # --------------------------------------------------------

    masked_logits = logits.float().masked_fill(
        ~legal,
        torch.finfo(torch.float32).min,
    )

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    policy = torch.softmax(
        masked_logits,
        dim=1,
    )


# ============================================================
# METRICS
# ============================================================

safe_policy = policy.clamp_min(1e-12)

entropy = -(
    safe_policy * safe_policy.log()
).sum(dim=1)

num_legal = legal.sum(
    dim=1
).float()

normalized_entropy = (
    entropy /
    torch.log(num_legal.clamp_min(2))
)

max_probability = policy.max(
    dim=1
).values


# ============================================================
# RESULTS
# ============================================================

print(
    f"\nAverage legal moves: "
    f"{num_legal.mean().item():.2f}"
)

print(
    f"Entropy: "
    f"{entropy.mean().item():.4f}"
)

print(
    f"Normalized entropy: "
    f"{normalized_entropy.mean().item():.4f}"
)

print(
    f"Max probability: "
    f"{max_probability.mean().item():.4f}"
)

print(
    f"Max probability min: "
    f"{max_probability.min().item():.4f}"
)

print(
    f"Max probability max: "
    f"{max_probability.max().item():.4f}"
)


# ============================================================
# POSITION-BY-POSITION RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("POSITION-BY-POSITION RESULTS")
print("=" * 90)

print(
    f"{'Pos':>5}"
    f"{'Legal':>10}"
    f"{'Entropy':>12}"
    f"{'Norm Ent':>12}"
    f"{'Max Prob':>12}"
)

print("-" * 90)

for i in range(NUM_POSITIONS):

    print(
        f"{i + 1:>5}"
        f"{int(num_legal[i].item()):>10}"
        f"{entropy[i].item():>12.4f}"
        f"{normalized_entropy[i].item():>12.4f}"
        f"{max_probability[i].item():>12.4f}"
    )


# ============================================================
# TOP MOVES
# ============================================================

print("\n")
print("=" * 70)
print("TOP NETWORK MOVES")
print("=" * 70)

for position in range(
    min(5, NUM_POSITIONS)
):

    legal_actions = torch.nonzero(
        legal[position],
        as_tuple=False,
    ).flatten()

    legal_probs = policy[
        position,
        legal_actions,
    ]

    top_k = min(
        10,
        legal_actions.numel(),
    )

    top_probs, top_indices = torch.topk(
        legal_probs,
        k=top_k,
    )

    top_actions = legal_actions[
        top_indices
    ]

    print(
        f"\nPosition {position + 1}:"
    )

    for rank in range(top_k):

        action = top_actions[
            rank
        ].item()

        probability = top_probs[
            rank
        ].item()

        print(
            f"  {rank + 1:2d}. "
            f"Action {action:4d} "
            f"Probability "
            f"{probability:.4f}"
        )


# ============================================================
# INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("WHAT THIS TEST TELLS US")
print("=" * 70)

print(
    """
This test uses real positions sampled from the RL53
replay buffer rather than the initial chess position.

If normalized entropy remains around 0.95-1.00
across real positions, the RL53 policy is very diffuse.

If entropy becomes substantially lower on real
middle-game/tactical positions, the opening-position
result was not representative.

The next step after this test is to compare these
network policies against the MCTS target policies
stored in the replay buffer.
"""
)

print("=" * 70)