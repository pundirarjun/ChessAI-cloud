import torch
import numpy as np
import random

from model.chess_net import ChessNet


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


print(
    f"Total replay samples: {len(samples)}"
)


if len(samples) < NUM_POSITIONS:

    raise RuntimeError(
        f"Replay buffer contains only "
        f"{len(samples)} samples."
    )


# ============================================================
# VERIFY SAMPLE STRUCTURE
# ============================================================

print("\n")
print("=" * 70)
print("REPLAY SAMPLE STRUCTURE")
print("=" * 70)

example = samples[0]

print(
    f"Sample type: {type(example)}"
)

if not isinstance(example, (tuple, list)):

    raise RuntimeError(
        "Expected replay sample to be a tuple/list."
    )


if len(example) != 3:

    raise RuntimeError(
        f"Expected 3 elements but found "
        f"{len(example)}"
    )


example_state = example[0]
example_policy = example[1]
example_value = example[2]

print(
    f"State:  shape={example_state.shape}, "
    f"dtype={example_state.dtype}"
)

print(
    f"Policy: shape={example_policy.shape}, "
    f"dtype={example_policy.dtype}"
)

print(
    f"Value:  {example_value}"
)


# ============================================================
# SELECT REAL REPLAY POSITIONS
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


# ============================================================
# EXTRACT STATE + TARGET POLICY + VALUE
# ============================================================

states_np = []
target_policies_np = []
values_np = []

for sample in selected_samples:

    # Replay format:
    #
    # sample[0] = state
    # sample[1] = MCTS target policy
    # sample[2] = value

    state = sample[0]

    target_policy = sample[1]

    value = sample[2]

    states_np.append(
        np.asarray(state)
    )

    target_policies_np.append(
        np.asarray(target_policy)
    )

    values_np.append(
        float(value)
    )


# ============================================================
# CONVERT TO TORCH
# ============================================================

states = torch.from_numpy(
    np.stack(states_np)
).float().to(DEVICE)

target_policy = torch.from_numpy(
    np.stack(target_policies_np)
).float().to(DEVICE)

target_values = torch.tensor(
    values_np,
    dtype=torch.float32,
    device=DEVICE,
)


print(
    f"\nState batch shape: "
    f"{tuple(states.shape)}"
)

print(
    f"Target policy shape: "
    f"{tuple(target_policy.shape)}"
)

print(
    f"Target values shape: "
    f"{tuple(target_values.shape)}"
)


# ============================================================
# VERIFY STATE SHAPE
# ============================================================

if states.ndim != 4:

    raise RuntimeError(
        f"Expected states to have 4 dimensions "
        f"[batch, channels, height, width]. "
        f"Got {tuple(states.shape)}"
    )


print(
    f"State channels: {states.shape[1]}"
)

print(
    f"Board size: "
    f"{states.shape[2]} x {states.shape[3]}"
)


# ============================================================
# RAW NETWORK POLICY
# ============================================================

print("\n")
print("=" * 70)
print("RAW RL53 POLICY ON REAL REPLAY POSITIONS")
print("=" * 70)


with torch.inference_mode():

    model_input = states

    if DEVICE.type == "cuda":

        model_input = model_input.contiguous(
            memory_format=torch.channels_last
        )

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


# ============================================================
# POLICY FROM NETWORK
# ============================================================

network_policy = torch.softmax(
    logits.float(),
    dim=1,
)


# ============================================================
# IMPORTANT:
# MASK ONLY ACTIONS THAT ARE ZERO IN THE STORED TARGET
# ============================================================
#
# The replay target policy contains non-zero probability
# only on legal moves that MCTS considered.
#
# Therefore we use its support as the legal-action mask.
#
# This avoids reconstructing GPUChess states.
# ============================================================

target_mask = target_policy > 0


# ============================================================
# NETWORK POLICY OVER TARGET'S LEGAL ACTIONS
# ============================================================

masked_logits = logits.float().masked_fill(
    ~target_mask,
    torch.finfo(torch.float32).min,
)

network_policy_legal = torch.softmax(
    masked_logits,
    dim=1,
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    policy,
    mask,
):

    safe_policy = policy.clamp_min(
        1e-12
    )

    entropy = -(
        policy * safe_policy.log()
    ).sum(dim=1)

    num_actions = mask.sum(
        dim=1
    ).float()

    normalized_entropy = (
        entropy /
        torch.log(
            num_actions.clamp_min(2)
        )
    )

    max_probability = policy.max(
        dim=1
    ).values

    return {
        "entropy":
            entropy.mean().item(),

        "normalized_entropy":
            normalized_entropy.mean().item(),

        "max_probability":
            max_probability.mean().item(),

        "num_actions":
            num_actions.mean().item(),

        "entropy_each":
            entropy,

        "normalized_each":
            normalized_entropy,

        "max_each":
            max_probability,
    }


# ============================================================
# RAW NETWORK METRICS
# ============================================================

network_metrics = calculate_metrics(
    network_policy_legal,
    target_mask,
)


# ============================================================
# TARGET POLICY METRICS
# ============================================================

target_metrics = calculate_metrics(
    target_policy,
    target_mask,
)


# ============================================================
# PRINT NETWORK RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("NETWORK POLICY")
print("=" * 70)

print(
    f"Average target-supported actions: "
    f"{network_metrics['num_actions']:.2f}"
)

print(
    f"Entropy: "
    f"{network_metrics['entropy']:.4f}"
)

print(
    f"Normalized entropy: "
    f"{network_metrics['normalized_entropy']:.4f}"
)

print(
    f"Max probability: "
    f"{network_metrics['max_probability']:.4f}"
)


# ============================================================
# PRINT MCTS TARGET RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("STORED MCTS TARGET POLICY")
print("=" * 70)

print(
    f"Average actions: "
    f"{target_metrics['num_actions']:.2f}"
)

print(
    f"Entropy: "
    f"{target_metrics['entropy']:.4f}"
)

print(
    f"Normalized entropy: "
    f"{target_metrics['normalized_entropy']:.4f}"
)

print(
    f"Max probability: "
    f"{target_metrics['max_probability']:.4f}"
)


# ============================================================
# POLICY CROSS-ENTROPY
# ============================================================

target_safe = target_policy.clamp_min(
    1e-12
)

network_log_probs = torch.log(
    network_policy_legal.clamp_min(1e-12)
)

cross_entropy = -(
    target_policy *
    network_log_probs
).sum(dim=1)


print("\n")
print("=" * 70)
print("NETWORK vs MCTS TARGET")
print("=" * 70)

print(
    f"Policy cross-entropy: "
    f"{cross_entropy.mean().item():.4f}"
)


# ============================================================
# KL DIVERGENCE
# ============================================================

kl_divergence = (
    target_policy *
    (
        torch.log(
            target_policy.clamp_min(1e-12)
        )
        -
        torch.log(
            network_policy_legal.clamp_min(1e-12)
        )
    )
).sum(dim=1)


print(
    f"KL(target || network): "
    f"{kl_divergence.mean().item():.4f}"
)


# ============================================================
# VALUE COMPARISON
# ============================================================

network_values = values.squeeze(-1).float()

value_mse = torch.mean(
    (
        network_values -
        target_values
    ) ** 2
)

print(
    f"Value MSE: "
    f"{value_mse.item():.4f}"
)


# ============================================================
# POSITION-BY-POSITION
# ============================================================

print("\n")
print("=" * 110)
print("POSITION-BY-POSITION RESULTS")
print("=" * 110)

print(
    f"{'Pos':>5}"
    f"{'Actions':>10}"
    f"{'Net Ent':>12}"
    f"{'Net Norm':>12}"
    f"{'Net Max':>12}"
    f"{'Target Ent':>13}"
    f"{'Target Norm':>14}"
    f"{'Target Max':>13}"
)

print("-" * 110)

for i in range(NUM_POSITIONS):

    print(
        f"{i + 1:>5}"
        f"{int(network_metrics['num_actions'] if False else target_mask[i].sum().item()):>10}"
        f"{network_metrics['entropy_each'][i].item():>12.4f}"
        f"{network_metrics['normalized_each'][i].item():>12.4f}"
        f"{network_metrics['max_each'][i].item():>12.4f}"
        f"{target_metrics['entropy_each'][i].item():>13.4f}"
        f"{target_metrics['normalized_each'][i].item():>14.4f}"
        f"{target_metrics['max_each'][i].item():>13.4f}"
    )


# ============================================================
# TOP NETWORK vs TARGET MOVES
# ============================================================

print("\n")
print("=" * 70)
print("TOP NETWORK vs MCTS TARGET")
print("=" * 70)


for position in range(
    min(5, NUM_POSITIONS)
):

    mask = target_mask[position]

    legal_actions = torch.nonzero(
        mask,
        as_tuple=False,
    ).flatten()

    # --------------------------------------------------------
    # Network probabilities
    # --------------------------------------------------------

    net_probs = network_policy_legal[
        position,
        legal_actions,
    ]

    net_top_k = min(
        5,
        legal_actions.numel(),
    )

    net_top_probs, net_indices = torch.topk(
        net_probs,
        k=net_top_k,
    )

    net_top_actions = legal_actions[
        net_indices
    ]

    # --------------------------------------------------------
    # Target probabilities
    # --------------------------------------------------------

    target_probs = target_policy[
        position,
        legal_actions,
    ]

    target_top_k = min(
        5,
        legal_actions.numel(),
    )

    target_top_probs, target_indices = torch.topk(
        target_probs,
        k=target_top_k,
    )

    target_top_actions = legal_actions[
        target_indices
    ]

    print(
        f"\nPosition {position + 1}"
    )

    print("\n  NETWORK:")

    for rank in range(net_top_k):

        print(
            f"    {rank + 1}. "
            f"Action "
            f"{net_top_actions[rank].item():4d} "
            f"Prob "
            f"{net_top_probs[rank].item():.4f}"
        )

    print("\n  MCTS TARGET:")

    for rank in range(target_top_k):

        print(
            f"    {rank + 1}. "
            f"Action "
            f"{target_top_actions[rank].item():4d} "
            f"Prob "
            f"{target_top_probs[rank].item():.4f}"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"""
RL53 network:
    Normalized entropy = "
    {network_metrics['normalized_entropy']:.4f}

    Max probability = "
    {network_metrics['max_probability']:.4f}


Stored MCTS targets:
    Normalized entropy = "
    {target_metrics['normalized_entropy']:.4f}

    Max probability = "
    {target_metrics['max_probability']:.4f}


Network vs target:
    Cross-entropy = "
    {cross_entropy.mean().item():.4f}

    KL(target || network) = "
    {kl_divergence.mean().item():.4f}

    Value MSE = "
    {value_mse.item():.4f}
"""
)

print("=" * 70)