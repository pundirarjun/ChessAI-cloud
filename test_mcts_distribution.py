import torch

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "/kaggle/working/chess-zero/checkpoints/rl_iteration_53.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_POSITIONS = 32

BATCH_SIZE = 16

TESTS = [
    {
        "name": "100 sims + noise",
        "simulations": 100,
        "noise": True,
    },
    {
        "name": "100 sims + no noise",
        "simulations": 100,
        "noise": False,
    },
    {
        "name": "200 sims + no noise",
        "simulations": 200,
        "noise": False,
    },
    {
        "name": "400 sims + no noise",
        "simulations": 400,
        "noise": False,
    },
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("MCTS DISTRIBUTION DIAGNOSTIC")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")

model = ChessNet().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded.")


# ============================================================
# CREATE TEST POSITIONS
# ============================================================

print("\nCreating test positions...")

states = GPUChess(DEVICE, NUM_POSITIONS)

# Use identical starting positions for every experiment.
states.reset()

print(f"Positions: {NUM_POSITIONS}")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(policy):
    """
    Calculate policy distribution statistics.

    policy shape:
        [batch, 4544]
    """

    nonzero = policy > 0

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    safe_policy = policy.clamp_min(1e-12)

    entropy = -(safe_policy * safe_policy.log()).sum(dim=1)

    # --------------------------------------------------------
    # Number of actions receiving probability / visits
    # --------------------------------------------------------

    num_visited = nonzero.sum(dim=1).float()

    # --------------------------------------------------------
    # Maximum probability
    # --------------------------------------------------------

    max_probability = policy.max(dim=1).values

    # --------------------------------------------------------
    # Normalized entropy
    # --------------------------------------------------------

    normalized_entropy = []

    for i in range(policy.shape[0]):

        p = policy[i]

        mask = p > 0

        n = mask.sum()

        if n > 1:

            max_entropy = torch.log(n.float())

            normalized = entropy[i] / max_entropy

        else:

            normalized = torch.tensor(
                0.0,
                device=policy.device,
            )

        normalized_entropy.append(normalized)

    normalized_entropy = torch.stack(normalized_entropy)

    return {
        "entropy": entropy.mean().item(),

        "normalized_entropy": normalized_entropy.mean().item(),

        "max_probability": max_probability.mean().item(),

        "visited_actions": num_visited.mean().item(),

        "max_probability_min": max_probability.min().item(),

        "max_probability_max": max_probability.max().item(),
    }


# ============================================================
# RAW NETWORK POLICY TEST
# ============================================================

print("\n")
print("=" * 70)
print("RAW NETWORK POLICY")
print("=" * 70)

print(
    "Testing the neural network policy BEFORE MCTS..."
)

with torch.inference_mode():

    model_input = states.to_model_input()

    if model_input.is_cuda:

        model_input = model_input.contiguous(
            memory_format=torch.channels_last
        )

    # --------------------------------------------------------
    # Neural network prediction
    # --------------------------------------------------------

    if DEVICE.type == "cuda":

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            logits, values = model(model_input)

    else:

        logits, values = model(model_input)

    # --------------------------------------------------------
    # Legal move mask
    # --------------------------------------------------------

    legal = states.legal_move_mask()

    # --------------------------------------------------------
    # Remove illegal moves from policy
    # --------------------------------------------------------

    masked_logits = logits.float().masked_fill(
        ~legal,
        torch.finfo(torch.float32).min,
    )

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    network_policy = torch.softmax(
        masked_logits,
        dim=1,
    )

raw_metrics = calculate_metrics(network_policy)

print(
    f"Entropy:              "
    f"{raw_metrics['entropy']:.4f}"
)

print(
    f"Normalized entropy:   "
    f"{raw_metrics['normalized_entropy']:.4f}"
)

print(
    f"Max probability:      "
    f"{raw_metrics['max_probability']:.4f}"
)

print(
    f"Visited actions:      "
    f"{raw_metrics['visited_actions']:.2f}"
)

print(
    f"Max probability min:  "
    f"{raw_metrics['max_probability_min']:.4f}"
)

print(
    f"Max probability max:  "
    f"{raw_metrics['max_probability_max']:.4f}"
)


# ============================================================
# SHOW EXAMPLE RAW NETWORK POLICY
# ============================================================

print("\n")
print("=" * 70)
print("EXAMPLE RAW NETWORK POLICY")
print("=" * 70)

example_policy = network_policy[0]

legal_actions = torch.nonzero(
    legal[0],
    as_tuple=False,
).flatten()

legal_probs = example_policy[legal_actions]

top_k = min(10, legal_actions.numel())

top_probs, top_indices = torch.topk(
    legal_probs,
    k=top_k,
)

top_actions = legal_actions[top_indices]

print("\nTop network actions for position 0:")

for rank in range(top_k):

    action = top_actions[rank].item()

    probability = top_probs[rank].item()

    print(
        f"{rank + 1:2d}. "
        f"Action {action:4d}  "
        f"Probability {probability:.4f}"
    )


# ============================================================
# RUN MCTS TESTS
# ============================================================

results = []

for test in TESTS:

    print("\n")
    print("=" * 70)
    print(test["name"])
    print("=" * 70)

    # --------------------------------------------------------
    # Fresh MCTS tree
    # --------------------------------------------------------

    mcts = GPUMCTS(
        model=model,
        device=DEVICE,
        c_puct=1.5,
    )

    # --------------------------------------------------------
    # Fresh identical positions
    # --------------------------------------------------------

    test_states = GPUChess(
        DEVICE,
        NUM_POSITIONS,
    )

    test_states.reset()

    # --------------------------------------------------------
    # Run MCTS
    # --------------------------------------------------------

    mcts.search(
        root_states=test_states,
        num_simulations=test["simulations"],
        dirichlet_alpha=(
            0.3 if test["noise"] else None
        ),
        dirichlet_epsilon=0.25,
        batch_size=BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Get MCTS visit-count policy
    # --------------------------------------------------------

    policy = mcts.root_visit_policy()

    metrics = calculate_metrics(policy)

    results.append(
        {
            "name": test["name"],
            **metrics,
        }
    )

    print(
        f"Simulations:          "
        f"{test['simulations']}"
    )

    print(
        f"Dirichlet noise:      "
        f"{test['noise']}"
    )

    print(
        f"Entropy:              "
        f"{metrics['entropy']:.4f}"
    )

    print(
        f"Normalized entropy:   "
        f"{metrics['normalized_entropy']:.4f}"
    )

    print(
        f"Max visit probability:"
        f"{metrics['max_probability']:.4f}"
    )

    print(
        f"Visited actions:      "
        f"{metrics['visited_actions']:.2f}"
    )

    print(
        f"Max probability min:  "
        f"{metrics['max_probability_min']:.4f}"
    )

    print(
        f"Max probability max:  "
        f"{metrics['max_probability_max']:.4f}"
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 100)
print("FINAL RESULTS")
print("=" * 100)

print(
    f"{'Test':<25}"
    f"{'Entropy':>12}"
    f"{'Norm Entropy':>15}"
    f"{'Max Prob':>12}"
    f"{'Visited':>12}"
)

print("-" * 100)


# ------------------------------------------------------------
# Raw network
# ------------------------------------------------------------

print(
    f"{'RAW NETWORK':<25}"
    f"{raw_metrics['entropy']:>12.4f}"
    f"{raw_metrics['normalized_entropy']:>15.4f}"
    f"{raw_metrics['max_probability']:>12.4f}"
    f"{raw_metrics['visited_actions']:>12.2f}"
)


# ------------------------------------------------------------
# MCTS results
# ------------------------------------------------------------

for r in results:

    print(
        f"{r['name']:<25}"
        f"{r['entropy']:>12.4f}"
        f"{r['normalized_entropy']:>15.4f}"
        f"{r['max_probability']:>12.4f}"
        f"{r['visited_actions']:>12.2f}"
    )

print("=" * 100)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    "\nRAW NETWORK tells us what RL53 itself believes."
)

print(
    "MCTS results tell us how the search changes that policy."
)

print(
    "\nIf RAW NETWORK is already highly uniform, "
    "the main issue is likely upstream of MCTS."
)

print(
    "If RAW NETWORK is concentrated but MCTS becomes "
    "highly uniform, we should investigate MCTS."
)

print(
    "\nCompare RAW NETWORK against the four MCTS tests "
    "before changing any training or MCTS parameters."
)

print("=" * 70)