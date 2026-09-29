import torch
import numpy as np

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
    {"name": "100 sims + noise", "simulations": 100, "noise": True},
    {"name": "100 sims + no noise", "simulations": 100, "noise": False},
    {"name": "200 sims + no noise", "simulations": 200, "noise": False},
    {"name": "400 sims + no noise", "simulations": 400, "noise": False},
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

# Initial positions.
# This gives us a controlled identical starting position
# for every experiment.
states.reset()

print(f"Positions: {NUM_POSITIONS}")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(policy):
    """
    policy shape:
        [batch, 4544]
    """

    # Only non-zero actions matter.
    nonzero = policy > 0

    # Entropy
    safe_policy = policy.clamp_min(1e-12)

    entropy = -(safe_policy * safe_policy.log()).sum(dim=1)

    # Number of actions receiving visits
    num_visited = nonzero.sum(dim=1).float()

    # Maximum visit probability
    max_probability = policy.max(dim=1).values

    # Normalized entropy.
    #
    # We normalize by log(number of legal actions).
    #
    # 0 = very concentrated
    # 1 = nearly uniform over legal moves
    entropy_per_position = []

    for i in range(policy.shape[0]):
        p = policy[i]
        mask = p > 0

        n = mask.sum()

        if n > 1:
            max_entropy = torch.log(n.float())
            normalized = entropy[i] / max_entropy
        else:
            normalized = torch.tensor(0.0, device=policy.device)

        entropy_per_position.append(normalized)

    normalized_entropy = torch.stack(entropy_per_position)

    return {
        "entropy": entropy.mean().item(),
        "normalized_entropy": normalized_entropy.mean().item(),
        "max_probability": max_probability.mean().item(),
        "visited_actions": num_visited.mean().item(),
        "max_probability_min": max_probability.min().item(),
        "max_probability_max": max_probability.max().item(),
    }


# ============================================================
# RUN TESTS
# ============================================================

results = []

for test in TESTS:

    print("\n" + "=" * 70)
    print(test["name"])
    print("=" * 70)

    # IMPORTANT:
    # Create a fresh MCTS tree for every test.
    mcts = GPUMCTS(
        model=model,
        device=DEVICE,
        c_puct=1.5,
    )

    # Fresh identical positions
    test_states = GPUChess(DEVICE, NUM_POSITIONS)
    test_states.reset()

    # Run MCTS
    mcts.search(
        root_states=test_states,
        num_simulations=test["simulations"],
        dirichlet_alpha=0.3 if test["noise"] else None,
        dirichlet_epsilon=0.25,
        batch_size=BATCH_SIZE,
    )

    # Get visit-count policy.
    policy = mcts.root_visit_policy()

    metrics = calculate_metrics(policy)

    results.append({
        "name": test["name"],
        **metrics,
    })

    print(f"Simulations:          {test['simulations']}")
    print(f"Dirichlet noise:      {test['noise']}")
    print(f"Entropy:              {metrics['entropy']:.4f}")
    print(f"Normalized entropy:   {metrics['normalized_entropy']:.4f}")
    print(f"Max visit probability:{metrics['max_probability']:.4f}")
    print(f"Visited actions:      {metrics['visited_actions']:.2f}")
    print(f"Max probability min:  {metrics['max_probability_min']:.4f}")
    print(f"Max probability max:  {metrics['max_probability_max']:.4f}")


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

for r in results:

    print(
        f"{r['name']:<25}"
        f"{r['entropy']:>12.4f}"
        f"{r['normalized_entropy']:>15.4f}"
        f"{r['max_probability']:>12.4f}"
        f"{r['visited_actions']:>12.2f}"
    )

print("=" * 100)