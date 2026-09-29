# ============================================================
# RL53 GPU MCTS BATCHED INTEGRATION TEST - FIXED
# ============================================================
#
# Tests the complete actual GPUMCTS.search() path:
#
#   root GPUChess state
#       -> root NN evaluation
#       -> root expansion
#       -> batched selection
#       -> virtual loss
#       -> batched NN evaluation
#       -> expansion
#       -> backup
#       -> repeated simulations
#
# IMPORTANT:
# GPUChess.reset() returns the GPUChess object itself.  GPUMCTS.search()
# expects a GPUChess object, NOT a model-input tensor.
# ============================================================

import sys
import traceback
import torch

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/kaggle/working/chess-zero"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

C_PUCT = 1.5
BATCH_SIZE = 16
SIMULATIONS = [100, 200, 400, 800]

print("=" * 100)
print("RL53 GPU MCTS BATCHED INTEGRATION TEST - FIXED")
print("=" * 100)
print()
print("Project:", PROJECT_DIR)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

sys.path.insert(0, PROJECT_DIR)

from model.chess_net import ChessNet
from mcts.gpu_mcts import GPUMCTS
from environment.gpu_chess import GPUChess


# ============================================================
# HELPERS
# ============================================================

def check(condition, message):
    if not bool(condition):
        raise AssertionError("FAILED: " + message)
    print("PASS:", message)


def check_close(value, expected, message, tol=1e-5):
    value = float(value)
    expected = float(expected)

    if abs(value - expected) > tol:
        raise AssertionError(
            f"FAILED: {message}: expected {expected:.8f}, "
            f"got {value:.8f}"
        )

    print(f"PASS: {message}: {value:.8f}")


def finite(tensor):
    return bool(torch.isfinite(tensor).all().item())


def root_stats(search):
    root = int(search._root_ids[0].item())

    n = int(search.visit_count[root].item())
    w = float(search.value_sum[root].item())

    q = w / n if n > 0 else 0.0

    return root, n, w, q


def root_edges(search):
    root = int(search._root_ids[0].item())

    start = int(search.edge_start[root].item())
    count = int(search.edge_count[root].item())
    end = start + count

    actions = search.edge_action[start:end].detach()
    priors = search.edge_prior[start:end].detach()

    child_ids = search.edge_child[start:end].long()
    visits = search.visit_count[child_ids].detach()
    values = search.value_sum[child_ids].detach()

    return actions, priors, child_ids, visits, values


def check_virtual_buffers(search, label):
    if hasattr(search, "virtual_visit_count"):
        if search.virtual_visit_count is not None:
            total = int(
                search.virtual_visit_count.abs().sum().item()
            )
            check(
                total == 0,
                f"{label}: virtual visit buffer is zero",
            )

    if hasattr(search, "virtual_value_sum"):
        if search.virtual_value_sum is not None:
            total = float(
                search.virtual_value_sum.abs().sum().item()
            )
            check(
                total < 1e-6,
                f"{label}: virtual value buffer is zero",
            )


def print_top_children(search, limit=10):
    actions, priors, child_ids, visits, values = root_edges(search)

    rows = []

    for i in range(len(actions)):
        rows.append(
            (
                int(actions[i].item()),
                int(child_ids[i].item()),
                float(priors[i].item()),
                int(visits[i].item()),
                float(values[i].item()),
            )
        )

    rows.sort(key=lambda x: x[3], reverse=True)

    print()
    print("Top root children:")
    print(
        "  action | child | prior      | visits | value_sum"
    )
    print("  " + "-" * 55)

    for action, child, prior, visits, value in rows[:limit]:
        print(
            f"  {action:6d} | "
            f"{child:5d} | "
            f"{prior:10.6f} | "
            f"{visits:6d} | "
            f"{value:+10.6f}"
        )


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 100)
print("CREATING MODEL")
print("=" * 100)

model = ChessNet().to(DEVICE)
model.eval()

print("Model created.")


# ============================================================
# TEST 1
# CREATE REAL GPUCHESS ROOT
# ============================================================

print()
print("=" * 100)
print("TEST 1: CREATE INITIAL GPUCHESS ROOT")
print("=" * 100)

try:
    root_states = GPUChess(
        device=DEVICE,
        batch_size=1,
    )

    print("GPUChess created.")
    print("Pieces shape:", tuple(root_states.pieces.shape))
    print("Turn shape:", tuple(root_states.turn.shape))
    print("Device:", root_states.device)

    check(
        root_states.pieces.shape == (1, 12),
        "Initial GPUChess has one 12-plane bitboard state",
    )

    check(
        root_states.device == DEVICE,
        "GPUChess is on the requested device",
    )

    check(
        int(root_states.turn[0].item()) == 0,
        "Initial side to move is White",
    )

except Exception:
    print()
    print("GPUChess creation failed.")
    traceback.print_exc()
    raise


# ============================================================
# TEST 2
# INITIAL LEGAL MOVES
# ============================================================

print()
print("=" * 100)
print("TEST 2: INITIAL LEGAL MOVE GENERATION")
print("=" * 100)

try:
    legal = root_states.legal_move_mask()

    print("Legal mask shape:", tuple(legal.shape))
    print("Legal dtype:", legal.dtype)

    legal_count = int(legal[0].sum().item())

    print("Initial legal moves:", legal_count)

    check(
        legal.shape == (1, 4544),
        "Legal move mask has 4544 action slots",
    )

    check(
        legal.dtype == torch.bool,
        "Legal move mask is boolean",
    )

    check(
        legal_count == 20,
        "Initial chess position has 20 legal moves",
    )

except Exception:
    print()
    print("Initial legal-move generation failed.")
    traceback.print_exc()
    raise


# ============================================================
# TEST 3
# 100-SIMULATION FULL BATCHED SEARCH
# ============================================================

print()
print("=" * 100)
print("TEST 3: 100-SIMULATION FULL BATCHED SEARCH")
print("=" * 100)

search = GPUMCTS(
    model=model,
    device=DEVICE,
    c_puct=C_PUCT,
)

try:
    search.search(
        root_states=root_states,
        num_simulations=100,
        dirichlet_alpha=None,
        dirichlet_epsilon=0.0,
        batch_size=BATCH_SIZE,
    )
except Exception:
    print()
    print("Full batched MCTS search failed.")
    traceback.print_exc()
    raise

root, n, w, q = root_stats(search)

print()
print("Root node:", root)
print("Root N:", n)
print("Root W:", w)
print("Root Q:", q)

check(
    n == 100,
    "Root visit count equals 100 simulations",
)

check(
    finite(search.visit_count),
    "All visit counts are finite",
)

check(
    finite(search.value_sum),
    "All value sums are finite",
)

check_virtual_buffers(
    search,
    "After 100 simulations",
)


# ============================================================
# TEST 4
# ROOT POLICY
# ============================================================

print()
print("=" * 100)
print("TEST 4: ROOT VISIT POLICY")
print("=" * 100)

policy = search.root_visit_policy()

print("Policy shape:", tuple(policy.shape))
print("Policy sum:", float(policy.sum().item()))
print("Policy max:", float(policy.max().item()))
print("Policy nonzero:", int((policy > 0).sum().item()))

check(
    policy.shape == (1, 4544),
    "Root policy shape is [1, 4544]",
)

check(
    finite(policy),
    "Root policy contains only finite values",
)

check_close(
    policy.sum().item(),
    1.0,
    "Root policy sums to 1",
    tol=1e-5,
)

check(
    bool((policy >= -1e-7).all().item()),
    "Root policy has no negative probabilities",
)

check(
    int((policy > 0).sum().item()) > 0,
    "Root policy has at least one nonzero action",
)


# ============================================================
# TEST 5
# ROOT CHILD VISIT CONSISTENCY
# ============================================================

print()
print("=" * 100)
print("TEST 5: ROOT CHILD VISIT CONSISTENCY")
print("=" * 100)

actions, priors, child_ids, visits, values = root_edges(search)

print("Root edge count:", len(actions))
print("Visited root children:", int((visits > 0).sum().item()))
print("Total child visits:", int(visits.sum().item()))

check(
    len(actions) > 0,
    "Root has children",
)

check(
    len(actions) == len(set(actions.cpu().tolist())),
    "Root actions are unique",
)

check(
    bool((visits >= 0).all().item()),
    "All child visit counts are non-negative",
)

check(
    int(visits.sum().item()) == 100,
    "Root child visits sum to 100",
)

check(
    bool((priors >= -1e-7).all().item()),
    "Root priors are non-negative",
)

print_top_children(search)


# ============================================================
# TEST 6
# 100 -> 200 -> 400 -> 800 FRESH SEARCHES
# ============================================================

print()
print("=" * 100)
print("TEST 6: SIMULATION-SCALE CONSISTENCY")
print("=" * 100)

results = []

for sims in SIMULATIONS:

    print()
    print("-" * 90)
    print(f"RUNNING {sims} SIMULATIONS")
    print("-" * 90)

    # Fresh root state and fresh tree for every simulation count.
    states = GPUChess(
        device=DEVICE,
        batch_size=1,
    )

    mcts = GPUMCTS(
        model=model,
        device=DEVICE,
        c_puct=C_PUCT,
    )

    try:
        mcts.search(
            root_states=states,
            num_simulations=sims,
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0,
            batch_size=BATCH_SIZE,
        )
    except Exception:
        print()
        print(f"{sims}-simulation search FAILED")
        traceback.print_exc()
        raise

    root, n, w, q = root_stats(mcts)

    policy = mcts.root_visit_policy()

    actions, priors, child_ids, visits, values = root_edges(mcts)

    policy_sum = float(policy.sum().item())
    child_visit_sum = int(visits.sum().item())

    print()
    print("Root N:", n)
    print("Root W:", w)
    print("Root Q:", q)
    print("Root children:", len(actions))
    print("Child visit sum:", child_visit_sum)
    print("Policy sum:", policy_sum)

    check(
        n == sims,
        f"Root N equals requested {sims} simulations",
    )

    check(
        child_visit_sum == sims,
        f"Child visits sum to {sims}",
    )

    check_close(
        policy_sum,
        1.0,
        f"Policy sums to 1 at {sims} simulations",
        tol=1e-5,
    )

    check(
        finite(mcts.visit_count),
        f"Visit counts finite at {sims} simulations",
    )

    check(
        finite(mcts.value_sum),
        f"Value sums finite at {sims} simulations",
    )

    check(
        finite(policy),
        f"Policy finite at {sims} simulations",
    )

    check_virtual_buffers(
        mcts,
        f"After {sims} simulations",
    )

    results.append(
        {
            "sims": sims,
            "root_n": n,
            "children": len(actions),
            "child_visit_sum": child_visit_sum,
            "policy_sum": policy_sum,
            "root_q": q,
        }
    )

    print_top_children(mcts)


# ============================================================
# TEST 7
# CHECK MONOTONIC ROOT VISITS
# ============================================================

print()
print("=" * 100)
print("TEST 7: ROOT VISIT COUNT SCALING")
print("=" * 100)

for i in range(1, len(results)):

    previous = results[i - 1]
    current = results[i]

    check(
        current["root_n"] > previous["root_n"],
        (
            f"Root N increases "
            f"{previous['sims']} -> {current['sims']}"
        ),
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

print()
print("=" * 100)
print("RL53 BATCHED MCTS INTEGRATION TEST COMPLETE")
print("=" * 100)

print()
print("Verified:")
print("  GPUChess initial state creation")
print("  Initial legal move generation")
print("  Full GPUMCTS.search()")
print("  Batched selection")
print("  Virtual loss")
print("  NN evaluation")
print("  Expansion")
print("  Backup")
print("  Root policy normalization")
print("  Root child visit accounting")
print("  100 / 200 / 400 / 800 simulation scaling")
print("  Virtual-loss cleanup")
print()
print("If all tests PASS, the next test is self-play integration.")
print("=" * 100)
