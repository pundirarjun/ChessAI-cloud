# ============================================================
# RL53 GPU MCTS BATCHED INTEGRATION TEST
# ============================================================
#
# Purpose:
#   Verify the complete batched MCTS loop used by self-play:
#
#       selection
#         -> virtual loss
#         -> batched NN evaluation
#         -> virtual-loss removal
#         -> expansion
#         -> backup
#         -> repeat
#
# This test does NOT use replay data or self-play.
# It uses a real chess position and the actual ChessNet + GPUMCTS.
#
# It checks:
#   1. Search completes without errors.
#   2. Root visit count == requested simulations.
#   3. Root policy is finite and normalized.
#   4. All selected root actions are legal.
#   5. Increasing simulations increases root visits correctly.
#   6. Virtual-loss buffers are zero after search.
#   7. Real tree statistics contain no NaN/Inf.
#   8. Search works with the actual batched search path.
# ============================================================

import sys
import math
import traceback
import torch

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/kaggle/working/chess-zero"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SIMULATIONS_LIST = [100, 200, 400, 800]
BATCH_SIZE = 16

print("=" * 100)
print("RL53 GPU MCTS BATCHED INTEGRATION TEST")
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

def fail(msg):
    raise AssertionError(msg)


def check(condition, msg):
    if not bool(condition):
        fail(msg)
    print("PASS:", msg)


def check_close(value, expected, msg, tol=1e-5):
    value = float(value)
    expected = float(expected)

    if abs(value - expected) > tol:
        fail(
            f"{msg}: expected {expected:.8f}, got {value:.8f}"
        )

    print(
        f"PASS: {msg}: {value:.8f}"
    )


def tensor_finite(t):
    return bool(torch.isfinite(t).all().item())


def tensor_sum(t):
    return float(t.detach().float().sum().item())


def make_initial_state():
    """
    Construct the standard initial chess position through GPUChess.
    This avoids relying on python-chess for the test position.
    """
    env = GPUChess(device=DEVICE)

    # Try the normal reset API first.
    if hasattr(env, "reset"):
        result = env.reset()

        if torch.is_tensor(result):
            state = result

            if state.ndim == 3:
                state = state.unsqueeze(0)

            return env, state

        if isinstance(result, tuple):
            for item in result:
                if torch.is_tensor(item):
                    state = item

                    if state.ndim == 3:
                        state = state.unsqueeze(0)

                    return env, state

    # Some implementations expose the state directly.
    for attr in ["state", "board_state", "states"]:
        if hasattr(env, attr):
            state = getattr(env, attr)

            if torch.is_tensor(state):
                if state.ndim == 3:
                    state = state.unsqueeze(0)

                return env, state

    raise RuntimeError(
        "Could not obtain the initial GPUChess state. "
        "Inspect GPUChess.reset()/state API."
    )


def get_root_policy(search, root_id=0):
    """
    Prefer the implementation's root_visit_policy().
    Fall back to root_policy() if necessary.
    """
    if hasattr(search, "root_visit_policy"):
        try:
            policy = search.root_visit_policy(
                torch.tensor(
                    [root_id],
                    dtype=torch.long,
                    device=DEVICE,
                )
            )
            return policy
        except Exception:
            pass

    if hasattr(search, "root_policy"):
        try:
            policy = search.root_policy(
                torch.tensor(
                    [root_id],
                    dtype=torch.long,
                    device=DEVICE,
                )
            )
            return policy
        except Exception:
            pass

    raise RuntimeError(
        "Could not obtain root visit policy from GPUMCTS."
    )


def get_root_actions_and_visits(search, root_id=0):
    start = int(search.edge_start[root_id].item())
    count = int(search.edge_count[root_id].item())

    if count <= 0:
        return [], [], []

    end = start + count

    actions = (
        search.edge_action[start:end]
        .detach()
        .cpu()
        .tolist()
    )

    child_ids = (
        search.edge_child[start:end]
        .long()
    )

    visits = (
        search.visit_count[child_ids]
        .detach()
        .cpu()
        .tolist()
    )

    return actions, visits, child_ids.detach().cpu().tolist()


def get_root_legal_actions(env):
    """
    Best-effort extraction of legal root actions.

    The exact GPUChess API has changed during development, so
    try the known legal-move interfaces without altering the
    actual MCTS implementation.
    """
    candidates = [
        "legal_actions",
        "get_legal_actions",
        "legal_move_mask",
    ]

    for name in candidates:
        if not hasattr(env, name):
            continue

        fn = getattr(env, name)

        try:
            result = fn()

            if torch.is_tensor(result):
                x = result

                # Boolean/action mask.
                if x.dtype == torch.bool:
                    if x.ndim > 1:
                        x = x[0]

                    return set(
                        torch.nonzero(
                            x,
                            as_tuple=False
                        ).flatten().cpu().tolist()
                    )

                # Integer action list.
                if x.ndim == 1:
                    return set(
                        x.long().cpu().tolist()
                    )

        except Exception:
            continue

    return None


# ============================================================
# CREATE MODEL + SEARCH
# ============================================================

print()
print("=" * 100)
print("CREATING MODEL + GPUMCTS")
print("=" * 100)

model = ChessNet().to(DEVICE)
model.eval()

search = GPUMCTS(
    model=model,
    device=DEVICE,
    c_puct=1.5,
)

print("Model and GPUMCTS created.")


# ============================================================
# CREATE INITIAL POSITION
# ============================================================

print()
print("=" * 100)
print("CREATING INITIAL CHESS POSITION")
print("=" * 100)

try:
    env, root_state = make_initial_state()

    print("Root state shape:", tuple(root_state.shape))
    print("Root state dtype:", root_state.dtype)
    print("Root state device:", root_state.device)

    check(
        root_state.device.type == DEVICE.type,
        "Root state is on the requested device",
    )

    check(
        root_state.shape[-2:] == (8, 8),
        "Root state has 8x8 board dimensions",
    )

except Exception:
    print()
    print("FAILED TO CREATE INITIAL GPUChess STATE")
    traceback.print_exc()
    raise


# ============================================================
# TEST 1
# SINGLE SEARCH AT 100 SIMULATIONS
# ============================================================

print()
print("=" * 100)
print("TEST 1: 100-SIMULATION BATCHED SEARCH")
print("=" * 100)

try:
    search.search(
        root_states=root_state,
        num_simulations=100,
        dirichlet_alpha=None,
        dirichlet_epsilon=0.0,
        batch_size=BATCH_SIZE,
    )
except Exception:
    print()
    print("100-SIMULATION SEARCH FAILED")
    traceback.print_exc()
    raise

root_n = int(search.visit_count[0].item())

print("Root visit count:", root_n)

check(
    root_n == 100,
    "Root visit count equals requested simulations",
)

check(
    tensor_finite(search.visit_count),
    "All visit counts are finite",
)

check(
    tensor_finite(search.value_sum),
    "All value sums are finite",
)

if hasattr(search, "virtual_visit_count"):
    if search.virtual_visit_count is not None:
        check(
            int(search.virtual_visit_count.abs().sum().item()) == 0,
            "Virtual visit buffer is zero after search",
        )

if hasattr(search, "virtual_value_sum"):
    if search.virtual_value_sum is not None:
        check(
            float(search.virtual_value_sum.abs().sum().item()) < 1e-6,
            "Virtual value buffer is zero after search",
        )


# ============================================================
# TEST 2
# ROOT POLICY VALIDATION
# ============================================================

print()
print("=" * 100)
print("TEST 2: ROOT POLICY VALIDATION")
print("=" * 100)

policy = get_root_policy(search, 0)

print("Root policy shape:", tuple(policy.shape))

check(
    tensor_finite(policy),
    "Root policy contains only finite values",
)

policy_sum = tensor_sum(policy)

print("Root policy sum:", policy_sum)

check_close(
    policy_sum,
    1.0,
    "Root visit policy sums to 1",
    tol=1e-4,
)

check(
    bool((policy >= -1e-6).all().item()),
    "Root policy contains no significant negative probabilities",
)

nonzero_policy = int(
    (policy > 1e-8).sum().item()
)

print("Non-zero root actions:", nonzero_policy)

check(
    nonzero_policy > 0,
    "Root policy contains at least one selected action",
)


# ============================================================
# TEST 3
# ROOT CHILD / VISIT CONSISTENCY
# ============================================================

print()
print("=" * 100)
print("TEST 3: ROOT CHILD / VISIT CONSISTENCY")
print("=" * 100)

actions, visits, child_ids = get_root_actions_and_visits(
    search,
    0,
)

print("Root edge count:", len(actions))
print("Root actions with visits > 0:", sum(v > 0 for v in visits))
print("Total child visits:", sum(visits))

check(
    len(actions) > 0,
    "Root has expanded children",
)

check(
    all(v >= 0 for v in visits),
    "All root child visit counts are non-negative",
)

check(
    sum(visits) == root_n,
    "Root child visit counts sum to root visit count",
)

check(
    len(actions) == len(set(actions)),
    "Root child actions are unique",
)

check(
    all(cid >= 0 for cid in child_ids),
    "All root child IDs are valid",
)


# ============================================================
# TEST 4
# LEGALITY CROSS-CHECK
# ============================================================

print()
print("=" * 100)
print("TEST 4: ROOT ACTION LEGALITY CROSS-CHECK")
print("=" * 100)

legal_actions = get_root_legal_actions(env)

if legal_actions is None:
    print(
        "WARNING: Could not obtain a compatible legal-action "
        "interface from GPUChess."
    )
    print(
        "Skipping direct action-set comparison; "
        "other MCTS consistency tests remain active."
    )
else:
    illegal_actions = [
        action
        for action in actions
        if action not in legal_actions
    ]

    print("GPUChess legal action count:", len(legal_actions))
    print("MCTS root action count:", len(actions))
    print("Illegal MCTS root actions:", illegal_actions[:20])

    check(
        len(illegal_actions) == 0,
        "Every MCTS root action is legal",
    )


# ============================================================
# TEST 5
# MULTI-DEPTH SIMULATION CONSISTENCY
# ============================================================

print()
print("=" * 100)
print("TEST 5: 100 / 200 / 400 / 800 SIMULATION CONSISTENCY")
print("=" * 100)

results = []

for sims in SIMULATIONS_LIST:

    print()
    print("-" * 90)
    print(f"Running {sims} simulations")
    print("-" * 90)

    # Fresh search object for each depth so the test measures
    # exactly the requested number of simulations.
    depth_search = GPUMCTS(
        model=model,
        device=DEVICE,
        c_puct=1.5,
    )

    try:
        depth_search.search(
            root_states=root_state,
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

    n = int(
        depth_search.visit_count[0].item()
    )

    policy_depth = get_root_policy(
        depth_search,
        0,
    )

    p_sum = tensor_sum(policy_depth)

    actions_d, visits_d, child_ids_d = (
        get_root_actions_and_visits(
            depth_search,
            0,
        )
    )

    total_child_visits = sum(visits_d)

    print("Root N:", n)
    print("Policy sum:", p_sum)
    print("Root child count:", len(actions_d))
    print("Total child visits:", total_child_visits)

    check(
        n == sims,
        f"Root visit count equals {sims}",
    )

    check_close(
        p_sum,
        1.0,
        f"Root policy sums to 1 at {sims} simulations",
        tol=1e-4,
    )

    check(
        total_child_visits == sims,
        f"Child visits sum to {sims}",
    )

    check(
        tensor_finite(depth_search.visit_count),
        f"Visit counts finite at {sims} simulations",
    )

    check(
        tensor_finite(depth_search.value_sum),
        f"Value sums finite at {sims} simulations",
    )

    if hasattr(depth_search, "virtual_visit_count"):
        if depth_search.virtual_visit_count is not None:
            check(
                int(
                    depth_search.virtual_visit_count
                    .abs()
                    .sum()
                    .item()
                ) == 0,
                f"Virtual visits cleared at {sims} simulations",
            )

    if hasattr(depth_search, "virtual_value_sum"):
        if depth_search.virtual_value_sum is not None:
            check(
                float(
                    depth_search.virtual_value_sum
                    .abs()
                    .sum()
                    .item()
                ) < 1e-6,
                f"Virtual values cleared at {sims} simulations",
            )

    top_k = min(10, len(actions_d))

    if top_k > 0:
        pairs = sorted(
            zip(actions_d, visits_d),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        print("Top root actions:")
        for action, visit in pairs:
            print(
                f"  action={action:4d} "
                f"visits={visit:4d}"
            )

    results.append(
        {
            "simulations": sims,
            "root_visits": n,
            "policy_sum": p_sum,
            "child_visits": total_child_visits,
            "children": len(actions_d),
        }
    )


# ============================================================
# TEST 6
# ROOT POLICY EVOLUTION
# ============================================================

print()
print("=" * 100)
print("TEST 6: ROOT POLICY EVOLUTION")
print("=" * 100)

print()
print("Simulation summary:")

for r in results:
    print(
        f"{r['simulations']:4d} sims | "
        f"root N={r['root_visits']:4d} | "
        f"children={r['children']:3d} | "
        f"child visits={r['child_visits']:4d} | "
        f"policy sum={r['policy_sum']:.6f}"
    )

# Root visits must grow exactly with simulations.
for i in range(1, len(results)):
    previous = results[i - 1]
    current = results[i]

    check(
        current["root_visits"] > previous["root_visits"],
        (
            f"Root visit count increases from "
            f"{previous['simulations']} to "
            f"{current['simulations']} simulations"
        ),
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

print()
print("=" * 100)
print("BATCHED MCTS INTEGRATION TEST COMPLETE")
print("=" * 100)

print()
print("Verified:")
print("  1. Complete batched MCTS search executes.")
print("  2. Root visits equal requested simulations.")
print("  3. Root policy is finite and normalized.")
print("  4. Root child visits are internally consistent.")
print("  5. MCTS tree statistics remain finite.")
print("  6. Virtual-loss buffers are cleared after search.")
print("  7. Search remains consistent from 100 -> 800 simulations.")
print()
print("If all checks above PASS, the next diagnostic should")
print("move to the actual self-play integration path.")
print("=" * 100)
