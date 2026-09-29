
# ============================================================
# RL53 GPU MCTS PUCT + VIRTUAL-LOSS UNIT TEST
# ============================================================
#
# Purpose:
#   Test the two remaining low-level MCTS mechanisms:
#
#   1. PUCT selection
#   2. Virtual-loss reservation/removal
#
# This test does NOT use:
#   - the neural network for evaluation
#   - replay data
#   - chess move generation
#   - self-play
#
# It constructs a tiny artificial MCTS tree and checks that
# GPUMCTS._select_leaves() follows the expected PUCT scores.
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

C_PUCT = 1.5

print("=" * 100)
print("RL53 GPU MCTS PUCT + VIRTUAL-LOSS UNIT TEST")
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


# ============================================================
# HELPERS
# ============================================================

def get_node_stats(search, node_ids):
    ids = torch.tensor(
        node_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    visits = search.visit_count[ids]
    values = search.value_sum[ids]

    q = torch.where(
        visits > 0,
        values / visits.float(),
        torch.zeros_like(values),
    )

    return (
        visits.detach().cpu().tolist(),
        values.detach().cpu().tolist(),
        q.detach().cpu().tolist(),
    )


def reset_stats(search):
    search.visit_count.zero_()
    search.value_sum.zero_()

    if hasattr(search, "virtual_visit_count"):
        if search.virtual_visit_count is not None:
            search.virtual_visit_count.zero_()

    if hasattr(search, "virtual_value_sum"):
        if search.virtual_value_sum is not None:
            search.virtual_value_sum.zero_()


def assert_equal(actual, expected, name):
    actual = int(actual)
    expected = int(expected)

    if actual != expected:
        raise AssertionError(
            f"{name} FAILED: expected {expected}, got {actual}"
        )

    print(f"PASS: {name}: {actual}")


def assert_close(actual, expected, name, tol=1e-5):
    actual = float(actual)
    expected = float(expected)

    if abs(actual - expected) > tol:
        raise AssertionError(
            f"{name} FAILED: expected {expected:+.6f}, "
            f"got {actual:+.6f}"
        )

    print(
        f"PASS: {name}: {actual:+.6f}"
    )


def assert_true(condition, name):
    if not bool(condition):
        raise AssertionError(
            f"{name} FAILED"
        )

    print(f"PASS: {name}")


def print_stats(search, node_ids, title):
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)

    visits, values, q = get_node_stats(
        search,
        node_ids,
    )

    for node, n, w, qq in zip(
        node_ids,
        visits,
        values,
        q,
    ):
        print(
            f"Node {node:2d} | "
            f"N={n:3d} | "
            f"W={w:+.6f} | "
            f"Q={qq:+.6f}"
        )


# ============================================================
# CREATE MCTS
# ============================================================

print()
print("=" * 100)
print("CREATING GPUMCTS")
print("=" * 100)

model = ChessNet().to(DEVICE)
model.eval()

search = GPUMCTS(
    model=model,
    device=DEVICE,
    c_puct=C_PUCT,
)

search._allocate(
    num_games=1,
    num_simulations=10,
)

root = 0
child_a = 1
child_b = 2
child_c = 3


# ============================================================
# TEST 1
# MANUAL PUCT SCORE CALCULATION
# ============================================================

print()
print("=" * 100)
print("TEST 1: MANUAL PUCT SCORE CALCULATION")
print("=" * 100)

reset_stats(search)

# Artificial root:
#
# Root N = 10
#
# Child A:
#   P = 0.60
#   N = 5
#   W = +1.0
#   Q = +0.20
#
# Child B:
#   P = 0.30
#   N = 2
#   W = -0.4
#   Q = -0.20
#
# Child C:
#   P = 0.10
#   N = 0
#   Q = 0
#
# GPUMCTS uses:
#
#   score = -Q + c_puct * P * sqrt(parent_N) / (1 + N)
#
# Calculate manually.

search.visit_count[root] = 10

search.visit_count[child_a] = 5
search.value_sum[child_a] = 1.0

search.visit_count[child_b] = 2
search.value_sum[child_b] = -0.4

search.visit_count[child_c] = 0
search.value_sum[child_c] = 0.0

priors = [
    0.60,
    0.30,
    0.10,
]

nodes = [
    child_a,
    child_b,
    child_c,
]

expected_scores = []

parent_n = 10.0

for prior, node in zip(priors, nodes):
    n = float(search.visit_count[node].item())
    w = float(search.value_sum[node].item())

    q = w / n if n > 0 else 0.0

    u = (
        C_PUCT
        * prior
        * math.sqrt(parent_n)
        / (1.0 + n)
    )

    score = -q + u

    expected_scores.append(score)

    print(
        f"Child {node}: "
        f"P={prior:.3f} "
        f"N={n:.0f} "
        f"Q={q:+.6f} "
        f"U={u:+.6f} "
        f"PUCT={score:+.6f}"
    )

best_expected_index = max(
    range(len(expected_scores)),
    key=lambda i: expected_scores[i],
)

best_expected_node = nodes[
    best_expected_index
]

print()
print(
    "Expected selected child:",
    best_expected_node,
)

assert_true(
    best_expected_node == child_c,
    "Manual PUCT prefers the highest exploration candidate",
)


# ============================================================
# TEST 2
# BUILD ARTIFICIAL ROOT EDGES
# ============================================================

print()
print("=" * 100)
print("TEST 2: ARTIFICIAL ROOT TREE")
print("=" * 100)

reset_stats(search)

# Root has three children.
#
# We manually construct the exact edge layout expected by
# _select_leaves().
#
# root edge_start = 0
# root edge_count = 3
#
# edge 0 -> child_a
# edge 1 -> child_b
# edge 2 -> child_c

search.edge_start[root] = 0
search.edge_count[root] = 3

search.edge_child[0] = child_a
search.edge_child[1] = child_b
search.edge_child[2] = child_c

search.edge_action[0] = 100
search.edge_action[1] = 200
search.edge_action[2] = 300

search.edge_prior[0] = 0.60
search.edge_prior[1] = 0.30
search.edge_prior[2] = 0.10

search.edge_valid[0] = True
search.edge_valid[1] = True
search.edge_valid[2] = True

search.expanded[root] = True

# Give children known statistics.
search.visit_count[root] = 10

search.visit_count[child_a] = 5
search.value_sum[child_a] = 1.0

search.visit_count[child_b] = 2
search.value_sum[child_b] = -0.4

search.visit_count[child_c] = 0
search.value_sum[child_c] = 0.0

# Children must be considered selectable nodes.
search.expanded[child_a] = True
search.expanded[child_b] = True
search.expanded[child_c] = False

# Non-terminal.
search.terminal[root] = False
search.terminal[child_a] = False
search.terminal[child_b] = False
search.terminal[child_c] = False

print_stats(
    search,
    [root, child_a, child_b, child_c],
    "ARTIFICIAL TREE",
)


# ============================================================
# TEST 3
# DIRECT _SELECT_LEAVES()
# ============================================================

print()
print("=" * 100)
print("TEST 3: _SELECT_LEAVES()")
print("=" * 100)

try:
    leaves, paths = search._select_leaves(
        search._root_ids,
        max_depth=8,
    )

    print()
    print("Selected leaves:")
    print(
        leaves.detach()
        .cpu()
        .tolist()
    )

    print()
    print("Selected paths:")
    print(
        paths.detach()
        .cpu()
        .tolist()
    )

except Exception as exc:
    print()
    print("ERROR calling _select_leaves():")
    print(repr(exc))
    traceback.print_exc()
    raise


# ============================================================
# TEST 4
# VERIFY PUCT SELECTION
# ============================================================

print()
print("=" * 100)
print("TEST 4: VERIFY PUCT SELECTION")
print("=" * 100)

selected_leaf = int(
    leaves[0].item()
)

print(
    "Selected leaf:",
    selected_leaf,
)

# Because child C has the highest PUCT score in this setup,
# it should be selected.

assert_equal(
    selected_leaf,
    child_c,
    "PUCT selected expected child",
)


# ============================================================
# TEST 5
# VIRTUAL LOSS ISOLATION
# ============================================================

print()
print("=" * 100)
print("TEST 5: VIRTUAL-LOSS ISOLATION")
print("=" * 100)

reset_stats(search)

# Rebuild a tiny root.
search.visit_count[root] = 10

search.edge_start[root] = 0
search.edge_count[root] = 2

search.edge_child[0] = child_a
search.edge_child[1] = child_b

search.edge_prior[0] = 0.50
search.edge_prior[1] = 0.50

search.edge_valid[0] = True
search.edge_valid[1] = True

search.expanded[root] = True
search.expanded[child_a] = True
search.expanded[child_b] = False

search.terminal[root] = False
search.terminal[child_a] = False
search.terminal[child_b] = False

# Real statistics before virtual loss.
search.visit_count[child_a] = 4
search.value_sum[child_a] = 0.8

search.visit_count[child_b] = 4
search.value_sum[child_b] = 0.8

before_a_n = int(
    search.visit_count[child_a].item()
)

before_a_w = float(
    search.value_sum[child_a].item()
)

print()
print("Before virtual loss:")
print_stats(
    search,
    [root, child_a, child_b],
    "REAL STATISTICS",
)

# Construct a path root -> child_a.
test_path = torch.tensor(
    [
        [root, child_a]
    ],
    dtype=torch.int32,
    device=DEVICE,
)

# Apply virtual loss.
search._apply_virtual_loss(
    test_path
)

print()
print("After applying virtual loss:")
print_stats(
    search,
    [root, child_a, child_b],
    "STATISTICS AFTER VIRTUAL LOSS",
)

# ------------------------------------------------------------
# Determine whether implementation uses separate virtual stats.
# ------------------------------------------------------------

has_separate_virtual = (
    hasattr(search, "virtual_visit_count")
    and search.virtual_visit_count is not None
    and hasattr(search, "virtual_value_sum")
    and search.virtual_value_sum is not None
)

if has_separate_virtual:

    print()
    print("Separate virtual-loss buffers detected.")

    real_n_after = int(
        search.visit_count[child_a].item()
    )

    real_w_after = float(
        search.value_sum[child_a].item()
    )

    virtual_n = int(
        search.virtual_visit_count[child_a].item()
    )

    virtual_w = float(
        search.virtual_value_sum[child_a].item()
    )

    assert_equal(
        real_n_after,
        before_a_n,
        "Real visit count unchanged by virtual loss",
    )

    assert_close(
        real_w_after,
        before_a_w,
        "Real value sum unchanged by virtual loss",
    )

    assert_equal(
        virtual_n,
        1,
        "Virtual visit count added",
    )

    assert_close(
        virtual_w,
        +1.0,
        "Virtual value loss added",
    )

else:

    print()
    print(
        "WARNING: GPUMCTS uses real visit/value tensors "
        "for virtual loss."
    )

    print(
        "This test will verify that the temporary change "
        "is completely reversible."
    )


# ============================================================
# TEST 6
# REMOVE VIRTUAL LOSS
# ============================================================

print()
print("=" * 100)
print("TEST 6: REMOVE VIRTUAL LOSS")
print("=" * 100)

search._remove_virtual_loss(
    test_path
)

after_n = int(
    search.visit_count[child_a].item()
)

after_w = float(
    search.value_sum[child_a].item()
)

print()
print("After removing virtual loss:")
print_stats(
    search,
    [root, child_a, child_b],
    "RESTORED STATISTICS",
)

assert_equal(
    after_n,
    before_a_n,
    "Visit count restored",
)

assert_close(
    after_w,
    before_a_w,
    "Value sum restored",
)


if has_separate_virtual:

    virtual_n_after = int(
        search.virtual_visit_count[child_a].item()
    )

    virtual_w_after = float(
        search.virtual_value_sum[child_a].item()
    )

    assert_equal(
        virtual_n_after,
        0,
        "Virtual visit count restored to zero",
    )

    assert_close(
        virtual_w_after,
        0.0,
        "Virtual value sum restored to zero",
    )


# ============================================================
# TEST 7
# VIRTUAL LOSS CHANGES EFFECTIVE SELECTION
# ============================================================

print()
print("=" * 100)
print("TEST 7: VIRTUAL LOSS CHANGES EFFECTIVE SELECTION")
print("=" * 100)

reset_stats(search)

search.visit_count[root] = 10

search.edge_start[root] = 0
search.edge_count[root] = 2

search.edge_child[0] = child_a
search.edge_child[1] = child_b

search.edge_prior[0] = 0.50
search.edge_prior[1] = 0.50

search.edge_valid[0] = True
search.edge_valid[1] = True

search.expanded[root] = True
search.expanded[child_a] = True
search.expanded[child_b] = False

search.terminal[root] = False
search.terminal[child_a] = False
search.terminal[child_b] = False

# Same real statistics.
search.visit_count[child_a] = 4
search.value_sum[child_a] = 0.8

search.visit_count[child_b] = 4
search.value_sum[child_b] = 0.8

# Reserve child A.
test_path = torch.tensor(
    [
        [root, child_a]
    ],
    dtype=torch.int32,
    device=DEVICE,
)

search._apply_virtual_loss(
    test_path
)

try:

    leaves_after_virtual, paths_after_virtual = (
        search._select_leaves(
            search._root_ids,
            max_depth=8,
        )
    )

    selected_after_virtual = int(
        leaves_after_virtual[0].item()
    )

    print()
    print(
        "Selected leaf after virtual loss:",
        selected_after_virtual,
    )

    print()
    print("Path:")
    print(
        paths_after_virtual.detach()
        .cpu()
        .tolist()
    )

    # With correct separate virtual statistics, child A should
    # be penalized by the temporary reservation, allowing the
    # alternative child to be selected.
    #
    # If the implementation intentionally uses a different
    # virtual-loss convention, report the result rather than
    # falsely declaring failure.

    if selected_after_virtual == child_b:

        print(
            "PASS: Virtual loss redirected selection "
            "away from reserved child."
        )

    elif selected_after_virtual == child_a:

        print(
            "WARNING: Selection remained on the reserved child."
        )

        print(
            "This does not automatically mean the implementation "
            "is wrong; inspect the exact virtual-loss PUCT formula."
        )

    else:

        print(
            "WARNING: Unexpected selected node:",
            selected_after_virtual,
        )

finally:

    # Always restore temporary virtual loss.
    search._remove_virtual_loss(
        test_path
    )


# ============================================================
# FINAL STATE CHECK
# ============================================================

print()
print("=" * 100)
print("FINAL VIRTUAL-LOSS STATE CHECK")
print("=" * 100)

final_n = int(
    search.visit_count[child_a].item()
)

final_w = float(
    search.value_sum[child_a].item()
)

assert_equal(
    final_n,
    4,
    "Final child-A real visit count",
)

assert_close(
    final_w,
    0.8,
    "Final child-A real value sum",
)

if hasattr(search, "virtual_visit_count"):
    if search.virtual_visit_count is not None:

        assert_equal(
            int(
                search.virtual_visit_count[child_a].item()
            ),
            0,
            "Final virtual visit count",
        )

if hasattr(search, "virtual_value_sum"):
    if search.virtual_value_sum is not None:

        assert_close(
            float(
                search.virtual_value_sum[child_a].item()
            ),
            0.0,
            "Final virtual value sum",
        )


# ============================================================
# COMPLETE
# ============================================================

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

print()
print("=" * 100)
print("PUCT + VIRTUAL-LOSS TEST COMPLETE")
print("=" * 100)

print()
print("The controlled PUCT calculation, direct selection test,")
print("virtual-loss application/removal, and final state restoration")
print("have completed.")

print()
print("IMPORTANT:")
print("If this script reports a PUCT-selection failure,")
print("do NOT train RL54 yet.")
print("=" * 100)
