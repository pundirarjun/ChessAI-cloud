
# ============================================================
# GPU MCTS BACKUP-SIGN UNIT TEST
# ============================================================
#
# Purpose:
#   Verify the sign convention used by GPUMCTS._backup_batched()
#   independently of:
#       - neural-network predictions
#       - PUCT selection
#       - chess move generation
#       - replay data
#       - MCTS expansion
#
# Expected AlphaZero-style perspective convention:
#
#   Leaf value = +V
#
#   Path [root, child]:
#       child = +V
#       root  = -V
#
#   Path [root, child, grandchild]:
#       grandchild = +V
#       child      = -V
#       root       = +V
#
# If these assertions pass, the backup sign logic itself is correct.
# ============================================================

import os
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

TEST_VALUE = 0.50

print("=" * 100)
print("GPU MCTS BACKUP-SIGN UNIT TEST")
print("=" * 100)

print()
print("Project:", PROJECT_DIR)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

# ============================================================
# IMPORT PROJECT
# ============================================================

sys.path.insert(0, PROJECT_DIR)

from model.chess_net import ChessNet
from mcts.gpu_mcts import GPUMCTS


# ============================================================
# HELPERS
# ============================================================

def reset_stats(search):
    """
    Clear real MCTS visit/value statistics.

    We deliberately do not run search() here.
    This test targets _backup_batched() directly.
    """
    search.visit_count.zero_()
    search.value_sum.zero_()

    # Some versions of the corrected MCTS implementation have
    # separate virtual-loss statistics. Clear them if present.
    if hasattr(search, "virtual_visit_count"):
        if search.virtual_visit_count is not None:
            search.virtual_visit_count.zero_()

    if hasattr(search, "virtual_value_sum"):
        if search.virtual_value_sum is not None:
            search.virtual_value_sum.zero_()


def get_stats(search, node_ids):
    """
    Read visit count and value sum for selected node IDs.
    """
    node_ids = torch.tensor(
        node_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    visits = search.visit_count[node_ids]
    values = search.value_sum[node_ids]

    return (
        visits.detach().cpu().tolist(),
        values.detach().cpu().tolist(),
    )


def assert_close(actual, expected, name, tolerance=1e-6):
    """
    Assert a scalar floating-point result.
    """
    actual = float(actual)
    expected = float(expected)

    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{name} FAILED: "
            f"expected {expected:+.6f}, "
            f"got {actual:+.6f}"
        )

    print(
        f"PASS: {name}: "
        f"{actual:+.6f}"
    )


def assert_equal(actual, expected, name):
    """
    Assert an integer result.
    """
    actual = int(actual)
    expected = int(expected)

    if actual != expected:
        raise AssertionError(
            f"{name} FAILED: "
            f"expected {expected}, "
            f"got {actual}"
        )

    print(
        f"PASS: {name}: "
        f"{actual}"
    )


def print_node_stats(search, node_ids, title):
    """
    Print node statistics.
    """
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)

    visits, values = get_stats(
        search,
        node_ids,
    )

    for node, visit, value in zip(
        node_ids,
        visits,
        values,
    ):
        if visit > 0:
            q = value / visit
        else:
            q = 0.0

        print(
            f"Node {node:2d} | "
            f"N = {visit:3d} | "
            f"W = {value:+.6f} | "
            f"Q = {q:+.6f}"
        )


# ============================================================
# CREATE MCTS OBJECT
# ============================================================

print()
print("=" * 100)
print("CREATING GPUMCTS")
print("=" * 100)

# The neural network is NOT used during this test.
# We only need a valid model object because GPUMCTS expects one.
model = ChessNet().to(DEVICE)
model.eval()

search = GPUMCTS(
    model=model,
    device=DEVICE,
    c_puct=1.5,
)

# ============================================================
# ALLOCATE A TINY TEST TREE
# ============================================================

print()
print("=" * 100)
print("ALLOCATING TEST TREE")
print("=" * 100)

# One game, one simulation is enough for the direct backup test.
search._allocate(
    num_games=1,
    num_simulations=1,
)

root = 0
child = 1
grandchild = 2

print("Root node:", root)
print("Child node:", child)
print("Grandchild node:", grandchild)


# ============================================================
# TEST 1
# ROOT -> CHILD
# ============================================================

print()
print("=" * 100)
print("TEST 1: ROOT -> CHILD")
print("=" * 100)

reset_stats(search)

# One leaf path:
#
#   root -> child
#
# The value +0.50 is from the perspective of the player
# at the leaf node.
#
# Distance from leaf:
#
#   child = 0  -> +0.50
#   root  = 1  -> -0.50

paths = torch.tensor(
    [
        [
            [root, child],
        ]
    ],
    dtype=torch.int32,
    device=DEVICE,
)

values = torch.tensor(
    [
        [TEST_VALUE],
    ],
    dtype=torch.float32,
    device=DEVICE,
)

print()
print("Path:")
print("  root -> child")

print()
print("Leaf value:")
print(f"  {TEST_VALUE:+.6f}")

search._backup_batched(
    paths,
    values,
)

print_node_stats(
    search,
    [root, child],
    "RESULT — TEST 1",
)

visits, sums = get_stats(
    search,
    [root, child],
)

assert_equal(
    visits[0],
    1,
    "Root visit count",
)

assert_equal(
    visits[1],
    1,
    "Child visit count",
)

assert_close(
    sums[0],
    -TEST_VALUE,
    "Root value sum",
)

assert_close(
    sums[1],
    +TEST_VALUE,
    "Child value sum",
)


# ============================================================
# TEST 2
# ROOT -> CHILD -> GRANDCHILD
# ============================================================

print()
print("=" * 100)
print("TEST 2: ROOT -> CHILD -> GRANDCHILD")
print("=" * 100)

reset_stats(search)

# Path:
#
#   root -> child -> grandchild
#
# Leaf value = +0.50
#
# Signs:
#
#   grandchild = +0.50
#   child      = -0.50
#   root       = +0.50

paths = torch.tensor(
    [
        [
            [root, child, grandchild],
        ]
    ],
    dtype=torch.int32,
    device=DEVICE,
)

values = torch.tensor(
    [
        [TEST_VALUE],
    ],
    dtype=torch.float32,
    device=DEVICE,
)

print()
print("Path:")
print("  root -> child -> grandchild")

print()
print("Leaf value:")
print(f"  {TEST_VALUE:+.6f}")

search._backup_batched(
    paths,
    values,
)

print_node_stats(
    search,
    [root, child, grandchild],
    "RESULT — TEST 2",
)

visits, sums = get_stats(
    search,
    [root, child, grandchild],
)

assert_equal(
    visits[0],
    1,
    "Root visit count",
)

assert_equal(
    visits[1],
    1,
    "Child visit count",
)

assert_equal(
    visits[2],
    1,
    "Grandchild visit count",
)

assert_close(
    sums[0],
    +TEST_VALUE,
    "Root value sum",
)

assert_close(
    sums[1],
    -TEST_VALUE,
    "Child value sum",
)

assert_close(
    sums[2],
    +TEST_VALUE,
    "Grandchild value sum",
)


# ============================================================
# TEST 3
# TWO DIFFERENT PATHS INTO THE SAME ROOT
# ============================================================

print()
print("=" * 100)
print("TEST 3: MULTIPLE BACKUPS")
print("=" * 100)

reset_stats(search)

# Two simulations through the same root:
#
# Simulation 1:
#   root -> child
#   leaf value = +0.50
#
# Simulation 2:
#   root -> child
#   leaf value = -0.20
#
# Expected:
#
#   child W = +0.50 + (-0.20) = +0.30
#
#   root receives the opposite values:
#   root W = -0.50 + (+0.20) = -0.30
#
# Both nodes have N = 2.

paths = torch.tensor(
    [
        [
            [root, child],
        ],
        [
            [root, child],
        ],
    ],
    dtype=torch.int32,
    device=DEVICE,
)

values = torch.tensor(
    [
        [+0.50],
        [-0.20],
    ],
    dtype=torch.float32,
    device=DEVICE,
)

search._backup_batched(
    paths,
    values,
)

print_node_stats(
    search,
    [root, child],
    "RESULT — TEST 3",
)

visits, sums = get_stats(
    search,
    [root, child],
)

assert_equal(
    visits[0],
    2,
    "Root visit count",
)

assert_equal(
    visits[1],
    2,
    "Child visit count",
)

assert_close(
    sums[0],
    -0.30,
    "Root accumulated value",
)

assert_close(
    sums[1],
    +0.30,
    "Child accumulated value",
)

assert_close(
    sums[0] / visits[0],
    -0.15,
    "Root Q",
)

assert_close(
    sums[1] / visits[1],
    +0.15,
    "Child Q",
)


# ============================================================
# TEST 4
# BATCH + VARIABLE PATH LENGTHS
# ============================================================

print()
print("=" * 100)
print("TEST 4: BATCHED VARIABLE-LENGTH PATHS")
print("=" * 100)

reset_stats(search)

# Two games in the batch:
#
# Game 1:
#   root -> child
#   value = +0.50
#
# Game 2:
#   root -> child -> grandchild
#   value = +0.25
#
# Padded paths use -1.
#
# Expected:
#
# Root:
#   Game 1: -0.50
#   Game 2: +0.25
#   total = -0.25
#
# Child:
#   Game 1: +0.50
#   Game 2: -0.25
#   total = +0.25
#
# Grandchild:
#   Game 2: +0.25

paths = torch.tensor(
    [
        [
            [root, child, -1],
        ],
        [
            [root, child, grandchild],
        ],
    ],
    dtype=torch.int32,
    device=DEVICE,
)

values = torch.tensor(
    [
        [+0.50],
        [+0.25],
    ],
    dtype=torch.float32,
    device=DEVICE,
)

search._backup_batched(
    paths,
    values,
)

print_node_stats(
    search,
    [root, child, grandchild],
    "RESULT — TEST 4",
)

visits, sums = get_stats(
    search,
    [root, child, grandchild],
)

assert_equal(
    visits[0],
    2,
    "Root visit count",
)

assert_equal(
    visits[1],
    2,
    "Child visit count",
)

assert_equal(
    visits[2],
    1,
    "Grandchild visit count",
)

assert_close(
    sums[0],
    -0.25,
    "Root accumulated value",
)

assert_close(
    sums[1],
    +0.25,
    "Child accumulated value",
)

assert_close(
    sums[2],
    +0.25,
    "Grandchild accumulated value",
)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 100)
print("BACKUP-SIGN TEST COMPLETE")
print("=" * 100)

print()
print("ALL BACKUP SIGN TESTS PASSED.")

print()
print("Verified:")
print("  [1] One-edge path sign")
print("  [2] Two-edge path sign")
print("  [3] Multiple backups")
print("  [4] Batched variable-length paths")

print()
print("Expected sign rule:")
print("  Leaf       = +V")
print("  Parent     = -V")
print("  Grandparent= +V")
print()

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

print("Conclusion:")
print("  GPUMCTS._backup_batched() follows the expected alternating")
print("  perspective/value sign convention for these controlled tests.")

print()
print("=" * 100)
