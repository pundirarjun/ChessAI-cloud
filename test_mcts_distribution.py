# ============================================================
# FIND THE ACTUAL SELF-PLAY IMPLEMENTATION
# ============================================================

import os
import sys
import ast
import inspect
import importlib
from pathlib import Path

PROJECT = Path("/kaggle/working/chess-zero")

print("=" * 100)
print("ACTUAL SELF-PLAY SOURCE DIAGNOSTIC")
print("=" * 100)

print()
print("PROJECT:", PROJECT)

# ------------------------------------------------------------
# 1. CHECK FILES
# ------------------------------------------------------------

print()
print("=" * 100)
print("1. SELF-PLAY RELATED FILES")
print("=" * 100)

for path in sorted(PROJECT.rglob("*.py")):
    name = str(path.relative_to(PROJECT)).lower()

    if (
        "self_play" in name
        or "rl_training" in name
        or "iteration" in name
        or "main.py" in name
    ):
        print(path.relative_to(PROJECT))


# ------------------------------------------------------------
# 2. READ ACTUAL training/self_play.py
# ------------------------------------------------------------

SELF_PLAY = PROJECT / "training" / "self_play.py"

print()
print("=" * 100)
print("2. ACTUAL training/self_play.py")
print("=" * 100)

print("Exists:", SELF_PLAY.exists())

if SELF_PLAY.exists():

    text = SELF_PLAY.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    print("Lines:", len(lines))
    print("Bytes:", SELF_PLAY.stat().st_size)

    print()
    print("FIRST 80 LINES")
    print("-" * 100)

    for i, line in enumerate(lines[:80], start=1):
        print(
            f"{i:4d}: {line}"
        )

    print()
    print("FUNCTIONS / CLASSES")
    print("-" * 100)

    try:

        tree = ast.parse(text)

        for node in tree.body:

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):

                kind = (
                    "CLASS"
                    if isinstance(node, ast.ClassDef)
                    else "FUNCTION"
                )

                print(
                    f"{kind:8s} "
                    f"{node.name:40s} "
                    f"line={node.lineno}"
                )

    except Exception as e:

        print(
            "AST parse failed:",
            repr(e)
        )

else:

    print(
        "training/self_play.py DOES NOT EXIST"
    )


# ------------------------------------------------------------
# 3. SEARCH ALL PYTHON FILES FOR SELF-PLAY FUNCTIONS
# ------------------------------------------------------------

print()
print("=" * 100)
print("3. SEARCH ENTIRE PROJECT FOR SELF-PLAY FUNCTIONS")
print("=" * 100)

keywords = [
    "self_play",
    "selfplay",
    "play_games",
    "play_game",
    "generate_self_play",
    "SelfPlay",
]

matches = []

for path in PROJECT.rglob("*.py"):

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        continue

    lines = text.splitlines()

    for lineno, line in enumerate(
        lines,
        start=1,
    ):

        lower = line.lower()

        if any(
            keyword.lower() in lower
            for keyword in keywords
        ):

            matches.append(
                (
                    path.relative_to(PROJECT),
                    lineno,
                    line.strip(),
                )
            )


for path, lineno, line in matches:

    print(
        f"{str(path):60s} "
        f"line {lineno:5d}: "
        f"{line}"
    )

print()
print(
    "Total matching lines:",
    len(matches)
)


# ------------------------------------------------------------
# 4. INSPECT ACTUAL IMPORTED MODULE
# ------------------------------------------------------------

print()
print("=" * 100)
print("4. IMPORTED training.self_play MODULE")
print("=" * 100)

sys.path.insert(
    0,
    str(PROJECT)
)

import training.self_play as sp

print(
    "Module:",
    sp
)

print(
    "File:",
    getattr(
        sp,
        "__file__",
        None,
    )
)

print()
print("ALL PUBLIC / PRIVATE NAMES:")

names = [
    x
    for x in dir(sp)
    if not x.startswith("__")
]

for name in names:
    try:
        obj = getattr(
            sp,
            name,
        )

        if callable(obj):
            print(
                f"CALLABLE  {name}"
            )
        else:
            print(
                f"OBJECT    {name}"
            )

    except Exception:
        print(
            f"ERROR     {name}"
        )


# ------------------------------------------------------------
# 5. INSPECT ACTUAL rl_training.py
# ------------------------------------------------------------

RL_TRAINING = PROJECT / "training" / "rl_training.py"

print()
print("=" * 100)
print("5. ACTUAL training/rl_training.py")
print("=" * 100)

print(
    "Exists:",
    RL_TRAINING.exists()
)

if RL_TRAINING.exists():

    text = RL_TRAINING.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    print(
        "Lines:",
        len(lines)
    )

    print()
    print("SELF-PLAY RELATED LINES")
    print("-" * 100)

    found = False

    for i, line in enumerate(
        lines,
        start=1,
    ):

        lower = line.lower()

        if (
            "self_play" in lower
            or "selfplay" in lower
            or "play_games" in lower
            or "self play" in lower
        ):

            print(
                f"{i:5d}: {line}"
            )

            found = True

    if not found:
        print(
            "NO SELF-PLAY REFERENCES FOUND"
        )

    print()
    print("IMPORT LINES")
    print("-" * 100)

    for i, line in enumerate(
        lines,
        start=1,
    ):

        stripped = line.strip()

        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
        ):

            print(
                f"{i:5d}: {line}"
            )


# ------------------------------------------------------------
# 6. SEARCH FOR CALLABLES IN rl_training AST
# ------------------------------------------------------------

print()
print("=" * 100)
print("6. FUNCTIONS / CLASSES IN rl_training.py")
print("=" * 100)

if RL_TRAINING.exists():

    try:

        text = RL_TRAINING.read_text(
            encoding="utf-8",
            errors="replace",
        )

        tree = ast.parse(text)

        for node in tree.body:

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):

                kind = (
                    "CLASS"
                    if isinstance(
                        node,
                        ast.ClassDef,
                    )
                    else "FUNCTION"
                )

                print(
                    f"{kind:8s} "
                    f"{node.name:50s} "
                    f"line={node.lineno}"
                )

    except Exception as e:

        print(
            "AST error:",
            repr(e)
        )


# ------------------------------------------------------------
# 7. SEARCH FOR ALL IMPORTS OF self_play
# ------------------------------------------------------------

print()
print("=" * 100)
print("7. ALL IMPORTS REFERENCING SELF-PLAY")
print("=" * 100)

for path in PROJECT.rglob("*.py"):

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        continue

    for i, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        lower = line.lower()

        if (
            "from training.self_play" in lower
            or "import training.self_play" in lower
            or "from .self_play" in lower
            or "import self_play" in lower
        ):

            print(
                f"{str(path.relative_to(PROJECT)):60s} "
                f"line {i:5d}: "
                f"{line.strip()}"
            )


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print()
print("=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)

print()
print(
    "Do NOT change any project files yet."
)

print(
    "Send me the complete output."
)

print(
    "We will identify the exact self-play function "
    "from your actual Kaggle project before running "
    "another MCTS test."
)

print("=" * 100)