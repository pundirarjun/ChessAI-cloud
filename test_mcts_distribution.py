import os
import subprocess
from pathlib import Path

PROJECT = Path("/kaggle/working/chess-zero")

print("=" * 100)
print("SELF-PLAY FILE / GIT CONSISTENCY DIAGNOSTIC")
print("=" * 100)

def run(cmd):
    print()
    print("$", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=PROJECT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout)
    return result.stdout

# ------------------------------------------------------------
# 1. FILE IDENTITY
# ------------------------------------------------------------

print()
print("=" * 100)
print("1. CURRENT FILE IDENTITY")
print("=" * 100)

files = [
    PROJECT / "training" / "self_play.py",
    PROJECT / "mcts" / "gpu_mcts.py",
    PROJECT / "training" / "rl_training.py",
    PROJECT / "training" / "iteration.py",
]

for path in files:

    print()
    print("FILE:", path)

    if not path.exists():
        print("  DOES NOT EXIST")
        continue

    data = path.read_bytes()

    print("  Bytes:", len(data))
    print("  SHA256:", __import__("hashlib").sha256(data).hexdigest())

    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    print("  Lines:", len(lines))

    print("  First line:")
    print("   ", lines[0] if lines else "<empty>")

    print("  First 5 lines:")
    for line in lines[:5]:
        print("   ", line)


# ------------------------------------------------------------
# 2. CHECK WHETHER self_play.py == gpu_mcts.py
# ------------------------------------------------------------

print()
print("=" * 100)
print("2. IS training/self_play.py IDENTICAL TO mcts/gpu_mcts.py?")
print("=" * 100)

self_play = PROJECT / "training" / "self_play.py"
gpu_mcts = PROJECT / "mcts" / "gpu_mcts.py"

if self_play.exists() and gpu_mcts.exists():

    a = self_play.read_bytes()
    b = gpu_mcts.read_bytes()

    if a == b:
        print("!!! EXACT MATCH !!!")
        print(
            "training/self_play.py and mcts/gpu_mcts.py "
            "are byte-for-byte identical."
        )
    else:
        print(
            "They are NOT byte-for-byte identical."
        )

else:
    print("One or both files are missing.")


# ------------------------------------------------------------
# 3. GIT STATUS
# ------------------------------------------------------------

print()
print("=" * 100)
print("3. GIT STATUS")
print("=" * 100)

run([
    "git",
    "status",
    "--short",
])

run([
    "git",
    "branch",
    "--show-current",
])

run([
    "git",
    "log",
    "-5",
    "--oneline",
])


# ------------------------------------------------------------
# 4. WHAT DOES GIT THINK self_play.py IS?
# ------------------------------------------------------------

print()
print("=" * 100)
print("4. GIT VERSION OF training/self_play.py")
print("=" * 100)

run([
    "git",
    "show",
    "HEAD:training/self_play.py",
])


# ------------------------------------------------------------
# 5. GIT VERSION OF mcts/gpu_mcts.py
# ------------------------------------------------------------

print()
print("=" * 100)
print("5. GIT VERSION OF mcts/gpu_mcts.py")
print("=" * 100)

run([
    "git",
    "show",
    "HEAD:mcts/gpu_mcts.py",
])


# ------------------------------------------------------------
# 6. SEARCH ALL COMMITS FOR play_games_multi_gpu
# ------------------------------------------------------------

print()
print("=" * 100)
print("6. GIT HISTORY SEARCH FOR play_games_multi_gpu")
print("=" * 100)

run([
    "git",
    "log",
    "--all",
    "-Splay_games_multi_gpu",
    "--oneline",
    "--",
    "training/self_play.py",
])

# ------------------------------------------------------------
# 7. SEARCH ALL COMMITS FOR _play_games_gpu
# ------------------------------------------------------------

print()
print("=" * 100)
print("7. GIT HISTORY SEARCH FOR _play_games_gpu")
print("=" * 100)

run([
    "git",
    "log",
    "--all",
    "-S_play_games_gpu",
    "--oneline",
    "--",
    "training/self_play.py",
])

# ------------------------------------------------------------
# 8. CURRENT rl_training IMPORT
# ------------------------------------------------------------

print()
print("=" * 100)
print("8. rl_training.py SELF-PLAY IMPORT")
print("=" * 100)

rl = PROJECT / "training" / "rl_training.py"

if rl.exists():

    lines = rl.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    for i, line in enumerate(lines, 1):

        if (
            "self_play" in line.lower()
            or "play_games" in line.lower()
        ):

            print(
                f"{i:5d}: {line}"
            )


# ------------------------------------------------------------
# 9. TRY THE EXACT IMPORT USED BY RL TRAINING
# ------------------------------------------------------------

print()
print("=" * 100)
print("9. EXACT RL TRAINING IMPORT TEST")
print("=" * 100)

try:

    from training.self_play import play_games_multi_gpu

    print(
        "PASS: play_games_multi_gpu imported successfully"
    )

except Exception as e:

    print(
        "FAIL: exact RL training import failed"
    )

    print(
        type(e).__name__ + ":",
        str(e)
    )


# ------------------------------------------------------------
# 10. FINAL
# ------------------------------------------------------------

print()
print("=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)

print()
print(
    "Do NOT modify any files."
)

print(
    "Send the COMPLETE output."
)

print(
    "The next step depends on whether Git contains the "
    "correct self_play.py or whether it was overwritten."
)

print("=" * 100)